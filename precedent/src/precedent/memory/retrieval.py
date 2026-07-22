"""The three retrieval paths over institutional memory.

Read CONTEXT.md first.

Builds queries on top of :func:`precedent.memory.store.query_hybrid` and
:func:`precedent.memory.store.query_dense` — this module never imports
``qdrant_client`` itself; store.py owns the client and query mechanics.

Every :class:`~precedent.models.Precedent` returned here carries ``point_id``,
``clause_text``, ``counterparty_id``, ``negotiation_outcome``, ``date``, and
``score``. The Citation Critic (Phase 5+) verifies claims against exactly
these point IDs, so dropping one silently breaks citation traceability.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from precedent.memory import store
from precedent.models import NegotiationOutcome, Precedent


def _to_precedent(hit: Any) -> Precedent:
    payload = hit.payload or {}
    return Precedent(
        point_id=str(hit.id),
        clause_text=payload["clause_text"],
        counterparty_id=payload["counterparty_id"],
        negotiation_outcome=NegotiationOutcome(payload["negotiation_outcome"]),
        date=datetime.fromisoformat(payload["date"]),
        score=hit.score,
    )


COUNTERPARTY_MATCH_THRESHOLD = 0.9


def resolve_counterparty_id(name: str) -> str | None:
    """Resolve a counterparty display name to its internal counterparty_id.

    Bridges an incoming document's extracted party name (e.g. "Meridian
    Systems") to the slug clause_memory/counterparty_profiles key it's
    stored under (e.g. "meridian-systems"). A name we've genuinely seen
    before scores far above the threshold (empirically ~1.0 for an exact
    match vs. ~0.6-0.7 for an unrelated name); returns None below that
    threshold rather than guessing, since a low-confidence match would
    silently attribute a stranger's history to the wrong counterparty. None
    is the correct, expected result for a genuinely new counterparty with no
    negotiation history on file — not an error.
    """

    dense_vec = store.embed_dense_query(name)
    hits = store.query_dense(store.COUNTERPARTY_PROFILES, dense_vec, limit=1)
    if not hits or hits[0].score < COUNTERPARTY_MATCH_THRESHOLD:
        return None
    return hits[0].payload["counterparty_id"]


def find_similar_clauses(
    clause_text: str, clause_type: str, tenant_id: str, k: int = 5
) -> list[Precedent]:
    """Hybrid search for clauses of this type, scoped to one tenant.

    Filtered by ``clause_type`` and ``tenant_id`` in the same query as the
    dense+sparse RRF search (see :func:`store.query_hybrid`) — Qdrant applies
    both constraints during HNSW traversal rather than after.
    """

    dense_vec, sparse_vec = store.embed_query(clause_text)
    hits = store.query_hybrid(
        store.CLAUSE_MEMORY, dense_vec, sparse_vec, limit=k, clause_type=clause_type, tenant_id=tenant_id
    )
    return [_to_precedent(h) for h in hits]


def find_counterparty_history(
    clause_text: str,
    clause_type: str,
    counterparty_id: str,
    tenant_id: str,
    k: int = 5,
    outcome: str | None = None,
) -> list[Precedent]:
    """Hybrid search scoped to one counterparty's history for this clause type.

    ``outcome``, if given, adds a ``negotiation_outcome`` equality constraint
    to the same filtered query — so "clauses we previously rejected from this
    counterparty" is one filtered call, not a search followed by a Python-side
    sift over the results.
    """

    dense_vec, sparse_vec = store.embed_query(clause_text)
    hits = store.query_hybrid(
        store.CLAUSE_MEMORY,
        dense_vec,
        sparse_vec,
        limit=k,
        clause_type=clause_type,
        tenant_id=tenant_id,
        counterparty_id=counterparty_id,
        negotiation_outcome=outcome,
    )
    return [_to_precedent(h) for h in hits]


def get_playbook_position(clause_text: str, clause_type: str) -> dict[str, Any]:
    """Dense-only search of the playbook for this clause type.

    Returns ``{"preferred": {...} | None, "fallbacks": [...], "walk_away":
    {...} | None}``. Each entry carries ``point_id``, ``position_rank``,
    ``position_text``, ``walk_away``, and ``score``. There are only four
    playbook positions per clause type (Prompt 2.1), so ``limit=10``
    comfortably retrieves all of them in rank order.
    """

    dense_vec, _ = store.embed_query(clause_text)
    hits = store.query_dense(store.PLAYBOOK, dense_vec, limit=10, clause_type=clause_type)

    positions = sorted(
        (
            {
                "point_id": str(h.id),
                "position_rank": h.payload["position_rank"],
                "position_text": h.payload["position_text"],
                "walk_away": h.payload["walk_away"],
                "score": h.score,
            }
            for h in hits
        ),
        key=lambda p: p["position_rank"],
    )

    preferred = next((p for p in positions if p["position_rank"] == 0), None)
    fallbacks = [p for p in positions if not p["walk_away"] and p["position_rank"] != 0]
    walk_away = next((p for p in positions if p["walk_away"]), None)
    return {"preferred": preferred, "fallbacks": fallbacks, "walk_away": walk_away}


def playbook_distance(clause_text: str, clause_type: str) -> float:
    """Cosine distance from ``clause_text`` to the preferred playbook position.

    The numeric deviation signal the Deviation & Risk Worker (Phase 5)
    consumes: 0.0 means the clause matches our preferred position almost
    exactly, 1.0 means no similarity at all. Qdrant's COSINE distance metric
    returns a similarity score, so distance is ``1 - score``.

    Returns 1.0 (maximal deviation) if this clause type has no playbook
    position on file at all, rather than raising.
    """

    preferred = get_playbook_position(clause_text, clause_type)["preferred"]
    if preferred is None:
        return 1.0
    return round(1.0 - preferred["score"], 4)
