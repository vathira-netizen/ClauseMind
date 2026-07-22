"""Qdrant-backed institutional memory store.

This is the **only** module in this codebase that imports ``qdrant_client``.
Every other layer (agents, governance, api) reaches memory exclusively
through the functions exposed here, never through the client directly.
``memory/retrieval.py`` is the one exception, as a sibling within this same
subsystem: it builds queries on top of :func:`query_hybrid`/:func:`query_dense`
and never imports ``qdrant_client`` itself.

Three collections:

* ``clause_memory``         — every clause ever reviewed; dense + sparse
  named vectors for hybrid retrieval.
* ``playbook``               — the institution's standing negotiation
  positions per clause type.
* ``counterparty_profiles``  — per-counterparty negotiation history.

See CONTEXT.md for the architectural rationale behind hybrid retrieval and
the `redlined_then_accepted` outcome.
"""

from __future__ import annotations

import uuid
from datetime import date as date_type
from datetime import datetime
from functools import lru_cache
from typing import Any

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    ScoredPoint,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from precedent.config import get_settings

CLAUSE_MEMORY = "clause_memory"
PLAYBOOK = "playbook"
COUNTERPARTY_PROFILES = "counterparty_profiles"

_ALL_COLLECTIONS = (CLAUSE_MEMORY, PLAYBOOK, COUNTERPARTY_PROFILES)

VECTOR_SIZE = 384
DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"

# Loaded once per process, at import time, as module singletons. fastembed
# initialises an ONNX runtime session on construction, so re-instantiating
# per call would reload the model on every embed.
_dense_model = TextEmbedding(DENSE_MODEL_NAME)
_sparse_model = SparseTextEmbedding(SPARSE_MODEL_NAME)


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url)


def _embed_dense(texts: list[str]) -> list[list[float]]:
    return [vec.tolist() for vec in _dense_model.embed(texts)]


def _embed_sparse(texts: list[str]) -> list[SparseVector]:
    return [
        SparseVector(indices=vec.indices.tolist(), values=vec.values.tolist())
        for vec in _sparse_model.embed(texts)
    ]


def embed_dense_query(text: str) -> list[float]:
    """Embed a single query string as a dense vector only."""

    return _embed_dense([text])[0]


def embed_sparse_query(text: str) -> SparseVector:
    """Embed a single query string as a sparse vector only."""

    return _embed_sparse([text])[0]


def embed_query(text: str) -> tuple[list[float], SparseVector]:
    """Embed a single query string as both a dense and a sparse vector.

    Used by :mod:`precedent.memory.retrieval` to build hybrid queries
    without that module needing its own embedding models or fastembed import.
    """

    return embed_dense_query(text), embed_sparse_query(text)


def _coerce_date(value: Any) -> Any:
    if isinstance(value, (datetime, date_type)):
        return value.isoformat()
    return value


def init_collections() -> dict[str, bool]:
    """Create the three memory collections if they don't already exist.

    Idempotent: a collection that already exists is left completely alone
    (including its payload indexes), so this is safe to call on every
    deploy. Returns which collections were newly created this call.
    """

    client = _client()
    created: dict[str, bool] = {}

    if not client.collection_exists(CLAUSE_MEMORY):
        client.create_collection(
            collection_name=CLAUSE_MEMORY,
            vectors_config={"dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
        # These payload indexes are what let Qdrant apply a filter (tenant,
        # clause type, outcome, ...) DURING HNSW graph traversal instead of
        # running an unfiltered ANN search and discarding non-matching hits
        # afterwards. Without them, filtered recall degrades to
        # post-filtering and quality collapses as clause_memory grows. This
        # is a core architectural property of the memory layer, not a minor
        # optimisation.
        for field, schema in (
            ("clause_type", PayloadSchemaType.KEYWORD),
            ("counterparty_id", PayloadSchemaType.KEYWORD),
            ("negotiation_outcome", PayloadSchemaType.KEYWORD),
            ("tenant_id", PayloadSchemaType.KEYWORD),
            ("dpdp_relevant", PayloadSchemaType.BOOL),
        ):
            client.create_payload_index(CLAUSE_MEMORY, field_name=field, field_schema=schema)
        created[CLAUSE_MEMORY] = True
    else:
        created[CLAUSE_MEMORY] = False

    if not client.collection_exists(PLAYBOOK):
        client.create_collection(
            collection_name=PLAYBOOK,
            vectors_config={"dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)},
        )
        created[PLAYBOOK] = True
    else:
        created[PLAYBOOK] = False

    if not client.collection_exists(COUNTERPARTY_PROFILES):
        client.create_collection(
            collection_name=COUNTERPARTY_PROFILES,
            vectors_config={"dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)},
        )
        created[COUNTERPARTY_PROFILES] = True
    else:
        created[COUNTERPARTY_PROFILES] = False

    return created


def reset_collections() -> None:
    """Drop all three memory collections and recreate them from scratch.

    Unlike :func:`init_collections`, this is destructive: it always
    recreates every collection, discarding existing points and payload
    indexes. Used by the corpus seeder's ``--reset`` flag so a reseed starts
    from a clean slate rather than accumulating duplicate or stale points
    from a previous corpus version.
    """

    client = _client()
    for name in _ALL_COLLECTIONS:
        if client.collection_exists(name):
            client.delete_collection(name)
    init_collections()


def clause_point_id(contract_id: Any, clause_index: Any) -> str:
    """Derive a clause's deterministic clause_memory point ID.

    Shared by :func:`upsert_clauses` and by anything (e.g. the retrieval
    ablation harness) that needs to compute a clause's point ID without a
    round trip to Qdrant, so the derivation formula lives in exactly one
    place.
    """

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{contract_id}:{clause_index}"))


def upsert_clauses(clauses: list[dict[str, Any]], tenant_id: str) -> list[str]:
    """Embed and upsert clauses into ``clause_memory``.

    Point IDs are deterministic —
    ``uuid5(NAMESPACE_URL, f"{contract_id}:{clause_index}")`` — so retrying a
    failed write-back overwrites the existing point instead of duplicating
    it. Institutional memory must be idempotent to write to: a job retried
    after a timeout must not silently double a precedent's weight in future
    recall.

    Each dict in ``clauses`` is expected to provide: ``contract_id``,
    ``clause_text``, ``clause_type``, and optionally ``position`` (its
    ordinal index, used for ID derivation and defaulting to list order),
    ``counterparty_id``, ``date``, ``negotiation_outcome``,
    ``template_version``, ``governing_law``, ``risk_score``,
    ``dpdp_relevant``, ``original_text``, ``redline_rounds``.

    ``clause_text`` is embedded and is the text retrieval matches against —
    for historical corpus clauses this should be the final, actually-agreed
    text, since that is the useful precedent. ``original_text`` is carried
    through to the payload only (never embedded) so the demo can show the
    before/after negotiation delta. ``redline_rounds`` is likewise
    payload-only; :func:`precedent.memory.profiles.recompute_profile` reads
    it back to compute a counterparty's average rounds to close.
    """

    if not clauses:
        return []

    texts = [c["clause_text"] for c in clauses]
    dense_vectors = _embed_dense(texts)
    sparse_vectors = _embed_sparse(texts)

    points: list[PointStruct] = []
    point_ids: list[str] = []
    for i, (clause, dense_vec, sparse_vec) in enumerate(zip(clauses, dense_vectors, sparse_vectors)):
        contract_id = clause["contract_id"]
        clause_index = clause.get("position", i)
        point_id = clause_point_id(contract_id, clause_index)
        point_ids.append(point_id)

        payload = {
            "clause_type": clause.get("clause_type"),
            "contract_id": str(contract_id),
            "counterparty_id": clause.get("counterparty_id"),
            "date": _coerce_date(clause.get("date")),
            "negotiation_outcome": clause.get("negotiation_outcome"),
            "template_version": clause.get("template_version"),
            "governing_law": clause.get("governing_law"),
            "risk_score": clause.get("risk_score"),
            "dpdp_relevant": clause.get("dpdp_relevant", False),
            "tenant_id": tenant_id,
            "clause_text": clause["clause_text"],
            "original_text": clause.get("original_text"),
            "redline_rounds": clause.get("redline_rounds"),
        }
        points.append(
            PointStruct(id=point_id, vector={"dense": dense_vec, "sparse": sparse_vec}, payload=payload)
        )

    _client().upsert(collection_name=CLAUSE_MEMORY, points=points, wait=True)
    return point_ids


def upsert_playbook_positions(positions: list[dict[str, Any]]) -> list[str]:
    """Embed and upsert playbook positions into ``playbook``.

    One point per ``(clause_type, position_rank)`` slot, via a deterministic
    ID, so re-running a playbook update replaces that slot instead of
    accumulating stale duplicates alongside it.
    """

    if not positions:
        return []

    texts = [p["position_text"] for p in positions]
    dense_vectors = _embed_dense(texts)

    points: list[PointStruct] = []
    point_ids: list[str] = []
    for position, dense_vec in zip(positions, dense_vectors):
        clause_type = position["clause_type"]
        position_rank = position["position_rank"]
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"playbook:{clause_type}:{position_rank}"))
        point_ids.append(point_id)

        payload = {
            "clause_type": clause_type,
            "position_rank": position_rank,
            "position_text": position["position_text"],
            "walk_away": position.get("walk_away", False),
        }
        points.append(PointStruct(id=point_id, vector={"dense": dense_vec}, payload=payload))

    _client().upsert(collection_name=PLAYBOOK, points=points, wait=True)
    return point_ids


def upsert_counterparty_profile(profile: dict[str, Any]) -> str:
    """Embed and upsert a single counterparty profile.

    One point per ``counterparty_id``, via a deterministic ID, so re-upserting
    updates that counterparty's profile in place instead of creating a second
    record for the same counterparty.
    """

    counterparty_id = profile["counterparty_id"]
    text = profile.get("name") or str(counterparty_id)
    dense_vec = _embed_dense([text])[0]
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"counterparty:{counterparty_id}"))

    payload = {
        "counterparty_id": counterparty_id,
        "name": profile.get("name"),
        "contested_clauses": profile.get("contested_clauses", []),
        "concession_rate": profile.get("concession_rate"),
        "avg_rounds_to_close": profile.get("avg_rounds_to_close"),
        "deals_count": profile.get("deals_count", 0),
    }
    point = PointStruct(id=point_id, vector={"dense": dense_vec}, payload=payload)
    _client().upsert(collection_name=COUNTERPARTY_PROFILES, points=[point], wait=True)
    return point_id


def get_by_contract_id(contract_id: str) -> list[dict[str, Any]]:
    """Fetch every ``clause_memory`` point belonging to a contract.

    Used by the DPDP erasure path to locate everything tied to a contract
    before deleting it.
    """

    records, _ = _client().scroll(
        collection_name=CLAUSE_MEMORY,
        scroll_filter=Filter(
            must=[FieldCondition(key="contract_id", match=MatchValue(value=str(contract_id)))]
        ),
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    return [{"id": r.id, **(r.payload or {})} for r in records]


def get_by_counterparty_id(counterparty_id: str) -> list[dict[str, Any]]:
    """Fetch every ``clause_memory`` point belonging to a counterparty.

    Used by :func:`precedent.memory.profiles.recompute_profile` to compute a
    counterparty's concession rate, contested clause types, and average
    rounds to close from its full negotiation history.
    """

    records, _ = _client().scroll(
        collection_name=CLAUSE_MEMORY,
        scroll_filter=Filter(
            must=[FieldCondition(key="counterparty_id", match=MatchValue(value=str(counterparty_id)))]
        ),
        limit=5000,
        with_payload=True,
        with_vectors=False,
    )
    return [{"id": r.id, **(r.payload or {})} for r in records]


def delete_by_contract_id(contract_id: str) -> int:
    """Delete every ``clause_memory`` point belonging to a contract.

    This is the DPDP erasure path: a data-subject deletion request removes a
    contract's clauses from institutional memory by payload filter, since
    callers do not track individual point IDs long-term. Returns the number
    of points deleted.
    """

    contract_filter = Filter(
        must=[FieldCondition(key="contract_id", match=MatchValue(value=str(contract_id)))]
    )
    client = _client()
    matched = client.count(CLAUSE_MEMORY, count_filter=contract_filter, exact=True).count
    if matched:
        client.delete(collection_name=CLAUSE_MEMORY, points_selector=contract_filter, wait=True)
    return matched


def health() -> dict[str, int]:
    """Return each memory collection's name mapped to its point count.

    A collection that does not exist yet is reported with a count of 0
    rather than raising.
    """

    client = _client()
    counts: dict[str, int] = {}
    for name in _ALL_COLLECTIONS:
        counts[name] = client.count(name, exact=True).count if client.collection_exists(name) else 0
    return counts


def _equality_filter(**equals: Any) -> Filter | None:
    """Build an equality-only Filter from field=value keyword arguments.

    Fields whose value is None are omitted, so callers can pass an optional
    constraint straight through without an if-branch. Returns None (no
    filter) if every value was None.
    """

    conditions = [
        FieldCondition(key=field, match=MatchValue(value=value))
        for field, value in equals.items()
        if value is not None
    ]
    return Filter(must=conditions) if conditions else None


def query_hybrid(
    collection: str, dense_vector: list[float], sparse_vector: SparseVector, limit: int, **filters: Any
) -> list[ScoredPoint]:
    """Run a hybrid dense+sparse query fused by Reciprocal Rank Fusion.

    Both vector legs are prefetched to a wide candidate pool (30) and fused
    by RRF. ``filters`` are equality constraints (e.g. ``clause_type=...,
    tenant_id=...``); Qdrant applies them as a single ``query_filter`` DURING
    HNSW traversal on both legs rather than post-filtering an unfiltered
    top-k, which is what keeps filtered recall from degrading as
    clause_memory grows.
    """

    return _client().query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(query=dense_vector, using="dense", limit=30),
            Prefetch(query=sparse_vector, using="sparse", limit=30),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        query_filter=_equality_filter(**filters),
        limit=limit,
        with_payload=True,
    ).points


def query_dense(collection: str, dense_vector: list[float], limit: int, **filters: Any) -> list[ScoredPoint]:
    """Run a dense-only similarity query against a collection's ``dense`` vector.

    Used for the ``playbook`` collection, which carries no sparse leg.
    ``filters`` are equality constraints, same as :func:`query_hybrid`.
    """

    return _client().query_points(
        collection_name=collection,
        query=dense_vector,
        using="dense",
        query_filter=_equality_filter(**filters),
        limit=limit,
        with_payload=True,
    ).points


def query_sparse(collection: str, sparse_vector: SparseVector, limit: int, **filters: Any) -> list[ScoredPoint]:
    """Run a sparse-only (exact legal-term overlap) query against a collection's ``sparse`` vector.

    Used by the retrieval ablation harness to isolate the sparse leg's
    contribution from the dense/hybrid configurations. ``filters`` are
    equality constraints, same as :func:`query_hybrid`.
    """

    return _client().query_points(
        collection_name=collection,
        query=sparse_vector,
        using="sparse",
        query_filter=_equality_filter(**filters),
        limit=limit,
        with_payload=True,
    ).points
