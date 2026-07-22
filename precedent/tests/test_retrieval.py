"""Integration tests for the hybrid retrieval layer.

Read CONTEXT.md first.

Run against the live, already-seeded Qdrant instance (see docker-compose.yml
and scripts/seed_memory.py), for the same reason as test_store.py: store.py
is the only module that owns a qdrant_client, so there is nothing to
substitute it with without duplicating that client logic here.
"""

from __future__ import annotations

import pytest

from precedent.memory import retrieval, store

QUERY_TEXT = "A standard commercial contract clause under negotiation."


@pytest.fixture(scope="module")
def sample_record() -> dict:
    """One real clause_memory record.

    Test parameters are derived from this rather than a hardcoded
    counterparty/outcome, so the tests stay valid even if the corpus is
    regenerated.
    """

    records, _ = store._client().scroll(store.CLAUSE_MEMORY, limit=1, with_payload=True)
    assert records, "clause_memory is empty — run scripts/seed_memory.py first"
    return records[0].payload


def test_counterparty_filtered_query_returns_only_that_counterparty(sample_record) -> None:
    counterparty_id = sample_record["counterparty_id"]
    clause_type = sample_record["clause_type"]

    results = retrieval.find_counterparty_history(
        QUERY_TEXT, clause_type, counterparty_id, tenant_id="demo", k=25
    )

    assert results
    assert all(p.counterparty_id == counterparty_id for p in results)


def test_outcome_filtered_query_returns_only_requested_state(sample_record) -> None:
    counterparty_id = sample_record["counterparty_id"]
    clause_type = sample_record["clause_type"]
    outcome = sample_record["negotiation_outcome"]

    results = retrieval.find_counterparty_history(
        QUERY_TEXT, clause_type, counterparty_id, tenant_id="demo", k=25, outcome=outcome
    )

    assert results
    assert all(p.negotiation_outcome.value == outcome for p in results)


def test_counterparty_and_outcome_filters_execute_in_a_single_query(sample_record, monkeypatch) -> None:
    # find_counterparty_history must issue exactly one store.query_hybrid
    # call carrying clause_type, tenant_id, counterparty_id, AND
    # negotiation_outcome together as one query_filter. Qdrant evaluates that
    # single Filter DURING HNSW graph traversal on both the dense and sparse
    # legs, rather than running an unfiltered top-k search and sifting the
    # results in Python afterward — the latter would silently drop matches
    # that fall outside the unfiltered top-k entirely.
    counterparty_id = sample_record["counterparty_id"]
    clause_type = sample_record["clause_type"]
    outcome = sample_record["negotiation_outcome"]

    calls: list[dict] = []
    original_query_hybrid = store.query_hybrid

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original_query_hybrid(*args, **kwargs)

    monkeypatch.setattr(store, "query_hybrid", spy)

    results = retrieval.find_counterparty_history(
        QUERY_TEXT, clause_type, counterparty_id, tenant_id="demo", k=25, outcome=outcome
    )

    assert len(calls) == 1
    assert calls[0]["clause_type"] == clause_type
    assert calls[0]["tenant_id"] == "demo"
    assert calls[0]["counterparty_id"] == counterparty_id
    assert calls[0]["negotiation_outcome"] == outcome
    assert results
    assert all(
        p.counterparty_id == counterparty_id and p.negotiation_outcome.value == outcome
        for p in results
    )
