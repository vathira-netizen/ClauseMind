"""Integration tests for the Qdrant-backed memory store.

These run against a live Qdrant instance (see docker-compose.yml) rather than
a mock, since store.py is deliberately the only place in the codebase that
constructs a qdrant_client — there is nothing to substitute it with without
duplicating that client logic in the test suite.
"""

from __future__ import annotations

import uuid

import pytest

from precedent.memory import store


@pytest.fixture(scope="module", autouse=True)
def _ensure_collections() -> None:
    store.init_collections()


def test_clause_memory_has_dense_and_sparse_named_vectors() -> None:
    info = store._client().get_collection(store.CLAUSE_MEMORY)

    dense = info.config.params.vectors["dense"]
    assert dense.size == store.VECTOR_SIZE

    assert "sparse" in info.config.params.sparse_vectors


def test_clause_memory_payload_indexes_exist() -> None:
    info = store._client().get_collection(store.CLAUSE_MEMORY)

    assert set(info.payload_schema.keys()) == {
        "clause_type",
        "counterparty_id",
        "negotiation_outcome",
        "tenant_id",
        "dpdp_relevant",
    }


def test_upsert_clauses_is_idempotent() -> None:
    contract_id = str(uuid.uuid4())
    clause = {
        "contract_id": contract_id,
        "clause_text": "Liability shall not exceed fees paid in the preceding 12 months.",
        "clause_type": "limitation_of_liability",
        "counterparty_id": "acme-corp",
        "negotiation_outcome": "accepted",
    }

    try:
        first_ids = store.upsert_clauses([clause], tenant_id="test-tenant")
        second_ids = store.upsert_clauses([clause], tenant_id="test-tenant")

        assert first_ids == second_ids

        records = store.get_by_contract_id(contract_id)
        assert len(records) == 1
        assert records[0]["clause_text"] == clause["clause_text"]
    finally:
        store.delete_by_contract_id(contract_id)
