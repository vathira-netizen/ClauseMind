from __future__ import annotations

import uuid

import pytest

from precedent.governance import approval_gate
from precedent.memory import profiles, retrieval, store


@pytest.fixture(autouse=True)
def collections() -> None:
    store.init_collections()


def test_write_back_is_retrievable_idempotent_and_refreshes_profile() -> None:
    contract_id = str(uuid.uuid4())
    report = {
        "contract_id": contract_id,
        "counterparty_id": "meridian-systems",
        "counterparty_name": "Meridian Systems",
        "clauses": [{
            "position": 0,
            "clause_type": "payment",
            "text": "Invoices are payable net thirty days after receipt.",
            "final_text": "Invoices are payable net thirty days after receipt.",
            "negotiation_outcome": "redlined_then_accepted",
            "redline_rounds": 2,
        }],
    }
    before = profiles.recompute_profile("meridian-systems")["deals_count"]
    try:
        first = approval_gate.approve("review-1", report, "reviewer", "demo")
        second = approval_gate.approve("review-1", report, "reviewer", "demo")
        assert first == second == [store.clause_point_id(contract_id, 0)]
        hits = retrieval.find_counterparty_history(
            report["clauses"][0]["final_text"], "payment", "meridian-systems", tenant_id="demo"
        )
        assert any(hit.point_id == first[0] for hit in hits)
        assert profiles.recompute_profile("meridian-systems")["deals_count"] == before + 1
    finally:
        store.delete_by_contract_id(contract_id)
        profile = profiles.recompute_profile("meridian-systems")
        store.upsert_counterparty_profile({"counterparty_id": "meridian-systems", **profile})
