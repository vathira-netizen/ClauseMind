"""Approved-review write-back: the compounding institutional-memory loop."""

from __future__ import annotations

from datetime import date
from typing import Any

from precedent.governance.local_adapter import audit_log
from precedent.memory import profiles, retrieval, store

_approval_capability = object()
_OUTCOMES = {"accepted", "redlined_then_accepted", "rejected", "deal_lost"}


def _value(item: Any, name: str, default: Any = None) -> Any:
    return getattr(item, name, item.get(name, default) if isinstance(item, dict) else default)


def _counterparty_id(report: dict[str, Any]) -> str:
    candidate = report.get("counterparty_id")
    if candidate:
        return str(candidate)
    name = report.get("counterparty_name", "")
    resolved = retrieval.resolve_counterparty_id(name)
    if not resolved:
        raise ValueError("approved report must identify a known counterparty")
    return resolved


def write_back(
    review_id: str,
    approved_report: dict[str, Any],
    reviewer_id: str,
    tenant_id: str,
    *,
    _capability: object | None = None,
) -> list[str]:
    """Upsert final negotiated clauses and refresh their counterparty profile.

    This function is deliberately capability-guarded: only ApprovalGate owns
    the capability, so unapproved reviews cannot become institutional memory.
    """

    assert _capability is _approval_capability, "only ApprovalGate may call write_back"
    contract_id = str(approved_report["contract_id"])
    counterparty_id = _counterparty_id(approved_report)
    clauses = approved_report.get("clauses", [])
    records: list[dict[str, Any]] = []
    for index, clause in enumerate(clauses):
        outcome = _value(clause, "negotiation_outcome", "accepted")
        if hasattr(outcome, "value"):
            outcome = outcome.value
        if outcome not in _OUTCOMES:
            raise ValueError(f"invalid negotiation_outcome: {outcome}")
        final_text = _value(clause, "final_text") or _value(clause, "proposed_text") or _value(clause, "text")
        if not final_text:
            raise ValueError("approved clause requires final negotiated text")
        clause_type = _value(clause, "clause_type")
        if hasattr(clause_type, "value"):
            clause_type = clause_type.value
        position = _value(clause, "position", index)
        records.append({
            "contract_id": contract_id,
            "position": position,
            "clause_text": final_text,
            "original_text": _value(clause, "text"),
            "clause_type": clause_type,
            "counterparty_id": counterparty_id,
            "date": approved_report.get("sign_off_date", date.today().isoformat()),
            "negotiation_outcome": outcome,
            "redline_rounds": _value(clause, "redline_rounds", approved_report.get("iterations_used", 1)),
            "template_version": approved_report.get("template_version"),
            "governing_law": approved_report.get("governing_law"),
            "risk_score": _value(clause, "risk_score", 0),
            "dpdp_relevant": _value(clause, "dpdp_relevant", clause_type == "data_processing"),
            "approver_id": reviewer_id,
        })
    point_ids = store.upsert_clauses(records, tenant_id=tenant_id)
    profile = profiles.recompute_profile(counterparty_id)
    store.upsert_counterparty_profile({"counterparty_id": counterparty_id, "name": approved_report.get("counterparty_name"), **profile})
    audit_log.record("memory_write_back", review_id=review_id, reviewer_id=reviewer_id, point_ids=point_ids)
    return point_ids


def revoke_memory(contract_id: str) -> int:
    """Remove derived contract points and recompute every affected profile."""

    affected = {record.get("counterparty_id") for record in store.get_by_contract_id(contract_id)}
    removed = store.delete_by_contract_id(contract_id)
    for counterparty_id in affected - {None}:
        profile = profiles.recompute_profile(str(counterparty_id))
        store.upsert_counterparty_profile({"counterparty_id": str(counterparty_id), **profile})
    audit_log.record("memory_revoked", contract_id=contract_id, points_removed=removed)
    return removed
