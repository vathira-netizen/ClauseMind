"""Dependency-free governance adapter used by default."""

from __future__ import annotations

import re
from typing import Any

from precedent.governance.base import ApprovalGate, AuditEvent, AuditLog, HallucinationGate, PIIRedactor


class InMemoryAuditLog(AuditLog):
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, action: str, **details: Any) -> AuditEvent:
        event = AuditEvent(action=action, details=details)
        self.events.append(event)
        return event


class RegexPIIRedactor(PIIRedactor):
    _patterns = (
        (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
        (re.compile(r"\b(?:\+?\d[\d -]{8,}\d)\b"), "[REDACTED_PHONE]"),
    )

    def redact(self, text: str) -> str:
        for pattern, replacement in self._patterns:
            text = pattern.sub(replacement, text)
        return text


class IndependentCitationGate(HallucinationGate):
    """Re-check report citations without consulting Citation Critic output."""

    _point_id = re.compile(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", re.IGNORECASE)

    def evaluate(self, report: str, precedents: dict[str, Any]) -> tuple[bool, list[str]]:
        valid_ids: set[str] = set()
        evidence_text: dict[str, str] = {}
        for bundle in precedents.values():
            data = bundle.model_dump() if hasattr(bundle, "model_dump") else bundle
            for key in ("similar_clauses", "counterparty_history"):
                for item in data.get(key, []):
                    valid_ids.add(str(item["point_id"]))
                    evidence_text[str(item["point_id"])] = str(item.get("clause_text", ""))
            position = data.get("playbook_position", {})
            for item in [position.get("preferred"), position.get("walk_away"), *position.get("fallbacks", [])]:
                if item:
                    valid_ids.add(str(item["point_id"]))
                    evidence_text[str(item["point_id"])] = str(item.get("position_text", ""))

        cited = set(self._point_id.findall(report))
        unknown = sorted(cited - valid_ids)
        empty_support = sorted(point_id for point_id in cited & valid_ids if not evidence_text.get(point_id, "").strip())
        failures = [f"unknown point_id: {point_id}" for point_id in unknown]
        failures += [f"empty retrieved evidence: {point_id}" for point_id in empty_support]
        return not failures, failures


class LocalApprovalGate(ApprovalGate):
    def __init__(self, audit_log: AuditLog) -> None:
        self.audit_log = audit_log

    def approve(self, review_id: str, report: dict[str, Any], reviewer_id: str, tenant_id: str) -> list[str]:
        from precedent.memory.flywheel import _approval_capability, write_back

        point_ids = write_back(
            review_id, report, reviewer_id, tenant_id, _capability=_approval_capability
        )
        self.audit_log.record("approval", review_id=review_id, reviewer_id=reviewer_id, point_ids=point_ids)
        return point_ids


audit_log = InMemoryAuditLog()
pii_redactor = RegexPIIRedactor()
hallucination_gate = IndependentCitationGate()
approval_gate = LocalApprovalGate(audit_log)
