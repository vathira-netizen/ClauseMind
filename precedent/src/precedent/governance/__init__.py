"""Governance interfaces and the dependency-free default adapter."""

from precedent.governance.local_adapter import approval_gate, audit_log, hallucination_gate, pii_redactor

__all__ = ["approval_gate", "audit_log", "hallucination_gate", "pii_redactor"]
