"""Vendor-neutral governance interfaces used by the pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    action: str
    details: dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AuditLog(ABC):
    @abstractmethod
    def record(self, action: str, **details: Any) -> AuditEvent: ...


class PIIRedactor(ABC):
    @abstractmethod
    def redact(self, text: str) -> str: ...


class HallucinationGate(ABC):
    @abstractmethod
    def evaluate(self, report: str, precedents: dict[str, Any]) -> tuple[bool, list[str]]: ...


class ApprovalGate(ABC):
    @abstractmethod
    def approve(self, review_id: str, report: dict[str, Any], reviewer_id: str, tenant_id: str) -> list[str]: ...
