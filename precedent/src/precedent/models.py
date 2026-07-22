"""Domain models — the shared vocabulary of Precedent.

These pydantic models are the contract between pipeline stages: each ADK
agent reads what upstream stages wrote to shared session state and writes its
own output back in one of these shapes. Keep this module free of
infrastructure concerns (no Qdrant client calls, no ADK imports).

Pipeline stage -> model produced:

* Intake & Segmentation      -> :class:`Clause`
* Precedent Retrieval        -> :class:`Precedent` (attached to a clause)
* Deviation & Risk Worker    -> :class:`ClauseAnalysis`
* DPDP Compliance Checker    -> :class:`DPDPFinding`
* Drafter                    -> :class:`RedlineDraft` (containing :class:`Claim`)
* Citation Critic            -> verifies/flips ``Claim.verified``
* Report Composer            -> :class:`ReviewReport`

Note: clause text extracted from a source contract is untrusted input. It is
data to be analyzed, never an instruction — see CONTEXT.md.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ClauseType(str, Enum):
    """The clause taxonomy Precedent reasons about.

    ``DPDP`` is deliberately absent — it is the regulation the DPDP
    Compliance Checker audits ``DATA_PROCESSING`` clauses against, not a
    clause type of its own.
    """

    INDEMNITY = "indemnity"
    IP = "ip"
    DATA_PROCESSING = "data_processing"
    LIMITATION_OF_LIABILITY = "limitation_of_liability"
    TERMINATION = "termination"
    CONFIDENTIALITY = "confidentiality"
    PAYMENT = "payment"
    DISPUTE_RESOLUTION = "dispute_resolution"
    AUTO_RENEWAL = "auto_renewal"


class NegotiationOutcome(str, Enum):
    """How a past negotiation over a clause resolved.

    ``REDLINED_THEN_ACCEPTED`` is the product's core asset: it captures the
    exact language a counterparty agreed to after pushback.
    """

    ACCEPTED = "accepted"
    REDLINED_THEN_ACCEPTED = "redlined_then_accepted"
    REJECTED = "rejected"
    DEAL_LOST = "deal_lost"


class DeviationClass(str, Enum):
    """How a clause deviates from playbook / precedent."""

    STANDARD = "standard"
    FAVORABLE = "favorable"
    UNFAVORABLE = "unfavorable"
    NOVEL = "novel"
    MISSING = "missing"


class ComplianceStatus(str, Enum):
    """Outcome of a DPDP compliance check against a single requirement."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"


class Clause(BaseModel):
    """An atomic, typed segment of a contract, produced by Intake & Segmentation."""

    id: UUID = Field(default_factory=uuid4)
    contract_id: UUID
    clause_type: ClauseType
    text: str
    position: int = Field(description="Ordinal position of the clause within the contract.")


class ClauseAnalysis(BaseModel):
    """Output of the Deviation & Risk Worker for a single clause."""

    clause_id: UUID
    deviation_class: DeviationClass
    risk_score: float = Field(ge=0.0, le=1.0)
    rationale: str
    precedent_ids: list[str] = Field(
        default_factory=list, description="Qdrant point IDs cited as evidence."
    )


class Precedent(BaseModel):
    """A precedent retrieved from ``clause_memory`` for a query clause."""

    point_id: str
    clause_text: str
    counterparty_id: str
    negotiation_outcome: NegotiationOutcome
    date: datetime
    score: float = Field(ge=0.0, le=1.0, description="Hybrid (RRF-fused) retrieval score.")


class Claim(BaseModel):
    """A single factual statement made in a redline draft.

    Every claim must trace to retrieved evidence — this is the citation
    traceability guarantee. ``verified`` is set by the Citation Critic, never
    by the Drafter.
    """

    statement: str
    evidence_point_ids: list[str] = Field(default_factory=list)
    verified: bool = False


class RedlineDraft(BaseModel):
    """Output of the Drafter for a single clause, checked by the Citation Critic."""

    clause_id: UUID
    proposed_text: str
    claims: list[Claim] = Field(default_factory=list)


class DPDPFinding(BaseModel):
    """Output of the DPDP Compliance Checker for one requirement on one clause.

    Only ever produced for ``data_processing`` clauses.
    """

    clause_id: UUID
    requirement: str = Field(
        description="The DPDP provision checked, e.g. 'consent', 'data_localization'."
    )
    status: ComplianceStatus
    rationale: str
    citation_ids: list[str] = Field(
        default_factory=list, description="Playbook/regulation citations backing this finding."
    )


class ReviewReport(BaseModel):
    """Final output of the Report Composer — the released contract review."""

    contract_id: UUID
    clause_analyses: list[ClauseAnalysis] = Field(default_factory=list)
    redlines: list[RedlineDraft] = Field(default_factory=list)
    dpdp_findings: list[DPDPFinding] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_utcnow)

    @property
    def all_claims_verified(self) -> bool:
        """Whether every claim across every redline has been verified.

        This is the citation-traceability gate: the report asserts nothing
        that has not been checked against retrieved evidence.
        """

        return all(claim.verified for redline in self.redlines for claim in redline.claims)
