"""Smoke tests for the domain models."""

from __future__ import annotations

from uuid import uuid4

from precedent.models import (
    Claim,
    ClauseAnalysis,
    ClauseType,
    ComplianceStatus,
    DeviationClass,
    DPDPFinding,
    NegotiationOutcome,
    RedlineDraft,
    ReviewReport,
)


def test_negotiation_outcome_has_four_states() -> None:
    assert {o.value for o in NegotiationOutcome} == {
        "accepted",
        "redlined_then_accepted",
        "rejected",
        "deal_lost",
    }


def test_clause_taxonomy_excludes_dpdp() -> None:
    values = {c.value for c in ClauseType}

    assert "dpdp" not in values
    assert "data_processing" in values


def test_claim_defaults_to_unverified() -> None:
    claim = Claim(statement="Liability is capped at fees paid.", evidence_point_ids=["abc123"])

    assert claim.verified is False


def test_review_report_all_claims_verified_gate() -> None:
    clause_id = uuid4()
    unverified_report = ReviewReport(
        contract_id=uuid4(),
        redlines=[
            RedlineDraft(
                clause_id=clause_id,
                proposed_text="Liability shall not exceed fees paid in the preceding 12 months.",
                claims=[Claim(statement="Matches prior accepted redline.", evidence_point_ids=["p1"])],
            )
        ],
    )

    assert unverified_report.all_claims_verified is False

    verified_report = unverified_report.model_copy(deep=True)
    verified_report.redlines[0].claims[0].verified = True

    assert verified_report.all_claims_verified is True


def test_review_report_aggregates_pipeline_outputs() -> None:
    contract_id = uuid4()
    clause_id = uuid4()

    report = ReviewReport(
        contract_id=contract_id,
        clause_analyses=[
            ClauseAnalysis(
                clause_id=clause_id,
                deviation_class=DeviationClass.UNFAVORABLE,
                risk_score=0.8,
                rationale="Uncapped indemnity with no precedent for acceptance.",
                precedent_ids=["p1"],
            )
        ],
        dpdp_findings=[
            DPDPFinding(
                clause_id=clause_id,
                requirement="data_localization",
                status=ComplianceStatus.NEEDS_REVIEW,
                rationale="Clause does not specify storage location.",
            )
        ],
    )

    assert report.contract_id == contract_id
    assert report.clause_analyses[0].risk_score == 0.8
    assert report.dpdp_findings[0].status is ComplianceStatus.NEEDS_REVIEW
