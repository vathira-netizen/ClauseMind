"""Smoke tests for the domain models."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from precedent.models import (
    Claim,
    ClauseAnalysis,
    ClauseType,
    DeviationClass,
    DPDPFinding,
    DPDPSeverity,
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
                deviation_class=DeviationClass.NEVER_SEEN,
                risk_score=80,
                rationale="Uncapped indemnity with no precedent for acceptance.",
                precedent_ids=["p1"],
            )
        ],
        dpdp_findings=[
            DPDPFinding(
                clause_id=clause_id,
                requirement="data_localization",
                severity=DPDPSeverity.HIGH,
                rationale="Clause does not specify storage location.",
                remediation="Add a clause requiring in-country data storage.",
            )
        ],
    )

    assert report.contract_id == contract_id
    assert report.clause_analyses[0].risk_score == 80
    assert report.dpdp_findings[0].severity is DPDPSeverity.HIGH


def test_clause_analysis_rejects_risk_score_with_no_evidence() -> None:
    with pytest.raises(ValidationError):
        ClauseAnalysis(
            clause_id=uuid4(),
            deviation_class=DeviationClass.NEVER_SEEN,
            risk_score=50,
            rationale="Looks risky.",
        )


def test_clause_analysis_allows_no_precedent_found_marker_instead_of_ids() -> None:
    analysis = ClauseAnalysis(
        clause_id=uuid4(),
        deviation_class=DeviationClass.NEVER_SEEN,
        risk_score=50,
        rationale="No comparable clause exists in memory.",
        no_precedent_found=True,
    )

    assert analysis.precedent_ids == []
    assert analysis.no_precedent_found is True
