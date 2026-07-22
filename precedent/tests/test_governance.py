from precedent.governance.local_adapter import IndependentCitationGate, RegexPIIRedactor


def test_redactor_removes_common_pii() -> None:
    assert "alice@example.com" not in RegexPIIRedactor().redact("Email alice@example.com; call +91 98765 43210")


def test_hallucination_gate_rejects_unknown_point() -> None:
    passed, failures = IndependentCitationGate().evaluate(
        "Citation: 00000000-0000-0000-0000-000000000000", {}
    )
    assert not passed
    assert failures
