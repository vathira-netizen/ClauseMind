from precedent.agents.redline import strip_unverified_claims


def test_fabricated_point_id_is_stripped_and_counted() -> None:
    drafts, stripped = strip_unverified_claims([
        {"clause_id": "a", "claims": [
            {"statement": "supported", "evidence_point_ids": ["real"], "verified": True},
            {"statement": "fabricated", "evidence_point_ids": ["00000000-0000-0000-0000-000000000000"], "verified": False},
        ]}
    ])
    assert stripped == 1
    assert [claim["statement"] for claim in drafts[0]["claims"]] == ["supported"]
