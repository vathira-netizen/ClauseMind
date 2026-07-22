from fastapi.testclient import TestClient

from precedent.api.main import app


def test_review_endpoint_returns_pipeline_result(monkeypatch):
    async def fake_run_review_pipeline(document_path: str):
        return {
            "review_report": "# Review\n\nPipeline completed.",
            "clauses": [],
            "session_id": "test-session",
        }

    monkeypatch.setattr("precedent.api.main.run_review_pipeline", fake_run_review_pipeline)

    client = TestClient(app)
    response = client.post(
        "/review",
        json={"document_path": "data/synthetic/incoming/incoming_meridian.json"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["review_id"]
    assert body["document_path"] == "data/synthetic/incoming/incoming_meridian.json"
    assert body["review_report"].startswith("# Review")
