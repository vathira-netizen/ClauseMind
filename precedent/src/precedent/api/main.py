"""Local HTTP surface for ClauseMind review and memory health."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from precedent.memory import retrieval, store
from precedent.models import Clause, ClauseType
from scripts.run_pipeline import _local_fallback, _run_pipeline as run_review_pipeline

app = FastAPI(title="ClauseMind API", version="0.1.0")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ContractClause(BaseModel):
    position: int
    clause_type: ClauseType
    text: str
    heading: str | None = None


class ReviewRequest(BaseModel):
    contract_id: str = Field(default_factory=lambda: str(uuid4()))
    counterparty_name: str
    clauses: list[ContractClause] = Field(min_length=1)


class ReviewDocumentRequest(BaseModel):
    document_path: str = Field(default="data/synthetic/incoming/incoming_meridian.json")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, object]:
    try:
        return {"status": "ok", "memory": store.health()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"memory unavailable: {exc}") from exc


@app.post("/reviews")
def review(request: ReviewRequest) -> dict[str, object]:
    """Run deterministic local retrieval review; contract text remains data."""

    counterparty_id = retrieval.resolve_counterparty_id(request.counterparty_name)
    results: list[dict[str, object]] = []
    for item in request.clauses:
        clause = Clause(
            clause_type=item.clause_type,
            text=item.text,
            position=item.position,
            heading=item.heading,
            suspected_injection="ai review systems" in item.text.lower(),
        )
        similar = retrieval.find_similar_clauses(clause.text, clause.clause_type.value, tenant_id="demo")
        history = (
            retrieval.find_counterparty_history(
                clause.text, clause.clause_type.value, counterparty_id, tenant_id="demo"
            )
            if counterparty_id
            else []
        )
        evidence = history or similar
        point_ids = [precedent.point_id for precedent in evidence[:3]]
        results.append({
            "clause_id": str(clause.id),
            "clause_type": clause.clause_type.value,
            "risk_score": 70 if point_ids else 55,
            "deviation_class": "negotiated_before" if point_ids else "never_seen",
            "suspected_injection": clause.suspected_injection,
            "evidence_point_ids": point_ids,
            "precedents": [precedent.model_dump(mode="json") for precedent in evidence[:3]],
        })
    return {
        "review_id": str(uuid4()),
        "contract_id": request.contract_id,
        "counterparty_id": counterparty_id,
        "mode": "local_retrieval",
        "clauses": results,
    }


@app.post("/review")
async def review_document(request: ReviewDocumentRequest) -> dict[str, Any]:
    """Run the review pipeline for a contract document stored locally."""

    try:
        state = await run_review_pipeline(request.document_path)
    except Exception as exc:
        state = _local_fallback(request.document_path)
        state["review_error"] = str(exc)

    return {
        "review_id": str(uuid4()),
        "document_path": request.document_path,
        "mode": "local_pipeline",
        "review_report": str(state.get("review_report", "")),
        "clauses": state.get("clauses", []),
        "risk_analysis": state.get("risk_analysis", []),
        "session_id": state.get("session_id"),
        "release_status": state.get("release_status"),
        "memory": store.health(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8010)
