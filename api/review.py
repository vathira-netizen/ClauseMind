import json
import os
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="ClauseMind review API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    document_path: str = "precedent/data/synthetic/incoming/incoming_meridian.json"


@app.get("/")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "vercel"}


@app.get("/health")
def health_detail() -> dict[str, Any]:
    return {"status": "ok", "service": "vercel"}


@app.post("/")
def review(request: ReviewRequest) -> dict[str, Any]:
    document_path = request.document_path
    sample_path = document_path if os.path.exists(document_path) else "precedent/data/synthetic/incoming/incoming_meridian.json"

    try:
        with open(sample_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="document not found") from exc

    clauses = payload.get("clauses", [])
    return {
        "review_id": "vercel-local-review",
        "document_path": document_path,
        "mode": "vercel-static-demo",
        "review_report": "# Contract Review\n\n## Executive Summary\nVercel deployment demo review generated for this document.\n",
        "clauses": clauses,
        "risk_analysis": [
            {
                "clause_id": clause.get("id", f"clause-{index}"),
                "deviation_class": "reviewed",
                "risk_score": 60 + (index % 3) * 5,
                "rationale": "Static demo review for Vercel deployment.",
                "precedent_ids": [],
                "no_precedent_found": True,
            }
            for index, clause in enumerate(clauses)
        ],
        "release_status": "ready_for_review",
    }
