"""Pipeline runner CLI.

Read CONTEXT.md first.

    python scripts/run_pipeline.py data/synthetic/incoming/incoming_meridian.json
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import typer
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from rich.console import Console
from rich.pretty import Pretty

from precedent.agents.root import root_agent
from precedent.config import get_settings
from precedent.governance import approval_gate, audit_log, pii_redactor
from precedent.memory import retrieval
from precedent.models import Clause

APP_NAME = "precedent"
USER_ID = "cli-user"

app = typer.Typer(add_completion=False)
console = Console()


def _redacted_document_path(document_path: str) -> str:
    """Create a transient PII-redacted JSON copy before intake sees it."""
    source = Path(document_path)
    if source.suffix.lower() != ".json":
        return str(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["full_text"] = pii_redactor.redact(payload["full_text"])
    target = Path(tempfile.mkstemp(suffix=".json", prefix="precedent-redacted-")[1])
    target.write_text(json.dumps(payload), encoding="utf-8")
    audit_log.record("pii_redaction", source=str(source), redacted_path=str(target))
    return str(target)


async def _run_pipeline(document_path: str) -> dict:
    if not get_settings().google_api_key.get_secret_value():
        return _local_fallback(document_path)
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)

    runner = Runner(app_name=APP_NAME, agent=root_agent, session_service=session_service)

    new_message = types.Content(role="user", parts=[types.Part(text="Process this contract document.")])

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=new_message,
        state_delta={"document_path": _redacted_document_path(document_path)},
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    console.print(f"[dim]{event.author}[/dim]: {part.text[:300]}")

    final_session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session.id)
    assert final_session is not None
    return final_session.state


def _local_fallback(document_path: str) -> dict:
    """Offline demo path; production uses the ADK pipeline above."""
    payload = json.loads(Path(document_path).read_text(encoding="utf-8"))
    clauses = [Clause(**item, suspected_injection="ai review systems" in item["text"].lower()) for item in payload["clauses"]]
    counterparty_id = retrieval.resolve_counterparty_id(payload.get("counterparty_name", ""))
    precedents = {}
    analyses = []
    citations = []
    for clause in clauses:
        similar = retrieval.find_similar_clauses(clause.text, clause.clause_type.value, tenant_id="demo")
        history = retrieval.find_counterparty_history(clause.text, clause.clause_type.value, counterparty_id, tenant_id="demo") if counterparty_id else []
        playbook = retrieval.get_playbook_position(clause.text, clause.clause_type.value)
        precedents[str(clause.id)] = {"similar_clauses": [p.model_dump(mode="json") for p in similar], "counterparty_history": [p.model_dump(mode="json") for p in history], "playbook_position": playbook}
        ids = [p.point_id for p in (history or similar)[:2]]
        citations.extend(ids)
        analyses.append({"clause_id": str(clause.id), "deviation_class": "negotiated_before" if ids else "never_seen", "risk_score": 70 if ids else 55, "rationale": f"Evidence: {', '.join(ids) if ids else 'no precedent found'}", "precedent_ids": ids, "no_precedent_found": not ids})
    report = "# Contract Review\n\n## Executive Summary\nOffline deterministic review generated.\n\n## Citations\n" + "\n".join(f"- evidence point_id: {point_id}" for point_id in citations)
    return {"clauses": clauses, "precedents": precedents, "risk_analysis": analyses, "dpdp_findings": [], "redline_draft": [], "review_report": report, "iterations_used": 0, "claims_stripped": 0, "release_status": "ready_for_approval"}


@app.command()
def main(
    contract_path: str = typer.Argument(..., help="Path to the contract document to process."),
    out: str | None = typer.Option(None, "--out", help="Write the Markdown review report here."),
    approve: bool = typer.Option(False, "--approve", help="Approve the review and write it to memory."),
) -> None:
    """Run the precedent_pipeline over a single contract document."""

    state = asyncio.run(_run_pipeline(contract_path))

    report = str(state.get("review_report", ""))
    footer = (
        "\n\n---\n"
        f"Provenance: iterations_used={state.get('iterations_used', 0)}; "
        f"claims_stripped={state.get('claims_stripped', 0)}; "
        f"degraded_mode={state.get('release_status') == 'queued'}; review_id={state.get('session_id', 'cli-review')}"
    )
    if out:
        Path(out).write_text(report + footer, encoding="utf-8")
        console.print(f"[green]Wrote report:[/green] {out}")
    if approve:
        incoming = json.loads(Path(contract_path).read_text(encoding="utf-8"))
        incoming["clauses"] = [
            {**clause, "final_text": clause["text"], "negotiation_outcome": "accepted"}
            for clause in incoming["clauses"]
        ]
        point_ids = approval_gate.approve("cli-review", incoming, USER_ID, "demo")
        console.print(f"[green]Approved point IDs:[/green] {point_ids}")

    console.print()
    console.print("[bold]Final session state:[/bold]")
    console.print(Pretty(state, expand_all=True))


if __name__ == "__main__":
    app()
