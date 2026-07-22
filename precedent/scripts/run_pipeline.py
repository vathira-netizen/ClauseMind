"""Pipeline runner CLI.

Read CONTEXT.md first.

    python scripts/run_pipeline.py data/synthetic/incoming/incoming_meridian.json
"""

from __future__ import annotations

import asyncio

import typer
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from rich.console import Console
from rich.pretty import Pretty

from precedent.agents.root import root_agent

APP_NAME = "precedent"
USER_ID = "cli-user"

app = typer.Typer(add_completion=False)
console = Console()


async def _run_pipeline(document_path: str) -> dict:
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)

    runner = Runner(app_name=APP_NAME, agent=root_agent, session_service=session_service)

    new_message = types.Content(role="user", parts=[types.Part(text="Process this contract document.")])

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=new_message,
        state_delta={"document_path": document_path},
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    console.print(f"[dim]{event.author}[/dim]: {part.text[:300]}")

    final_session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session.id)
    assert final_session is not None
    return final_session.state


@app.command()
def main(
    contract_path: str = typer.Argument(..., help="Path to the contract document to process."),
) -> None:
    """Run the precedent_pipeline over a single contract document."""

    state = asyncio.run(_run_pipeline(contract_path))

    console.print()
    console.print("[bold]Final session state:[/bold]")
    console.print(Pretty(state, expand_all=True))


if __name__ == "__main__":
    app()
