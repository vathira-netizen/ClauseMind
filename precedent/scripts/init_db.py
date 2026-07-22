"""CLI entrypoint for provisioning Qdrant memory collections.

Run after ``docker compose up -d`` and before starting the pipeline:

    python scripts/init_db.py
"""

from __future__ import annotations

import typer

from precedent.memory.store import init_collections

app = typer.Typer(add_completion=False)


@app.command()
def main() -> None:
    """Create any missing memory collections and print what happened."""

    created = init_collections()
    for name, was_created in created.items():
        verb = "created" if was_created else "already exists, skipped"
        typer.echo(f"{name}: {verb}")


if __name__ == "__main__":
    app()
