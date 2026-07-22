"""Corpus seeder.

Read CONTEXT.md first.

Loads data/synthetic/ (produced by generate_corpus.py) and populates all
three Qdrant memory collections through memory/store.py — the only module
that talks to Qdrant directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from precedent.memory import store
from precedent.memory.profiles import recompute_profile

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"

# The synthetic corpus has no notion of multi-tenancy; it all belongs to one
# demo tenant.
DEMO_TENANT_ID = "demo"

app = typer.Typer(add_completion=False)
console = Console()


def _load(filename: str) -> Any:
    return json.loads((DATA_DIR / filename).read_text(encoding="utf-8"))


def _seed_clauses(contracts: list[dict[str, Any]]) -> None:
    clause_dicts = []
    for contract in contracts:
        for clause in contract["clauses"]:
            # final_text is what was actually agreed and is the useful
            # precedent, so it's what gets embedded; original_text rides
            # along in the payload for the before/after delta. A deal_lost
            # clause has no final_text, so fall back to original_text —
            # there was never an agreed version to embed instead.
            clause_text = clause["final_text"] or clause["original_text"]
            clause_dicts.append(
                {
                    "contract_id": contract["contract_id"],
                    "position": clause["position"],
                    "clause_type": clause["clause_type"],
                    "clause_text": clause_text,
                    "original_text": clause["original_text"],
                    "counterparty_id": contract["counterparty_id"],
                    "date": contract["date"],
                    "negotiation_outcome": clause["negotiation_outcome"],
                    "redline_rounds": clause["redline_rounds"],
                    "template_version": contract["template_version"],
                    "governing_law": contract["governing_law"],
                    "dpdp_relevant": clause["dpdp_relevant"],
                }
            )
    store.upsert_clauses(clause_dicts, tenant_id=DEMO_TENANT_ID)


def _seed_profiles(counterparties: list[dict[str, Any]]) -> None:
    # Profiles are computed from clause_memory, so this must run after
    # _seed_clauses has populated it.
    for cp in counterparties:
        stats = recompute_profile(cp["id"])
        store.upsert_counterparty_profile({"counterparty_id": cp["id"], "name": cp["name"], **stats})


@app.command()
def main(
    reset: bool = typer.Option(
        False, "--reset", help="Drop and recreate all three collections before seeding."
    ),
) -> None:
    """Seed clause_memory, playbook, and counterparty_profiles from data/synthetic/."""

    if reset:
        store.reset_collections()
    else:
        store.init_collections()

    contracts = _load("contracts.json")
    playbook = _load("playbook.json")
    counterparties = _load("counterparties.json")

    _seed_clauses(contracts)
    store.upsert_playbook_positions(playbook)
    _seed_profiles(counterparties)

    table = Table(title="Precedent memory - seeded point counts")
    table.add_column("Collection")
    table.add_column("Points", justify="right")
    for name, count in store.health().items():
        table.add_row(name, str(count))
    console.print(table)


if __name__ == "__main__":
    app()
