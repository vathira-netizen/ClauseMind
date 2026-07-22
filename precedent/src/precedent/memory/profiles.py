"""Counterparty profile computation.

Read CONTEXT.md first.

``concession_rate``, ``contested_clauses``, and ``avg_rounds_to_close`` are
computed from a counterparty's actual ``clause_memory`` history rather than
hardcoded, so the seeder and the live write-back path — which appends a
freshly negotiated contract's clauses to ``clause_memory`` and then needs to
refresh that counterparty's profile — call the exact same function instead of
duplicating the computation.
"""

from __future__ import annotations

import statistics
from typing import Any

from precedent.memory import store

# Outcomes that indicate this clause type required pushback rather than
# being signed as-is.
CONTESTED_OUTCOMES = {"redlined_then_accepted", "rejected", "deal_lost"}


def recompute_profile(counterparty_id: str) -> dict[str, Any]:
    """Recompute a counterparty's profile stats from its clause_memory history.

    - ``concession_rate``: fraction of this counterparty's clauses that were
      redlined before being accepted — how often a deal with them requires
      negotiation rather than signing an opening position as-is.
    - ``contested_clauses``: clause types with at least one non-``accepted``
      outcome for this counterparty, i.e. types that have required pushback.
    - ``avg_rounds_to_close``: mean ``redline_rounds`` across clauses that
      actually closed (excludes ``deal_lost``, since those never closed).
    - ``deals_count``: number of distinct contracts on record.

    Returns zeroed stats if this counterparty has no clause_memory history
    yet, rather than raising.
    """

    records = store.get_by_counterparty_id(counterparty_id)
    if not records:
        return {
            "concession_rate": 0.0,
            "contested_clauses": [],
            "avg_rounds_to_close": 0.0,
            "deals_count": 0,
        }

    total = len(records)
    redlined = sum(1 for r in records if r["negotiation_outcome"] == "redlined_then_accepted")
    concession_rate = redlined / total

    contested_clauses = sorted(
        {r["clause_type"] for r in records if r["negotiation_outcome"] in CONTESTED_OUTCOMES}
    )

    closed_rounds = [r["redline_rounds"] for r in records if r["negotiation_outcome"] != "deal_lost"]
    avg_rounds_to_close = statistics.mean(closed_rounds) if closed_rounds else 0.0

    deals_count = len({r["contract_id"] for r in records})

    return {
        "concession_rate": round(concession_rate, 4),
        "contested_clauses": contested_clauses,
        "avg_rounds_to_close": round(avg_rounds_to_close, 2),
        "deals_count": deals_count,
    }
