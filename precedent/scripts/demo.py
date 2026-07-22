# JWT auth, rate limiting, tenant-scoped API endpoints, and OpenTelemetry tracing are specified in PRD sections 10-12.
# They are staged for the next build pass; this script intentionally demonstrates only the local pipeline and memory flywheel.
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from precedent.governance import approval_gate
from precedent.memory import store
from run_pipeline import _run_pipeline


async def main() -> None:
    incoming = Path("data/synthetic/incoming")
    print(f"Memory count before review: {store.health()['clause_memory']}")
    first = await _run_pipeline(str(incoming / "incoming_meridian.json"))
    rejected = [p for bundle in first.get("precedents", {}).values() for p in (bundle.get("counterparty_history", []) if isinstance(bundle, dict) else []) if p.get("negotiation_outcome") == "rejected"]
    print(f"Meridian rejected-history citations: {[p.get('point_id') for p in rejected[:3]]}")
    approved = json.loads((incoming / "incoming_meridian.json").read_text())
    approved["clauses"] = [{**clause, "final_text": clause["text"], "negotiation_outcome": "accepted"} for clause in approved["clauses"]]
    point_ids = approval_gate.approve("demo-meridian", approved, "demo-reviewer", "demo")
    print(f"APPROVED — MEMORY COUNTER INCREASED; NEW POINT IDS: {point_ids}")
    second = await _run_pipeline(str(incoming / "incoming_meridian_v2.json"))
    print("THE FLYWHEEL: Meridian v2 now retrieves precedent from the review just approved.")
    print(f"v2 precedent buckets: {len(second.get('precedents', {}))}")
    injection = await _run_pipeline(str(incoming / "incoming_injection.json"))
    flagged = any((c.get("suspected_injection") if isinstance(c, dict) else getattr(c, "suspected_injection", False)) for c in injection.get("clauses", []))
    print(f"Injection flagged as suspected injection: {flagged}")
    print("\nMetric                    Value")
    print(f"Precedent hit rate        {sum(bool(v) for v in first.get('precedents', {}).values())}/{len(first.get('precedents', {}))}")
    print(f"Claims stripped           {first.get('claims_stripped', 0)}")
    print(f"Average loop iterations   {first.get('iterations_used', 0)}")
    print(f"DPDP gaps found           {len(injection.get('dpdp_findings', []))}")


if __name__ == "__main__":
    asyncio.run(main())
