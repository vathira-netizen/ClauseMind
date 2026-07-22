"""Retrieval ablation harness: dense-only vs sparse-only vs hybrid RRF.

Read CONTEXT.md first.

Builds a held-out query set by sampling clauses from the seeded corpus and
paraphrasing them (preserving genuine legal terms of art), then measures
Recall@1, Recall@5, MRR, and retrieval latency for three configurations
against the live clause_memory collection.
"""

from __future__ import annotations

import json
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.table import Table

from precedent.memory import store

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_PATH = REPO_ROOT / "data" / "synthetic" / "contracts.json"
EVAL_DIR = REPO_ROOT / "data" / "eval"
OUTPUT_PATH = EVAL_DIR / "retrieval_ablation.json"

TENANT_ID = "demo"
SAMPLE_SIZE = 40
K = 5
SEED = 42

console = Console()

# ---------------------------------------------------------------------------
# Deterministic paraphrasing. Not an LLM call: a curated list of genuine
# legal terms of art is protected verbatim, and a much broader vocabulary of
# non-term-of-art words and boilerplate connectives elsewhere in the text is
# rewritten. This preserves the exact vocabulary sparse retrieval keys on
# while still producing real lexical drift, so the ablation actually
# exercises dense/sparse/hybrid differently instead of hitting a ceiling
# where every configuration finds an almost-verbatim match trivially.
# ---------------------------------------------------------------------------

PROTECTED_PHRASES = [
    "indemnify, defend, and hold harmless",
    "indemnify and hold harmless",
    "hold harmless",
    "gross negligence, willful misconduct, or material breach",
    "gross negligence",
    "willful misconduct",
    "material breach",
    "consequential, incidental, or punitive damages",
    "consequential damages",
    "incidental damages",
    "punitive damages",
    "sole discretion",
    "notwithstanding the foregoing",
    "notwithstanding any provision to the contrary",
    "reasonable attorneys' fees",
    "attorneys' fees",
    "binding arbitration",
    "aggregate liability",
    "Digital Personal Data Protection Act",
    "documented instructions",
    "without undue delay",
    "personal data breach",
    "technical and organisational security safeguards",
    "sub-processor",
    "Confidential Information",
    "confidential information",
    "cure period",
    "limitation of liability",
    "intellectual property",
    "third-party claims",
]

# (phrase, replacement). Sorted longest-phrase-first before compiling into a
# single alternation, so a more specific multi-word phrase is tried before a
# shorter one it contains — Python's re tries alternatives left-to-right and
# takes the first that matches at a position, not the longest.
RAW_SUBSTITUTIONS: list[tuple[str, str]] = [
    ("in no event shall", "under no circumstances will"),
    ("arising out of or relating to", "stemming from or connected with"),
    ("arising out of or related to", "stemming from or connected with"),
    ("in connection with", "relating to"),
    ("in accordance with", "pursuant to"),
    ("upon written notice", "following notice in writing"),
    ("written notice", "notice in writing"),
    ("provided that", "on the condition that"),
    ("without limitation", "without restriction"),
    ("for any reason", "for whatever reason"),
    ("as necessary to", "as needed to"),
    ("immediately upon", "as soon as"),
    ("whichever is greater", "whichever amount is larger"),
    ("in perpetuity", "for an unlimited duration"),
    ("any and all", "all"),
    ("each party's", "both parties'"),
    ("either party", "any party"),
    ("each party", "both parties"),
    ("set forth", "specified"),
    ("capped at", "limited to"),
    ("howsoever arising", "arising in whatever manner"),
    ("prior to", "before"),
    ("at least", "no fewer than"),
    ("successive", "consecutive"),
    ("shall not", "will not"),
    ("shall be", "will be"),
    ("shall", "will"),
    ("claims", "assertions"),
    ("claim", "assertion"),
    ("expenses", "outlays"),
    ("costs", "charges"),
    ("obligations", "duties"),
    ("obligation", "duty"),
    ("breaches", "violates"),
    ("breach", "violation"),
    ("agrees to", "undertakes to"),
    ("maintaining", "keeping"),
    ("maintains", "keeps"),
    ("maintain", "keep"),
    ("immediately", "right away"),
    ("expiration", "expiry"),
    ("delivered", "sent"),
    ("disclosed", "shared"),
    ("received", "obtained"),
    ("required", "necessitated"),
    ("require", "necessitate"),
    ("engaging", "retaining"),
    ("engage", "retain"),
    ("purpose", "aim"),
    ("terminating", "ending"),
    ("terminated", "ended"),
    ("terminate", "end"),
    ("reasonable", "sensible"),
    ("promptly", "without delay"),
    ("solely", "exclusively"),
    ("exclusively", "solely"),
    ("permits", "allows"),
    ("permit", "allow"),
    ("audits", "reviews"),
    ("audit", "review"),
    ("verify", "confirm"),
    ("notifies", "informs"),
    ("notify", "inform"),
    ("indefinitely", "without end"),
    ("exceeds", "surpasses"),
    ("exceed", "surpass"),
    ("deemed", "considered"),
    ("whatsoever", "of any kind"),
    ("consent", "approval"),
    ("security safeguards", "security measures"),
]


def _protect_phrases(text: str) -> tuple[str, dict[str, str]]:
    # Delimited, non-numeric placeholder tokens: a bare digit token (e.g.
    # "1") would collide with real numbers already in the clause text (e.g.
    # "one (1) month") once _restore_phrases does a plain substring replace.
    mapping: dict[str, str] = {}

    for phrase in PROTECTED_PHRASES:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)

        def _sub(m: re.Match) -> str:
            token = f"@@P{len(mapping)}@@"
            mapping[token] = m.group(0)
            return token

        text = pattern.sub(_sub, text)
    return text, mapping


def _restore_phrases(text: str, mapping: dict[str, str]) -> str:
    for token, original in mapping.items():
        text = text.replace(token, original)
    return text


_SUBSTITUTION_LOOKUP = {phrase.lower(): replacement for phrase, replacement in RAW_SUBSTITUTIONS}
_SUBSTITUTION_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p, _ in sorted(RAW_SUBSTITUTIONS, key=lambda x: -len(x[0]))) + r")\b",
    re.IGNORECASE,
)


def _apply_substitutions(text: str) -> str:
    # A single combined-alternation pass: every match is found against the
    # original text in one left-to-right scan, so an earlier substitution's
    # replacement text can never be re-matched and re-substituted by a later
    # pattern the way sequential per-phrase passes would risk.
    def _sub(m: re.Match) -> str:
        matched = m.group(0)
        replacement = _SUBSTITUTION_LOOKUP[matched.lower()]
        return replacement[:1].upper() + replacement[1:] if matched[:1].isupper() else replacement

    return _SUBSTITUTION_PATTERN.sub(_sub, text)


def paraphrase(text: str) -> str:
    protected, mapping = _protect_phrases(text)
    substituted = _apply_substitutions(protected)
    return _restore_phrases(substituted, mapping)


# ---------------------------------------------------------------------------
# Held-out query set
# ---------------------------------------------------------------------------


def build_query_set(rng: random.Random) -> list[dict[str, Any]]:
    contracts = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))

    candidates: dict[str, dict[str, Any]] = {}
    duplicate_texts: set[str] = set()
    for contract in contracts:
        for clause in contract["clauses"]:
            embedded_text = clause["final_text"] or clause["original_text"]
            if embedded_text in candidates:
                duplicate_texts.add(embedded_text)
                continue
            candidates[embedded_text] = {
                "contract_id": contract["contract_id"],
                "position": clause["position"],
                "clause_text": embedded_text,
            }
    # Many templates repeat verbatim across counterparties/contracts (only 2-3
    # phrasing variants per clause type). Sampling only from globally-unique
    # text keeps ground truth unambiguous — otherwise a paraphrase could
    # legitimately match a different point carrying identical text and get
    # scored as a miss.
    unique_pool = [c for text, c in candidates.items() if text not in duplicate_texts]
    assert len(unique_pool) >= SAMPLE_SIZE, (
        f"only {len(unique_pool)} clauses have globally-unique text, need {SAMPLE_SIZE}"
    )

    sampled = rng.sample(unique_pool, SAMPLE_SIZE)
    queries = []
    for c in sampled:
        queries.append(
            {
                "point_id": store.clause_point_id(c["contract_id"], c["position"]),
                "original_text": c["clause_text"],
                "query_text": paraphrase(c["clause_text"]),
            }
        )
    return queries


# ---------------------------------------------------------------------------
# Retrieval configurations under test
# ---------------------------------------------------------------------------


def _run_dense(text: str) -> tuple[list[str], float]:
    t0 = time.perf_counter()
    dense_vec = store.embed_dense_query(text)
    hits = store.query_dense(store.CLAUSE_MEMORY, dense_vec, limit=K, tenant_id=TENANT_ID)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return [str(h.id) for h in hits], elapsed_ms


def _run_sparse(text: str) -> tuple[list[str], float]:
    t0 = time.perf_counter()
    sparse_vec = store.embed_sparse_query(text)
    hits = store.query_sparse(store.CLAUSE_MEMORY, sparse_vec, limit=K, tenant_id=TENANT_ID)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return [str(h.id) for h in hits], elapsed_ms


def _run_hybrid(text: str) -> tuple[list[str], float]:
    t0 = time.perf_counter()
    dense_vec, sparse_vec = store.embed_query(text)
    hits = store.query_hybrid(store.CLAUSE_MEMORY, dense_vec, sparse_vec, limit=K, tenant_id=TENANT_ID)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return [str(h.id) for h in hits], elapsed_ms


CONFIGS: dict[str, Callable[[str], tuple[list[str], float]]] = {
    "dense": _run_dense,
    "sparse": _run_sparse,
    "hybrid": _run_hybrid,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def evaluate(queries: list[dict[str, Any]]) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    raw: dict[str, dict[str, list[float]]] = {
        name: {"recall_at_1": [], "recall_at_5": [], "reciprocal_ranks": [], "latencies_ms": []}
        for name in CONFIGS
    }
    per_query_records = []

    for q in queries:
        record: dict[str, Any] = {"point_id": q["point_id"], "query_text": q["query_text"]}
        for name, run in CONFIGS.items():
            retrieved_ids, elapsed_ms = run(q["query_text"])
            rank = retrieved_ids.index(q["point_id"]) + 1 if q["point_id"] in retrieved_ids else None

            raw[name]["recall_at_1"].append(1.0 if rank == 1 else 0.0)
            raw[name]["recall_at_5"].append(1.0 if rank is not None else 0.0)
            raw[name]["reciprocal_ranks"].append(1.0 / rank if rank else 0.0)
            raw[name]["latencies_ms"].append(elapsed_ms)
            record[name] = {"rank": rank, "latency_ms": round(elapsed_ms, 2)}
        per_query_records.append(record)

    summary = {
        name: {
            "recall_at_1": statistics.mean(m["recall_at_1"]),
            "recall_at_5": statistics.mean(m["recall_at_5"]),
            "mrr": statistics.mean(m["reciprocal_ranks"]),
            "latency_p50_ms": _percentile(m["latencies_ms"], 50),
            "latency_p95_ms": _percentile(m["latencies_ms"], 95),
        }
        for name, m in raw.items()
    }
    return summary, per_query_records


def _render_table(summary: dict[str, dict[str, float]]) -> Table:
    table = Table(title=f"Retrieval ablation: dense vs sparse vs hybrid (k={K}, n={SAMPLE_SIZE})")
    table.add_column("Configuration")
    table.add_column("Recall@1", justify="right")
    table.add_column("Recall@5", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("p50 latency (ms)", justify="right")
    table.add_column("p95 latency (ms)", justify="right")
    for name in ("dense", "sparse", "hybrid"):
        s = summary[name]
        table.add_row(
            name,
            f"{s['recall_at_1']:.3f}",
            f"{s['recall_at_5']:.3f}",
            f"{s['mrr']:.3f}",
            f"{s['latency_p50_ms']:.1f}",
            f"{s['latency_p95_ms']:.1f}",
        )
    return table


def _conclusion(summary: dict[str, dict[str, float]]) -> str:
    dense_r5 = summary["dense"]["recall_at_5"]
    hybrid_r5 = summary["hybrid"]["recall_at_5"]

    if dense_r5 == 0:
        return "Hybrid Recall@5 lift over dense-only: N/A (dense-only scored 0.0 Recall@5)"

    lift_pct = (hybrid_r5 - dense_r5) / dense_r5 * 100
    if lift_pct > 0:
        verdict = f"beats dense-only by {lift_pct:.1f}%"
    elif lift_pct < 0:
        verdict = f"trails dense-only by {abs(lift_pct):.1f}%"
    else:
        verdict = "ties dense-only"
    return f"Hybrid RRF {verdict} on Recall@5 ({dense_r5:.3f} -> {hybrid_r5:.3f})"


def main() -> None:
    rng = random.Random(SEED)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    console.print(f"Sampling {SAMPLE_SIZE} held-out clauses and paraphrasing them...")
    queries = build_query_set(rng)

    console.print(f"Running {len(CONFIGS)} configurations over {len(queries)} queries at k={K}...")
    summary, per_query = evaluate(queries)

    console.print(_render_table(summary))

    dump = {
        "sample_size": SAMPLE_SIZE,
        "k": K,
        "seed": SEED,
        "summary": summary,
        "queries": per_query,
    }
    OUTPUT_PATH.write_text(json.dumps(dump, indent=2), encoding="utf-8")
    console.print(f"Wrote {OUTPUT_PATH}")

    console.print()
    console.print(f"[bold]{_conclusion(summary)}[/bold]")


if __name__ == "__main__":
    main()
