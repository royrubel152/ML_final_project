"""
select_k.py — Stage K: choose (pool_k, rerank_k, context_k) by the recall-vs-latency knee.

On the single locked pipeline (Stage C winner), sweep the k-triples in
configs.DEFAULT_K_TRIPLES, then:
  * plot Recall@k and nDCG@5 vs context_k AND p50/p95 latency vs context_k,
  * pick the knee: smallest triple whose marginal Recall@5 gain over the previous
    triple is < --knee-threshold AND whose p95 latency is within --latency-budget.

Keeps final context_k small (latency + faithfulness); only the reranker sees the
larger pool. Writes a chart and prints the recommended triple. Runs real
retrieval, so the friend executes it; not run at build time.

Usage:
    python -m experiments.v2.select_k --chunking breadcrumb --retrieval rerank_ce --embedding gemini-embedding-2
    python -m experiments.v2.select_k ... --latency-budget-ms 800 --knee-threshold 0.02
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.v2 import configs, harness
from experiments.v2.configs import ArmConfig

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "experiments" / "results"
CHART_PATH = RESULTS_DIR / "stage_k_knee.png"


def choose_knee(rows: list[dict], knee_threshold: float, latency_budget_ms: float) -> dict:
    """Smallest triple within latency budget whose marginal Recall@5 gain < threshold."""
    rows_sorted = sorted(rows, key=lambda r: r["context_k"])
    within = [r for r in rows_sorted if r["latency_p95_ms"] <= latency_budget_ms] or rows_sorted
    chosen = within[0]
    for prev, cur in zip(within, within[1:]):
        if (cur["recall@5"] - prev["recall@5"]) < knee_threshold:
            chosen = prev
            break
        chosen = cur
    return chosen


def plot(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["context_k"])
    xs = [r["context_k"] for r in rows]
    recall = [r["recall@5"] for r in rows]
    ndcg = [r["ndcg@5"] for r in rows]
    p95 = [r["latency_p95_ms"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(xs, recall, "o-", color="#2ecc71", label="Recall@5")
    ax1.plot(xs, ndcg, "s-", color="#3498db", label="nDCG@5")
    ax1.set_xlabel("context_k")
    ax1.set_ylabel("quality")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(xs, p95, "^--", color="#e74c3c", label="p95 latency (ms)")
    ax2.set_ylabel("latency p95 (ms)")
    ax1.grid(True, alpha=0.3, linestyle="--")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    ax1.set_title("Stage K — recall/nDCG vs latency (knee selection)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(CHART_PATH), dpi=150, bbox_inches="tight")
    print(f"  chart -> {CHART_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage K: latency-aware k selection")
    ap.add_argument("--chunking", required=True)
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--embedding", default=configs.FIXED_EMBEDDER)
    ap.add_argument("--latency-budget-ms", type=float, default=1000.0)
    ap.add_argument("--knee-threshold", type=float, default=0.02)
    args = ap.parse_args()

    locked = ArmConfig(
        arm_id="K_locked", chunking=args.chunking,
        embedding=args.embedding, retrieval=args.retrieval,
    )
    arms = configs.stage_k_arms(locked, configs.DEFAULT_K_TRIPLES)

    rows: list[dict] = []
    for arm in arms:
        m = harness.run_arm(arm)
        rows.append({
            "context_k": arm.context_k, "pool_k": arm.pool_k, "rerank_k": arm.rerank_k,
            "recall@5": m.get("recall@5", 0.0), "ndcg@5": m.get("ndcg@5", 0.0),
            "latency_p50_ms": m.get("latency_p50_ms", 0.0),
            "latency_p95_ms": m.get("latency_p95_ms", 0.0),
        })

    plot(rows)
    knee = choose_knee(rows, args.knee_threshold, args.latency_budget_ms)
    print("\nRecommended k-triple (knee rule):")
    print(f"  pool_k={knee['pool_k']}  rerank_k={knee['rerank_k']}  context_k={knee['context_k']}")
    print(f"  Recall@5={knee['recall@5']:.3f}  p95_latency={knee['latency_p95_ms']:.0f}ms")


if __name__ == "__main__":
    main()
