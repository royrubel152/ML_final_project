"""
report.py — generate IMPROVEMENT_REPORT.md (+ charts) from experiments/results/runs.csv.

All numbers come from the runs table written by harness.run_arm, so the report
can never drift from the actual runs. The document is chronological (Stage A ->
B -> C -> K), with one comparison table per stage, the recommended winner, and an
overall Recall@k curve. This is the GitHub/PR-friendly twin of RESULTS.ipynb.

Usage:
    python -m experiments.v2.report
Safe to run after any subset of stages; missing stages are simply omitted.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"
RUNS_CSV = RESULTS_DIR / "runs.csv"
REPORT_MD = RESULTS_DIR / "IMPROVEMENT_REPORT.md"
RECALL_CHART = RESULTS_DIR / "recall_at_k.png"

K_VALUES = [1, 3, 5, 10]
_TABLE_METRICS = [f"recall@{k}" for k in K_VALUES] + ["mrr", "ndcg@5", "latency_p50_ms", "latency_p95_ms"]
_STAGE_NAMES = {"A": "Chunking", "B": "Retrieval", "C": "Embedding", "K": "k selection"}


def _aggregate(runs):
    """Return {arm_id: {metric: mean_score, ...signature}} from the long runs table."""
    import pandas as pd

    df = pd.DataFrame(runs)
    agg: dict[str, dict] = {}
    for arm_id, grp in df.groupby("arm_id"):
        metrics = grp.groupby("metric")["score_norm"].mean().to_dict()
        # latency rows store raw ms in raw_score; recompute p50/p95 from per-sample latency
        lat = grp[grp["metric"] == "latency_ms"]["raw_score"].astype(float).tolist()
        if lat:
            s = sorted(lat)
            metrics["latency_p50_ms"] = s[len(s) // 2]
            metrics["latency_p95_ms"] = s[min(len(s) - 1, int(0.95 * (len(s) - 1)))]
        sig = grp.iloc[0][["chunking", "embedding", "retrieval", "pool_k", "rerank_k", "context_k"]].to_dict()
        agg[arm_id] = {**metrics, **sig}
    return agg


def _stage_of(arm_id: str) -> str:
    return arm_id[0] if arm_id and arm_id[0] in _STAGE_NAMES else "?"


def _winner(arms: dict) -> str:
    return max(arms, key=lambda a: (arms[a].get("recall@5", 0.0), arms[a].get("mrr", 0.0)))


def _md_table(arms: dict) -> str:
    header = "| arm | " + " | ".join(_TABLE_METRICS) + " |"
    sep = "|" + "---|" * (len(_TABLE_METRICS) + 1)
    win = _winner(arms)
    lines = [header, sep]
    for arm_id, m in sorted(arms.items()):
        mark = " (winner)" if arm_id == win else ""
        cells = []
        for metric in _TABLE_METRICS:
            v = m.get(metric, 0.0)
            cells.append(f"{v:.1f}" if "latency" in metric else f"{v:.3f}")
        lines.append(f"| {arm_id}{mark} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _plot_recall(agg: dict) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for arm_id, m in sorted(agg.items()):
        ys = [m.get(f"recall@{k}", 0.0) for k in K_VALUES]
        ax.plot(K_VALUES, ys, marker="o", label=arm_id)
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(K_VALUES)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.set_title("Recall@k by arm — Hebrew MBA RAG")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(RECALL_CHART), dpi=150, bbox_inches="tight")
    return True


def build_report() -> str:
    import csv

    if not RUNS_CSV.exists():
        return f"No runs found at {RUNS_CSV}. Run a stage first (run_stage.py)."

    with open(RUNS_CSV, encoding="utf-8") as f:
        runs = list(csv.DictReader(f))
    # score_norm/raw_score are strings from CSV; coerce numerics
    for r in runs:
        for col in ("score_norm", "raw_score", "pool_k", "rerank_k", "context_k"):
            try:
                r[col] = float(r[col])
            except (TypeError, ValueError):
                pass

    agg = _aggregate(runs)
    has_chart = _plot_recall(agg)

    by_stage: dict[str, dict] = {}
    for arm_id, m in agg.items():
        by_stage.setdefault(_stage_of(arm_id), {})[arm_id] = m

    lines = ["# RAG Pipeline Improvement — Results", ""]
    lines.append("Generated from `experiments/results/runs.csv` (all numbers data-derived).")
    lines.append("")

    # Headline
    all_arms = agg
    if all_arms:
        best = _winner(all_arms)
        lines += [
            "## Headline", "",
            f"- Best arm overall: **{best}** "
            f"(R@5={all_arms[best].get('recall@5', 0):.3f}, "
            f"MRR={all_arms[best].get('mrr', 0):.3f}, "
            f"p95 latency={all_arms[best].get('latency_p95_ms', 0):.0f}ms).",
            f"- Pipeline: chunking=`{all_arms[best].get('chunking')}`, "
            f"embedding=`{all_arms[best].get('embedding')}`, "
            f"retrieval=`{all_arms[best].get('retrieval')}`.",
            "",
        ]

    for stage in ["A", "B", "C", "K"]:
        if stage not in by_stage:
            continue
        arms = by_stage[stage]
        win = _winner(arms)
        lines += [
            f"## Stage {stage} — {_STAGE_NAMES[stage]}", "",
            f"Variable changed: {_STAGE_NAMES[stage].lower()} only.", "",
            _md_table(arms), "",
            f"Winner: **{win}**. (Explanation: fill in why this arm won, tying the "
            f"number to a cause — see RESULTS.ipynb narrative.)", "",
        ]

    if has_chart:
        lines += ["## Recall@k curve", "", f"![Recall@k]({RECALL_CHART.name})", ""]

    lines += [
        "## Secretariat-only questions", "",
        "Questions with no source URL are excluded from retrieval metrics and "
        "must be answered by a human / secretariat hand-off (see plan Stage 0).", "",
        "## LLM-judge scores (Stage P)", "",
        "Populated after the full evaluation runs the 5 judges (faithfulness, "
        "correctness, context_relevance, answer_relevance, completeness).", "",
    ]
    return "\n".join(lines)


def main() -> None:
    text = build_report()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(text, encoding="utf-8")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
