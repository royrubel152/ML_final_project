"""
Runs P2, P3, P4 each in their own subprocess (so RAM is released between runs),
then combines the saved JSON results into a single comparison table + charts.

Usage:
  python experiments/orchestrate.py              # all three pipelines
  python experiments/orchestrate.py --skip-p2    # skip Gemini (no API cost)
  python experiments/orchestrate.py --only P3    # run just one pipeline
"""
import argparse
import json
import subprocess
import sys
import os
from pathlib import Path

os.environ["USE_TF"]  = "0"
os.environ["USE_JAX"] = "0"

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

K_VALUES = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
PIPELINES = ["P2", "P3", "P4"]

LABELS = {
    "P2": "P2: Gemini-001 + Semantic (prod)",
    "P3": "P3: E5-Large + BM25 Hybrid",
    "P4": "P4: Parent-Child + bge-reranker",
}


def run_pipeline(name: str) -> Path | None:
    out_path = RESULTS_DIR / f"{name.lower()}.json"
    if out_path.exists():
        print(f"\n[{name}] Using cached results from {out_path.name}")
        return out_path

    print(f"\n{'='*60}")
    print(f"  Running pipeline {name}...")
    print(f"{'='*60}")
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_pipeline.py"),
        "--pipeline", name,
        "--out", str(out_path),
    ]
    env = dict(os.environ)
    env["USE_TF"]  = "0"
    env["USE_JAX"] = "0"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.run(cmd, env=env, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(f"  [{name}] FAILED with exit code {proc.returncode}")
        return None
    return out_path


def load_results(paths: dict[str, Path]) -> dict[str, dict]:
    results = {}
    for name, path in paths.items():
        if path is None or not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            results[name] = json.load(f)
    return results


def print_table(results: dict[str, dict]):
    headers = ["Pipeline"] + [f"R@{k}" for k in K_VALUES] + ["MRR"]
    rows = []
    for name, r in results.items():
        row = [LABELS.get(name, name)]
        for k in K_VALUES:
            row.append(f"{r.get(f'recall@{k}', 0):.3f}")
        row.append(f"{r.get('mrr', 0):.3f}")
        rows.append(row)

    col_w = [max(len(h), max(len(r[i]) for r in rows)) + 2
             for i, h in enumerate(headers)]

    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    hdr = "|" + "|".join(h.center(w) for h, w in zip(headers, col_w)) + "|"

    print("\n" + "═" * 70)
    print("  RETRIEVAL EVALUATION — 100 GT questions, URL-level match")
    print("═" * 70)
    print(sep); print(hdr); print(sep)
    for row in rows:
        print("|" + "|".join(v.center(w) for v, w in zip(row, col_w)) + "|")
    print(sep)

    # per-category MRR
    categories = set()
    for r in results.values():
        categories |= set(r.get("per_category", {}).keys())
    categories = sorted(categories)

    if categories:
        print("\n  Per-Category MRR")
        cw = max(len(c) for c in categories) + 2
        cat_headers = ["Category"] + list(results.keys())
        cw2 = 8
        print(f"  {'Category':<{cw}}" + "".join(f"  {n:^{cw2}}" for n in results))
        print("  " + "-" * (cw + cw2 * len(results) + 2 * len(results)))
        for cat in categories:
            row_s = f"  {cat:<{cw}}"
            for r in results.values():
                v = r.get("per_category", {}).get(cat, {}).get("mrr", float("nan"))
                row_s += f"  {v:^{cw2}.3f}"
            print(row_s)


def plot_charts(results: dict[str, dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping charts")
        return

    colors  = {"P2": "#4e8ef7", "P3": "#f7a44e", "P4": "#5dbf72"}
    markers = {"P2": "o", "P3": "s", "P4": "^"}

    # ── Overall Recall@k ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in results.items():
        ys = [r.get(f"recall@{k}", 0) for k in K_VALUES]
        ax.plot(K_VALUES, ys,
                label=LABELS.get(name, name),
                color=colors.get(name, None),
                marker=markers.get(name, "o"),
                linewidth=2.2, markersize=7)
    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("Recall@k", fontsize=12)
    ax.set_title("Recall@k — MBA Hebrew RAG Retrieval Comparison", fontsize=13)
    ax.set_xticks(K_VALUES)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    path1 = RESULTS_DIR / "recall_at_k.png"
    fig.savefig(path1, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Chart saved: {path1}")

    # ── Per-Category subplots ─────────────────────────────────────────────────
    categories = set()
    for r in results.values():
        categories |= set(r.get("per_category", {}).keys())
    categories = sorted(categories)

    if not categories:
        return

    ncols = 4
    nrows = (len(categories) + ncols - 1) // ncols
    fig2, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
    fig2.suptitle("Recall@k by Category", fontsize=14, fontweight="bold")

    for i, cat in enumerate(categories):
        ax = axes[i // ncols][i % ncols]
        for name, r in results.items():
            cat_data = r.get("per_category", {}).get(cat, {})
            ys = [cat_data.get(f"recall@{k}", 0) for k in K_VALUES]
            ax.plot(K_VALUES, ys,
                    label=name,
                    color=colors.get(name, None),
                    marker=markers.get(name, "o"),
                    linewidth=1.8, markersize=5)
        ax.set_title(cat, fontsize=9)
        ax.set_xticks(K_VALUES)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    # hide unused subplots
    for j in range(len(categories), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    path2 = RESULTS_DIR / "recall_by_category.png"
    fig2.savefig(path2, bbox_inches="tight", dpi=150)
    plt.close(fig2)
    print(f"  Chart saved: {path2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-p2", action="store_true", help="Skip P2 (Gemini)")
    parser.add_argument("--only",    type=str,            help="Run only this pipeline, e.g. P3")
    parser.add_argument("--no-cache", action="store_true", help="Re-run even if results exist")
    args = parser.parse_args()

    pipelines = PIPELINES.copy()
    if args.skip_p2:
        pipelines = [p for p in pipelines if p != "P2"]
    if args.only:
        pipelines = [args.only.upper()]

    if args.no_cache:
        for p in pipelines:
            path = RESULTS_DIR / f"{p.lower()}.json"
            if path.exists():
                path.unlink()
                print(f"  Cleared cache: {path.name}")

    result_paths: dict[str, Path] = {}
    for p in pipelines:
        result_paths[p] = run_pipeline(p)

    results = load_results(result_paths)
    if not results:
        print("\nNo results to display.")
        return

    print_table(results)
    plot_charts(results)

    print("\n" + "═" * 70)
    print("  Done. Results in experiments/results/")
    print("═" * 70)


if __name__ == "__main__":
    main()
