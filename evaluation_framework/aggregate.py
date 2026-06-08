"""
aggregate.py — summarize results, save/load parquets, plot recall@k curves.
"""
from __future__ import annotations

import pathlib

import pandas as pd

_RESULTS_DIR = pathlib.Path(__file__).parent / "results"


# ── Save / Load ───────────────────────────────────────────────────

def save_results(results_df: pd.DataFrame, run_id: str) -> pathlib.Path:
    _RESULTS_DIR.mkdir(exist_ok=True)
    path = _RESULTS_DIR / f"{run_id}.parquet"
    results_df.to_parquet(path, index=False)
    print(f"[aggregate] Saved {len(results_df)} rows → {path}")
    return path


def load_results(run_id: str) -> pd.DataFrame:
    path = _RESULTS_DIR / f"{run_id}.parquet"
    return pd.read_parquet(path)


def load_all_results() -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in _RESULTS_DIR.glob("*.parquet")]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Summarize ─────────────────────────────────────────────────────

def summarize(results_df: pd.DataFrame) -> dict:
    """
    Returns a dict with:
      - per_metric: {metric -> mean score_norm}
      - per_category: {category -> {metric -> mean}}
      - per_archetype: {archetype -> {metric -> mean}}
      - per_source_type: {source_type -> {metric -> mean}}
    """
    def _pivot(df: pd.DataFrame, groupby: str) -> dict:
        out = {}
        for key, grp in df.groupby(groupby, dropna=False):
            out[str(key)] = grp.groupby("metric")["score_norm"].mean().to_dict()
        return out

    return {
        "per_metric":      results_df.groupby("metric")["score_norm"].mean().to_dict(),
        "per_category":    _pivot(results_df, "category"),
        "per_archetype":   _pivot(results_df, "archetype"),
        "per_source_type": _pivot(results_df, "source_type"),
    }


# ── Recall@k plots ────────────────────────────────────────────────

def plot_recall_at_k(
    run_ids: list[str] | None = None,
    save_dir: str | pathlib.Path = _RESULTS_DIR,
) -> None:
    """
    Produce two PNG charts and save them to results/:

    Chart 1 — Overall recall@k: one line per run_id.
    Chart 2 — Per-category recall@k: one line per category (latest run or specified run).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting: pip install matplotlib")

    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    if run_ids:
        frames = [load_results(rid) for rid in run_ids]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = load_all_results()

    if df.empty:
        print("[aggregate] No results to plot.")
        return

    recall_df = df[df["metric"].str.startswith("recall@")].copy()
    if recall_df.empty:
        print("[aggregate] No recall@k metrics found.")
        return

    recall_df["k"] = recall_df["metric"].str.extract(r"recall@(\d+)").astype(int)

    # Chart 1: Overall recall@k, one line per run_id
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for rid, grp in recall_df.groupby("run_id"):
        curve = grp.groupby("k")["score_norm"].mean().sort_index()
        ax1.plot(curve.index, curve.values, marker="o", label=str(rid))
    ax1.set_xlabel("k")
    ax1.set_ylabel("Recall@k (avg)")
    ax1.set_title("Overall Recall@k by Experiment Run")
    ax1.legend(title="run_id", fontsize=8)
    ax1.set_ylim(0, 1.05)
    out1 = save_dir / "recall_at_k_overall.png"
    fig1.tight_layout()
    fig1.savefig(out1, dpi=150)
    plt.close(fig1)
    print(f"[aggregate] Chart 1 saved → {out1}")

    # Chart 2: Per-category recall@k (latest run or first in list)
    target_run = run_ids[-1] if run_ids else recall_df["run_id"].iloc[-1]
    cat_df = recall_df[recall_df["run_id"] == target_run]
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for cat, grp in cat_df.groupby("category", dropna=False):
        curve = grp.groupby("k")["score_norm"].mean().sort_index()
        ax2.plot(curve.index, curve.values, marker="o", label=str(cat))
    ax2.set_xlabel("k")
    ax2.set_ylabel("Recall@k (avg)")
    ax2.set_title(f"Per-Category Recall@k — run: {target_run}")
    ax2.legend(title="category", fontsize=8)
    ax2.set_ylim(0, 1.05)
    out2 = save_dir / "recall_at_k_per_category.png"
    fig2.tight_layout()
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    print(f"[aggregate] Chart 2 saved → {out2}")
