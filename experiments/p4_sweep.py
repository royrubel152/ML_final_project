"""
P4 K_CHILDREN sweep — find optimal number of children to retrieve in stage 1.

Keeps P4_K_PARENTS=20 and P4_K_FETCH=10 fixed.
Sweeps K_CHILDREN in [30, 50, 100, 150, 200].

Models loaded ONCE; only retrieve_p4() is called repeatedly with different k_children.

Usage:
  python experiments/p4_sweep.py
  python experiments/p4_sweep.py --k 30 50 100   # custom range
"""
import os
os.environ["USE_TF"]  = "0"
os.environ["USE_JAX"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]     = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"]    = "1"

import argparse, sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from experiments.retrieval_experiment import (
    K_VALUES, P4_K_PARENTS, P4_K_FETCH,
    load_ground_truth, load_semantic_chunks, build_parent_child_chunks,
    embed_with_e5, build_faiss_index, build_bm25_index,
    _fingerprint, _load_or_embed,
    retrieve_p4, evaluate_pipeline,
)

DEFAULT_K_CHILDREN = [30, 50, 100, 150, 200]


def run_sweep(k_children_values: list[int]):
    from sentence_transformers import SentenceTransformer, CrossEncoder

    print("\n[P4 Sweep] Loading data...")
    samples    = load_ground_truth()
    chunks_sem = load_semantic_chunks()

    print("  Loading E5-large (once)...")
    model_e5 = SentenceTransformer("intfloat/multilingual-e5-large")
    print("  Loading bge-reranker-base (once)...")
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    _, children = build_parent_child_chunks(chunks_sem, child_size=250)
    print(f"  Parent chunks: {len(chunks_sem)} | Child chunks: {len(children)}")

    fpc           = _fingerprint(children)
    vecs_children = _load_or_embed(f"p4_children_e5_{fpc}", children,
                                   lambda c: embed_with_e5(c, model_e5))
    idx_children  = build_faiss_index(vecs_children)
    bm25_children = build_bm25_index(children)

    state = {
        "model_e5":      model_e5,
        "reranker":      reranker,
        "chunks_sem":    chunks_sem,
        "children":      children,
        "idx_children":  idx_children,
        "bm25_children": bm25_children,
    }

    all_results: dict[str, dict] = {}

    for kc in k_children_values:
        label = f"P4 k_children={kc:>3d} | k_parents={P4_K_PARENTS} | k_fetch={P4_K_FETCH}"
        fn    = partial(retrieve_p4, k_children=kc, k_parents=P4_K_PARENTS, top_k=P4_K_FETCH)
        all_results[label] = evaluate_pipeline(label, fn, samples, state)

    _print_sweep_table(all_results, k_children_values)


def _print_sweep_table(all_results: dict, k_children_values: list[int]):
    cols       = [f"recall@{k}" for k in K_VALUES] + ["mrr"]
    col_labels = [f"R@{k}" for k in K_VALUES] + ["MRR"]

    print("\n" + "═" * 85)
    print("  P4 K_CHILDREN SWEEP  —  100 GT Questions  (k_parents=20, k_fetch=10 fixed)")
    print("═" * 85)
    header = f"  {'k_children':<10}" + "".join(f" {h:>7}" for h in col_labels)
    print(header)
    print("  " + "─" * 83)

    for kc, (label, metrics) in zip(k_children_values, all_results.items()):
        row = f"  {kc:<10}"
        for col in cols:
            v = metrics.get(col, 0.0)
            row += f" {v:>7.3f}"
        print(row)

    print("  " + "─" * 83)
    print()

    # Highlight best per column
    best: dict[str, tuple[int, float]] = {}
    for kc, (label, metrics) in zip(k_children_values, all_results.items()):
        for col in cols:
            v = metrics.get(col, 0.0)
            if col not in best or v > best[col][1]:
                best[col] = (kc, v)

    print("  Best k_children per metric:")
    for col, label in zip(cols, col_labels):
        kc, val = best[col]
        print(f"    {label:>5}: k_children={kc}  ({val:.3f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="P4 K_CHILDREN sweep")
    parser.add_argument(
        "--k", nargs="+", type=int, default=DEFAULT_K_CHILDREN,
        metavar="N", help="k_children values to sweep (default: 30 50 100 150 200)",
    )
    args = parser.parse_args()
    run_sweep(args.k)


if __name__ == "__main__":
    main()
