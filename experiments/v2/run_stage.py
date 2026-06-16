"""
run_stage.py — driver for Stages A / B / C (the OVAT ablations).

Each invocation runs one stage's arms (which differ in exactly one dimension,
enforced by configs.check_ovat), prints a comparison table, and prints the
recommended winner (highest Recall@5, tie-broken by MRR). The winner is NOT
auto-applied — you pass it explicitly into the next stage so the locked context
is always visible and intentional.

Usage:
    # Stage A: chunking
    python -m experiments.v2.run_stage --stage A

    # Stage B: retrieval (pass Stage A winner)
    python -m experiments.v2.run_stage --stage B --chunking breadcrumb

    # Stage C: embedding (pass A + B winners)
    python -m experiments.v2.run_stage --stage C --chunking breadcrumb --retrieval rerank_ce

Runs real embedding/retrieval, so the friend executes it; not run at build time.
"""
from __future__ import annotations

import argparse

from experiments.v2 import configs, harness


def _winner(results: dict[str, dict]) -> str:
    """Pick the arm with the best Recall@5, tie-broken by MRR."""
    return max(results, key=lambda a: (results[a].get("recall@5", 0.0), results[a].get("mrr", 0.0)))


def _print_table(results: dict[str, dict]) -> None:
    cols = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@5",
            "latency_p50_ms", "latency_p95_ms"]
    header = f"{'arm':<24}" + "".join(f"{c:>16}" for c in cols)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for arm, m in results.items():
        row = f"{arm:<24}"
        for c in cols:
            row += f"{m.get(c, 0.0):>16.3f}"
        print(row)
    print("=" * len(header))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an OVAT stage (A/B/C)")
    ap.add_argument("--stage", required=True, choices=["A", "B", "C"])
    ap.add_argument("--chunking", help="locked Stage A winner (required for B and C)")
    ap.add_argument("--retrieval", help="locked Stage B winner (required for C)")
    args = ap.parse_args()

    if args.stage == "A":
        arms = configs.stage_a_arms()
    elif args.stage == "B":
        if not args.chunking:
            ap.error("--chunking (Stage A winner) is required for Stage B")
        arms = configs.stage_b_arms(args.chunking)
    else:  # C
        if not (args.chunking and args.retrieval):
            ap.error("--chunking and --retrieval (A and B winners) are required for Stage C")
        arms = configs.stage_c_arms(args.chunking, args.retrieval)

    print(f"Stage {args.stage}: {len(arms)} arms (OVAT guard passed)")

    results: dict[str, dict] = {}
    for arm in arms:
        try:
            results[arm.arm_id] = harness.run_arm(arm)
        except RuntimeError as exc:
            # e.g. ColBERT-v2 optional dependency missing for arm B2b
            print(f"  [skip] {arm.arm_id}: {exc}")

    _print_table(results)
    if results:
        win = _winner(results)
        print(f"\nRecommended Stage {args.stage} winner: {win}")
        print(f"  -> {results[win]}")
        print("Pass this winner into the next stage's --chunking/--retrieval flag.")


if __name__ == "__main__":
    main()
