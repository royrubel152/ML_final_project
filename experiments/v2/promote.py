"""
promote.py — Stage P helper: capture the winning configuration for rag.py.

This does NOT edit rag.py automatically (promotion happens only after the
results exist and have been reviewed). It records the winning ArmConfig to
``experiments/results/winning_config.json`` and prints the exact constants to
change in rag.py, including the gemini-embedding-2 migration note (different
vector space + inline task instructions vs the older `001`).

Usage:
    python -m experiments.v2.promote --chunking breadcrumb --retrieval rerank_ce \
        --embedding neodictabert --pool-k 30 --rerank-k 20 --context-k 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.v2.configs import ArmConfig

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "experiments" / "results"
OUT_JSON = RESULTS_DIR / "winning_config.json"

_EMBED_MODEL_HINT = {
    "gemini-embedding-2": "models/gemini-embedding-2  (no task_type; inline instructions; re-embed — different space from 001)",
    "neodictabert": "dicta-il/neodictabert-bilingual-embed  (local, 768-dim, trust_remote_code=True)",
    "bge-m3": "BAAI/bge-m3  (local, 1024-dim, FlagEmbedding)",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Record winning config for rag.py promotion")
    ap.add_argument("--chunking", required=True)
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--pool-k", type=int, default=30)
    ap.add_argument("--rerank-k", type=int, default=20)
    ap.add_argument("--context-k", type=int, default=5)
    args = ap.parse_args()

    config = ArmConfig(
        arm_id="winner", chunking=args.chunking, embedding=args.embedding,
        retrieval=args.retrieval, pool_k=args.pool_k, rerank_k=args.rerank_k,
        context_k=args.context_k,
    )
    config.validate()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(config.as_row(), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {OUT_JSON}\n")
    print("Apply these to rag.py (manually, after reviewing results):")
    print(f"  EMBED_MODEL  -> {_EMBED_MODEL_HINT.get(args.embedding, args.embedding)}")
    print(f"  chunking     -> {args.chunking} strategy (see experiments/v2/chunkers.py)")
    print(f"  retrieval    -> {args.retrieval} (see experiments/v2/retrievers.py)")
    print(f"  POOL_K       -> {args.pool_k}")
    print(f"  RERANK_K     -> {args.rerank_k}")
    print(f"  CONTEXT_K    -> {args.context_k}   (replaces single TOP_K)")
    print("\nThen re-run the full evaluation (deterministic + 5 LLM judges) and report.py.")


if __name__ == "__main__":
    main()
