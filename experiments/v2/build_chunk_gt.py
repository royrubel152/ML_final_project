"""
build_chunk_gt.py — Stage 0: semi-automated chunk-level ground truth.

Why: URL-level matching is too coarse for near-duplicate specialization pages,
producing the R@1=0 artifact. Chunk-level gold labels fix this by recording,
per question, the chunk_id(s) that actually contain the answer.

How (semi-automated, human-confirmed):
  1. Build baseline chunks and a dense index (gemini-embedding-2).
  2. For each in-corpus question, retrieve a high-recall pool (top-30).
  3. Extract key facts from the gold answer (course codes, credit values, dates,
     salient Hebrew tokens) and score each retrieved chunk by how many facts it
     contains.
  4. Write a DRAFT CSV (sample_id, chunk_ids, match_score, matched_preview,
     needs_review). A human confirms/edits, then sets needs_review=FALSE.

Output: evaluation_framework/chunk_ground_truth.csv (draft).
This script performs embedding/retrieval, so the friend runs it; it is not
executed at build time.

Usage:
    python -m experiments.v2.build_chunk_gt
    python -m experiments.v2.build_chunk_gt --pool-k 30 --min-facts 1
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from experiments.v2 import harness, retrievers
from experiments.v2.configs import ArmConfig

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_CSV = ROOT / "evaluation_framework" / "chunk_ground_truth.csv"

_COURSE_CODE = re.compile(r"\b\d{5}\b")
_CREDITS = re.compile(r"\d+(?:\.\d+)?\s*נ[\"״׳']?ז")
_DATES = re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b")
_STOPWORDS = {"של", "על", "עם", "אם", "כי", "זה", "הוא", "היא", "אך", "גם", "או", "אל", "מה"}


def extract_facts(answer: str) -> list[str]:
    """Salient, matchable facts from a gold answer."""
    facts = set(_COURSE_CODE.findall(answer))
    facts |= set(_CREDITS.findall(answer))
    facts |= set(_DATES.findall(answer))
    for tok in re.findall(r"[א-ת]{4,}", answer):
        if tok not in _STOPWORDS:
            facts.add(tok)
    return sorted(facts)


def score_chunk(chunk_text: str, facts: list[str]) -> int:
    return sum(1 for fact in facts if fact in chunk_text)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build draft chunk-level ground truth")
    ap.add_argument("--pool-k", type=int, default=30, help="retrieval pool size per question")
    ap.add_argument("--min-facts", type=int, default=1, help="min fact overlap to propose a chunk")
    ap.add_argument("--max-chunks", type=int, default=3, help="max proposed chunks per question")
    args = ap.parse_args()

    # Reuse the harness machinery: baseline chunks + dense gemini index, top-K dense.
    config = ArmConfig(
        arm_id="gt_builder", chunking="baseline", embedding="gemini-embedding-2",
        retrieval="dense", pool_k=args.pool_k, rerank_k=args.pool_k, context_k=args.pool_k,
    )
    from experiments.v2 import chunkers
    chunks = chunkers.build_chunks(config.chunking, enrich_meta=config.enrich)
    ctx = harness.build_context(config, chunks)
    samples = harness.load_in_corpus_samples()

    rows: list[dict] = []
    for i, sample in enumerate(samples):
        hits = retrievers.run_retrieval(
            "dense", sample.question, ctx,
            pool_k=args.pool_k, rerank_k=args.pool_k, context_k=args.pool_k,
        )
        facts = extract_facts(sample.ground_truth_answer)
        scored = [(h, score_chunk(h["text"], facts)) for h in hits]
        scored = [(h, s) for h, s in scored if s >= args.min_facts]
        scored.sort(key=lambda x: -x[1])
        chosen = scored[: args.max_chunks]

        if chosen:
            chunk_ids = ";".join(h["chunk_id"] for h, _ in chosen)
            preview = chosen[0][0]["text"][:120].replace("\n", " ")
            top_score = chosen[0][1]
        else:
            chunk_ids, preview, top_score = "", "", 0

        rows.append({
            "sample_id": sample.sample_id,
            "chunk_ids": chunk_ids,
            "match_score": top_score,
            "matched_preview": preview,
            "needs_review": "TRUE",   # human confirms then flips to FALSE
        })
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(samples)} questions labeled")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "chunk_ids", "match_score", "matched_preview", "needs_review"],
        )
        writer.writeheader()
        writer.writerows(rows)

    labeled = sum(1 for r in rows if r["chunk_ids"])
    print(f"\nWrote {OUT_CSV}")
    print(f"  {labeled}/{len(rows)} questions got >=1 proposed chunk; review needs_review=TRUE rows.")


if __name__ == "__main__":
    main()
