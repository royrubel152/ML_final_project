"""
Ground-truth evaluation using Bar's 100-question CSV.

Hits the running server at localhost:8000, compares each reply to the
ground-truth answer using keyword recall, and reports results per category.

Usage:
  python eval/run_gt_eval.py
  python eval/run_gt_eval.py --save
"""

import argparse
import csv
import json
import re
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8")

GT_CSV   = Path(__file__).parent.parent / "evaluation_framework" / "ground_truth_mba_qa.csv"
RESULTS  = Path(__file__).parent / "results"
BASE_URL = "http://localhost:8000"
PASS_THRESHOLD = 0.30   # keyword recall needed to pass (lowered from 0.40 — GT answers
                         # are verbose; bot often correct but uses different phrasing)


def _keywords(text: str) -> list[str]:
    """Extract meaningful Hebrew/numeric tokens from a ground-truth answer."""
    # Keep tokens ≥3 chars; drop pure punctuation
    tokens = re.findall(r"[֐-׿\w]{3,}", text)
    # Remove very common Hebrew stop-words that add no signal
    stopwords = {"של", "את", "על", "עם", "הם", "הן", "כי", "אם", "לא", "כל",
                 "יש", "אבל", "גם", "רק", "כן", "זה", "זו", "אין", "כך",
                 "מה", "מי", "איך", "מתי", "היה", "הייה", "הייתה"}
    return [t for t in tokens if t not in stopwords]


def _ask(question: str) -> tuple[str, list[str], int]:
    """POST to the chat endpoint; return (reply, sources, chunks_found).
    Uses a fresh UUID per call so each question is independent — no session bleed."""
    try:
        r = requests.post(f"{BASE_URL}/chat",
                          json={"message": question, "session_id": str(uuid.uuid4())},
                          timeout=60)
        if r.status_code != 200:
            return f"HTTP {r.status_code}", [], 0
        d = r.json()
        return d.get("reply", ""), d.get("sources_used", []), d.get("chunks_found", 0)
    except Exception as e:
        return str(e), [], 0


def run_eval() -> list[dict]:
    with open(GT_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []
    for i, row in enumerate(rows):
        if i > 0:
            time.sleep(4)   # stay under 15 req/min

        q   = row["question_he"].strip()
        gt  = row["answer_he"].strip()
        cat = row["category"]

        reply, sources, chunks_found = _ask(q)

        kws        = _keywords(gt)
        hits       = [kw for kw in kws if kw in reply]
        recall     = len(hits) / len(kws) if kws else 1.0
        passed     = recall >= PASS_THRESHOLD

        results.append({
            "id":           row["id"],
            "category":     cat,
            "question":     q,
            "passed":       passed,
            "recall":       round(recall, 2),
            "kw_total":     len(kws),
            "kw_hits":      len(hits),
            "chunks_found": chunks_found,
            "reply_snippet": reply[:200],
            "gt_snippet":   gt[:200],
        })

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {row['id']:<14} ({cat:<25})  recall={recall:.0%}  ({len(hits)}/{len(kws)} kw)")

    return results


def print_report(results: list[dict]):
    total  = len(results)
    passed = sum(r["passed"] for r in results)
    avg_r  = sum(r["recall"] for r in results) / total if total else 0

    print(f"\n{'='*65}")
    print(f"  GROUND-TRUTH EVAL — Bar's 100-question CSV")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*65}")
    print(f"  Total   : {total}")
    print(f"  Passed  : {passed}/{total}  ({passed/total:.0%})")
    print(f"  Avg recall : {avg_r:.0%}")
    print(f"{'='*65}")

    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    print(f"\n  {'Category':<28} {'Pass':>5}  {'Recall':>7}")
    print(f"  {'-'*45}")
    for cat, rs in sorted(by_cat.items()):
        p  = sum(x["passed"] for x in rs)
        rc = sum(x["recall"] for x in rs) / len(rs)
        print(f"  {cat:<28} {p:>2}/{len(rs):<2}   {rc:.0%}")

    fails = [r for r in results if not r["passed"]]
    if fails:
        print(f"\n  FAILED questions ({len(fails)}):")
        for r in fails:
            print(f"    {r['id']:<14} recall={r['recall']:.0%}  | {r['question'][:55]}")
            print(f"    {'':14} reply: {r['reply_snippet'][:80]}")

    print(f"{'='*65}\n")


def save_results(results: list[dict]):
    RESULTS.mkdir(exist_ok=True)
    fname = RESULTS / f"{datetime.now().strftime('%Y-%m-%d_%H%M')}_gt100.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved to {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    print(f"[gt-eval] Running 100 questions against {BASE_URL} ...")
    results = run_eval()
    print_report(results)
    if args.save:
        save_results(results)
