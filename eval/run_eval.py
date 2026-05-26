"""
Eval runner for the MBA Academic Advisor chatbot.

Two modes:
  --mock   (default) Uses frozen fixture chunks + mock Gemini. Fast, free, no API calls.
  --live   Uses real FAISS index + real Gemini API. Requires GEMINI_API_KEY.

Usage:
  python eval/run_eval.py           # mock mode
  python eval/run_eval.py --live    # live mode
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

QUESTIONS_FILE = Path(__file__).parent / "eval_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"

VALID_SESSION_ID = "00000000-0000-0000-0000-000000000001"


def load_questions() -> list[dict]:
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def run_mock_eval() -> list[dict]:
    from eval.mocks import load_fixture_chunks, MockRAGRetriever, make_mock_gemini_model
    from fastapi.testclient import TestClient

    fixture_chunks = load_fixture_chunks()
    retriever = MockRAGRetriever(fixture_chunks)
    mock_model = make_mock_gemini_model()

    with patch("app.load_all_sources", return_value="fake content"), \
         patch("app.get_or_build_index", return_value=(None, fixture_chunks)), \
         patch("app.genai.GenerativeModel", return_value=mock_model), \
         patch("app.retrieve", side_effect=lambda q, idx, cks: retriever.retrieve(q)):

        from app import app as fastapi_app
        client = TestClient(fastapi_app)
        return _run_questions(client, load_questions())


def run_live_eval() -> list[dict]:
    from dotenv import load_dotenv
    load_dotenv()

    from fastapi.testclient import TestClient
    from app import app as fastapi_app
    client = TestClient(fastapi_app)
    return _run_questions(client, load_questions())


def _run_questions(client, questions: list[dict]) -> list[dict]:
    results = []
    for q in questions:
        resp = client.post("/chat", json={
            "message": q["question"],
            "session_id": VALID_SESSION_ID,
        })

        if resp.status_code != 200:
            results.append({
                "id": q["id"],
                "category": q.get("category", ""),
                "question": q["question"],
                "passed": False,
                "error": f"HTTP {resp.status_code}",
            })
            continue

        data = resp.json()
        reply = data.get("reply", "")
        sources = data.get("sources_used", [])
        chunks_found = data.get("chunks_found", 0)

        # Off-topic questions: must be short-circuited (chunks_found == 0)
        if q.get("expected_off_topic"):
            off_topic_triggered = chunks_found == 0
            contains_required = any(phrase in reply for phrase in q.get("must_contain_one_of", []))
            hallucinated = [w for w in q.get("should_not_hallucinate", []) if w in reply]
            passed = off_topic_triggered and contains_required and not hallucinated
            results.append({
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "passed": passed,
                "off_topic_triggered": off_topic_triggered,
                "contains_required": contains_required,
                "hallucination_count": len(hallucinated),
                "hallucinations": hallucinated,
            })
            continue

        # In-scope questions: keyword recall + source match + hallucination check
        keyword_hits = [kw for kw in q.get("expected_keywords", []) if kw in reply]
        keyword_recall = (
            len(keyword_hits) / len(q["expected_keywords"])
            if q.get("expected_keywords") else 1.0
        )
        source_match = any(s in " ".join(sources) for s in q.get("expected_sources", []))
        hallucinated = [w for w in q.get("should_not_hallucinate", []) if w in reply]

        passed = keyword_recall >= 0.5 and not hallucinated

        results.append({
            "id": q["id"],
            "category": q.get("category", ""),
            "question": q["question"],
            "passed": passed,
            "keyword_recall": round(keyword_recall, 2),
            "keyword_hits": keyword_hits,
            "source_match": source_match,
            "hallucination_count": len(hallucinated),
            "hallucinations": hallucinated,
            "chunks_found": chunks_found,
        })

    return results


def print_report(results: list[dict], mode: str):
    total = len(results)
    passed = sum(r["passed"] for r in results)
    in_scope = [r for r in results if "keyword_recall" in r]
    off_topic = [r for r in results if "off_topic_triggered" in r]

    avg_recall = (
        sum(r["keyword_recall"] for r in in_scope) / len(in_scope)
        if in_scope else 0.0
    )
    total_hallucinations = sum(r.get("hallucination_count", 0) for r in results)

    print(f"\n{'='*55}")
    print(f"  MBA ADVISOR EVAL — {mode.upper()} MODE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}")
    print(f"  Total questions : {total}")
    print(f"  Passed          : {passed}/{total}  ({passed/total:.0%})")
    print(f"  Avg recall      : {avg_recall:.0%}  (in-scope only)")
    print(f"  Hallucinations  : {total_hallucinations}")
    print(f"  Off-topic gate  : {sum(r.get('off_topic_triggered',False) for r in off_topic)}/{len(off_topic)} triggered correctly")
    print(f"{'='*55}")

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        cat = r.get("category", "")
        if "keyword_recall" in r:
            detail = f"recall={r['keyword_recall']:.0%}  hallu={r['hallucination_count']}  src={'ok' if r.get('source_match') else 'miss'}"
        elif "off_topic_triggered" in r:
            detail = f"gate={'ok' if r['off_topic_triggered'] else 'MISSED'}  hallu={r['hallucination_count']}"
        else:
            detail = r.get("error", "")
        print(f"  [{status}] {r['id']:<14} ({cat:<12})  {detail}")

    print(f"{'='*55}\n")


def save_results(results: list[dict], mode: str):
    RESULTS_DIR.mkdir(exist_ok=True)
    filename = RESULTS_DIR / f"{datetime.now().strftime('%Y-%m-%d_%H%M')}_{mode}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MBA Advisor eval runner")
    parser.add_argument("--live", action="store_true", help="Use real FAISS + Gemini API")
    parser.add_argument("--save", action="store_true", help="Save results to eval/results/")
    args = parser.parse_args()

    mode = "live" if args.live else "mock"
    print(f"[eval] Running in {mode} mode...")

    results = run_live_eval() if args.live else run_mock_eval()
    print_report(results, mode)

    if args.save:
        save_results(results, mode)
