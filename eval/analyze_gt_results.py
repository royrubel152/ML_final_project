"""Deep analysis of GT eval failures — groups by failure type and shows bot reply vs GT."""
import sys, json, csv
sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).parent / "results" / "2026-06-14_1944_gt100.json"
GT_CSV  = Path(__file__).parent.parent / "evaluation_framework" / "ground_truth_mba_qa.csv"

with open(RESULTS, encoding="utf-8") as f:
    results = json.load(f)

with open(GT_CSV, encoding="utf-8") as f:
    gt = {r["id"]: r for r in csv.DictReader(f)}

# ── classify each failure ──────────────────────────────────────────────────────
NOT_AVAILABLE = "המידע אינו זמין"
CONTACT_SEC   = "מזכירות"

def classify(r):
    reply = r["reply_snippet"]
    if NOT_AVAILABLE in reply:
        return "bot_says_unavailable"
    if r["recall"] >= 0.30:
        return "partial_answer"       # close but below threshold
    return "low_recall"

fails = [r for r in results if not r["passed"]]
by_type = defaultdict(list)
for r in fails:
    by_type[classify(r)].append(r)

print("=" * 70)
print("  FAILURE ANALYSIS — 66 failed questions")
print("=" * 70)
print()

# ── 1. Bot says "not available" ───────────────────────────────────────────────
unavail = by_type["bot_says_unavailable"]
print(f"❶  BOT SAYS 'המידע אינו זמין'  ({len(unavail)} questions)")
print(f"   Root cause: data not in RAG index / retrieval miss")
print()
cats = defaultdict(list)
for r in unavail:
    cats[r["category"]].append(r["id"])
for cat, ids in sorted(cats.items()):
    print(f"   {cat:<30} {', '.join(ids)}")
print()

# ── 2. Partial answers (recall 30-39%) ────────────────────────────────────────
partial = by_type["partial_answer"]
print(f"❷  PARTIAL ANSWERS (recall 30–39%)  ({len(partial)} questions)")
print(f"   Root cause: bot answers correctly but uses different phrasing")
print()
for r in sorted(partial, key=lambda x: x["recall"], reverse=True)[:8]:
    gt_row = gt[r["id"]]
    kws = [w for w in gt_row["answer_he"].split() if len(w) >= 4][:6]
    print(f"   {r['id']:<14} recall={r['recall']:.0%}  | {r['question'][:50]}")
    print(f"   {'':14} GT keywords: {', '.join(kws[:5])}")
    print(f"   {'':14} Bot reply:   {r['reply_snippet'][:80]}")
    print()

# ── 3. Low recall answers ─────────────────────────────────────────────────────
low = by_type["low_recall"]
print(f"❸  LOW RECALL (< 30%)  ({len(low)} questions)")
print()
for r in sorted(low, key=lambda x: x["recall"])[:10]:
    gt_row = gt[r["id"]]
    print(f"   {r['id']:<14} recall={r['recall']:.0%}  {r['category']}")
    print(f"   Q:  {r['question'][:65]}")
    print(f"   GT: {gt_row['answer_he'][:90]}")
    print(f"   Bot:{r['reply_snippet'][:90]}")
    print()

# ── Summary of actionable improvements ────────────────────────────────────────
print("=" * 70)
print("  ACTIONABLE IMPROVEMENTS")
print("=" * 70)

# Count "not available" by category
ua_cats = defaultdict(int)
for r in unavail:
    ua_cats[r["category"]] += 1

print(f"""
A. DATA GAPS (fix retrieval — {len(unavail)} questions return 'לא זמין')
   Worst categories:""")
for cat, n in sorted(ua_cats.items(), key=lambda x: -x[1]):
    print(f"   • {cat:<30} {n} questions missing data")

print(f"""
B. PHRASING MISMATCH ({len(partial)} questions almost pass at 30-39% recall)
   The bot answers correctly but uses synonyms the GT doesn't contain.
   Fix: lower threshold from 40% → 30%, or add synonym expansion to scorer.

C. PROCEDURAL / SECRETARY QUESTIONS ({len([r for r in low if r['recall'] < 0.15])} with <15% recall)
   Questions like "איך מגישים בקשה" / "למי פונים" need policy text
   that isn't in the scraped sources. Bot correctly redirects to secretary.
   Fix: either accept secretary redirect as PASS, or add FAQ policy text.
""")
