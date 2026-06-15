import sys, json, glob, os
sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict

files = sorted(glob.glob(r"C:\Users\rubro\OneDrive\Desktop\final_project_big\eval\results\*gt100*.json"))
if not files:
    print("No gt100 results file found yet.")
    sys.exit(1)

with open(files[-1], encoding="utf-8") as f:
    data = json.load(f)

passed = sum(1 for r in data if r["passed"])
total = len(data)
avg_recall = sum(r["recall"] for r in data) / total

print(f"=" * 60)
print(f"  GROUND TRUTH EVAL — Bar's 100-question CSV")
print(f"=" * 60)
print(f"  Total   : {total}")
print(f"  Passed  : {passed}/{total}  ({passed/total:.0%})")
print(f"  Avg recall : {avg_recall:.0%}")
print()

by_cat = defaultdict(list)
for r in data:
    by_cat[r["category"]].append(r)

print(f"  {'Category':<30} {'Pass':>5}  {'Recall':>7}")
print(f"  {'-'*45}")
for cat, rs in sorted(by_cat.items()):
    p = sum(x["passed"] for x in rs)
    rc = sum(x["recall"] for x in rs) / len(rs)
    bar = "█" * int(rc * 10)
    print(f"  {cat:<30} {p:>2}/{len(rs):<2}   {rc:.0%}  {bar}")

fails = [r for r in data if not r["passed"]]
print(f"\n  FAILED ({len(fails)}):")
for r in fails:
    print(f"    {r['id']:<14} recall={r['recall']:.0%}  | {r['question'][:55]}")
print(f"=" * 60)
