import json, sys
sys.stdout.reconfigure(encoding="utf-8")
with open("data/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)
faq = [c for c in chunks if c.get("source", "").startswith("FAQ")]
print(f"Total chunks: {len(chunks)}")
print(f"FAQ chunks  : {len(faq)}")
if faq:
    print("\nSample FAQ chunk:")
    print(faq[0]["text"][:200])
    print(f"\nsource: {faq[0]['source']}")
else:
    # Show last 3 chunks
    print("\nLast 3 chunks:")
    for c in chunks[-3:]:
        print(f"  source={c['source']}  text={c['text'][:80]}")
