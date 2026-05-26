"""
One-time script: scrapes HUJI content and saves the resulting chunks
to eval/fixtures/chunks.json so eval runs never need network/API access.

Run ONCE (requires GEMINI_API_KEY and network):
    python eval/build_fixtures.py
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from scraper import load_all_sources
from rag import chunk_content

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


async def main():
    print("[build_fixtures] Scraping HUJI content...")
    content = await load_all_sources()
    print(f"[build_fixtures] Scraped {len(content)} chars")

    chunks = chunk_content(content)
    print(f"[build_fixtures] Created {len(chunks)} chunks")

    os.makedirs(FIXTURES_DIR, exist_ok=True)
    out_path = os.path.join(FIXTURES_DIR, "chunks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[build_fixtures] Saved to {out_path}")
    print("[build_fixtures] Commit eval/fixtures/chunks.json to git for future eval runs.")


if __name__ == "__main__":
    asyncio.run(main())
