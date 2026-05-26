import subprocess
import sys
import os
import json
import time

CACHE_FILE = os.path.join(os.path.dirname(__file__), "scraped_content.json")
CACHE_TTL_HOURS = 24


def _load_cache() -> str | None:
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        age_hours = (time.time() - data["timestamp"]) / 3600
        if age_hours > CACHE_TTL_HOURS:
            print(f"[cache] Expired ({age_hours:.1f}h old) — will re-scrape")
            return None
        print(f"[cache] Using cached content ({data['char_count']} chars, {age_hours:.1f}h old)")
        return data["content"]
    except Exception as e:
        print(f"[cache] Read error: {e}")
        return None


def _save_cache(content: str):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "content": content,
                "timestamp": time.time(),
                "char_count": len(content)
            }, f, ensure_ascii=False)
        print(f"[cache] Saved ({len(content)} chars)")
    except Exception as e:
        print(f"[cache] Write error: {e}")


def _run_subprocess() -> str:
    script = os.path.join(os.path.dirname(__file__), "scrape_runner.py")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
    )
    for line in result.stderr.splitlines():
        print(line)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"Scraper failed: {result.stderr[:300]}")
    return result.stdout


async def load_all_sources(force: bool = False) -> str:
    import asyncio

    if not force:
        cached = _load_cache()
        if cached:
            return cached

    print("[scraper] Fetching fresh content from HUJI...")
    loop = asyncio.get_event_loop()
    try:
        content = await loop.run_in_executor(None, _run_subprocess)
        _save_cache(content)
        return content
    except Exception as e:
        print(f"[scraper] Scraping failed: {e}")
        fallback = _load_cache.__wrapped__() if hasattr(_load_cache, '__wrapped__') else None
        # Try reading cache file directly as fallback regardless of age
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print("[scraper] Using stale cache as fallback")
                return data["content"]
            except Exception:
                pass
        raise RuntimeError("No content available — scraping failed and no cache exists")
