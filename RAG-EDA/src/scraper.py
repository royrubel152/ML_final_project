"""
scraper.py — Fetches Hebrew University MBA data from the Shnaton and bschool APIs.

Strategy:
  - shnaton.huji.ac.il: React SPA — data is served via a JSON REST API at /api/.
    We call the API endpoints directly and cache the JSON responses.
  - bschool.huji.ac.il HTML pages: Drupal CMS — fetch real HTML with browser headers.
  - bschool.huji.ac.il PDF: download directly.

All responses are saved to data/raw/ as .json or .html or .pdf.
The manifest records each URL → local_path mapping.
Re-running is safe: files already on disk are skipped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
POLITE_DELAY_SECONDS = 0.5
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3

# HUJI uses institutional TLS cert chains not in Python's default CA store.
# Disabling verification is acceptable for read-only academic research scraping.
SSL_VERIFY = False

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "he-IL,he;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

JSON_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "he-IL,he;q=0.9,en;q=0.8",
    "Referer": "https://shnaton.huji.ac.il/",
    "Origin": "https://shnaton.huji.ac.il",
}

MANIFEST_FILENAME = "manifest.json"

SHNATON_API_BASE = "https://shnaton.huji.ac.il/api"
YEAR = 2026  # Academic year to fetch

# ── Shnaton roadmap code → (learning_program_id, yearly_roadmap_id) mappings
# Discovered by probing /api/learning-programs/{id}/yearly-roadmaps?year=2026
ROADMAP_CODE_TO_PROGRAM_ID: Dict[str, int] = {
    "322-3220": 4,   # Masters
    "322-3222": 4,
    "826-3254": 4,
    "826-3255": 4,
    "943-3894": 4,
    "322-3792": 12,  # Supplementary Masters
    "322-8168": 5,   # PhD (research track)
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_sources(sources_path: Path) -> List[Dict]:
    """
    Parse sources.txt into a list of {group, url} dicts.
    Lines starting with '#' or that are blank are ignored.
    """
    sources = []
    with open(sources_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                logger.warning("Skipping malformed line in sources.txt: %r", line)
                continue
            group, url = parts
            sources.append({"group": group.strip(), "url": url.strip()})
    return sources


def fetch_all(sources: List[Dict], raw_dir: Path) -> Dict:
    """
    Fetch all sources, routing each to the appropriate strategy:
      - shnaton_roadmap  → Shnaton JSON API (yearly roadmap endpoint)
      - shnaton_specialization → Shnaton JSON API (specialization endpoint)
      - bschool HTML     → Direct HTTP with browser headers
      - bschool PDF      → Direct HTTP binary download

    Parameters
    ----------
    sources : list of {group, url} dicts (from load_sources)
    raw_dir : directory where raw files are saved

    Returns
    -------
    manifest : dict mapping url → {
        group, url, local_path, extension,
        content_type, last_modified, fetched_at, size_bytes, status
    }
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = raw_dir / MANIFEST_FILENAME
    manifest: Dict = {}

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)

    session = requests.Session()
    session.verify = SSL_VERIFY

    # Pre-fetch all Shnaton roadmaps once (avoids N API calls per roadmap URL)
    shnaton_roadmap_cache: Dict[str, Dict] = {}
    shnaton_spec_cache: Dict[str, Dict] = {}

    roadmap_sources = [s for s in sources if s["group"] == "shnaton_roadmap"]
    spec_sources = [s for s in sources if s["group"] == "shnaton_specialization"]

    if roadmap_sources:
        logger.info("Pre-fetching Shnaton roadmap data from API...")
        shnaton_roadmap_cache = _fetch_all_roadmaps(session)

    if spec_sources:
        logger.info("Pre-fetching Shnaton specialization data from API...")
        shnaton_spec_cache = _fetch_all_specializations(session)

    for source in sources:
        url = source["url"]
        group = source["group"]

        if url in manifest and manifest[url].get("status") in ("cached", "fetched"):
            local_path = Path(manifest[url]["local_path"])
            if local_path.exists():
                manifest[url]["status"] = "cached"
                logger.info("Skipping (cached): %s", url)
                continue

        logger.info("Fetching: %s", url)

        if group == "shnaton_roadmap":
            entry = _save_shnaton_roadmap(url, raw_dir, shnaton_roadmap_cache, session)
        elif group == "shnaton_specialization":
            entry = _save_shnaton_specialization(url, raw_dir, shnaton_spec_cache, session)
        elif url.lower().endswith(".pdf"):
            entry = _save_binary(url, raw_dir, session)
        else:
            entry = _save_html(url, raw_dir, session)

        entry["group"] = group
        entry["url"] = url
        manifest[url] = entry

        _save_manifest(manifest, manifest_path)
        time.sleep(POLITE_DELAY_SECONDS)

    fetched = sum(1 for v in manifest.values() if v["status"] == "fetched")
    cached = sum(1 for v in manifest.values() if v["status"] == "cached")
    errors = sum(1 for v in manifest.values() if v["status"] == "error")
    print(f"\nFetch summary: {fetched} fetched, {cached} cached, {errors} errors")
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Shnaton API fetchers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_all_roadmaps(session: requests.Session) -> Dict[str, Dict]:
    """
    Fetch all relevant MBA yearly-roadmap records from the Shnaton API.
    Returns a dict keyed by roadmap code (e.g. "322-3220").
    """
    cache: Dict[str, Dict] = {}
    fetched_prog_ids: set = set()

    for code, prog_id in ROADMAP_CODE_TO_PROGRAM_ID.items():
        if prog_id in fetched_prog_ids:
            continue
        url = f"{SHNATON_API_BASE}/learning-programs/{prog_id}/yearly-roadmaps"
        try:
            resp = session.get(url, headers=JSON_HEADERS, params={"year": YEAR}, timeout=REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
            for item in resp.json():
                if "roadmapCode" in item:
                    cache[item["roadmapCode"]] = item
            fetched_prog_ids.add(prog_id)
            logger.info("Fetched %d roadmaps for learning-program %d", len([k for k in cache]), prog_id)
        except Exception as exc:
            logger.error("Failed to fetch roadmaps for program %d: %s", prog_id, exc)

    return cache


def _fetch_all_specializations(session: requests.Session) -> Dict[str, Dict]:
    """
    Fetch all MBA specialization records from the Shnaton API.

    The API requires a non-empty 'name' query parameter. We issue multiple
    searches across common Hebrew term roots to ensure full coverage of the
    17 MBA specialization codes.

    Returns a dict keyed by specialization code string (e.g. "3111").
    """
    # Search terms chosen to cover all 17 MBA specialization names
    SEARCH_TERMS = [
        "התמחות", "מנהל", "מימון", "שיווק", "אסטרטגיה", "ניהול",
        "כספים", "בינלאומי", "מידע", "חשבונאות", "יזמות",
        "כלכלה", "טכנולוגיה", "אנליטיקה", "בריאות", "ספורט",
        "נדלן", "תפעול", "לוגיסטיקה",
    ]

    cache: Dict[str, Dict] = {}
    url = f"{SHNATON_API_BASE}/specializations/search"

    for term in SEARCH_TERMS:
        try:
            resp = session.get(
                url,
                headers=JSON_HEADERS,
                params={"name": term, "year": YEAR},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            for item in resp.json():
                code = str(item.get("code", ""))
                if code and code not in cache:
                    cache[code] = item
        except Exception as exc:
            logger.warning("Specialization search failed for term '%s': %s", term, exc)

    logger.info("Fetched %d total specializations from search API", len(cache))
    return cache


def _save_shnaton_roadmap(
    url: str, raw_dir: Path, cache: Dict[str, Dict], session: requests.Session
) -> Dict:
    """Save a Shnaton roadmap URL as a JSON file using cached API data."""
    # Extract roadmap code from URL: /roadmap/322-3220 → 322-3220
    code = url.rstrip("/").split("/")[-1]
    data = cache.get(code)

    if data is None:
        # Try fetching directly as a fallback
        prog_id = ROADMAP_CODE_TO_PROGRAM_ID.get(code, 4)
        api_url = f"{SHNATON_API_BASE}/learning-programs/{prog_id}/yearly-roadmaps"
        try:
            resp = session.get(api_url, headers=JSON_HEADERS, params={"year": YEAR}, timeout=REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
            for item in resp.json():
                if item.get("roadmapCode") == code:
                    data = item
                    break
        except Exception as exc:
            logger.error("API fallback failed for roadmap %s: %s", code, exc)

    if data is None:
        logger.error("No data found for roadmap code: %s", code)
        return _error_entry()

    # Enrich with thresholds (course groups with full course lists).
    # The 'year' query param is required — omitting it returns ERR_1101.
    rm_id = data.get("id")
    if rm_id:
        try:
            thresh_url = f"{SHNATON_API_BASE}/yearly-roadmaps/{rm_id}/thresholds"
            thresh_resp = session.get(thresh_url, headers=JSON_HEADERS,
                                      params={"year": YEAR}, timeout=REQUEST_TIMEOUT_SECONDS)
            if thresh_resp.ok:
                data["thresholds_detail"] = thresh_resp.json()
                n_courses = sum(len(t.get("courses", [])) for t in data["thresholds_detail"])
                logger.info("  Fetched %d groups, %d courses for roadmap %s",
                            len(data["thresholds_detail"]), n_courses, code)
            else:
                logger.warning("Thresholds fetch failed for roadmap %s: %s %s",
                               code, thresh_resp.status_code, thresh_resp.text[:80])
        except Exception as exc:
            logger.warning("Failed to fetch thresholds for roadmap %s: %s", code, exc)

    sha1 = _url_to_sha1(url)
    local_path = raw_dir / f"{sha1}.json"
    json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    local_path.write_bytes(json_bytes)

    logger.info("Saved roadmap %s → %s (%.1f KB)", code, local_path.name, len(json_bytes) / 1024)
    return {
        "local_path": str(local_path),
        "extension": ".json",
        "content_type": "application/json",
        "last_modified": None,
        "fetched_at": _now_iso(),
        "size_bytes": len(json_bytes),
        "status": "fetched",
    }


def _save_shnaton_specialization(
    url: str, raw_dir: Path, cache: Dict[str, Dict], session: requests.Session
) -> Dict:
    """Save a Shnaton specialization URL as a JSON file using cached API data."""
    # Extract specialization code from URL: /specialization/3111 → 3111
    code = url.rstrip("/").split("/")[-1]
    data = cache.get(code)

    if data is None:
        # Fallback: search broadly and filter by code
        api_url = f"{SHNATON_API_BASE}/specializations/search"
        for term in ["התמחות", "מנהל", "מימון", "שיווק", "אסטרטגיה"]:
            try:
                resp = session.get(
                    api_url,
                    headers=JSON_HEADERS,
                    params={"name": term, "year": YEAR},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                resp.raise_for_status()
                for item in resp.json():
                    if str(item.get("code")) == code:
                        data = item
                        break
                if data:
                    break
            except Exception as exc:
                logger.warning("Fallback search failed for term '%s': %s", term, exc)

    if data is None:
        logger.error("No data found for specialization code: %s", code)
        return _error_entry()

    # Enrich with full course lists via /specializations/{code}/thresholds.
    # Must use the USER-FACING CODE (e.g. "3111"), not the internal db id (e.g. 45).
    # The internal id resolves to a generic yearly-roadmap entity that lacks the MBA
    # course data; the code-based endpoint correctly scopes to the specialization.
    # Parameters activeYear + thresholdYear are both required; include='courses' adds
    # full course detail. The response shape is {name, thresholds: [...], academicInfo}.
    try:
        thresh_url = f"{SHNATON_API_BASE}/specializations/{code}/thresholds"
        thresh_resp = session.get(
            thresh_url,
            headers=JSON_HEADERS,
            params={"activeYear": YEAR, "thresholdYear": YEAR, "include": "courses"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if thresh_resp.ok:
            resp_data = thresh_resp.json()
            # Response is always a dict with a 'thresholds' key (never bare array)
            thresholds = resp_data.get("thresholds", []) if isinstance(resp_data, dict) else resp_data
            data["thresholds_detail"] = thresholds
            if isinstance(resp_data, dict) and resp_data.get("academicInfo"):
                data["academicInfo"] = resp_data["academicInfo"]
            n_courses = sum(len(t.get("courses", [])) for t in thresholds)
            logger.info("  Fetched %d groups, %d courses for spec %s",
                        len(thresholds), n_courses, code)
        else:
            logger.warning("Thresholds fetch failed for spec %s: %s %s",
                           code, thresh_resp.status_code, thresh_resp.text[:80])
    except Exception as exc:
        logger.warning("Failed to fetch thresholds for specialization %s: %s", code, exc)

    sha1 = _url_to_sha1(url)
    local_path = raw_dir / f"{sha1}.json"
    json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    local_path.write_bytes(json_bytes)

    logger.info("Saved specialization %s → %s (%.1f KB)", code, local_path.name, len(json_bytes) / 1024)
    return {
        "local_path": str(local_path),
        "extension": ".json",
        "content_type": "application/json",
        "last_modified": None,
        "fetched_at": _now_iso(),
        "size_bytes": len(json_bytes),
        "status": "fetched",
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTML / PDF fetchers
# ─────────────────────────────────────────────────────────────────────────────

def _save_html(url: str, raw_dir: Path, session: requests.Session) -> Dict:
    """Fetch an HTML page with browser-like headers and save to disk."""
    response = _fetch_with_retry(url, session, headers=BROWSER_HEADERS)
    if response is None:
        return _error_entry()

    sha1 = _url_to_sha1(url)
    local_path = raw_dir / f"{sha1}.html"
    local_path.write_bytes(response.content)

    logger.info("Saved HTML → %s (%.1f KB)", local_path.name, len(response.content) / 1024)
    return {
        "local_path": str(local_path),
        "extension": ".html",
        "content_type": response.headers.get("Content-Type", ""),
        "last_modified": response.headers.get("Last-Modified"),
        "fetched_at": _now_iso(),
        "size_bytes": len(response.content),
        "status": "fetched",
    }


def _save_binary(url: str, raw_dir: Path, session: requests.Session) -> Dict:
    """Download a binary file (PDF) and save to disk."""
    response = _fetch_with_retry(url, session, headers=BROWSER_HEADERS)
    if response is None:
        return _error_entry()

    sha1 = _url_to_sha1(url)
    local_path = raw_dir / f"{sha1}.pdf"
    local_path.write_bytes(response.content)

    logger.info("Saved PDF → %s (%.1f KB)", local_path.name, len(response.content) / 1024)
    return {
        "local_path": str(local_path),
        "extension": ".pdf",
        "content_type": response.headers.get("Content-Type", ""),
        "last_modified": response.headers.get("Last-Modified"),
        "fetched_at": _now_iso(),
        "size_bytes": len(response.content),
        "status": "fetched",
    }


def _fetch_with_retry(
    url: str, session: requests.Session, headers: Dict
) -> Optional[requests.Response]:
    """Attempt to fetch a URL up to MAX_RETRIES times."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, MAX_RETRIES, url, exc)
            if attempt < MAX_RETRIES:
                time.sleep(POLITE_DELAY_SECONDS * attempt)
    logger.error("All %d attempts failed for: %s", MAX_RETRIES, url)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _url_to_sha1(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()


def _error_entry() -> Dict:
    return {
        "local_path": None,
        "extension": None,
        "content_type": None,
        "last_modified": None,
        "fetched_at": _now_iso(),
        "size_bytes": 0,
        "status": "error",
    }


def _save_manifest(manifest: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
