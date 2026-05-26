"""
File-based session store.
Each session is saved as data/sessions/<session_id>.json
Sessions expire after TTL_HOURS and are cleaned up on load.
"""

import os
import json
import time

SESSIONS_DIR = os.path.join("data", "sessions")
TTL_HOURS = 2


def _path(session_id: str) -> str:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    safe_id = session_id.replace("/", "_").replace("\\", "_")[:64]
    return os.path.join(SESSIONS_DIR, f"{safe_id}.json")


def load_history(session_id: str) -> list[dict]:
    path = _path(session_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        age_hours = (time.time() - data.get("updated_at", 0)) / 3600
        if age_hours > TTL_HOURS:
            os.remove(path)
            return []
        return data.get("history", [])
    except Exception:
        return []


def save_history(session_id: str, history: list[dict]):
    path = _path(session_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "history": history,
                "updated_at": time.time(),
            }, f, ensure_ascii=False)
    except Exception as e:
        print(f"[sessions] Save error: {e}")


def delete_session(session_id: str):
    path = _path(session_id)
    if os.path.exists(path):
        os.remove(path)


def cleanup_expired():
    """Remove expired session files. Called at startup."""
    if not os.path.exists(SESSIONS_DIR):
        return
    removed = 0
    for fname in os.listdir(SESSIONS_DIR):
        fpath = os.path.join(SESSIONS_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            age_hours = (time.time() - data.get("updated_at", 0)) / 3600
            if age_hours > TTL_HOURS:
                os.remove(fpath)
                removed += 1
        except Exception:
            pass
    if removed:
        print(f"[sessions] Cleaned up {removed} expired sessions")
