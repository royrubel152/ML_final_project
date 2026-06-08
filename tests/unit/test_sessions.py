"""
Unit tests for the file-based session store.
Tests TTL expiry, history capping, save/load roundtrip, and cleanup.
Run with: pytest tests/unit/test_sessions.py -v
"""

import os
import json
import time
import tempfile
import pytest

import sessions as sess_module
from sessions import load_history, save_history, delete_session, cleanup_expired, MAX_HISTORY_TURNS


@pytest.fixture(autouse=True)
def tmp_sessions_dir(tmp_path, monkeypatch):
    """Redirect all session file I/O to a temp directory for each test."""
    sessions_dir = str(tmp_path / "sessions")
    monkeypatch.setattr(sess_module, "SESSIONS_DIR", sessions_dir)
    return sessions_dir


VALID_UUID = "123e4567-e89b-12d3-a456-426614174000"
SAMPLE_HISTORY = [
    {"role": "user", "parts": ["שאלה ראשונה"]},
    {"role": "model", "parts": ["תשובה ראשונה"]},
]


class TestSaveAndLoad:

    def test_save_and_load_roundtrip(self):
        save_history(VALID_UUID, SAMPLE_HISTORY)
        loaded = load_history(VALID_UUID)
        assert loaded == SAMPLE_HISTORY

    def test_load_nonexistent_session_returns_empty(self):
        result = load_history("aaaaaaaa-0000-0000-0000-000000000000")
        assert result == []

    def test_delete_removes_session(self):
        save_history(VALID_UUID, SAMPLE_HISTORY)
        delete_session(VALID_UUID)
        assert load_history(VALID_UUID) == []


class TestHistoryCapping:

    def test_history_capped_to_max_turns(self):
        # Create 20 turns (10 user + 10 model)
        long_history = []
        for i in range(20):
            role = "user" if i % 2 == 0 else "model"
            long_history.append({"role": role, "parts": [f"הודעה {i}"]})
        save_history(VALID_UUID, long_history)
        loaded = load_history(VALID_UUID)
        assert len(loaded) == MAX_HISTORY_TURNS

    def test_history_shorter_than_cap_loads_fully(self):
        short_history = [
            {"role": "user", "parts": ["שאלה"]},
            {"role": "model", "parts": ["תשובה"]},
        ]
        save_history(VALID_UUID, short_history)
        loaded = load_history(VALID_UUID)
        assert len(loaded) == 2

    def test_capped_history_keeps_most_recent_turns(self):
        long_history = [{"role": "user", "parts": [f"הודעה {i}"]} for i in range(20)]
        save_history(VALID_UUID, long_history)
        loaded = load_history(VALID_UUID)
        # Should have turns 10-19 (the last MAX_HISTORY_TURNS)
        assert loaded[0]["parts"][0] == "הודעה 10"
        assert loaded[-1]["parts"][0] == "הודעה 19"


class TestTTLExpiry:

    def test_expired_session_returns_empty(self, tmp_sessions_dir):
        # Write a session file with an old timestamp
        os.makedirs(tmp_sessions_dir, exist_ok=True)
        safe_id = VALID_UUID[:64]
        path = os.path.join(tmp_sessions_dir, f"{safe_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "history": SAMPLE_HISTORY,
                "updated_at": time.time() - (sess_module.TTL_HOURS + 1) * 3600,
            }, f)
        result = load_history(VALID_UUID)
        assert result == []
        assert not os.path.exists(path)

    def test_fresh_session_is_not_expired(self):
        save_history(VALID_UUID, SAMPLE_HISTORY)
        result = load_history(VALID_UUID)
        assert result == SAMPLE_HISTORY


class TestCleanup:

    def test_cleanup_removes_expired_files(self, tmp_sessions_dir):
        os.makedirs(tmp_sessions_dir, exist_ok=True)
        old_path = os.path.join(tmp_sessions_dir, "old_session.json")
        with open(old_path, "w") as f:
            json.dump({"history": [], "updated_at": time.time() - 99999}, f)
        cleanup_expired()
        assert not os.path.exists(old_path)

    def test_cleanup_keeps_fresh_files(self, tmp_sessions_dir):
        save_history(VALID_UUID, SAMPLE_HISTORY)
        cleanup_expired()
        assert load_history(VALID_UUID) == SAMPLE_HISTORY
