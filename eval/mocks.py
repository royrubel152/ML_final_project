"""
Mock implementations for eval runs.
Replaces FAISS retrieval and Gemini calls so eval is free, fast, and deterministic.
"""

import json
import os
from unittest.mock import MagicMock

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
CHUNKS_FIXTURE = os.path.join(FIXTURES_DIR, "chunks.json")


def load_fixture_chunks() -> list[dict]:
    """Load committed frozen chunks for deterministic eval."""
    if not os.path.exists(CHUNKS_FIXTURE):
        raise FileNotFoundError(
            f"Fixture file not found: {CHUNKS_FIXTURE}\n"
            "Run: python eval/build_fixtures.py  (requires real scraped data)"
        )
    with open(CHUNKS_FIXTURE, "r", encoding="utf-8") as f:
        return json.load(f)


class MockRAGRetriever:
    """
    Simulates FAISS retrieval using keyword overlap.
    Returns the top-k chunks from the frozen fixture by keyword match score.
    """

    def __init__(self, fixture_chunks: list[dict], top_k: int = 5, min_score: float = 0.1):
        self.chunks = fixture_chunks
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, query: str) -> list[dict]:
        query_words = set(query.split())
        scored = []
        for chunk in self.chunks:
            overlap = sum(1 for w in query_words if w in chunk["text"])
            if overlap > 0:
                scored.append({**chunk, "score": overlap / len(query_words)})
        scored.sort(key=lambda c: c["score"], reverse=True)
        results = [c for c in scored[:self.top_k] if c["score"] >= self.min_score]
        return results


def make_mock_gemini_model(canned_reply: str = "תשובה מדומה מהמודל.") -> MagicMock:
    """
    Returns a MagicMock that behaves like genai.GenerativeModel.
    start_chat() returns a session whose send_message() returns canned_reply.
    """
    mock_response = MagicMock()
    mock_response.text = canned_reply

    mock_session = MagicMock()
    mock_session.send_message.return_value = mock_response
    mock_session.history = []

    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_session
    return mock_model
