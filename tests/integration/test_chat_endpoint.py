"""
Integration tests for the /chat, /feedback, and /reset endpoints.
Uses FastAPI TestClient with mocked RAG and Gemini — no real API calls.
Run with: pytest tests/integration/ -v
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

VALID_UUID = "123e4567-e89b-12d3-a456-426614174000"
VALID_QUESTION = "מה דרישות הקבלה לתוכנית?"

MOCK_CHUNK = {
    "text": "תנאי הקבלה לתוכנית MBA כוללים תואר ראשון בממוצע 80 ולפחות שנתיים ניסיון עבודה.",
    "source": "קבלה",
    "url": "https://bschool.huji.ac.il/mba/admittance",
    "score": 0.87,
}


@pytest.fixture(scope="module")
def client():
    """
    Create a TestClient with the full app but mock out:
    - load_all_sources (scraper — no network)
    - get_or_build_index (FAISS build — no embedding API)
    - retrieve (returns a fake chunk for in-scope questions)
    - genai.GenerativeModel (Gemini — no LLM API)
    """
    mock_response = MagicMock()
    mock_response.text = "על פי המידע: תנאי הקבלה כוללים תואר ראשון."

    mock_session = MagicMock()
    mock_session.send_message.return_value = mock_response
    mock_session.history = []

    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_session

    mock_index = MagicMock()
    mock_chunks = [MOCK_CHUNK]

    with patch("app.load_all_sources", return_value="fake scraped content"), \
         patch("app.get_or_build_index", return_value=(mock_index, mock_chunks)), \
         patch("app.genai.GenerativeModel", return_value=mock_model):

        from app import app as fastapi_app
        with TestClient(fastapi_app) as c:
            yield c


# ── /chat validation ──────────────────────────────────────────────

class TestChatValidation:

    def test_missing_message_returns_422(self, client):
        resp = client.post("/chat", json={"session_id": VALID_UUID})
        assert resp.status_code == 422

    def test_empty_message_returns_422(self, client):
        resp = client.post("/chat", json={"message": "", "session_id": VALID_UUID})
        assert resp.status_code == 422

    def test_message_too_long_returns_422(self, client):
        resp = client.post("/chat", json={"message": "א" * 501, "session_id": VALID_UUID})
        assert resp.status_code == 422

    def test_html_injection_returns_422(self, client):
        resp = client.post("/chat", json={
            "message": "<script>evil()</script>",
            "session_id": VALID_UUID
        })
        assert resp.status_code == 422

    def test_invalid_uuid_returns_422(self, client):
        resp = client.post("/chat", json={"message": VALID_QUESTION, "session_id": "bad-id"})
        assert resp.status_code == 422

    def test_missing_session_id_returns_422(self, client):
        resp = client.post("/chat", json={"message": VALID_QUESTION})
        assert resp.status_code == 422


# ── /chat off-topic gate ──────────────────────────────────────────

class TestOffTopicGate:

    def test_off_topic_question_returns_off_topic_reply(self, client):
        with patch("app.retrieve", return_value=[]):
            resp = client.post("/chat", json={
                "message": "מה בירת צרפת?",
                "session_id": VALID_UUID
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunks_found"] == 0
        assert data["sources_used"] == []
        assert "MBA" in data["reply"] or "מזכירות" in data["reply"]

    def test_off_topic_does_not_call_gemini(self, client):
        with patch("app.retrieve", return_value=[]) as mock_retrieve, \
             patch("app.build_rag_prompt") as mock_prompt:
            client.post("/chat", json={"message": "כתוב שיר", "session_id": VALID_UUID})
            mock_prompt.assert_not_called()


# ── /chat happy path ──────────────────────────────────────────────

class TestChatHappyPath:

    def test_valid_request_returns_reply(self, client):
        with patch("app.retrieve", return_value=[MOCK_CHUNK]):
            resp = client.post("/chat", json={
                "message": VALID_QUESTION,
                "session_id": VALID_UUID
            })
        assert resp.status_code == 200
        data = resp.json()
        assert "reply" in data
        assert "sources_used" in data
        assert data["chunks_found"] == 1

    def test_sources_used_contains_chunk_url(self, client):
        with patch("app.retrieve", return_value=[MOCK_CHUNK]):
            resp = client.post("/chat", json={
                "message": VALID_QUESTION,
                "session_id": VALID_UUID
            })
        assert "https://bschool.huji.ac.il/mba/admittance" in resp.json()["sources_used"]


# ── /feedback validation ──────────────────────────────────────────

class TestFeedback:

    def test_thumbs_up_returns_ok(self, client):
        resp = client.post("/feedback", json={"session_id": VALID_UUID, "value": "up"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_invalid_feedback_value_returns_422(self, client):
        resp = client.post("/feedback", json={"session_id": VALID_UUID, "value": "meh"})
        assert resp.status_code == 422

    def test_missing_value_returns_422(self, client):
        resp = client.post("/feedback", json={"session_id": VALID_UUID})
        assert resp.status_code == 422


# ── /reset ────────────────────────────────────────────────────────

class TestReset:

    def test_reset_valid_uuid_returns_ok(self, client):
        resp = client.post("/reset", json={"session_id": VALID_UUID})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_reset_invalid_uuid_returns_422(self, client):
        resp = client.post("/reset", json={"session_id": "not-a-uuid"})
        assert resp.status_code == 422


# ── /health ───────────────────────────────────────────────────────

class TestHealth:

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "chunks_indexed" in data
        assert "rag_ready" in data
