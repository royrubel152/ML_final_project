"""
Unit tests for Pydantic validation schemas.
Run with: pytest tests/unit/test_schemas.py -v
"""

import pytest
from pydantic import ValidationError
from models.schemas import ChatRequest, FeedbackRequest, ResetRequest

VALID_UUID = "123e4567-e89b-12d3-a456-426614174000"
VALID_QUESTION = "מה דרישות הקבלה לתוכנית MBA?"


# ── ChatRequest — question field ──────────────────────────────────

class TestChatRequestQuestion:

    def test_valid_request_passes(self):
        req = ChatRequest(message=VALID_QUESTION, session_id=VALID_UUID)
        assert req.message == VALID_QUESTION

    def test_question_too_short_raises(self):
        with pytest.raises(ValidationError) as exc:
            ChatRequest(message="?", session_id=VALID_UUID)
        assert "min_length" in str(exc.value) or "קצרה" in str(exc.value)

    def test_question_too_long_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="א" * 501, session_id=VALID_UUID)

    def test_empty_question_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="", session_id=VALID_UUID)

    def test_whitespace_only_question_raises(self):
        # strip() turns "   " into "" which then fails min_length
        with pytest.raises(ValidationError):
            ChatRequest(message="   ", session_id=VALID_UUID)

    def test_script_tag_raises(self):
        with pytest.raises(ValidationError) as exc:
            ChatRequest(message="<script>alert(1)</script>", session_id=VALID_UUID)
        assert "HTML" in str(exc.value)

    def test_html_tag_in_question_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="<b>שאלה</b>?", session_id=VALID_UUID)

    def test_leading_trailing_whitespace_is_stripped(self):
        req = ChatRequest(message="  מה השכר?  ", session_id=VALID_UUID)
        assert req.message == "מה השכר?"

    def test_missing_question_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(session_id=VALID_UUID)


# ── ChatRequest — session_id field ───────────────────────────────

class TestChatRequestSessionId:

    def test_valid_uuid_passes(self):
        req = ChatRequest(message=VALID_QUESTION, session_id=VALID_UUID)
        assert req.session_id == VALID_UUID

    def test_invalid_uuid_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(message=VALID_QUESTION, session_id="not-a-uuid")

    def test_empty_session_id_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(message=VALID_QUESTION, session_id="")

    def test_missing_session_id_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(message=VALID_QUESTION)

    def test_short_string_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(message=VALID_QUESTION, session_id="abc")


# ── FeedbackRequest ───────────────────────────────────────────────

class TestFeedbackRequest:

    def test_thumbs_up_passes(self):
        req = FeedbackRequest(session_id=VALID_UUID, value="up")
        assert req.value == "up"

    def test_thumbs_down_passes(self):
        req = FeedbackRequest(session_id=VALID_UUID, value="down")
        assert req.value == "down"

    def test_invalid_value_raises(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(session_id=VALID_UUID, value="maybe")

    def test_missing_value_raises(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(session_id=VALID_UUID)

    def test_invalid_session_id_raises(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(session_id="bad-id", value="up")


# ── ResetRequest ──────────────────────────────────────────────────

class TestResetRequest:

    def test_valid_passes(self):
        req = ResetRequest(session_id=VALID_UUID)
        assert req.session_id == VALID_UUID

    def test_invalid_uuid_raises(self):
        with pytest.raises(ValidationError):
            ResetRequest(session_id="not-a-uuid")
