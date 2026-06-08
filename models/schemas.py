"""
Pydantic request/response models for the MBA Advisor API.
FastAPI uses these to validate incoming JSON and return 422 automatically on bad input.
"""

import re
from typing import Literal
from pydantic import BaseModel, Field, field_validator


UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=2, max_length=500)
    session_id: str = Field(..., min_length=36, max_length=36)

    @field_validator("message")
    @classmethod
    def strip_and_reject_html(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("השאלה קצרה מדי לאחר הסרת רווחים")
        if HTML_TAG_PATTERN.search(v):
            raise ValueError("השאלה לא יכולה להכיל תגי HTML")
        return v

    @field_validator("session_id")
    @classmethod
    def must_be_uuid(cls, v: str) -> str:
        if not UUID_PATTERN.match(v):
            raise ValueError("session_id חייב להיות UUID תקני")
        return v


class ChatResponse(BaseModel):
    reply: str
    sources_used: list[str]
    chunks_found: int


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., min_length=36, max_length=36)
    value: Literal["up", "down"]

    @field_validator("session_id")
    @classmethod
    def must_be_uuid(cls, v: str) -> str:
        if not UUID_PATTERN.match(v):
            raise ValueError("session_id חייב להיות UUID תקני")
        return v


class ResetRequest(BaseModel):
    session_id: str = Field(..., min_length=36, max_length=36)

    @field_validator("session_id")
    @classmethod
    def must_be_uuid(cls, v: str) -> str:
        if not UUID_PATTERN.match(v):
            raise ValueError("session_id חייב להיות UUID תקני")
        return v
