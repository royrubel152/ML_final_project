from __future__ import annotations

from typing import Protocol, runtime_checkable

from evaluation_framework.schemas import JudgeVerdict


@runtime_checkable
class LLMJudge(Protocol):
    name: str

    def score(self, prompt: str) -> JudgeVerdict:
        ...


class DummyJudge:
    """Stub judge that returns score=3 for every prompt. Used in tests and CI."""

    name: str = "dummy"

    def score(self, prompt: str) -> JudgeVerdict:
        return JudgeVerdict(
            score=3,
            explanation="DummyJudge: fixed score — wire a real LLM provider to replace this.",
            judge_name=self.name,
            metric="",
        )
