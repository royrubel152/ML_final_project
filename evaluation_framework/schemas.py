from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalSample:
    sample_id: str
    question: str
    ground_truth_answer: str
    ground_truth_sources: list[str]

    ground_truth_chunk_ids: list[str] = field(default_factory=list)
    chat_history: list[dict] = field(default_factory=list)

    retrieved_sources: Optional[list[str]] = None
    retrieved_chunk_ids: Optional[list[str]] = None
    answer: Optional[str] = None
    category: Optional[str] = None
    archetype: Optional[str] = None
    source_type: str = "single_turn"
    secretary_tag: Optional[str] = None


@dataclass
class JudgeVerdict:
    score: int
    explanation: str
    judge_name: str
    metric: str
