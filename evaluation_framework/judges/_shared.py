"""Shared helpers for judge modules: template loading, chat_history formatting, verdict parsing."""
from __future__ import annotations

import pathlib
import re

from evaluation_framework.schemas import EvalSample, JudgeVerdict

_PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"


def load_template(metric: str) -> str:
    path = _PROMPTS_DIR / f"{metric}.txt"
    return path.read_text(encoding="utf-8")


def format_chat_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = ["Prior conversation turns:"]
    for turn in history:
        role = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def format_sources(sources: list[str] | None) -> str:
    if not sources:
        return "(no sources retrieved)"
    return "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(sources))


def parse_verdict(raw: str, judge_name: str, metric: str) -> JudgeVerdict:
    score_match = re.search(r"SCORE:\s*([1-5])", raw)
    expl_match = re.search(r"EXPLANATION:\s*(.+)", raw, re.DOTALL)
    if not score_match:
        raise ValueError(f"Judge response missing 'SCORE: <1-5>':\n{raw[:300]}")
    score = int(score_match.group(1))
    explanation = expl_match.group(1).strip() if expl_match else ""
    return JudgeVerdict(score=score, explanation=explanation, judge_name=judge_name, metric=metric)
