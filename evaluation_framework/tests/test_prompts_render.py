import pytest
from evaluation_framework.schemas import EvalSample
from evaluation_framework.judges import (
    context_relevance,
    faithfulness,
    correctness,
    answer_relevance,
    completeness,
)

_SAMPLE = EvalSample(
    sample_id="fixture_01",
    question="מה קורסי החובה?",
    ground_truth_answer="רשימת קורסי חובה...",
    ground_truth_sources=["https://shnaton.huji.ac.il/roadmap/322-3220"],
    retrieved_sources=["פסקת טקסט מקורס א", "פסקת טקסט מקורס ב"],
    answer="קורסי החובה הם: חשבונאות, מימון וסטטיסטיקה.",
    chat_history=[
        {"role": "user", "content": "שאלה קודמת"},
        {"role": "assistant", "content": "תשובה קודמת"},
    ],
    category="program_structure",
    source_type="single_turn",
)

_SAMPLE_NO_HISTORY = EvalSample(
    sample_id="fixture_02",
    question="האם יש קורסים בשישי?",
    ground_truth_answer="לא, אין קורסים בשישי.",
    ground_truth_sources=[],
    answer="לא.",
    source_type="single_turn",
)


@pytest.mark.parametrize("mod,sample", [
    (context_relevance, _SAMPLE),
    (faithfulness, _SAMPLE),
    (correctness, _SAMPLE),
    (answer_relevance, _SAMPLE),
    (completeness, _SAMPLE),
])
def test_prompt_renders_no_unfilled_placeholders(mod, sample):
    prompt = mod.build_prompt(sample)
    # No leftover {placeholder} in the output
    assert "{" not in prompt, f"{mod.__name__}: unfilled placeholder in prompt"


@pytest.mark.parametrize("mod,sample", [
    (context_relevance, _SAMPLE),
    (faithfulness, _SAMPLE),
    (correctness, _SAMPLE),
    (answer_relevance, _SAMPLE),
    (completeness, _SAMPLE),
])
def test_prompt_contains_score_instruction(mod, sample):
    prompt = mod.build_prompt(sample)
    assert "SCORE" in prompt


@pytest.mark.parametrize("mod,sample", [
    (context_relevance, _SAMPLE_NO_HISTORY),
    (faithfulness, _SAMPLE_NO_HISTORY),
    (correctness, _SAMPLE_NO_HISTORY),
    (answer_relevance, _SAMPLE_NO_HISTORY),
    (completeness, _SAMPLE_NO_HISTORY),
])
def test_prompt_renders_without_history(mod, sample):
    prompt = mod.build_prompt(sample)
    assert "{" not in prompt
