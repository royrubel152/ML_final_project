from evaluation_framework.schemas import EvalSample, JudgeVerdict


def test_evalsample_defaults():
    s = EvalSample(
        sample_id="s1",
        question="מה שעות הקבלה?",
        ground_truth_answer="ראה האתר",
        ground_truth_sources=["https://bschool.huji.ac.il/mba/admittance"],
    )
    assert s.ground_truth_chunk_ids == []
    assert s.chat_history == []
    assert s.retrieved_sources is None
    assert s.retrieved_chunk_ids is None
    assert s.answer is None
    assert s.category is None
    assert s.archetype is None
    assert s.source_type == "single_turn"
    assert s.secretary_tag is None


def test_evalsample_full():
    s = EvalSample(
        sample_id="s2",
        question="מה ההתמחויות?",
        ground_truth_answer="מימון, שיווק, מדע המידע",
        ground_truth_sources=["https://shnaton.huji.ac.il/specialization/3662"],
        ground_truth_chunk_ids=["chunk_42"],
        chat_history=[{"role": "user", "content": "שאלה קודמת"}],
        retrieved_sources=["https://shnaton.huji.ac.il/specialization/3662"],
        retrieved_chunk_ids=["chunk_42"],
        answer="ישנן מספר התמחויות",
        category="program_structure",
        archetype="clarification_loop",
        source_type="conversation",
        secretary_tag="approved",
    )
    assert s.source_type == "conversation"
    assert s.secretary_tag == "approved"
    assert len(s.chat_history) == 1


def test_judgevardict_fields():
    v = JudgeVerdict(score=4, explanation="Good answer", judge_name="dummy", metric="faithfulness")
    assert 1 <= v.score <= 5
    assert v.metric == "faithfulness"
