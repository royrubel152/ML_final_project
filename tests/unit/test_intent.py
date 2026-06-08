"""
Regression tests for intent detection and specialization context tracking.
Tests the exact failure scenario described in the issue report.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from app import detect_intent, SPEC_ALIASES, CORRECTION_PATTERNS


# ── Intent detection ──────────────────────────────────────────────

class TestSpecializationDetection:
    def test_big_data_hebrew(self):
        intent = detect_intent("איזה סמינריום יש בהתמחות של אנליטיקה ונתוני עתק?", {})
        assert intent["spec_code"] == "3663"
        assert "אנליטיקה" in intent["spec_name"]

    def test_big_data_short_alias(self):
        intent = detect_intent("מה קורסי החובה באנליטיקה?", {})
        assert intent["spec_code"] == "3663"

    def test_fintech(self):
        intent = detect_intent("אילו קורסים יש בפינטק?", {})
        assert intent["spec_code"] == "3798"

    def test_information_science(self):
        intent = detect_intent("מה יש במדע המידע?", {})
        assert intent["spec_code"] == "3662"

    def test_marketing(self):
        intent = detect_intent("קורסי חובה בשיווק", {})
        assert intent["spec_code"] == "3123"

    def test_unknown_specialization_returns_none(self):
        intent = detect_intent("מה מזג האוויר?", {})
        assert intent["spec_code"] is None


class TestTopicDetection:
    def test_seminar_topic(self):
        intent = detect_intent("איזה סמינריון יש בהתמחות?", {})
        assert intent["topic"] == "seminar"

    def test_lecturer_topic(self):
        intent = detect_intent("מי מרצה את הקורס?", {})
        assert intent["topic"] == "lecturer"

    def test_schedule_topic(self):
        intent = detect_intent("מתי מתקיים הקורס?", {})
        assert intent["topic"] == "schedule"

    def test_general_topic(self):
        intent = detect_intent("מה תנאי הקבלה?", {})
        assert intent["topic"] in ("admission", "general")


class TestCorrectionDetection:
    def test_correction_why_not_mentioned(self):
        """Regression: 'למה לא אמרת שזאת גם אופציה?' must trigger correction."""
        intent = detect_intent("אז למה לא אמרת שזאת גם אופציה?", {
            "active_spec_code": "3663",
            "active_spec_name": "התמחות באנליטיקה של נתוני עתק - ראשית",
        })
        assert intent["is_correction"] is True

    def test_correction_three_options(self):
        """Regression: user says there are 3 options → correction flag."""
        intent = detect_intent("שאלתי לגבי ההתמחות איזה סמינריונים אפשר לקחת ויש 3 אופציות", {
            "active_spec_code": "3663",
        })
        assert intent["is_correction"] is True

    def test_correction_missed_item(self):
        intent = detect_intent("לא הזכרת את הסמינריון של רונן", {
            "active_spec_code": "3663",
        })
        assert intent["is_correction"] is True

    def test_no_correction_on_normal_question(self):
        intent = detect_intent("מי מרצה בקורס פינטק?", {})
        assert intent["is_correction"] is False


class TestContextCarryOver:
    """
    Regression: follow-up questions must carry active specialization
    from session state when no new specialization is mentioned.
    """
    active_state = {
        "active_spec_code": "3663",
        "active_spec_name": "התמחות באנליטיקה של נתוני עתק - ראשית",
        "active_topic": "seminar",
    }

    def test_follow_up_course_name_not_spec_switch(self):
        """'אין מידע לסמינריון מדע המידע?' — 'מדע המידע' is a course name here,
        not an explicit spec switch → active spec 3663 must be preserved."""
        intent = detect_intent("אין מידע לסמינריון מדע המידע?", self.active_state)
        assert intent["spec_code"] == "3663"  # must NOT switch to 3662

    def test_explicit_spec_reference_does_switch(self):
        """'מה יש בהתמחות מדע המידע?' — explicit 'בהתמחות' marker → allowed to switch."""
        intent = detect_intent("מה יש בהתמחות מדע המידע?", self.active_state)
        assert intent["spec_code"] == "3662"

    def test_correction_carries_spec(self):
        """Correction message with no new spec keyword → keep active spec."""
        intent = detect_intent("אז למה לא אמרת שזאת גם אופציה?", self.active_state)
        assert intent["spec_code"] == "3663"
        assert intent["is_correction"] is True

    def test_explicit_switch_clears_spec(self):
        """'שאלה אחרת — מה יש בפינטק?' → switch to FinTech."""
        intent = detect_intent("שאלה אחרת, מה יש בפינטק?", self.active_state)
        assert intent["spec_code"] == "3798"

    def test_short_followup_without_spec_keyword(self):
        """A short generic follow-up → carry state."""
        intent = detect_intent("ומה עוד יש?", self.active_state)
        assert intent["spec_code"] == "3663"

    def test_reset_state_clears_spec(self):
        """Empty state → no spec carried."""
        intent = detect_intent("ומה עוד יש?", {})
        assert intent["spec_code"] is None


# ── Regression scenario from issue report ─────────────────────────

class TestFullConversationScenario:
    """
    Full 4-turn regression scenario from the issue report.
    Validates that intent detection produces correct signals at each turn.
    """

    def test_turn1_list_seminars(self):
        """Turn 1: 'איזה סמינריום יש בהתמחות של אנליטיקה ונתוני עתק?'"""
        intent = detect_intent(
            "איזה סמינריום יש בהתמחות של אנליטיקה ונתוני עתק?", {}
        )
        assert intent["spec_code"] == "3663"
        assert intent["topic"] == "seminar"
        assert intent["is_correction"] is False

    def test_turn2_follow_up_ronen(self):
        """Turn 2: 'ומה עם סמינריון מדע המידע של רונן?' — context from state."""
        state = {"active_spec_code": "3663", "active_spec_name": "התמחות באנליטיקה של נתוני עתק - ראשית"}
        intent = detect_intent("ומה עם סמינריון מדע המידע של רונן?", state)
        assert intent["topic"] == "seminar"
        # Must NOT switch to FinTech or research track
        assert intent["spec_code"] != "3798"
        assert intent["spec_code"] is not None

    def test_turn3_correction(self):
        """Turn 3: 'אז למה לא אמרת שזאת גם אופציה?' — correction."""
        state = {"active_spec_code": "3663", "active_spec_name": "התמחות באנליטיקה של נתוני עתק - ראשית"}
        intent = detect_intent("אז למה לא אמרת שזאת גם אופציה?", state)
        assert intent["is_correction"] is True
        assert intent["spec_code"] == "3663"  # must NOT switch to research track

    def test_turn4_full_list_correction(self):
        """Turn 4: 'שאלתי אבל לגבי ההתמחות איזה סמינריונים אפשר לקחת ויש 3 אופציות'"""
        state = {"active_spec_code": "3663", "active_spec_name": "התמחות באנליטיקה של נתוני עתק - ראשית"}
        intent = detect_intent(
            "שאלתי אבל לגבי ההתמחות איזה סמינריונים אפשר לקחת ויש 3 אופציות",
            state
        )
        assert intent["is_correction"] is True
        assert intent["topic"] == "seminar"
        assert intent["spec_code"] == "3663"  # must NOT switch to FinTech
