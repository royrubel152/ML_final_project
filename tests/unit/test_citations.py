"""
Unit tests for the verify_citations() function in app.py.
Tests that hallucinated quotes are stripped and real quotes are preserved.
Run with: pytest tests/unit/test_citations.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app import verify_citations

REAL_CONTEXT = (
    "תנאי הקבלה לתוכנית MBA כוללים תואר ראשון בממוצע של 80 לפחות. "
    "הגשת מועמדות מתבצעת דרך אתר האוניברסיטה העברית. "
    "שכר הלימוד עומד על 45,000 ש\"ח לשנה."
)


class TestVerifyCitations:

    def test_real_quote_is_preserved(self):
        reply = 'על פי המסמך: "תנאי הקבלה לתוכנית MBA כוללים תואר ראשון בממוצע של 80 לפחות"'
        result = verify_citations(reply, REAL_CONTEXT)
        assert "תנאי הקבלה לתוכנית MBA כוללים תואר ראשון בממוצע של 80 לפחות" in result
        assert "[ציטוט לא אומת" not in result

    def test_hallucinated_quote_is_replaced(self):
        reply = 'על פי המסמך: "ניתן לקבל מלגה מלאה לסטודנטים מצטיינים"'
        result = verify_citations(reply, REAL_CONTEXT)
        assert "[ציטוט לא אומת — פנה למזכירות]" in result
        assert "ניתן לקבל מלגה מלאה" not in result

    def test_multiple_quotes_mixed(self):
        reply = (
            '"תנאי הקבלה לתוכנית MBA כוללים תואר ראשון בממוצע של 80 לפחות" '
            'ו"ניתן לקבל מלגה מלאה לסטודנטים מצטיינים"'
        )
        result = verify_citations(reply, REAL_CONTEXT)
        assert "תנאי הקבלה לתוכנית MBA כוללים תואר ראשון בממוצע של 80 לפחות" in result
        assert "[ציטוט לא אומת — פנה למזכירות]" in result

    def test_no_quotes_returns_reply_unchanged(self):
        reply = "על פי המידע הזמין, יש לפנות למזכירות התלמידים."
        result = verify_citations(reply, REAL_CONTEXT)
        assert result == reply

    def test_short_quote_under_20_chars_is_ignored(self):
        # verify_citations only checks quotes of 20+ chars
        reply = 'כתוב: "ממוצע של 80"'
        result = verify_citations(reply, REAL_CONTEXT)
        assert "[ציטוט לא אומת" not in result

    def test_empty_context_replaces_all_long_quotes(self):
        reply = '"תנאי הקבלה לתוכנית MBA כוללים תואר ראשון בממוצע של 80 לפחות"'
        result = verify_citations(reply, "")
        assert "[ציטוט לא אומת — פנה למזכירות]" in result

    def test_empty_reply_returns_empty(self):
        result = verify_citations("", REAL_CONTEXT)
        assert result == ""
