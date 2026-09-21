import pytest

from scripts.export_evaluation_review import spreadsheet_text


@pytest.mark.parametrize("text", ["=SUM(A1:A2)", "+1+1", "-2+3", "@SUM(A1)", " \t=1+1"])
def test_spreadsheet_formula_like_content_is_plain_text(text):
    assert spreadsheet_text(text) == "'" + text


def test_plain_question_is_not_changed():
    text = "Why does regularization help?"
    assert spreadsheet_text(text) == text
