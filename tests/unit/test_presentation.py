from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pytest

from src.bm25 import BM25_PARAMETERS
from src.config import ROOT, load_config
from src.evaluation import CODE_FILES, corpus_fingerprint
from src.indexing import IndexStore
from src.ingestion import DocumentInput
from src.presentation import load_saved_experiment, markdown_label, reading_html, reading_view
from tests.helpers import FakeEncoder


REGULARIZATION = (
    ". Regularization problems are typically formulated as optimization\n"
    "problems involving classi\ufb01cation loss and a regularization penalty.\n"
    "The penalty stabilizes the ob\u00ad\njective.\n"
    "\n"
    "a)\n\u22123 \u22122 \u22121 0 1 2 3\n\u22121\n\u22120.5\n0\n0.5\n1\n2\n3 b)\n"
    "\u22123 \u22122 \u22121 0 1 2 3\n\u22121\n0\n1\n2\n3\n"
    "Figure 1: Hinge loss and logistic loss log[1 + exp(-z)]."
)


def test_reader_folds_axis_runs_but_preserves_original_and_caption():
    view = reading_view(REGULARIZATION)
    assert view.original == REGULARIZATION
    assert len(view.numeric_fragments) == 1
    assert "\u22123 \u22122" in view.numeric_fragments[0]
    assert view.folded_lines >= 12
    assert "classification loss" in view.paragraphs[0]
    assert "objective." in view.paragraphs[0]
    assert view.paragraphs[-1].startswith("Figure 1:")
    assert "log[1 + exp(-z)]" in view.paragraphs[-1]


def test_reader_keeps_variable_equations_and_isolated_numbers_visible():
    original = "Loss = mse + lambda * penalty\nw = -0.5\n\nThreshold:\n0.5\n\nL\u00b2 and \u03bb"
    view = reading_view(original)
    assert view.original == original
    assert not view.numeric_fragments
    assert "Loss = mse + lambda * penalty" in reading_html(view)
    assert "w = -0.5" in reading_html(view)
    assert view.blocks[0].kind == "formula"
    assert "\n" in view.blocks[0].text
    assert "Threshold: 0.5" in view.paragraphs
    assert view.paragraphs[-1] == "L\u00b2 and \u03bb"


def test_numeric_only_passage_is_not_discarded():
    text = "1 0\n0 1\n2 3"
    view = reading_view(text)
    assert not view.paragraphs
    assert view.numeric_fragments == (text,)
    assert view.original == text


def test_two_numeric_lines_are_not_folded():
    view = reading_view("Values:\n1 2\n3 4")
    assert not view.numeric_fragments
    assert view.paragraphs == ("Values: 1 2 3 4",)


def test_bullets_remain_separate_and_visible_hyphens_are_not_guessed():
    view = reading_view("- Task-specific\nmethods.\n- Non-\nlinear models.")
    assert view.paragraphs == ("- Task-specific methods.", "- Non- linear models.")


def test_reader_handles_crlf_without_changing_the_original():
    original = "One line.\r\nAnother line.\r\n\r\nA paragraph."
    view = reading_view(original)
    assert view.original == original
    assert view.paragraphs == ("One line. Another line.", "A paragraph.")


def test_replacement_glyphs_are_reported_not_reconstructed():
    view = reading_view("A \ufffd B")
    assert view.replacement_characters == 1
    assert view.paragraphs == ("A \ufffd B",)


def test_damaged_pdf_formulas_are_separated_from_the_explanatory_paragraph():
    prefix = (
        "X\n\ufffd \u03b8\u2217\n\u03b8\u2217\n0\n\ufffd\n\ufffd \ufffd\n(49)\n"
        "= (\u03bbI + XT X)\u22121(XT X + \u03bbI \u2212 \u03bbI) \u03b8\u2217\n"
        "\u03b8\u2217\n0\n(50)\n\ufffd \ufffd \ufffd bias\ufffd\ufffd \ufffd \ufffd\ufffd\n"
        "=\n=\n\u03b8\u2217\n\u02c6\u03b8\u2217\n0\n"
        "\u2212\u03bb(\u03bbI + XT X)\u22121 \u03b8\u2217\n(51)\n(52)\n\ufffd \ufffd\n"
    )
    prose = (
        "It is straightforward to check that I \u2212 \u03bb(\u03bbI + XT X)\u22121 is a positive de\ufb01nite matrix with\n"
        "eigenvalues all less than one. The parameter estimates are therefore shrunk towards zero."
    )
    tail = "\n\u03b8\u02c6\nCov \u02c6 |X = \u03c3\u22172(\u03bb"
    original = prefix + prose + tail
    view = reading_view(original)
    assert view.original == original
    assert len(view.formula_fragments) == 2
    assert view.formula_fragments[0] == prefix.strip()
    assert view.formula_fragments[1] == tail.strip()
    assert view.paragraphs[0].startswith("It is straightforward")
    assert "positive definite matrix" in view.paragraphs[0]
    assert "\ufffd" not in reading_html(view)
    assert view.replacement_characters == original.count("\ufffd")


def test_inline_math_in_natural_language_stays_in_prose():
    text = "The slope is y = a*x + b.\nIt is positive when x > 0."
    view = reading_view(text)
    assert not view.formula_fragments
    assert view.paragraphs == ("The slope is y = a*x + b. It is positive when x > 0.",)
    assert all(block.kind == "prose" for block in view.blocks)


def test_short_prose_does_not_become_an_equation():
    text = "I\ncan explain the method.\n\nA\nshort introduction follows."
    view = reading_view(text)
    assert not view.formula_fragments
    assert view.paragraphs == ("I can explain the method.", "A short introduction follows.")


@pytest.mark.parametrize("value", [None, "", " \n\t", 42])
def test_reader_rejects_invalid_or_blank_input(value):
    with pytest.raises(ValueError, match="neprazan"):
        reading_view(value)


def test_reader_html_treats_documents_as_literal_text():
    text = '<script>alert("x")</script>\n![remote](https://example.invalid/image)'
    html = reading_html(reading_view(text))
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "<img" not in html
    assert reading_view(text).original == text


def test_expander_labels_cannot_create_markdown_images_or_links():
    label = markdown_label("![remote](https://example.invalid) **bold** <img>")
    assert label.startswith(r"\!\[remote\]\(")
    assert r"\*\*bold\*\*" in label
    assert r"\<img\>" in label


@pytest.fixture
def experiment(tmp_path):
    config = load_config()
    store = IndexStore(tmp_path / "index")
    store.add([DocumentInput("sample.txt", b"Regularization controls overfitting.")], FakeEncoder(config))
    index = store.load(config)
    identity = "a" * 64
    code = {
        str(Path("src") / name): hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
        for name in CODE_FILES
    }
    methods = {}
    per_type = {}
    for method, hit, mrr in (("tfidf", 0.85, 0.6854), ("semantic", 0.95, 0.8017),
                              ("bm25", 0.90, 0.725)):
        methods[method] = {
            "status": "completed", "method": method, "split": "test",
            "query_count": 40, "generation": index.generation,
            "config": asdict(config), "corpus_fingerprint": corpus_fingerprint(index),
            "bm25_parameters": dict(BM25_PARAMETERS),
            "code_sha256": code, "dataset_sha256": identity,
            "annotation_mode": "ai_source_reviewed", "human_review_complete": False,
            "hit_at_5": hit, "mrr_at_5": mrr,
            "finished_at_utc": "2026-09-17T11:23:03+00:00",
        }
        per_type[method] = {
            kind: {"query_count": count, "hit_at_5": hit, "mrr_at_5": mrr}
            for kind, count in (("direct", 10), ("paraphrase", 15), ("synonym", 15))
        }
    payload = {
        "annotation_mode": "ai_source_reviewed", "human_review_complete": False,
        "metrics_recomputed_from_raw_rankings": True, "test_queries": 40,
        "distinct_test_intent_groups": 35, "dataset_sha256": identity,
        "method_summaries": methods,
        "quality_by_query_type": per_type,
    }
    return tmp_path / "analysis.json", payload, index


def save_report(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_saved_metrics_are_final_and_explicitly_ai_reviewed(experiment):
    path, payload, index = experiment
    save_report(path, payload)
    report = load_saved_experiment(path, index)
    assert report.query_count == 40 and report.intent_count == 35
    assert [row.method for row in report.rows] == ["semantic", "tfidf", "bm25"]
    assert report.rows[0].hit_at_5 == 0.95
    assert report.rows[2].hit_at_5 == 0.90
    assert report.measured_on == "17.09.2026."
    assert all(row.precision_at_5 is None for row in report.rows)


def expanded_report(payload):
    payload.update(
        protocol_id="ml-basics-evaluation-v2", protocol_sha256="b" * 64,
        metrics_schema_version=2,
        precision_label_coverage="pooled_top5", test_queries=100, distinct_test_intent_groups=95,
        protocol={"protocol_id": "ml-basics-evaluation-v2", "expected_splits": {"test": 100},
                  "k": 5, "precision_denominator": 5},
    )
    for method, summary in payload["method_summaries"].items():
        summary.update(
            protocol_id=payload["protocol_id"], protocol_sha256=payload["protocol_sha256"],
            metrics_schema_version=2,
            precision_label_coverage="pooled_top5", query_count=100, precision_at_5=0.6,
        )
        for kind, count in (("direct", 45), ("paraphrase", 37), ("synonym", 18)):
            payload["quality_by_query_type"][method][kind].update(query_count=count, precision_at_5=0.6)


def test_expanded_saved_report_includes_precision_without_relabeling_the_legacy_run(experiment):
    path, payload, index = experiment
    expanded_report(payload)
    save_report(path, payload)
    report = load_saved_experiment(path, index)
    assert report.query_count == 100 and report.intent_count == 95
    assert all(row.precision_at_5 == 0.6 for row in report.rows)
    assert report.human_review_complete is False


@pytest.mark.parametrize("change", ["missing_precision", "invalid_precision", "unjudged", "different_protocol"])
def test_expanded_report_rejects_incomplete_precision(experiment, change):
    path, payload, index = experiment
    expanded_report(payload)
    if change == "missing_precision":
        del payload["method_summaries"]["bm25"]["precision_at_5"]
    elif change == "invalid_precision":
        payload["method_summaries"]["bm25"]["precision_at_5"] = 1.1
    elif change == "unjudged":
        payload["precision_label_coverage"] = "not_adjudicated"
    else:
        payload["method_summaries"]["semantic"]["protocol_sha256"] = "c" * 64
    save_report(path, payload)
    with pytest.raises(ValueError):
        load_saved_experiment(path, index)


def test_legacy_report_cannot_gain_validated_precision_by_adding_only_a_flag(experiment):
    path, payload, index = experiment
    payload["precision_label_coverage"] = "pooled_top5"
    save_report(path, payload)
    with pytest.raises(ValueError, match="prosireni protokol"):
        load_saved_experiment(path, index)


@pytest.mark.parametrize("method", ["semantic", "tfidf", "bm25"])
@pytest.mark.parametrize("field,value", [
    ("status", "running"),
    ("generation", "b" * 32),
    ("corpus_fingerprint", "wrong"),
    ("dataset_sha256", "b" * 64),
    ("config", {}),
    ("code_sha256", {}),
    ("bm25_parameters", {}),
    ("bm25_parameters", {**BM25_PARAMETERS, "b": 0.5}),
    ("hit_at_5", True),
    ("hit_at_5", float("nan")),
    ("mrr_at_5", 0.99),
    ("human_review_complete", True),
    ("finished_at_utc", "2026-09-17"),
])
def test_partial_stale_or_invalid_metric_rows_are_not_shown(experiment, method, field, value):
    path, payload, index = experiment
    payload["method_summaries"][method][field] = value
    save_report(path, payload)
    with pytest.raises(ValueError):
        load_saved_experiment(path, index)


@pytest.mark.parametrize("method", ["semantic", "tfidf", "bm25"])
def test_two_finished_methods_are_not_a_three_method_comparison(experiment, method):
    path, payload, index = experiment
    del payload["method_summaries"][method]
    save_report(path, payload)
    with pytest.raises(ValueError):
        load_saved_experiment(path, index)


@pytest.mark.parametrize("field,value", [
    ("annotation_mode", "reviewed"),
    ("human_review_complete", True),
    ("metrics_recomputed_from_raw_rankings", False),
    ("test_queries", 39),
    ("test_queries", True),
    ("distinct_test_intent_groups", 34),
    ("dataset_sha256", "not-a-checksum"),
])
def test_saved_report_requires_the_frozen_ai_reviewed_protocol(experiment, field, value):
    path, payload, index = experiment
    payload[field] = value
    save_report(path, payload)
    with pytest.raises(ValueError):
        load_saved_experiment(path, index)


def test_saved_report_requires_bm25_code_provenance(experiment):
    path, payload, index = experiment
    del payload["method_summaries"]["bm25"]["code_sha256"][str(Path("src") / "bm25.py")]
    save_report(path, payload)
    with pytest.raises(ValueError, match="drugoj verziji"):
        load_saved_experiment(path, index)


def test_saved_report_requires_query_type_results_for_bm25(experiment):
    path, payload, index = experiment
    del payload["quality_by_query_type"]["bm25"]
    save_report(path, payload)
    with pytest.raises(ValueError, match="po tipu upita"):
        load_saved_experiment(path, index)


@pytest.mark.parametrize("field,value", [
    ("query_count", True),
    ("query_count", 9),
    ("hit_at_5", float("nan")),
    ("mrr_at_5", 0.99),
])
def test_incomplete_or_invalid_bm25_query_type_results_are_not_shown(experiment, field, value):
    path, payload, index = experiment
    payload["quality_by_query_type"]["bm25"]["direct"][field] = value
    save_report(path, payload)
    with pytest.raises(ValueError):
        load_saved_experiment(path, index)


def test_saved_report_requires_the_same_query_types_and_counts_for_all_methods(experiment):
    path, payload, index = experiment
    typed = payload["quality_by_query_type"]["bm25"]
    typed["direct"]["query_count"] -= 1
    typed["paraphrase"]["query_count"] += 1
    save_report(path, payload)
    with pytest.raises(ValueError, match="isti potpuni test skup"):
        load_saved_experiment(path, index)


def test_legacy_index_cannot_display_three_method_measurements(experiment):
    path, payload, index = experiment
    save_report(path, payload)
    legacy = replace(index, bm25_matrix=None, bm25_vectorizer=None)
    with pytest.raises(ValueError, match="pripremite BM25"):
        load_saved_experiment(path, legacy)


def test_historic_two_method_report_is_rejected_without_rewriting_it(experiment):
    path, payload, index = experiment
    del payload["method_summaries"]["bm25"]
    del payload["quality_by_query_type"]["bm25"]
    for summary in payload["method_summaries"].values():
        del summary["bm25_parameters"]
    save_report(path, payload)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="sve tri metode"):
        load_saved_experiment(path, index)
    assert path.read_bytes() == original
