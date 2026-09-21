import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.config import ROOT, load_config
from src.indexing import IndexStore
from src.ingestion import DocumentInput
from src import presentation
from tests.helpers import FakeEncoder


REPORT_PATH = ROOT / "artifacts" / "experiments" / "test_three_methods_v2" / "analysis.json"
REPORT_LABEL = "Sa\u010duvani eksperiment \u00b7 nije ocena ovog upita"


def set_report_presence(monkeypatch, present):
    original_exists = Path.exists

    def exists(path, *args, **kwargs):
        return present if path == REPORT_PATH else original_exists(path, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", exists)


def html_output(app):
    return "\n".join(element.proto.body for element in app.get("html"))


@pytest.fixture
def ui(tmp_path, monkeypatch):
    monkeypatch.delenv("SEMANTIC_SEARCH_REPORT", raising=False)
    st.cache_resource.clear()
    st.cache_data.clear()
    config = load_config()
    store = IndexStore(tmp_path / "ui_index")
    original = (
        "Regularization helps control overfitting.\n"
        "A regularization penalty changes the training objective.\n"
        "a)\n-3 -2 -1 0 1 2 3\n-1\n0\n1\n2\n3\n"
        "Figure 1: A regularization diagram."
    )
    sources = [DocumentInput("regularization.txt", original.encode("utf-8"),
                             "Regularization", "https://example.org/regularization", "Test source")]
    sources.extend(DocumentInput(f"notes_{number}.txt",
                                 f"Regression and classification example number {number}.".encode())
                   for number in range(5))
    store.add(sources, FakeEncoder(config))
    created_encoders = []

    def create_encoder(settings):
        encoder = FakeEncoder(settings)
        created_encoders.append(encoder)
        return encoder

    monkeypatch.setenv("SEMANTIC_SEARCH_INDEX_DIR", str(store.root))
    monkeypatch.setitem(sys.modules, "src.model", SimpleNamespace(SemanticEncoder=create_encoder))
    set_report_presence(monkeypatch, False)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
    assert not app.exception
    yield app, store, original, created_encoders
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.fixture
def legacy_ui(ui):
    app, store, _, _ = ui
    pointer = json.loads(store.active.read_text(encoding="utf-8"))
    directory = store.root / pointer["generation"]
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest.pop("bm25_parameters")
    for name in ("bm25.npz", "bm25_vectorizer.pkl"):
        del manifest["files"][name]
        (directory / name).unlink()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    store.active.write_text(json.dumps(pointer), encoding="utf-8")
    app.run()
    assert not app.exception
    return ui


def perform_search(app, query="What is regularization?"):
    app.text_input(key="query_input").set_value(query)
    app.button(key="search_submit").click().run()
    assert not app.exception
    assert sum(expander.label == REPORT_LABEL for expander in app.expander) == 1


def test_comparison_keeps_five_hits_per_method_and_exact_original(ui):
    app, store, original, _ = ui
    before = store.active.read_bytes()
    perform_search(app)
    responses = app.session_state["responses"]
    assert [response.method for response in responses] == ["semantic", "tfidf", "bm25"]
    assert all(len(response.hits) == 5 for response in responses)
    assert any(block.value == original for block in app.code)
    assert store.active.read_bytes() == before
    assert app.radio(key="search_method").options == ["Pore\u0111enje", "Semanti\u010dka", "TF-IDF", "BM25"]
    html = html_output(app)
    assert html.count('class="role-tag"') == 3
    assert "TERMINI / DU\u017dINA" in html
    assert "Skorovi razli\u010ditih metoda nisu na zajedni\u010dkoj skali" in html
    assert any("Nema sa\u010duvanih" in message.value for message in app.info)


def test_clear_removes_query_and_results_without_touching_collection(ui):
    app, store, _, _ = ui
    perform_search(app)
    before = store.active.read_bytes()
    app.button(key="clear_query").click().run()
    assert not app.exception
    assert app.text_input(key="query_input").value == ""
    assert "responses" not in app.session_state
    assert store.active.read_bytes() == before


def test_example_fills_query_but_does_not_search_or_load_model(ui):
    app, _, _, encoders = ui
    app.button(key="example_0").click().run()
    assert app.text_input(key="query_input").value == "What is regularization?"
    assert "responses" not in app.session_state
    assert not encoders


def test_empty_query_has_visible_error_and_no_stale_results(ui):
    app, store, _, _ = ui
    perform_search(app)
    before = store.active.read_bytes()
    perform_search(app, "   ")
    assert any("Pretraga nije uspela" in error.value for error in app.error)
    assert "responses" not in app.session_state
    assert store.active.read_bytes() == before


@pytest.mark.parametrize("label,method", [("TF-IDF", "tfidf"), ("BM25", "bm25")])
def test_lexical_oov_warning_and_zero_scores_are_preserved_without_model(ui, label, method):
    app, store, _, encoders = ui
    before = store.active.read_bytes()
    app.radio(key="search_method").set_value(label)
    perform_search(app, "zzzzunseenword")
    response, = app.session_state["responses"]
    assert response.method == method
    assert all(hit.score == 0 for hit in response.hits)
    assert any(f"Nema poznatih termina u {label}" in warning.value for warning in app.warning)
    assert not encoders
    assert store.active.read_bytes() == before


def test_bm25_only_keeps_raw_text_scores_and_rerun_without_loading_model(ui):
    app, store, original, encoders = ui
    before = store.active.read_bytes()
    app.radio(key="search_method").set_value("BM25")
    perform_search(app, "regularization")
    response, = app.session_state["responses"]
    assert response.method == "bm25" and len(response.hits) == 5
    assert response.hits[0].score > 1
    assert any(block.value == original for block in app.code)
    assert html_output(app).count('class="role-tag"') == 1
    app.toggle(key="technical_details").set_value(True).run()
    assert not app.exception
    assert app.session_state["responses"] == (response,)
    assert store.active.read_bytes() == before
    assert not encoders


def test_legacy_comparison_uses_only_prepared_methods_without_rebuilding(legacy_ui):
    app, store, _, _ = legacy_ui
    before = store.active.read_bytes()
    assert any("BM25 jo\u0161 nije pripremljen" in message.value for message in app.info)
    assert not app.button(key="prepare_bm25").disabled
    perform_search(app)
    assert [response.method for response in app.session_state["responses"]] == ["semantic", "tfidf"]
    assert html_output(app).count('class="role-tag"') == 2
    assert store.active.read_bytes() == before
    assert store.load(load_config()).bm25_matrix is None


def test_legacy_bm25_query_requires_explicit_preparation_without_model(legacy_ui):
    app, store, _, encoders = legacy_ui
    before = store.active.read_bytes()
    app.radio(key="search_method").set_value("BM25")
    perform_search(app)
    assert any("Pretraga nije uspela" in error.value for error in app.error)
    assert any("Koristite dugme Pripremi BM25" in message.value for message in app.text)
    assert "responses" not in app.session_state
    assert store.active.read_bytes() == before
    assert not encoders


def test_legacy_tfidf_remains_readable_without_upgrade_or_model(legacy_ui):
    app, store, original, encoders = legacy_ui
    before = store.active.read_bytes()
    app.radio(key="search_method").set_value("TF-IDF")
    perform_search(app)
    response, = app.session_state["responses"]
    assert response.method == "tfidf"
    assert any(block.value == original for block in app.code)
    assert store.active.read_bytes() == before
    assert not encoders


def test_explicit_preparation_preserves_old_artifacts_and_does_not_load_model(legacy_ui):
    app, store, original, encoders = legacy_ui
    old_index = store.load(load_config())
    old_directory = store.root / old_index.generation
    original_files = {
        path.relative_to(old_directory): path.read_bytes()
        for path in old_directory.rglob("*") if path.is_file()
    }
    sources = store.source_inputs()
    app.radio(key="search_method").set_value("TF-IDF")
    perform_search(app)
    before = app.session_state["responses"][0]
    app.button(key="prepare_bm25").click().run()
    assert not app.exception
    assert "responses" not in app.session_state
    assert app.radio(key="search_method").value == "TF-IDF"
    assert app.text_input(key="query_input").value == before.query
    assert any("BM25 je pripremljen" in message.value for message in app.success)
    assert not any(button.key == "prepare_bm25" for button in app.button)
    upgraded = store.load(load_config())
    assert upgraded.generation != old_index.generation
    assert upgraded.bm25_matrix is not None and upgraded.bm25_vectorizer is not None
    assert store.source_inputs() == sources
    for relative, content in original_files.items():
        assert (old_directory / relative).read_bytes() == content
        if relative != Path("manifest.json"):
            assert (store.root / upgraded.generation / relative).read_bytes() == content
    perform_search(app)
    assert app.session_state["responses"][0].hits == before.hits
    app.radio(key="search_method").set_value("BM25")
    perform_search(app)
    assert app.session_state["responses"][0].method == "bm25"
    assert any(block.value == original for block in app.code)
    assert not encoders


def test_failed_preparation_preserves_active_collection_and_results(legacy_ui, monkeypatch):
    app, store, _, encoders = legacy_ui
    app.radio(key="search_method").set_value("TF-IDF")
    perform_search(app)
    responses = app.session_state["responses"]
    before = store.active.read_bytes()

    def fail_upgrade(self, config):
        raise RuntimeError("Test preparation failure")

    monkeypatch.setattr(IndexStore, "upgrade_bm25", fail_upgrade)
    app.button(key="prepare_bm25").click().run()
    assert not app.exception
    assert any("Priprema BM25 nije uspela" in error.value for error in app.error)
    assert app.session_state["responses"] == responses
    assert store.active.read_bytes() == before
    assert not encoders
    assert sum(expander.label == REPORT_LABEL for expander in app.expander) == 1


def test_presentation_rerun_does_not_reencode_or_change_rankings(ui):
    app, _, _, encoders = ui
    perform_search(app)
    before = app.session_state["responses"]
    calls = encoders[0].calls
    app.toggle(key="technical_details").set_value(True).run()
    assert not app.exception
    assert app.session_state["responses"] == before
    assert encoders[0].calls == calls


def test_collection_change_invalidates_visible_results(ui):
    app, store, _, _ = ui
    perform_search(app)
    store.add([DocumentInput("new.txt", b"A distinct new regression material.")],
              FakeEncoder(load_config()))
    app.run()
    assert not app.exception
    assert "responses" not in app.session_state
    assert any("Kolekcija je promenjena" in message.value for message in app.info)


def test_results_from_previous_ui_version_do_not_require_new_timing_state(ui):
    app, _, _, encoders = ui
    perform_search(app)
    before = app.session_state["responses"]
    calls = encoders[0].calls
    del app.session_state["search_timings"]
    app.run()
    assert not app.exception
    assert app.session_state["responses"] == before
    assert encoders[0].calls == calls


def test_missing_three_method_report_is_explicit_and_not_fabricated(ui):
    app, _, _, encoders = ui
    assert any("Nema sa\u010duvanih zavr\u0161nih rezultata eksperimenta za sve tri metode."
               == message.value for message in app.info)
    assert 'class="metric-row"' not in html_output(app)
    assert not encoders


def test_invalid_saved_report_does_not_render_any_metrics(ui, monkeypatch):
    app, _, _, encoders = ui
    set_report_presence(monkeypatch, True)

    def invalid_report(path, index):
        assert path == REPORT_PATH
        raise ValueError("Sacuvana merenja pripadaju drugoj verziji kolekcije.")

    monkeypatch.setattr(presentation, "load_saved_experiment", invalid_report)
    app.run()
    assert not app.exception
    assert any("drugoj verziji" in message.value for message in app.info)
    assert 'class="metric-row"' not in html_output(app)
    assert not encoders


def test_complete_saved_report_displays_three_methods_and_ai_disclosure(ui, monkeypatch):
    app, _, _, encoders = ui
    set_report_presence(monkeypatch, True)
    report = presentation.SavedExperiment(
        tuple(presentation.ExperimentRow(method, 0.5, 0.25)
              for method in ("semantic", "tfidf", "bm25")),
        40, 35, "a" * 64, "18.09.2026.",
    )

    def completed_report(path, index):
        assert path == REPORT_PATH
        return report

    monkeypatch.setattr(presentation, "load_saved_experiment", completed_report)
    app.run()
    assert not app.exception
    html = html_output(app)
    assert html.count('class="metric-row"') == 6
    assert html.count("<span>BM25</span>") == 2
    assert "40 test pitanja / 35 informacionih potreba" in html
    assert "AI-pregledane oznake; bez ljudske validacije." in html
    assert "Precision@5 nije objavljen" in html
    assert not encoders


def test_completed_expanded_report_displays_precision_for_all_methods(ui, monkeypatch):
    app, _, _, encoders = ui
    set_report_presence(monkeypatch, True)
    report = presentation.SavedExperiment(
        tuple(presentation.ExperimentRow(method, 0.9, 0.7, 0.6)
              for method in ("semantic", "tfidf", "bm25")),
        100, 95, "b" * 64, "20.09.2026.", human_review_complete=True,
    )
    monkeypatch.setattr(presentation, "load_saved_experiment", lambda path, index: report)
    app.run()
    assert not app.exception
    html = html_output(app)
    assert html.count('class="metric-row"') == 9
    assert "Precision@5" in html
    assert "100 test pitanja / 95 informacionih potreba" in html
    assert "Ljudski pregled oznaka evidentiran." in html
    assert "AI-pregledane oznake; bez ljudske validacije." not in html
    assert not encoders


def test_reproduced_report_can_be_selected_without_replacing_the_original(ui, tmp_path, monkeypatch):
    app, _, _, encoders = ui
    selected = tmp_path / "reproduced_analysis.json"
    selected.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SEMANTIC_SEARCH_REPORT", str(selected))
    report = presentation.SavedExperiment(
        tuple(presentation.ExperimentRow(method, 0.9, 0.7, 0.6)
              for method in ("semantic", "tfidf", "bm25")),
        100, 95, "b" * 64, "21.09.2026.",
    )

    def load_selected(path, index):
        assert path == selected
        return report

    monkeypatch.setattr(presentation, "load_saved_experiment", load_selected)
    app.run()
    assert not app.exception
    assert html_output(app).count('class="metric-row"') == 9
    assert not encoders


def test_empty_collection_disables_search(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIC_SEARCH_INDEX_DIR", str(tmp_path / "empty_index"))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
    assert not app.exception
    assert app.button(key="search_submit").disabled
    assert any("Prvo dodajte materijale" in message.value for message in app.info)
