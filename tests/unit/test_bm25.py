import hashlib
import json
import math

import numpy as np
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

import src.indexing as indexing
from src.bm25 import build_bm25
from src.cli import main as cli_main
from src.config import load_config
from src.indexing import CorruptIndexError, IndexBusyError, IndexStore, StaleIndexError
from src.ingestion import DocumentInput
from src.search import resolve_methods, search
from tests.helpers import FakeEncoder


def test_bm25_matches_independently_calculated_formula():
    texts = ["apple apple banana", "apple carrot carrot carrot", "banana"]
    vectorizer, matrix = build_bm25(texts, TfidfVectorizer())
    assert matrix.dtype == np.float32
    average = 8 / 3
    for row, text in enumerate(texts):
        for term, column in vectorizer.vocabulary_.items():
            frequency = text.split().count(term)
            document_frequency = sum(term in document.split() for document in texts)
            idf = math.log(1 + (3 - document_frequency + 0.5) / (document_frequency + 0.5))
            expected = idf * frequency * 2.5 / (
                frequency + 1.5 * (0.25 + 0.75 * len(text.split()) / average)
            )
            assert matrix[row, column] == pytest.approx(expected, rel=1e-6)


def test_term_frequency_saturates_and_common_terms_stay_positive():
    vectorizer, matrix = build_bm25(["apple", "apple apple"], TfidfVectorizer(), b=0)
    first, second = matrix.toarray()[:, vectorizer.vocabulary_["apple"]]
    assert 0 < first < second < 2 * first
    assert second < math.log(1 + 0.5 / 2.5) * 2.5


def test_length_normalization_penalizes_longer_equal_frequency_passage():
    vectorizer, matrix = build_bm25(
        ["apple", "apple banana banana banana"], TfidfVectorizer()
    )
    column = vectorizer.vocabulary_["apple"]
    assert matrix[0, column] > matrix[1, column] > 0


def test_lexical_analysis_matches_tfidf_and_retains_empty_token_rows():
    texts = ["Regularization, MODEL! x", "a i", "model cafe\u0301"]
    template = TfidfVectorizer(lowercase=True, strip_accents=None, stop_words=None)
    template.fit(texts)
    vectorizer, matrix = build_bm25(texts, template)
    assert list(vectorizer.get_feature_names_out()) == list(template.get_feature_names_out())
    assert vectorizer.build_analyzer()(texts[0]) == template.build_analyzer()(texts[0])
    assert matrix.shape[0] == 3 and matrix[1].nnz == 0
    assert np.isfinite(matrix.data).all()


@pytest.mark.parametrize("kwargs", [
    {"k1": 0}, {"k1": -1}, {"k1": float("nan")}, {"k1": True},
    {"b": -0.1}, {"b": 1.1}, {"b": float("inf")}, {"b": False},
])
def test_invalid_parameters_are_explicit_errors(kwargs):
    with pytest.raises(ValueError, match="BM25"):
        build_bm25(["some text"], TfidfVectorizer(), **kwargs)


@pytest.mark.parametrize("texts", [[], "not a collection", ["a i"]])
def test_empty_or_unusable_corpus_is_rejected(texts):
    with pytest.raises(ValueError):
        build_bm25(texts, TfidfVectorizer())


@pytest.fixture
def corpus(tmp_path):
    config = load_config()
    encoder = FakeEncoder(config)
    store = IndexStore(tmp_path / "indexes")
    store.add([
        DocumentInput("regularization.txt", b"Regularization reduces overfitting."),
        DocumentInput("regression.txt", b"Linear regression predicts a continuous value."),
    ], encoder)
    return store, encoder


def rewrite_manifest(store, generation, mutate):
    path = store.root / generation / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    content = indexing._json_bytes(manifest)
    path.write_bytes(content)
    store.active.write_bytes(indexing._json_bytes({
        "generation": generation, "manifest_sha256": hashlib.sha256(content).hexdigest(),
    }))


def legacy_bundle(store, encoder):
    generation = store.load(encoder.config).generation
    for name in ("bm25.npz", "bm25_vectorizer.pkl"):
        (store.root / generation / name).unlink()

    def make_legacy(manifest):
        manifest["schema_version"] = 1
        manifest.pop("bm25_parameters")
        for name in ("bm25.npz", "bm25_vectorizer.pkl"):
            manifest["files"].pop(name)

    rewrite_manifest(store, generation, make_legacy)
    return store.load(encoder.config)


def test_persistence_oov_binary_query_terms_and_shared_result_contract(corpus):
    store, encoder = corpus
    index = IndexStore(store.root).load(encoder.config)
    response = search(index, "regularization", "bm25")
    repeated = search(index, "regularization regularization", "bm25")
    assert response.hits == repeated.hits
    assert response.hits[0].file_name == "regularization.txt"
    assert response.hits[0].score > 0 and response.hits[1].score == 0
    assert all(hit.method == "bm25" for hit in response.hits)
    unknown = search(index, "zzzzunseenword", "bm25")
    assert all(hit.score == 0 for hit in unknown.hits)
    assert [hit.chunk_id for hit in unknown.hits] == sorted(hit.chunk_id for hit in unknown.hits)
    assert "BM25 recniku" in unknown.warnings[0]
    with pytest.raises(ValueError, match="neprazan"):
        search(index, " ", "bm25")


def test_upgrade_preserves_all_legacy_artifacts_ids_and_original_rankings(corpus):
    store, encoder = corpus
    old = legacy_bundle(store, encoder)
    before_files = {
        str(path.relative_to(store.root / old.generation)): path.read_bytes()
        for path in (store.root / old.generation).rglob("*") if path.is_file()
    }
    before_scores = {method: search(old, "regularization", method, encoder).hits
                     for method in ("semantic", "tfidf")}
    with pytest.raises(StaleIndexError, match="upgrade-bm25"):
        search(old, "regularization", "bm25")
    calls = encoder.calls
    result = store.upgrade_bm25(encoder.config)
    new = store.load(encoder.config)
    assert result.changed and new.generation != old.generation
    assert encoder.calls == calls and old.chunks == new.chunks
    for relative, content in before_files.items():
        assert (store.root / old.generation / relative).read_bytes() == content
        if relative != "manifest.json":
            assert (store.root / new.generation / relative).read_bytes() == content
    for method, hits in before_scores.items():
        assert search(new, "regularization", method, encoder).hits == hits
    pointer = store.active.read_bytes()
    assert not store.upgrade_bm25(encoder.config).changed
    assert store.active.read_bytes() == pointer


def test_failed_upgrade_keeps_the_previous_active_bundle(corpus, monkeypatch):
    store, encoder = corpus
    old = legacy_bundle(store, encoder)
    pointer = store.active.read_bytes()
    actual_replace = indexing.os.replace

    def fail_publish(source, destination):
        if destination == store.active:
            raise PermissionError("controlled pointer failure")
        return actual_replace(source, destination)

    monkeypatch.setattr(indexing.os, "replace", fail_publish)
    with pytest.raises(PermissionError):
        store.upgrade_bm25(encoder.config)
    assert store.active.read_bytes() == pointer
    assert store.load(encoder.config).generation == old.generation


def test_upgrade_respects_writer_lock(corpus):
    store, encoder = corpus
    legacy_bundle(store, encoder)
    with store.lock:
        with pytest.raises(IndexBusyError):
            IndexStore(store.root).upgrade_bm25(encoder.config)


def test_stale_bm25_parameters_are_rejected_but_rebuild_is_possible(corpus):
    store, encoder = corpus
    generation = store.load(encoder.config).generation
    rewrite_manifest(store, generation, lambda manifest: manifest["bm25_parameters"].update(k1=99))
    with pytest.raises(StaleIndexError, match="BM25"):
        store.load(encoder.config)
    store.rebuild(encoder)
    assert store.load(encoder.config).bm25_matrix is not None


def test_negative_or_corrupt_bm25_data_is_rejected(corpus):
    store, encoder = corpus
    index = store.load(encoder.config)
    path = store.root / index.generation / "bm25.npz"
    matrix = index.bm25_matrix.copy()
    matrix.data[0] = -1
    sparse.save_npz(path, matrix)
    rewrite_manifest(store, index.generation, lambda manifest: manifest["files"].update(
        {"bm25.npz": hashlib.sha256(path.read_bytes()).hexdigest()}
    ))
    with pytest.raises(CorruptIndexError, match="BM25"):
        store.load(encoder.config)


def test_cli_upgrade_and_bm25_search_do_not_require_neural_encoder(corpus, monkeypatch, capsys):
    store, encoder = corpus
    legacy_bundle(store, encoder)
    monkeypatch.setattr("sys.argv", [
        "search", "--index-dir", str(store.root), "upgrade-bm25",
    ])
    assert cli_main() == 0
    assert json.loads(capsys.readouterr().out)["bm25_ready"]
    monkeypatch.setattr("sys.argv", [
        "search", "--index-dir", str(store.root), "search", "regularization", "--method", "bm25",
    ])
    assert cli_main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert [result["method"] for result in payload["results"]] == ["bm25"]


def test_original_pair_alias_and_new_all_are_distinct():
    assert resolve_methods("both") == ("semantic", "tfidf")
    assert resolve_methods("all") == ("semantic", "tfidf", "bm25")
    assert resolve_methods("bm25") == ("bm25",)
    with pytest.raises(ValueError):
        resolve_methods("unknown")
