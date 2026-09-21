from dataclasses import replace

import numpy as np
import pytest
from filelock import FileLock

import src.indexing as indexing
from src.config import load_config
from src.indexing import (
    CorruptIndexError, IndexBusyError, IndexErrorBase, IndexStore, NoIndexError,
    StaleIndexError,
)
from src.ingestion import BatchImportError, DocumentInput
from src.search import search
from tests.helpers import FakeEncoder


@pytest.fixture
def setup(tmp_path):
    config = replace(load_config(), chunk_tokens=20, overlap_tokens=3)
    encoder = FakeEncoder(config)
    store = IndexStore(tmp_path / "indexes")
    store.add([DocumentInput("regularization.txt", b"Regularization reduces overfitting.")], encoder)
    return store, encoder


def test_load_persists_sources_vectors_and_tfidf(setup):
    store, encoder = setup
    index = IndexStore(store.root).load(encoder.config)
    assert len(index.documents) == len(index.chunks) == 1
    assert index.embeddings.shape == (1, 4)
    assert not index.embeddings.flags.writeable
    assert store.source_inputs()[0].content == b"Regularization reduces overfitting."
    assert index.tfidf_matrix.shape[0] == 1


def test_append_and_same_name_different_content_preserve_existing(setup):
    store, encoder = setup
    old = store.load(encoder.config)
    result = store.add([
        DocumentInput("regularization.txt", b"Linear regression predicts numeric values.")
    ], encoder)
    index = store.load(encoder.config)
    assert result.changed and len(result.added_ids) == 1
    assert len(index.documents) == 2
    assert old.documents[0].document_id in {doc.document_id for doc in index.documents}
    assert len({doc.document_id for doc in index.documents}) == 2
    assert (store.root / old.generation).is_dir()


def test_entire_invalid_group_is_rejected_without_publication(setup):
    store, encoder = setup
    before = store.active.read_bytes()
    with pytest.raises(BatchImportError):
        store.add([
            DocumentInput("good.txt", b"Classification predicts labels."),
            DocumentInput("broken.pdf", b"not a PDF"),
        ], encoder)
    assert store.active.read_bytes() == before
    assert len(store.load(encoder.config).documents) == 1


def test_duplicate_is_visible_noop_and_does_not_encode_again(setup):
    store, encoder = setup
    before = store.active.read_bytes()
    calls = encoder.calls
    result = store.add([
        DocumentInput("renamed.txt", b"Regularization reduces overfitting.")
    ], encoder)
    assert not result.changed and result.added_ids == ()
    assert result.duplicate_file_names == ("renamed.txt",)
    assert store.active.read_bytes() == before
    assert encoder.calls == calls


def test_failed_encoding_preserves_current_generation(setup, monkeypatch):
    store, encoder = setup
    before = store.active.read_bytes()

    def fail(texts):
        raise RuntimeError("model failure")

    monkeypatch.setattr(encoder, "encode", fail)
    with pytest.raises(RuntimeError, match="model failure"):
        store.add([DocumentInput("new.txt", b"Classification predicts labels.")], encoder)
    assert store.active.read_bytes() == before
    assert not list(store.root.glob(".building-*"))


def test_failed_pointer_publication_preserves_active_collection(setup, monkeypatch):
    store, encoder = setup
    before = store.active.read_bytes()
    original_replace = indexing.os.replace

    def fail_pointer(source, destination):
        if destination == store.active:
            raise PermissionError("simulated pointer lock")
        return original_replace(source, destination)

    monkeypatch.setattr(indexing.os, "replace", fail_pointer)
    with pytest.raises(PermissionError):
        store.add([DocumentInput("new.txt", b"Classification predicts labels.")], encoder)
    assert store.active.read_bytes() == before
    assert len(store.load(encoder.config).documents) == 1
    assert not list(store.root.glob(".active-*.tmp"))


def test_nonfinite_vectors_cannot_be_published(setup, monkeypatch):
    store, encoder = setup
    before = store.active.read_bytes()
    monkeypatch.setattr(encoder, "encode",
                        lambda texts: np.full((len(texts), 4), np.nan, dtype=np.float32))
    with pytest.raises(CorruptIndexError):
        store.add([DocumentInput("new.txt", b"Classification predicts labels.")], encoder)
    assert store.active.read_bytes() == before


def test_empty_tfidf_vocabulary_does_not_create_an_active_index(tmp_path):
    store = IndexStore(tmp_path / "indexes")
    with pytest.raises(IndexErrorBase, match="TF-IDF"):
        store.add([DocumentInput("short.txt", b"a i")], FakeEncoder(load_config()))
    assert not store.active.exists()


def test_stale_configuration_and_rebuild(setup):
    store, encoder = setup
    updated = replace(encoder.config, overlap_tokens=0)
    with pytest.raises(StaleIndexError):
        store.load(updated)
    result = store.rebuild(FakeEncoder(updated))
    assert result.changed
    assert len(store.load(updated).documents) == 1


def test_library_mismatch_rejected_before_unpickling(setup, monkeypatch):
    store, encoder = setup
    monkeypatch.setattr(indexing, "_versions", lambda: {"numpy": "different"})
    monkeypatch.setattr(indexing.pickle, "load",
                        lambda stream: pytest.fail("Do not unpickle incompatible artifacts"))
    with pytest.raises(StaleIndexError):
        store.load(encoder.config)


def test_corrupt_matrix_is_rejected_but_sources_can_rebuild(setup):
    store, encoder = setup
    generation = store.load(encoder.config).generation
    (store.root / generation / "embeddings.npy").write_bytes(b"corrupt")
    with pytest.raises(CorruptIndexError):
        store.load(encoder.config)
    store.rebuild(encoder)
    assert len(store.load(encoder.config).chunks) == 1


def test_invalid_active_pointer_and_missing_index(tmp_path):
    store = IndexStore(tmp_path / "indexes")
    with pytest.raises(NoIndexError):
        store.load(load_config())
    store.active.write_text('{"generation": "../escape"}', encoding="utf-8")
    with pytest.raises(CorruptIndexError):
        store.load(load_config())


def test_concurrent_writer_is_reported_as_busy(setup):
    store, encoder = setup
    with FileLock(str(store.root / ".write.lock")):
        with pytest.raises(IndexBusyError):
            IndexStore(store.root).add([DocumentInput("new.txt", b"new text")], encoder)


@pytest.mark.parametrize("method", ["semantic", "tfidf", "bm25"])
def test_shared_result_format_ranking_and_small_corpus(setup, method):
    store, encoder = setup
    store.add([DocumentInput("regression.txt", b"Linear regression predicts values.")], encoder)
    index = store.load(encoder.config)
    response = search(index, "regularization", method, encoder)
    assert len(response.hits) == 2
    assert response.hits[0].file_name == "regularization.txt"
    assert [hit.rank for hit in response.hits] == [1, 2]
    assert all(hit.method == method and hit.page_start is None for hit in response.hits)
    assert response.generation == index.generation


def test_tfidf_unknown_terms_have_zero_scores_and_deterministic_ties(setup):
    store, encoder = setup
    store.add([DocumentInput("regression.txt", b"Linear regression predicts values.")], encoder)
    response = search(store.load(encoder.config), "unseenzephyrword", "tfidf")
    assert all(hit.score == 0 for hit in response.hits)
    assert [hit.chunk_id for hit in response.hits] == sorted(hit.chunk_id for hit in response.hits)
    assert any("Nema poznatih termina" in warning for warning in response.warnings)


@pytest.mark.parametrize("query", ["", " \n\t", None])
def test_empty_query_rejected(setup, query):
    store, encoder = setup
    with pytest.raises(ValueError, match="neprazan"):
        search(store.load(encoder.config), query, "tfidf")


def test_semantic_encoder_and_index_must_match(setup):
    store, encoder = setup
    with pytest.raises(StaleIndexError):
        search(store.load(encoder.config), "regularization", "semantic",
               FakeEncoder(replace(encoder.config, overlap_tokens=0)))


def test_search_does_not_accept_an_empty_index(setup):
    store, encoder = setup
    empty = replace(store.load(encoder.config), chunks=(),
                    embeddings=np.empty((0, 4), dtype=np.float32))
    with pytest.raises(NoIndexError):
        search(empty, "query", "tfidf")


def test_known_cosine_values_keep_opposite_direction_negative(setup):
    store, encoder = setup
    store.add([
        DocumentInput("regression.txt", b"Regression predicts values."),
        DocumentInput("classification.txt", b"Classification predicts labels."),
    ], encoder)
    index = replace(store.load(encoder.config), embeddings=np.array(
        [[1, 0], [0, 1], [-1, 0]], dtype=np.float32
    ))

    class FixedQueryEncoder(FakeEncoder):
        def encode(self, texts):
            return np.array([[1, 0]], dtype=np.float32)

    response = search(index, "query", "semantic", FixedQueryEncoder(encoder.config))
    assert [hit.score for hit in response.hits] == [1.0, 0.0, -1.0]
    assert [hit.chunk_id for hit in response.hits] == [chunk.chunk_id for chunk in index.chunks]


def test_top_five_is_capped_without_duplicate_hits(setup):
    store, encoder = setup
    store.add([
        DocumentInput(f"notes-{number}.txt", f"Regression example number {number}.".encode())
        for number in range(7)
    ], encoder)
    response = search(store.load(encoder.config), "regression", "tfidf")
    assert len(response.hits) == 5
    assert len({hit.chunk_id for hit in response.hits}) == 5
