from dataclasses import replace

import numpy as np
import pytest

import src.model as runtime
from src.config import load_config
from tests.helpers import FakeEncoder, WordTokenizer


@pytest.fixture
def encoder(monkeypatch):
    config = replace(load_config(), chunk_tokens=3, overlap_tokens=1)

    class LocalModel:
        tokenizer = WordTokenizer()
        max_seq_length = 8

        def __init__(self, *args, **kwargs):
            assert kwargs["device"] == "cpu"
            assert kwargs["token"] is False
            assert kwargs["trust_remote_code"] is False
            assert kwargs["local_files_only"] is True

        def get_embedding_dimension(self):
            return 4

        def encode(self, texts, **kwargs):
            assert kwargs["normalize_embeddings"] is True
            return FakeEncoder(config).encode(texts)

    monkeypatch.setattr(runtime, "SentenceTransformer", LocalModel)
    return runtime.SemanticEncoder(config)


def test_model_wrapper_returns_finite_normalized_vectors(encoder):
    vectors = encoder.encode(["regularization", "regression"])
    assert vectors.shape == (2, 4)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0)


@pytest.mark.parametrize("texts", [[], [""], ["  "], "not a batch"])
def test_invalid_model_inputs_are_rejected(encoder, texts):
    with pytest.raises(ValueError):
        encoder.encode(texts)


def test_overlong_model_input_is_rejected_without_truncation(encoder):
    with pytest.raises(ValueError, match="nije odsecen"):
        encoder.encode(["one two three four five six seven"])


@pytest.mark.parametrize("values", [
    np.zeros((1, 4), dtype=np.float32),
    np.full((1, 4), np.nan, dtype=np.float32),
    np.ones((1, 2), dtype=np.float32),
])
def test_invalid_model_outputs_are_rejected(encoder, monkeypatch, values):
    monkeypatch.setattr(encoder.model, "encode", lambda *args, **kwargs: values)
    with pytest.raises(ValueError):
        encoder.encode(["regularization"])


def test_missing_model_is_not_replaced_by_a_fake_encoder(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("model cache is missing")

    monkeypatch.setattr(runtime, "SentenceTransformer", unavailable)
    with pytest.raises(OSError, match="model cache"):
        runtime.SemanticEncoder(load_config())
