from dataclasses import replace

import pytest

from src.config import SearchConfig, load_config


def test_current_config_is_valid_and_deterministic():
    first = load_config()
    assert first.device == "cpu"
    assert first.chunk_tokens == 200
    assert first.overlap_tokens == 40
    assert first.top_k == 5
    assert first.fingerprint() == load_config().fingerprint()
    assert replace(first, overlap_tokens=0).fingerprint() != first.fingerprint()


@pytest.mark.parametrize("changes", [
    {"chunk_tokens": 0}, {"overlap_tokens": -1}, {"overlap_tokens": 200},
    {"chunk_tokens": True}, {"top_k": 10}, {"device": "cuda"},
    {"model_revision": "main"}, {"schema_version": 2}, {"model_name": ""},
])
def test_invalid_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(load_config(), **changes)


def test_unknown_configuration_fields_are_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"unexpected": true}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)
