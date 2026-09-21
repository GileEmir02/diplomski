import hashlib

import pytest

from src import provenance


def test_current_code_remains_exact_and_never_accepts_missing_keys():
    current = {"src\\search.py": "search", "src\\evaluation.py": "new"}
    assert provenance.code_provenance_profile(dict(current), current, "dataset") == "current"
    assert provenance.code_provenance_profile({}, current, "dataset") is None


def test_frozen_evaluator_exception_is_exact_and_does_not_ignore_retrieval_changes(tmp_path, monkeypatch):
    snapshot = tmp_path / "evaluation.py"
    snapshot.write_bytes(b"original evaluator")
    checksum = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    monkeypatch.setattr(provenance, "V1_EVALUATOR", snapshot)
    monkeypatch.setattr(provenance, "V1_EVALUATOR_SHA256", checksum)
    recorded = {"src\\evaluation.py": checksum, "src\\search.py": "unchanged"}
    current = {**recorded, "src\\evaluation.py": "extended"}
    dataset = provenance.V1_DATASET_SHA256
    assert provenance.code_provenance_profile(recorded, current, dataset) == "frozen_v1"
    assert provenance.code_provenance_profile(recorded, current, "different dataset") is None
    assert provenance.code_provenance_profile(recorded, {**current, "src\\search.py": "changed"}, dataset) is None
    assert provenance.code_provenance_profile({**recorded, "src\\evaluation.py": "unknown"}, current, dataset) is None
    snapshot.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="snapshot"):
        provenance.code_provenance_profile(recorded, current, dataset)
