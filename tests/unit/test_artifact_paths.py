from pathlib import Path

import pytest

from scripts.artifact_paths import PATH_RENAMES, relocated_hashes, relocated_path


@pytest.mark.parametrize(("old", "new"), PATH_RENAMES.items())
def test_recorded_path_resolves_to_neutral_location(tmp_path, old, new):
    assert relocated_path(old, root=tmp_path) == tmp_path / new
    assert relocated_path(tmp_path / old, root=tmp_path) == tmp_path / new


def test_experiment_children_are_relocated_but_similar_names_are_not(tmp_path):
    old = Path(r"artifacts\experiments\test_bm25_ai_reviewed_v1")
    new = Path(r"artifacts\experiments\test_three_methods_v1")
    child = Path(r"bm25\rankings.csv")
    assert relocated_path(old / child, root=tmp_path) == tmp_path / new / child
    unrelated = Path(str(old) + "_another_run") / child
    assert relocated_path(unrelated, root=tmp_path) == tmp_path / unrelated


def test_current_and_external_paths_are_unchanged(tmp_path):
    current = tmp_path / r"data\evaluation\queries_v1.json"
    external = tmp_path.parent / "separate-experiment" / "analysis.json"
    assert relocated_path(current, root=tmp_path) == current
    assert relocated_path(external, root=tmp_path) == external


def test_hash_keys_move_without_changing_checksums_or_original_record(tmp_path):
    old = r"artifacts\experiments\test_bm25_ai_reviewed_v1\analysis.json"
    new = r"artifacts\experiments\test_three_methods_v1\analysis.json"
    recorded = {old: "a" * 64, r"config\bm25_experiment.json": "b" * 64}
    normalized = relocated_hashes(recorded, root=tmp_path)
    assert normalized == {new: "a" * 64, r"config\bm25_experiment.json": "b" * 64}
    assert recorded[old] == "a" * 64
    assert new not in recorded


def test_ambiguous_recorded_hash_paths_are_rejected(tmp_path):
    recorded = {
        r"data\evaluation\queries_ai_reviewed.json": "a" * 64,
        r"data\evaluation\queries_v1.json": "a" * 64,
    }
    with pytest.raises(ValueError, match="Duplicate evidence path"):
        relocated_hashes(recorded, root=tmp_path)
