import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.evaluation as evaluation
from src.config import load_config
from src.evaluation import corpus_fingerprint, evaluate_rankings, load_queries
from src.indexing import IndexStore
from src.ingestion import DocumentInput
from tests.helpers import FakeEncoder
from src.bm25 import BM25_PARAMETERS


def test_hit_and_mrr_exact_rank_example():
    rankings = {
        "q1": ["a"],
        "q2": ["x", "a"],
        "q3": ["x1", "x2", "x3", "x4", "a"],
        "q4": ["x1", "x2", "x3", "x4", "x5", "a"],
        "q5": ["x"],
    }
    result = evaluate_rankings(rankings, {query: ["a"] for query in rankings})
    assert result["hit_at_5"] == pytest.approx(0.60)
    assert result["mrr_at_5"] == pytest.approx(0.34)
    assert result["precision_at_5"] == pytest.approx(0.12)
    assert [row["first_relevant_rank"] for row in result["per_query"]] == [1, 2, 5, None, None]


def test_first_of_multiple_relevant_results_is_used():
    result = evaluate_rankings({"q": ["x", "a", "b"]}, {"q": ["a", "b"]})
    assert result["mrr_at_5"] == 0.5
    assert result["hit_at_5"] == 1
    assert result["precision_at_5"] == pytest.approx(0.4)


def test_precision_distinguishes_lists_with_the_same_hit_and_first_rank():
    relevant = {"q": ["a", "b", "c", "d"]}
    one = evaluate_rankings({"q": ["a", "x", "y", "z", "w"]}, relevant)
    four = evaluate_rankings({"q": ["a", "b", "c", "d", "x"]}, relevant)
    assert one["hit_at_5"] == four["hit_at_5"] == 1
    assert one["mrr_at_5"] == four["mrr_at_5"] == 1
    assert one["precision_at_5"] == 0.2
    assert four["precision_at_5"] == 0.8
    assert four["per_query"][0]["relevant_count_at_5"] == 4


def test_precision_uses_fixed_five_cutoff_macro_average_and_zero_misses():
    result = evaluate_rankings(
        {"short": ["a"], "empty": [], "late": ["x", "x2", "x3", "x4", "x5", "a"]},
        {"short": ["a"], "empty": ["a"], "late": ["a"]},
    )
    assert [row["precision_at_5"] for row in result["per_query"]] == [0.2, 0, 0]
    assert result["precision_at_5"] == pytest.approx(1 / 15)


def test_precision_can_be_larger_than_mrr_without_being_invalid():
    result = evaluate_rankings({"q": ["x", "a", "b", "c", "d"]}, {"q": ["a", "b", "c", "d"]})
    assert result["precision_at_5"] == 0.8
    assert result["mrr_at_5"] == 0.5


@pytest.mark.parametrize("rankings,relevance", [
    ({}, {}), ({"q": ["a"]}, {}), ({"q": ["a"]}, {"q": []}),
    ({"q": ["a", "a"]}, {"q": ["a"]}),
])
def test_invalid_metric_inputs_are_rejected(rankings, relevance):
    with pytest.raises(ValueError):
        evaluate_rankings(rankings, relevance)


@pytest.fixture
def dataset(tmp_path):
    config = load_config()
    store = IndexStore(tmp_path / "index")
    store.add([DocumentInput("notes.txt", b"Regularization reduces overfitting.")], FakeEncoder(config))
    index = store.load(config)
    chunk_id = index.chunks[0].chunk_id
    payload = {
        "schema_version": 1,
        "annotation_status": "draft",
        "corpus_fingerprint": corpus_fingerprint(index),
        "config_fingerprint": config.fingerprint(),
        "queries": [{
            "query_id": "dev-001", "query": "Why use regularization?", "split": "dev",
            "query_type": "direct", "topic": "regularization", "intent_group": "regularization-purpose",
            "relevant_chunk_ids": [chunk_id], "answer_summary": "It reduces overfitting.",
            "evidence": [{"chunk_id": chunk_id, "quote": "Regularization reduces overfitting."}],
            "review_status": "draft",
        }],
    }
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return index, payload, path


def test_draft_requires_explicit_exploratory_mode(dataset):
    index, _, path = dataset
    with pytest.raises(ValueError, match="nacrt"):
        load_queries(path, index)
    assert len(load_queries(path, index, allow_draft=True)) == 1


def test_reviewed_dataset_can_be_loaded(dataset):
    index, payload, path = dataset
    payload["annotation_status"] = "reviewed"
    payload["queries"][0]["review_status"] = "reviewed"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert len(load_queries(path, index)) == 1


def test_ai_review_requires_explicit_permission_and_is_not_human_review(dataset):
    index, payload, path = dataset
    payload["annotation_status"] = "ai_reviewed"
    payload["human_review_complete"] = False
    payload["queries"][0]["review_status"] = "ai_reviewed"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="AI"):
        load_queries(path, index)
    result = load_queries(path, index, allow_ai_reviewed=True)
    assert result[0].review_status == "ai_reviewed"
    payload["annotation_status"] = "reviewed"
    payload["queries"][0]["review_status"] = "reviewed"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="ljudskim"):
        load_queries(path, index, allow_ai_reviewed=True)


def test_ai_permission_does_not_approve_a_draft(dataset):
    index, _, path = dataset
    with pytest.raises(ValueError, match="nacrt"):
        load_queries(path, index, allow_ai_reviewed=True)


def test_ai_reviewed_root_cannot_hide_unreviewed_questions(dataset):
    index, payload, path = dataset
    payload["annotation_status"] = "ai_reviewed"
    payload["human_review_complete"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="nije dovrsen"):
        load_queries(path, index, allow_draft=True)


def test_bm25_worker_rebuilds_and_scores_without_a_neural_model(dataset, tmp_path, monkeypatch):
    index, _, path = dataset
    monkeypatch.setattr(evaluation, "IndexStore", lambda: SimpleNamespace(load=lambda config: index))
    monkeypatch.setattr(evaluation, "load_config", lambda: index.config)
    args = SimpleNamespace(
        method="bm25", dataset=path, allow_draft=True, allow_ai_reviewed=False,
        profile_only=False, split="dev", repeats=2, warmup=1, seed=42,
        output=tmp_path / "bm25_run",
    )
    evaluation.run_method(args)
    summary = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    assert summary["method"] == "bm25" and summary["status"] == "completed"
    assert summary["hit_at_5"] == summary["mrr_at_5"] == 1
    assert summary["precision_at_5"] == 0.2
    assert summary["precision_label_coverage"] == "not_adjudicated"
    assert summary["model_loading_ms"] == 0
    assert summary["timing_samples"] == 2
    assert summary["bm25_parameters"] == BM25_PARAMETERS
    assert (args.output / "dataset.json").read_bytes() == path.read_bytes()


@pytest.mark.parametrize("change", ["fingerprint", "quote", "missing_evidence", "unknown_chunk", "duplicate_query", "leaked_intent"])
def test_dataset_consistency_checks(dataset, change):
    index, payload, path = dataset
    if change == "fingerprint":
        payload["corpus_fingerprint"] = "wrong"
    elif change == "quote":
        payload["queries"][0]["evidence"][0]["quote"] = "Not present in the source."
    elif change == "missing_evidence":
        payload["queries"][0]["evidence"] = []
    elif change == "unknown_chunk":
        payload["queries"][0]["relevant_chunk_ids"] = ["unknown"]
    elif change == "duplicate_query":
        payload["queries"].append(dict(payload["queries"][0]))
    else:
        other = dict(payload["queries"][0])
        other.update(query_id="test-001", query="How can regularization help?", split="test")
        payload["queries"].append(other)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_queries(path, index, allow_draft=True)


@pytest.mark.parametrize("tamper_summary", [False, True])
@pytest.mark.parametrize("selection,workers", [("both", 2), ("all", 3)])
def test_both_methods_use_one_immutable_dataset_snapshot(
    dataset, tmp_path, monkeypatch, tamper_summary, selection, workers,
):
    index, _, original = dataset
    original_bytes = original.read_bytes()
    output = tmp_path / "run"
    monkeypatch.setattr(evaluation, "IndexStore", lambda: SimpleNamespace(load=lambda config: index))
    monkeypatch.setattr(evaluation, "load_config", lambda: index.config)
    monkeypatch.setattr(evaluation.sys, "argv", [
        "evaluation", "--dataset", str(original), "--allow-draft",
        "--method", selection, "--output", str(output),
    ])
    seen = []

    def fake_worker(command, *, cwd, check):
        assert check is True
        snapshot = Path(command[command.index("--dataset") + 1])
        assert snapshot == (output / "dataset.json").resolve()
        content = snapshot.read_bytes()
        seen.append(content)
        original.write_text("changed during the run", encoding="utf-8")
        folder = Path(command[command.index("--output") + 1])
        folder.mkdir()
        checksum = hashlib.sha256(content).hexdigest()
        if tamper_summary and len(seen) == workers:
            checksum = "different"
        (folder / "summary.json").write_text(json.dumps({
            "dataset_sha256": checksum,
            "corpus_fingerprint": corpus_fingerprint(index),
            "config": asdict(index.config),
            "code_sha256": {"evaluation.py": "same-code-for-both-workers"},
            "generation": index.generation,
            "bm25_parameters": dict(BM25_PARAMETERS),
        }), encoding="utf-8")

    monkeypatch.setattr(evaluation.subprocess, "run", fake_worker)
    if tamper_summary:
        with pytest.raises(ValueError, match="istim podacima"):
            evaluation.main()
        assert not (output / "comparison.json").exists()
    else:
        evaluation.main()
        assert (output / "comparison.json").is_file()
    assert seen == [original_bytes] * workers
