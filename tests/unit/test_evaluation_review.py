from copy import deepcopy
from dataclasses import asdict
import csv
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import evaluation_review as review
from src import evaluation
from src.config import load_config
from src.evaluation import corpus_fingerprint, require_judged_results
from src.indexing import IndexStore
from src.ingestion import DocumentInput
from tests.helpers import FakeEncoder


def save(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def packet_inputs(tmp_path):
    config = load_config()
    encoder = FakeEncoder(config)
    store = IndexStore(tmp_path / "index")
    store.add([DocumentInput(f"doc{n}.txt", f"Example {n}. Regularization controls overfitting.".encode())
               for n in range(5)], encoder)
    index = store.load(config)
    chunk = index.chunks[0]

    def question(identity, split="test"):
        return {
            "query_id": identity, "query": f"What is described by {identity}?",
            "split": split, "query_type": "out_of_scope" if split == "no_answer" else "direct",
            "topic": "test", "intent_group": identity, "review_status": "draft",
            "answer_summary": "No answer." if split == "no_answer" else "Regularization controls overfitting.",
            "relevant_chunk_ids": [] if split == "no_answer" else [chunk.chunk_id],
            "evidence": [] if split == "no_answer" else [{"chunk_id": chunk.chunk_id, "quote": chunk.text}],
        }

    base = {
        "schema_version": 1, "annotation_status": "draft", "human_review_complete": False,
        "corpus_fingerprint": corpus_fingerprint(index), "config_fingerprint": config.fingerprint(),
        "queries": [question("dev-001", "dev"), question("test-001"), question("none-001", "no_answer")],
    }
    base_path, additional_path, protocol_path = (tmp_path / name for name in
                                                 ("base.json", "new.json", "protocol.json"))
    save(base_path, base)
    parent_sha = review.digest(base_path)
    additions = {
        "schema_version": 1, "annotation_status": "draft", "human_review_complete": False,
        "parent_dataset_sha256": parent_sha,
        "queries": [question("test-002"), question("test-003")],
    }
    save(additional_path, additions)
    protocol = json.loads((evaluation.ROOT / "config" / "evaluation_v2.json").read_bytes())
    protocol.update(
        parent_dataset_sha256=parent_sha, corpus_fingerprint=corpus_fingerprint(index),
        config_fingerprint=config.fingerprint(), expected_splits={"dev": 1, "test": 3, "no_answer": 1},
        new_test_questions=2, timing={"repeats": 1, "warmup": 0, "seed": 42},
    )
    save(protocol_path, protocol)
    return SimpleNamespace(
        index=index, encoder=encoder, base=base_path, additions=additional_path, protocol=protocol_path,
        output=tmp_path / "packet", positive=chunk.chunk_id,
    )


def prepare(inputs):
    return review.prepare(inputs.base, inputs.additions, inputs.protocol, inputs.output,
                          index=inputs.index, encoder=inputs.encoder)


def complete_rows(inputs):
    for name, fields in (
        ("questions.csv", review.QUESTION_FIELDS + review.QUERY_REVIEW_FIELDS),
        ("judgments.csv", review.CANDIDATE_FIELDS + review.REVIEW_FIELDS),
    ):
        path = inputs.output / name
        rows = review.read_csv(path, fields)
        for row in rows:
            row.update(reviewer="Test reviewer", reason="Explicit fixture decision.")
            if name == "questions.csv":
                row["decision"] = "accepted"
            else:
                positive = row["query_id"] != "none-001" and row["chunk_id"] == inputs.positive
                row["decision"] = "relevant" if positive else "not_relevant"
                row["evidence_quote"] = "Regularization controls overfitting." if positive else ""
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def test_prepare_keeps_original_bytes_and_creates_a_blind_unapproved_packet(packet_inputs):
    p = packet_inputs
    original = p.base.read_bytes()
    report = prepare(p)
    assert p.base.read_bytes() == original
    assert report["split_counts"] == {"dev": 1, "test": 3, "no_answer": 1}
    assert report["candidate_pairs"] == 25
    assert report["no_aggregate_quality_metrics_computed"] is True
    rows = review.read_csv(p.output / "judgments.csv", review.CANDIDATE_FIELDS + review.REVIEW_FIELDS)
    assert all(row["decision"] == row["reviewer"] == "" for row in rows)
    assert not {"method", "rank", "score", "suggested_label"} & set(rows[0])
    assert "reviewer_kind" not in rows[0]
    assert report["review_csv_schema_version"] == 2
    assert review.status(p.output)["candidate_decisions"] == {"pending": 25}
    with pytest.raises(FileExistsError):
        prepare(p)


def test_freeze_rejects_pending_review_without_writing_a_final_dataset(packet_inputs, tmp_path):
    prepare(packet_inputs)
    output = tmp_path / "not_created.json"
    with pytest.raises(ValueError, match="not accepted"):
        review.freeze(packet_inputs.output, output, index=packet_inputs.index)
    assert not output.exists()


@pytest.mark.parametrize("kind,human,status", [("human", True, "reviewed"), ("assistant", False, "ai_reviewed")])
def test_freeze_records_review_source_once_and_explicit_negative_judgments(packet_inputs, tmp_path, kind, human, status):
    p = packet_inputs
    prepare(p)
    complete_rows(p)
    output = tmp_path / "reviewed.json"
    result = review.freeze(p.output, output, review_source=kind, index=p.index)
    assert result["human_review_complete"] is human
    assert result["annotation_status"] == status
    payload = json.loads(output.read_bytes())
    assert payload["review_provenance"] == {"source": kind, "reviewers": ["Test reviewer"]}
    assert all("reviewer_kind" not in item["query_review"] for item in payload["queries"])
    assert all("reviewer_kind" not in row for item in payload["queries"] for row in item["judgments"])
    assert all(len(item["judgments"]) == 5 for item in payload["queries"])
    assert all(len(item["relevant_chunk_ids"]) == (0 if item["split"] == "no_answer" else 1)
               for item in payload["queries"])
    with pytest.raises(FileExistsError):
        review.freeze(p.output, output, review_source=kind, index=p.index)


@pytest.mark.parametrize("change,match", [
    ("uncertain", "Unresolved"),
    ("blank", "Unresolved"),
    ("missing_row", "missing"),
    ("duplicate", "duplicate"),
    ("changed_source", "protected"),
    ("bad_quote", "exact source"),
])
def test_review_cannot_silently_promote_missing_or_changed_candidates(packet_inputs, tmp_path, change, match):
    p = packet_inputs
    prepare(p)
    complete_rows(p)
    path = p.output / "judgments.csv"
    fields = review.CANDIDATE_FIELDS + review.REVIEW_FIELDS
    rows = review.read_csv(path, fields)
    if change == "uncertain":
        rows[0]["decision"] = "uncertain"
    elif change == "blank":
        rows[0]["decision"] = ""
    elif change == "missing_row":
        rows.pop()
    elif change == "duplicate":
        rows.append(dict(rows[0]))
    elif change == "changed_source":
        rows[0]["text"] = "Changed source."
    else:
        next(row for row in rows if row["decision"] == "relevant")["evidence_quote"] = "Not in the source."
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match=match):
        review.freeze(p.output, tmp_path / "must_not_exist.json", review_source="human", index=p.index)
    assert not (tmp_path / "must_not_exist.json").exists()


def test_query_needs_must_not_reuse_an_old_group(packet_inputs):
    p = packet_inputs
    payload = json.loads(p.additions.read_bytes())
    payload["queries"][0]["intent_group"] = "dev-001"
    save(p.additions, payload)
    with pytest.raises(ValueError, match="information needs"):
        prepare(p)
    assert not p.output.exists()


def test_unjudged_results_are_not_counted_as_negative():
    payload = {"queries": [{"query_id": "q", "judgments": [
        {"chunk_id": "a", "decision": "relevant"},
        {"chunk_id": "b", "decision": "not_relevant"},
    ]}]}
    require_judged_results(payload, {"q": ["a", "b"]})
    with pytest.raises(ValueError, match="nepregledan"):
        require_judged_results(payload, {"q": ["a", "unknown"]})


def test_expanded_worker_exports_all_three_metrics_and_protocol_snapshot(packet_inputs, tmp_path, monkeypatch):
    p = packet_inputs
    prepare(p)
    complete_rows(p)
    dataset = tmp_path / "reviewed.json"
    review.freeze(p.output, dataset, review_source="human", index=p.index)
    monkeypatch.setattr(evaluation, "IndexStore", lambda: SimpleNamespace(load=lambda config: p.index))
    args = SimpleNamespace(
        method="bm25", dataset=dataset, protocol=p.protocol,
        allow_draft=False, allow_ai_reviewed=False, profile_only=False,
        split="test", repeats=1, warmup=0, seed=42, output=tmp_path / "results",
    )
    evaluation.run_method(args)
    summary = json.loads((args.output / "summary.json").read_bytes())
    assert summary["query_count"] == 3
    assert summary["precision_at_5"] == pytest.approx(0.2)
    assert summary["precision_label_coverage"] == "pooled_top5"
    assert summary["human_review_complete"] is True
    assert (args.output / "protocol.json").read_bytes() == p.protocol.read_bytes()
    with (args.output / "per_query.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert all(float(row["precision_at_5"]) == 0.2 and int(row["relevant_count_at_5"]) == 1 for row in rows)


def test_spreadsheet_source_formula_is_escaped():
    assert review.csv_value("=1+1") == "'=1+1"
    assert review.csv_value("Ordinary prose.") == "Ordinary prose."
    assert review.verified_quote("'=x-y", "The example is =x-y") == "=x-y"
    assert review.verified_quote("'-2", "A signed value is -2.") == "-2"
    with pytest.raises(ValueError, match="exact source"):
        review.verified_quote("'=invented", "No such formula.")


def test_new_csv_never_infers_human_review_from_reviewer_identity(packet_inputs, tmp_path):
    p = packet_inputs
    prepare(p)
    complete_rows(p)
    output = tmp_path / "not_created.json"
    with pytest.raises(ValueError, match="review-source"):
        review.freeze(p.output, output, index=p.index)
    assert not output.exists()


def test_legacy_csv_still_loads_without_rewriting_its_reviewer_metadata(packet_inputs, tmp_path):
    p = packet_inputs
    prepare(p)
    complete_rows(p)
    for name, fields in (
        ("questions.csv", review.QUESTION_FIELDS + review.QUERY_REVIEW_FIELDS),
        ("judgments.csv", review.CANDIDATE_FIELDS + review.REVIEW_FIELDS),
    ):
        path = p.output / name
        rows = review.read_csv(path, fields)
        for row in rows:
            row["reviewer_kind"] = "assistant"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*fields, "reviewer_kind"])
            writer.writeheader()
            writer.writerows(rows)
    assert review.status(p.output)["question_decisions"] == {"accepted": 5}
    output = tmp_path / "legacy_reviewed.json"
    result = review.freeze(p.output, output, index=p.index)
    assert result["human_review_complete"] is False
    assert "review_provenance" not in json.loads(output.read_bytes())
    with pytest.raises(ValueError, match="conflicts"):
        review.freeze(p.output, tmp_path / "wrong.json", review_source="human", index=p.index)


@pytest.mark.parametrize("corruption", ["missing", "unknown_source", "wrong_reviewer", "conflicting_row"])
def test_packet_level_provenance_is_required_and_cannot_be_overridden(packet_inputs, tmp_path, corruption):
    p = packet_inputs
    prepare(p)
    complete_rows(p)
    output = tmp_path / "assistant_review.json"
    review.freeze(p.output, output, review_source="assistant", index=p.index)
    payload = json.loads(output.read_bytes())
    queries = evaluation.validate_queries(payload, p.index, allow_ai_reviewed=True)
    protocol = json.loads(p.protocol.read_bytes())
    if corruption == "missing":
        del payload["review_provenance"]
    elif corruption == "unknown_source":
        payload["review_provenance"]["source"] = "automatic_human"
    elif corruption == "wrong_reviewer":
        payload["review_provenance"]["reviewers"] = ["Another person"]
    else:
        payload["queries"][0]["query_review"]["reviewer_kind"] = "human"
    with pytest.raises(ValueError):
        evaluation.validate_expanded_protocol(protocol, payload, queries, p.index, payload["protocol_sha256"])


@pytest.mark.parametrize("field,value", [
    ("expected_splits", {"dev": 1, "test": 100, "no_answer": 1}),
    ("k", 10),
    ("precision_denominator", 3),
    ("corpus_fingerprint", "different"),
    ("metrics", ["hit_at_5", "mrr_at_5"]),
])
def test_expanded_protocol_cannot_be_silently_changed(packet_inputs, field, value):
    p = packet_inputs
    prepare(p)
    payload = json.loads((p.output / "queries_draft.json").read_bytes())
    queries = evaluation.validate_queries(payload, p.index, allow_draft=True)
    protocol = json.loads(p.protocol.read_bytes())
    protocol[field] = value
    with pytest.raises(ValueError):
        evaluation.validate_expanded_protocol(protocol, payload, queries, p.index, payload["protocol_sha256"])


def test_final_protocol_refuses_unfinished_pool_or_false_human_status(packet_inputs, tmp_path):
    p = packet_inputs
    prepare(p)
    complete_rows(p)
    output = tmp_path / "reviewed.json"
    review.freeze(p.output, output, review_source="assistant", index=p.index)
    payload = json.loads(output.read_bytes())
    protocol = json.loads(p.protocol.read_bytes())
    queries = evaluation.validate_queries(payload, p.index, allow_ai_reviewed=True)
    incomplete = deepcopy(payload)
    incomplete["label_review"]["pool_review_complete"] = False
    with pytest.raises(ValueError, match="pregled"):
        evaluation.validate_expanded_protocol(protocol, incomplete, queries, p.index, payload["protocol_sha256"])
    payload["human_review_complete"] = True
    with pytest.raises(ValueError, match="procenjivac"):
        evaluation.validate_expanded_protocol(protocol, payload, queries, p.index, payload["protocol_sha256"])


def test_expanded_runs_and_analysis_preserve_cohorts_precision_and_no_answer_separation(packet_inputs, tmp_path, monkeypatch):
    from scripts import analyze_evaluation

    p = packet_inputs
    prepare(p)
    complete_rows(p)
    dataset = tmp_path / "reviewed.json"
    review.freeze(p.output, dataset, review_source="human", index=p.index)
    store = SimpleNamespace(root=tmp_path / "index", load=lambda config: p.index)
    monkeypatch.setattr(evaluation, "IndexStore", lambda: store)
    monkeypatch.setattr(analyze_evaluation, "IndexStore", lambda: store)
    monkeypatch.setitem(sys.modules, "src.model", SimpleNamespace(SemanticEncoder=FakeEncoder))
    run, outside = tmp_path / "run", tmp_path / "outside"
    for directory, split in ((run, "test"), (outside, "no_answer")):
        directory.mkdir()
        (directory / "dataset.json").write_bytes(dataset.read_bytes())
        (directory / "protocol.json").write_bytes(p.protocol.read_bytes())
        summaries = []
        for method in ("tfidf", "bm25", "semantic"):
            args = SimpleNamespace(
                method=method, dataset=dataset, protocol=p.protocol,
                allow_draft=False, allow_ai_reviewed=False, profile_only=False,
                split=split, repeats=1, warmup=0, seed=42, output=directory / method,
            )
            evaluation.run_method(args)
            summary = json.loads((args.output / "summary.json").read_bytes())
            summaries.append(summary)
            if split == "no_answer":
                assert not set(evaluation.QUALITY_METRICS) & set(summary)
        save(directory / "comparison.json", summaries)
    result = analyze_evaluation.analyze(run, outside)
    assert result["test_queries"] == 3
    assert result["human_review_complete"] is True
    assert result["precision_label_coverage"] == "pooled_top5"
    assert result["quality_by_cohort"]["original"]["bm25"]["query_count"] == 1
    assert result["quality_by_cohort"]["new"]["bm25"]["query_count"] == 2
    assert all(summary["precision_at_5"] == pytest.approx(0.2)
               for summary in result["method_summaries"].values())
    per_query = run / "bm25" / "per_query.csv"
    original_export = per_query.read_bytes()
    with per_query.open(encoding="utf-8", newline="") as stream:
        exported = list(csv.DictReader(stream))
    exported[0]["precision_at_5"] = "0.8"
    with per_query.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    with pytest.raises(ValueError, match="Per-query metric"):
        analyze_evaluation.analyze(run, outside)
    per_query.write_bytes(original_export)
    path = run / "bm25" / "summary.json"
    summary = json.loads(path.read_bytes())
    summary["precision_at_5"] = 0.8
    save(path, summary)
    with pytest.raises(ValueError, match="raw rankings"):
        analyze_evaluation.analyze(run, outside)


def test_all_workers_receive_the_same_frozen_protocol_even_if_original_changes(packet_inputs, tmp_path, monkeypatch):
    from pathlib import Path

    p = packet_inputs
    prepare(p)
    complete_rows(p)
    dataset = tmp_path / "reviewed.json"
    review.freeze(p.output, dataset, review_source="human", index=p.index)
    original_protocol = p.protocol.read_bytes()
    output = tmp_path / "frozen_run"
    monkeypatch.setattr(evaluation, "IndexStore", lambda: SimpleNamespace(load=lambda config: p.index))
    monkeypatch.setattr(evaluation.sys, "argv", [
        "evaluation", "--dataset", str(dataset), "--protocol", str(p.protocol),
        "--method", "all", "--repeats", "1", "--warmup", "0", "--seed", "42", "--output", str(output),
    ])
    seen = []

    def worker(command, *, cwd, check):
        assert check is True
        protocol = Path(command[command.index("--protocol") + 1])
        snapshot = Path(command[command.index("--dataset") + 1])
        assert protocol == (output / "protocol.json").resolve()
        seen.append(protocol.read_bytes())
        p.protocol.write_text("changed original", encoding="utf-8")
        folder = Path(command[command.index("--output") + 1])
        folder.mkdir()
        save(folder / "summary.json", {
            "dataset_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            "corpus_fingerprint": corpus_fingerprint(p.index), "config": asdict(p.index.config),
            "code_sha256": {"same": "version"}, "generation": p.index.generation,
            "protocol_id": "ml-basics-evaluation-v2", "protocol_sha256": hashlib.sha256(seen[-1]).hexdigest(),
            "metrics_schema_version": 2, "precision_label_coverage": "pooled_top5",
        })

    monkeypatch.setattr(evaluation.subprocess, "run", worker)
    evaluation.main()
    assert seen == [original_protocol] * 3
    assert (output / "protocol.json").read_bytes() == original_protocol
