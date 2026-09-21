import csv
from dataclasses import asdict
import hashlib
import json

import pytest

import scripts.analyze_evaluation as analysis
from src.bm25 import BM25_PARAMETERS
from src.config import load_config
from src.evaluation import corpus_fingerprint
from src.indexing import IndexStore
from src.ingestion import DocumentInput
from tests.helpers import FakeEncoder


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def three_method_run(tmp_path, monkeypatch):
    config = load_config()
    store = IndexStore(tmp_path / "index")
    store.add([DocumentInput(f"doc{number}.txt", f"Example number {number} regularization.".encode())
               for number in range(5)], FakeEncoder(config))
    index = store.load(config)
    ids = [chunk.chunk_id for chunk in index.chunks]
    queries = [{
        "query_id": f"test-{number:03}", "query": f"Question {number}", "split": "test",
        "query_type": "direct", "intent_group": f"need-{number}",
        "relevant_chunk_ids": [ids[0]], "answer_summary": "A test answer.",
    } for number in range(40)]
    queries.extend({"query_id": f"none-{number}", "query": f"Outside {number}", "split": "no_answer"}
                   for number in range(5))
    data = json.dumps({"queries": queries}).encode()
    checksum = hashlib.sha256(data).hexdigest()
    run, no_answer = tmp_path / "run", tmp_path / "no_answer"
    run.mkdir()
    no_answer.mkdir()
    (run / "dataset.json").write_bytes(data)
    summaries = []
    for method, first_rank in (("tfidf", 1), ("bm25", 2), ("semantic", 3)):
        folder, outside = run / method, no_answer / method
        folder.mkdir()
        outside.mkdir()
        (folder / "dataset.json").write_bytes(data)
        (outside / "dataset.json").write_bytes(data)
        summary = {
            "method": method, "status": "completed", "dataset_sha256": checksum,
            "generation": index.generation, "config": asdict(config),
            "corpus_fingerprint": corpus_fingerprint(index),
            "annotation_mode": "ai_source_reviewed", "human_review_complete": False,
            "code_sha256": {"same": "version"}, "bm25_parameters": dict(BM25_PARAMETERS),
            "hit_at_5": 1, "mrr_at_5": 1 / first_rank, "repeats": 2,
            "latency_median_ms": 1, "latency_p95_ms": 1, "query_count": 40,
        }
        summaries.append(summary)
        (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        outside_summary = {key: value for key, value in summary.items()
                           if key not in {"hit_at_5", "mrr_at_5"}}
        outside_summary["query_count"] = 5
        (outside / "summary.json").write_text(json.dumps(outside_summary), encoding="utf-8")
        order = ids[1:].copy()
        order.insert(first_rank - 1, ids[0])
        rows = [{"query_id": query["query_id"], "rank": rank,
                 "chunk_id": chunk_id, "score": 1 / rank}
                for query in queries if query["split"] == "test"
                for rank, chunk_id in enumerate(order, 1)]
        write_csv(folder / "rankings.csv", rows)
        write_csv(folder / "timings.csv", [
            {"query_id": query["query_id"], "repetition": repeat, "elapsed_ms": 1}
            for query in queries if query["split"] == "test" for repeat in range(2)
        ])
        write_csv(outside / "rankings.csv", [
            {"query_id": query["query_id"], "rank": rank, "chunk_id": chunk_id, "score": 1 / rank}
            for query in queries if query["split"] == "no_answer"
            for rank, chunk_id in enumerate(order, 1)
        ])
    (run / "comparison.json").write_text(json.dumps(summaries), encoding="utf-8")
    monkeypatch.setattr(analysis, "IndexStore", lambda: store)
    return run, no_answer


def test_all_three_methods_are_recomputed_and_disk_is_not_double_counted(three_method_run):
    result = analysis.analyze(*three_method_run)
    assert set(result["method_summaries"]) == {"tfidf", "bm25", "semantic"}
    assert result["method_summaries"]["bm25"]["mrr_at_5"] == 0.5
    assert len(result["pairwise"]["tfidf_vs_bm25"]["left_better"]) == 40
    assert len(result["no_answer_examples"]) == 15
    sizes = result["disk_bytes"]
    assert sizes["bm25_vectorizer_and_matrix"] > 0
    assert sum(value for key, value in sizes.items() if key != "complete_index_bundle") == sizes["complete_index_bundle"]


def test_analysis_rejects_wrong_recorded_bm25_metric(three_method_run):
    run, outside = three_method_run
    path = run / "bm25" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["mrr_at_5"] = 0.9
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="raw rankings"):
        analysis.analyze(run, outside)
