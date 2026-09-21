import argparse
import csv
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from src.config import load_config
from src.evaluation import (
    QUALITY_METRICS, corpus_fingerprint, evaluate_rankings, require_judged_results,
    validate_expanded_protocol, validate_queries,
)
from src.indexing import IndexStore
from src.search import SEARCH_METHODS


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def analyze(run: Path, no_answer: Path) -> dict[str, object]:
    store = IndexStore()
    index = store.load(load_config())
    chunks = {chunk.chunk_id: chunk for chunk in index.chunks}
    data = (run / "dataset.json").read_bytes()
    checksum = hashlib.sha256(data).hexdigest()
    dataset = json.loads(data)
    protocol_path = run / "protocol.json"
    protocol = None
    protocol_sha256 = None
    if protocol_path.exists():
        protocol_bytes = protocol_path.read_bytes()
        protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
        protocol = json.loads(protocol_bytes)
        checked_queries = validate_queries(dataset, index, allow_draft=True, allow_ai_reviewed=True)
        validate_expanded_protocol(protocol, dataset, checked_queries, index, protocol_sha256)
    elif dataset.get("protocol_id"):
        raise ValueError("Expanded analysis requires its frozen protocol snapshot.")
    queries = {item["query_id"]: item for item in dataset["queries"] if item["split"] == "test"}
    expected_test = protocol["expected_splits"]["test"] if protocol else 40
    expected_no_answer = protocol["expected_splits"]["no_answer"] if protocol else 5
    if len(queries) != expected_test:
        raise ValueError("Test query count differs from the declared protocol.")
    rankings = {}
    metrics = {}
    summaries = {}
    per_type = {}
    comparison = json.loads((run / "comparison.json").read_text(encoding="utf-8"))
    if (not isinstance(comparison, list) or not comparison
            or any(not isinstance(row, dict) or row.get("method") not in SEARCH_METHODS for row in comparison)):
        raise ValueError("Invalid comparison method list.")
    methods = tuple(row["method"] for row in comparison)
    if len(set(methods)) != len(methods) or not {"tfidf", "semantic"}.issubset(methods):
        raise ValueError("Comparison needs distinct methods including the original pair.")
    if protocol and set(methods) != set(protocol["methods"]):
        raise ValueError("The expanded protocol requires all three methods.")
    baseline = comparison[0]
    annotation_mode = baseline.get("annotation_mode")
    human_review = baseline.get("human_review_complete")
    if (not isinstance(annotation_mode, str)
            or annotation_mode not in {"reviewed", "ai_source_reviewed", "exploratory_draft"}
            or human_review is not (annotation_mode == "reviewed")):
        raise ValueError("Invalid annotation provenance.")
    has_precision = baseline.get("metrics_schema_version", 1) == 2
    if (type(baseline.get("metrics_schema_version", 1)) is not int
            or baseline.get("metrics_schema_version", 1) not in {1, 2}
            or (protocol and not has_precision)):
        raise ValueError("Expanded analysis requires the three-metric schema.")
    metric_names = QUALITY_METRICS if has_precision else QUALITY_METRICS[:2]
    for method in methods:
        folder = run / method
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if (summary.get("status") != "completed" or summary.get("method") != method
                or summary["generation"] != index.generation
                or summary["dataset_sha256"] != checksum
                or (folder / "dataset.json").read_bytes() != data
                or summary["corpus_fingerprint"] != corpus_fingerprint(index)
                or summary["annotation_mode"] != annotation_mode
                or summary["human_review_complete"] is not human_review
                or summary["query_count"] != expected_test):
            raise ValueError("Evaluation provenance does not match.")
        if protocol and (summary.get("protocol_id") != protocol["protocol_id"]
                         or summary.get("protocol_sha256") != protocol_sha256
                         or (folder / "protocol.json").read_bytes() != protocol_bytes):
            raise ValueError("Method protocol differs from the frozen snapshot.")
        if protocol and summary.get("precision_label_coverage") != (
            "not_adjudicated" if annotation_mode == "exploratory_draft" else "pooled_top5"
        ):
            raise ValueError("Precision annotation coverage differs from the review status.")
        rows = read_csv(folder / "rankings.csv")
        if len(rows) != expected_test * 5 or {row["query_id"] for row in rows} != set(queries):
            raise ValueError("Unexpected test ranking rows.")
        method_rankings = {}
        for query_id in queries:
            hits = sorted((row for row in rows if row["query_id"] == query_id),
                          key=lambda row: int(row["rank"]))
            if [int(hit["rank"]) for hit in hits] != [1, 2, 3, 4, 5]:
                raise ValueError("Invalid rank sequence.")
            if any(hit["chunk_id"] not in chunks for hit in hits):
                raise ValueError("Ranking references an unknown chunk.")
            method_rankings[query_id] = [hit["chunk_id"] for hit in hits]
        if protocol and annotation_mode != "exploratory_draft":
            require_judged_results(dataset, method_rankings)
        result = evaluate_rankings(method_rankings, {
            query_id: item["relevant_chunk_ids"] for query_id, item in queries.items()
        })
        for metric in metric_names:
            value = summary.get(metric)
            if (type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1
                    or not np.isclose(result[metric], value, rtol=0, atol=1e-12)):
                raise ValueError("Stored metric differs from raw rankings.")
        if has_precision:
            exported = read_csv(folder / "per_query.csv")
            by_id = {row["query_id"]: row for row in exported}
            if len(exported) != len(queries) or set(by_id) != set(queries):
                raise ValueError("Per-query export has missing or duplicate queries.")
            for expected in result["per_query"]:
                actual = by_id[expected["query_id"]]
                if actual.get("first_relevant_rank") != (
                    str(expected["first_relevant_rank"]) if expected["first_relevant_rank"] else ""
                ):
                    raise ValueError("Per-query rank differs from raw rankings.")
                for field in ("hit_at_5", "rr_at_5", "relevant_count_at_5", "precision_at_5"):
                    if field not in actual or not np.isclose(
                        float(actual[field]), expected[field], rtol=0, atol=1e-12
                    ):
                        raise ValueError("Per-query metric differs from raw rankings.")
        if not has_precision:
            result.pop("precision_at_5")
            for row in result["per_query"]:
                row.pop("precision_at_5")
                row.pop("relevant_count_at_5")
        times = read_csv(folder / "timings.csv")
        counts = Counter(row["query_id"] for row in times)
        if counts != {query_id: summary["repeats"] for query_id in queries}:
            raise ValueError("Timing repetitions are inconsistent.")
        values = np.array([float(row["elapsed_ms"]) for row in times])
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Invalid timing sample.")
        if not np.isclose(np.median(values), summary["latency_median_ms"]):
            raise ValueError("Median cannot be reproduced.")
        if not np.isclose(np.percentile(values, 95, method="linear"), summary["latency_p95_ms"]):
            raise ValueError("P95 cannot be reproduced.")
        rankings[method] = method_rankings
        metrics[method] = {row["query_id"]: row for row in result["per_query"]}
        summaries[method] = summary
        per_type[method] = {}
        for query_type in sorted({item["query_type"] for item in queries.values()}):
            ids = [query_id for query_id, item in queries.items() if item["query_type"] == query_type]
            typed = evaluate_rankings(
                {query_id: method_rankings[query_id] for query_id in ids},
                {query_id: queries[query_id]["relevant_chunk_ids"] for query_id in ids},
            )
            del typed["per_query"]
            if not has_precision:
                typed.pop("precision_at_5")
            per_type[method][query_type] = typed
    shared = ("code_sha256", "generation", "config", "bm25_parameters", "metrics_schema_version",
              "protocol_id", "protocol_sha256", "precision_label_coverage")
    if any(summaries[method].get(key) != summaries[methods[0]].get(key)
           for method in methods for key in shared):
        raise ValueError("Methods used different code versions or parameters.")
    semantic_better = []
    tfidf_better = []
    equal = []
    both_miss = []
    for query_id in queries:
        left = metrics["tfidf"][query_id]["rr_at_5"]
        right = metrics["semantic"][query_id]["rr_at_5"]
        if right > left:
            semantic_better.append(query_id)
        elif left > right:
            tfidf_better.append(query_id)
        else:
            equal.append(query_id)
        if left == right == 0:
            both_miss.append(query_id)
    pairwise = {}
    for left_method, right_method in combinations(methods, 2):
        left_better, right_better, tied, shared_misses = [], [], [], []
        for query_id in queries:
            left = metrics[left_method][query_id]["rr_at_5"]
            right = metrics[right_method][query_id]["rr_at_5"]
            (left_better if left > right else right_better if right > left else tied).append(query_id)
            if left == right == 0:
                shared_misses.append(query_id)
        pairwise[f"{left_method}_vs_{right_method}"] = {
            "left": left_method, "right": right_method,
            "left_better": left_better, "right_better": right_better,
            "equal_rr": tied, "both_miss": shared_misses,
        }
    examples = {}
    chosen = sorted(set(semantic_better[:3] + tfidf_better[:3] + both_miss))
    for query_id in chosen:
        item = queries[query_id]
        examples[query_id] = {
            "query": item["query"],
            "answer_summary": item["answer_summary"],
            "methods": {
                method: {
                    **metrics[method][query_id],
                    "top_file": chunks[rankings[method][query_id][0]].file_name,
                    "top_page": chunks[rankings[method][query_id][0]].page_start,
                    "top_text": chunks[rankings[method][query_id][0]].text,
                } for method in methods
            },
        }
    no_answer_queries = {item["query_id"]: item for item in dataset["queries"]
                         if item["split"] == "no_answer"}
    if len(no_answer_queries) != expected_no_answer:
        raise ValueError("No-answer query count differs from the protocol.")
    no_answer_examples = []
    for method in methods:
        folder = no_answer / method
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if (summary.get("status") != "completed" or summary.get("method") != method
                or summary["dataset_sha256"] != checksum or summary["query_count"] != expected_no_answer
                or (folder / "dataset.json").read_bytes() != data
                or any(summary.get(key) != summaries[method].get(key)
                       for key in (*shared, "corpus_fingerprint", "annotation_mode", "human_review_complete"))
                or any(metric in summary for metric in QUALITY_METRICS)):
            raise ValueError("No-answer queries were mixed into quality metrics.")
        if protocol and (folder / "protocol.json").read_bytes() != protocol_bytes:
            raise ValueError("No-answer protocol snapshot differs from the test run.")
        rows = read_csv(folder / "rankings.csv")
        if len(rows) != expected_no_answer * 5 or {row["query_id"] for row in rows} != set(no_answer_queries):
            raise ValueError("Invalid no-answer rankings.")
        if protocol and annotation_mode != "exploratory_draft":
            require_judged_results(dataset, {
                query_id: [row["chunk_id"] for row in sorted(
                    (r for r in rows if r["query_id"] == query_id), key=lambda r: int(r["rank"])
                )] for query_id in no_answer_queries
            })
        for row in rows:
            if int(row["rank"]) == 1:
                chunk = chunks[row["chunk_id"]]
                no_answer_examples.append({
                    "query_id": row["query_id"],
                    "query": no_answer_queries[row["query_id"]]["query"],
                    "method": method, "score": float(row["score"]),
                    "file_name": chunk.file_name,
                    "text_preview": chunk.text[:300],
                })
    directory = store.root / index.generation
    semantic_bytes = (directory / "embeddings.npy").stat().st_size
    tfidf_bytes = sum((directory / name).stat().st_size for name in ("tfidf.npz", "vectorizer.pkl"))
    bm25_bytes = sum((directory / name).stat().st_size
                    for name in ("bm25.npz", "bm25_vectorizer.pkl")) if index.bm25_matrix is not None else 0
    all_bytes = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    cohorts = {}
    if protocol:
        for cohort in ("original", "new"):
            ids = [qid for qid, item in queries.items() if item.get("cohort") == cohort]
            if not ids:
                raise ValueError("Expanded analysis requires original/new question cohorts.")
            cohorts[cohort] = {
                method: {metric: value for metric, value in evaluate_rankings(
                    {qid: rankings[method][qid] for qid in ids},
                    {qid: queries[qid]["relevant_chunk_ids"] for qid in ids},
                ).items() if metric != "per_query"} for method in methods
            }
    return {
        "annotation_mode": annotation_mode,
        "human_review_complete": human_review,
        "metrics_schema_version": 2 if has_precision else 1,
        "protocol_id": protocol["protocol_id"] if protocol else "legacy-v1",
        "protocol_sha256": protocol_sha256,
        "protocol": protocol,
        "precision_label_coverage": baseline.get("precision_label_coverage", "not_adjudicated"),
        "dataset_sha256": checksum,
        "test_queries": len(queries),
        "distinct_test_intent_groups": len({item["intent_group"] for item in queries.values()}),
        "metrics_recomputed_from_raw_rankings": True,
        "timing_aggregates_recomputed": True,
        "method_summaries": summaries,
        "quality_by_query_type": per_type,
        "quality_by_cohort": cohorts,
        "semantic_better": semantic_better,
        "tfidf_better": tfidf_better,
        "equal_rr": equal,
        "both_miss": both_miss,
        "pairwise": pairwise,
        "examples": examples,
        "no_answer_examples": no_answer_examples,
        "disk_bytes": {
            "semantic_matrix": semantic_bytes,
            "tfidf_vectorizer_and_matrix": tfidf_bytes,
            "bm25_vectorizer_and_matrix": bm25_bytes,
            "shared_sources_metadata_and_chunks": all_bytes - semantic_bytes - tfidf_bytes - bm25_bytes,
            "complete_index_bundle": all_bytes,
        },
        "limitations": [
            f"Small {len(index.documents)}-document corpus and {len(queries)} test formulations.",
            f"Annotation mode: {annotation_mode}; human_review_complete={human_review}.",
            "Pooled review, when present, covers fixed methods' candidates, not every possible relevant chunk.",
            "Precision without pooled candidate adjudication is exploratory and must not be presented as validated.",
            "Some questions share information needs and cannot be treated as fully independent observations.",
            "Original questions can have re-adjudicated labels in v2; cross-version quality differences are not purely retrieval improvements.",
            "PDF glyph errors and chunk boundaries can affect both retrieval and judgments.",
            "The BM25 extension reuses an already-used test collection; BM25 parameters were fixed before its test results.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provera i analiza sacuvanih test rezultata.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--no-answer", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run, args.no_answer)
    output = args.run / "analysis.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
    print(json.dumps({
        "verified": True,
        "semantic_better": len(result["semantic_better"]),
        "tfidf_better": len(result["tfidf_better"]),
        "equal": len(result["equal_rr"]),
        "both_miss": result["both_miss"],
        "disk_bytes": result["disk_bytes"],
        "output": str(output),
    }))
