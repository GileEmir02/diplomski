"""Prepare a blinded review packet and freeze explicitly adjudicated labels."""

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from src.config import ROOT, load_config
from src.evaluation import CODE_FILES, resolve_review_source, validate_expanded_protocol, validate_queries
from src.indexing import IndexStore
from src.search import search


QUESTION_FIELDS = (
    "query_id", "query", "split", "query_type", "topic", "intent_group",
    "answer_summary", "source_evidence",
)
CANDIDATE_FIELDS = ("query_id", "query", "chunk_id", "file_name", "page", "text")
REVIEW_FIELDS = ("decision", "evidence_quote", "reason", "reviewer")
QUERY_REVIEW_FIELDS = ("decision", "reason", "reviewer")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True)
        stream.write("\n")


def csv_value(value: object) -> str:
    text = str(value)
    # Source passages must remain data, not spreadsheet formulas.
    return "'" + text if text.startswith(("=", "+", "-", "@", "\t", "\r")) else text


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: csv_value(row.get(key, "")) for key in fields} for row in rows)


def read_csv(path: Path, fields: tuple[str, ...], *, allow_legacy_reviewer: bool = False) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(fields) and not (
            allow_legacy_reviewer and reader.fieldnames == [*fields, "reviewer_kind"]
        ):
            raise ValueError(f"CSV columns were changed or the delimiter is not a comma: {path.name}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"CSV has an incomplete row: {path.name}")
    return rows


def question_fields(item: dict, chunks: dict) -> dict:
    evidence = []
    for support in item["evidence"]:
        chunk = chunks[support["chunk_id"]]
        evidence.append(f"{chunk.file_name}, page {chunk.page_start}: {support['quote']}")
    return {key: item[key] for key in QUESTION_FIELDS if key != "source_evidence"} | {
        "source_evidence": "\n".join(evidence) or "No answer proposed in this corpus."
    }


def candidate_fields(item: dict, chunk) -> dict:
    return {
        "query_id": item["query_id"], "query": item["query"], "chunk_id": chunk.chunk_id,
        "file_name": chunk.file_name, "page": chunk.page_start if chunk.page_start is not None else "",
        "text": chunk.text,
    }


def retrieval_hashes() -> dict[str, str]:
    return {name: digest(ROOT / "src" / name) for name in CODE_FILES if name != "evaluation.py"}


def merge_questions(base: dict, additions: dict, protocol: dict, protocol_sha256: str) -> dict:
    if (additions.get("annotation_status") != "draft"
            or additions.get("human_review_complete") is not False
            or additions.get("parent_dataset_sha256") != protocol["parent_dataset_sha256"]):
        raise ValueError("Additional questions must be unapproved proposals tied to the original dataset.")
    new = additions.get("queries")
    if not isinstance(new, list) or len(new) != protocol["new_test_questions"]:
        raise ValueError("The number of proposed new questions differs from the protocol.")
    original_test_count = sum(item["split"] == "test" for item in base["queries"])
    expected_ids = {f"test-{number:03d}" for number in
                    range(original_test_count + 1, original_test_count + len(new) + 1)}
    if ({item.get("query_id") for item in new} != expected_ids
            or any(item.get("split") != "test" or item.get("review_status") != "draft" for item in new)):
        raise ValueError("New test IDs or review statuses do not match the declared extension.")
    old_groups = {item["intent_group"].strip().casefold() for item in base["queries"]}
    new_groups = [item["intent_group"].strip().casefold() for item in new]
    if len(set(new_groups)) != len(new_groups) or set(new_groups) & old_groups:
        raise ValueError("New questions must add distinct information needs, not reuse old groups.")
    result = deepcopy(base)
    parent_review = {key: result.pop(key) for key in ("reviewed_at_utc", "reviewed_from_sha256") if key in result}
    result.update({
        "dataset_version": 2,
        "annotation_status": "draft",
        "annotation_method": "Original labels plus source-grounded assistant proposals; pending query and pooled-candidate review.",
        "human_review_complete": False,
        "parent_dataset_sha256": protocol["parent_dataset_sha256"],
        "parent_review_metadata": parent_review,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "label_review": {"query_review_complete": False, "pool_review_complete": False},
        "queries": [
            {**deepcopy(item), "cohort": cohort}
            for items, cohort in ((base["queries"], "original"), (new, "new")) for item in items
        ],
    })
    return result


def prepare(base_path: Path, additions_path: Path, protocol_path: Path, output: Path,
            *, index=None, encoder=None) -> dict:
    if output.exists():
        raise FileExistsError("Use a new review packet directory; do not overwrite reviews.")
    base_bytes = base_path.read_bytes()
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if hashlib.sha256(base_bytes).hexdigest() != protocol["parent_dataset_sha256"]:
        raise ValueError("The original dataset changed.")
    payload = merge_questions(json.loads(base_bytes), json.loads(additions_path.read_bytes()),
                              protocol, hashlib.sha256(protocol_bytes).hexdigest())
    if index is None:
        index = IndexStore().load(load_config())
    queries = validate_queries(payload, index, allow_draft=True, allow_ai_reviewed=True)
    validate_expanded_protocol(protocol, payload, queries, index, payload["protocol_sha256"])
    if encoder is None:
        from src.model import SemanticEncoder

        encoder = SemanticEncoder(index.config)
    chunks = {chunk.chunk_id: chunk for chunk in index.chunks}
    candidates, questions, pool, method_results = [], [], {}, {}
    for item in payload["queries"]:
        query_id = item["query_id"]
        questions.append(question_fields(item, chunks))
        chosen = set(item["relevant_chunk_ids"])
        method_results[query_id] = {}
        for method in protocol["methods"]:
            response = search(index, item["query"], method, encoder if method == "semantic" else None)
            if len(response.hits) != 5:
                raise ValueError("The review protocol requires five candidates per method.")
            chosen.update(hit.chunk_id for hit in response.hits)
            method_results[query_id][method] = [
                {"chunk_id": hit.chunk_id, "rank": hit.rank, "score": hit.score} for hit in response.hits
            ]
        order = sorted(chosen)
        random.Random(f"{protocol['timing']['seed']}:{query_id}").shuffle(order)
        pool[query_id] = order
        candidates.extend(candidate_fields(item, chunks[chunk_id]) for chunk_id in order)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "queries_draft.json", payload)
    with (output / "protocol.json").open("xb") as stream:
        stream.write(protocol_bytes)
    write_csv(output / "questions.csv", questions, QUESTION_FIELDS + QUERY_REVIEW_FIELDS)
    write_csv(output / "judgments.csv", candidates, CANDIDATE_FIELDS + REVIEW_FIELDS)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "awaiting_review",
        "review_csv_schema_version": 2,
        "generation": index.generation,
        "protocol_sha256": digest(output / "protocol.json"),
        "draft_sha256": digest(output / "queries_draft.json"),
        "additions_sha256": digest(additions_path),
        "parent_dataset_sha256": protocol["parent_dataset_sha256"],
        "retrieval_code_sha256": retrieval_hashes(),
        "preparation_script_sha256": digest(Path(__file__)),
        "query_count": len(queries),
        "split_counts": dict(Counter(query.split for query in queries)),
        "test_types": dict(Counter(query.query_type for query in queries if query.split == "test")),
        "distinct_test_intents": len({query.intent_group for query in queries if query.split == "test"}),
        "candidate_pairs": len(candidates),
        "pool": pool,
        "method_results": method_results,
        "blinding_note": "judgments.csv omits methods, ranks, scores and suggested relevance labels. questions.csv contains proposed source evidence for separate question verification. This audit manifest is unblinded.",
        "no_aggregate_quality_metrics_computed": True,
    }
    write_json(output / "pool_manifest.json", manifest)
    return manifest


def checked_row(row: dict, expected: dict, fields: tuple[str, ...]) -> None:
    if any(row.get(key) != csv_value(expected[key]) for key in fields):
        raise ValueError("A protected query/source cell changed. Revise the draft and generate a new pool.")


def reviewer(row: dict, review_source: str | None = None) -> dict:
    identity = row["reviewer"].strip()
    reason = row["reason"].strip()
    if not identity or not reason:
        raise ValueError("Every decision requires a reviewer and reason.")
    source = resolve_review_source(row, review_source)
    result = {"reviewer": identity, "reason": reason}
    if review_source is None:
        result["reviewer_kind"] = source
    return result


def verified_quote(value: str, source: str) -> str:
    quote = value.strip()
    original = " ".join(source.split())
    if quote and " ".join(quote.split()) in original:
        return quote
    if quote.startswith(("'=", "'+", "'-", "'@", "'\t", "'\r")):
        decoded = quote[1:]
        if " ".join(decoded.split()) in original:
            return decoded
    raise ValueError("A positive candidate needs an exact source quote.")


def freeze(directory: Path, output: Path, *, review_source: str | None = None, index=None) -> dict:
    if output.exists():
        raise FileExistsError("Refusing to overwrite a frozen dataset.")
    if review_source is not None:
        resolve_review_source({}, review_source)
    manifest = json.loads((directory / "pool_manifest.json").read_bytes())
    if (digest(directory / "queries_draft.json") != manifest["draft_sha256"]
            or digest(directory / "protocol.json") != manifest["protocol_sha256"]
            or retrieval_hashes() != manifest["retrieval_code_sha256"]):
        raise ValueError("Draft, protocol or retrieval code changed after pool preparation.")
    payload = json.loads((directory / "queries_draft.json").read_bytes())
    protocol = json.loads((directory / "protocol.json").read_bytes())
    if index is None:
        index = IndexStore().load(load_config())
    if index.generation != manifest["generation"]:
        raise ValueError("The active index changed after pool preparation.")
    chunks = {chunk.chunk_id: chunk for chunk in index.chunks}
    entries = {item["query_id"]: item for item in payload["queries"]}
    question_rows = read_csv(directory / "questions.csv", QUESTION_FIELDS + QUERY_REVIEW_FIELDS,
                             allow_legacy_reviewer=True)
    decisions = read_csv(directory / "judgments.csv", CANDIDATE_FIELDS + REVIEW_FIELDS,
                         allow_legacy_reviewer=True)
    if len(question_rows) != len(entries) or {row["query_id"] for row in question_rows} != set(entries):
        raise ValueError("The query review is incomplete or has duplicate rows.")
    for row in question_rows:
        item = entries[row["query_id"]]
        checked_row(row, question_fields(item, chunks), QUESTION_FIELDS)
        if row["decision"].strip() != "accepted":
            raise ValueError(f"Question {item['query_id']} is not accepted; review or revise it first.")
        item["query_review"] = {"decision": "accepted", **reviewer(row, review_source)}
        item["judgments"] = []
        item["relevant_chunk_ids"] = []
        item["evidence"] = []
    expected_pairs = {(query_id, chunk_id) for query_id, ids in manifest["pool"].items() for chunk_id in ids}
    received_pairs = [(row["query_id"], row["chunk_id"]) for row in decisions]
    if len(received_pairs) != len(set(received_pairs)) or set(received_pairs) != expected_pairs:
        raise ValueError("The candidate review has missing, duplicate or extra rows.")
    for row in decisions:
        item = entries[row["query_id"]]
        chunk = chunks[row["chunk_id"]]
        checked_row(row, candidate_fields(item, chunk), CANDIDATE_FIELDS)
        decision = row["decision"].strip()
        if decision not in {"relevant", "not_relevant"}:
            raise ValueError(f"Unresolved candidate for {item['query_id']}; blanks and uncertain are not negatives.")
        judgment = {"chunk_id": chunk.chunk_id, "decision": decision, **reviewer(row, review_source)}
        item["judgments"].append(judgment)
        if decision == "relevant":
            quote = verified_quote(row["evidence_quote"], chunk.text)
            item["relevant_chunk_ids"].append(chunk.chunk_id)
            item["evidence"].append({"chunk_id": chunk.chunk_id, "quote": quote})
    all_human = True
    identities = set()
    for item in payload["queries"]:
        human = (resolve_review_source(item["query_review"], review_source) == "human"
                 and all(resolve_review_source(row, review_source) == "human" for row in item["judgments"]))
        identities.add(item["query_review"]["reviewer"])
        identities.update(row["reviewer"] for row in item["judgments"])
        item["review_status"] = "reviewed" if human else "ai_reviewed"
        item["review_notes"] = "Explicit query and pooled-candidate decisions are recorded in this version."
        all_human &= human
    payload["annotation_status"] = "reviewed" if all_human else "ai_reviewed"
    payload["human_review_complete"] = all_human
    if review_source is not None:
        payload["review_provenance"] = {"source": review_source, "reviewers": sorted(identities)}
        payload["annotation_method"] = (
            "Explicit source-content review of questions and pooled candidates; "
            "review source recorded once in review_provenance, with per-decision identities and reasons."
        )
    else:
        payload["annotation_method"] = (
            "Explicit review of questions and pooled candidates; legacy per-decision reviewer metadata retained."
        )
    payload["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["label_review"] = {
        "query_review_complete": True, "pool_review_complete": True,
        "pool_manifest_sha256": digest(directory / "pool_manifest.json"),
        "questions_csv_sha256": digest(directory / "questions.csv"),
        "judgments_csv_sha256": digest(directory / "judgments.csv"),
        "candidate_pairs": len(decisions),
    }
    queries = validate_queries(payload, index, allow_ai_reviewed=True)
    validate_expanded_protocol(protocol, payload, queries, index, manifest["protocol_sha256"])
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    return {"queries": len(queries), "sha256": digest(output),
            "annotation_status": payload["annotation_status"], "human_review_complete": all_human}


def status(directory: Path) -> dict:
    questions = read_csv(directory / "questions.csv", QUESTION_FIELDS + QUERY_REVIEW_FIELDS,
                         allow_legacy_reviewer=True)
    judgments = read_csv(directory / "judgments.csv", CANDIDATE_FIELDS + REVIEW_FIELDS,
                         allow_legacy_reviewer=True)
    return {
        "questions": len(questions),
        "question_decisions": dict(Counter(row["decision"].strip() or "pending" for row in questions)),
        "candidate_pairs": len(judgments),
        "candidate_decisions": dict(Counter(row["decision"].strip() or "pending" for row in judgments)),
        "note": "Progress only. The freeze command validates completeness, evidence and provenance.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare")
    create.add_argument("--base", type=Path, default=ROOT / "data" / "evaluation" / "queries_v1.json")
    create.add_argument("--additions", type=Path, required=True)
    create.add_argument("--protocol", type=Path, default=ROOT / "config" / "evaluation_v2.json")
    create.add_argument("--output", type=Path, required=True)
    for command in ("status", "freeze"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--directory", type=Path, required=True)
        if command == "freeze":
            subparser.add_argument("--output", type=Path, required=True)
            subparser.add_argument("--review-source", choices=["human", "assistant"],
                                   help="Poreklo pregleda navodi se jednom, ne kao kolona u svakom CSV redu.")
    args = parser.parse_args()
    if args.command == "prepare":
        report = prepare(args.base, args.additions, args.protocol, args.output)
        result = {key: report[key] for key in
                  ("status", "query_count", "split_counts", "test_types", "distinct_test_intents", "candidate_pairs")}
    elif args.command == "freeze":
        result = freeze(args.directory, args.output, review_source=args.review_source)
    else:
        result = status(args.directory)
    print(json.dumps(result, indent=2, ensure_ascii=True))
