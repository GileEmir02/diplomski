import argparse
import csv
import hashlib
import json
from pathlib import Path

from src.config import ROOT, load_config
from src.evaluation import validate_queries
from src.indexing import IndexStore


def spreadsheet_text(value: str) -> str:
    if value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def export_review(dataset: Path, output: Path) -> int:
    index = IndexStore().load(load_config())
    content = dataset.read_bytes()
    payload = json.loads(content.decode("utf-8"))
    queries = validate_queries(payload, index, allow_draft=True)
    dataset_sha256 = hashlib.sha256(content).hexdigest()
    entries = {entry["query_id"]: entry for entry in payload["queries"]}
    chunks = {chunk.chunk_id: chunk for chunk in index.chunks}
    documents = {document.document_id: document for document in index.documents}
    rows = []
    for query in queries:
        entry = entries[query.query_id]
        evidence = []
        for support in entry["evidence"]:
            chunk = chunks[support["chunk_id"]]
            document = documents[chunk.document_id]
            location = "TXT" if chunk.page_start is None else f"PDF page {chunk.page_start}"
            evidence.append(
                f"{chunk.file_name} | {location}\n"
                f"Source: {document.source_url or 'User-provided document'}\n"
                f"License: {document.license_name or 'Not recorded'}\n"
                f"Quote: {support['quote']}"
            )
        rows.append({
            "query_id": query.query_id,
            "dataset_sha256": dataset_sha256,
            "split": query.split,
            "query_type": query.query_type,
            "topic": query.topic,
            "question_en": spreadsheet_text(query.query),
            "proposed_answer_en": spreadsheet_text(entry["answer_summary"]),
            "source_evidence": spreadsheet_text("\n\n".join(evidence)),
            "relevant_chunk_ids": " | ".join(query.relevant_chunk_ids),
            "review_status": query.review_status,
            "ai_review_note": spreadsheet_text(entry.get("review_notes", "")),
            "reviewer_comment": "",
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite a user's review decisions or comments.
    with output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Izvoz pitanja i dokaza za pregled u tabeli.")
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "data" / "evaluation" / "queries_draft.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data" / "evaluation" / "queries_review.csv")
    args = parser.parse_args()
    count = export_review(args.dataset, args.output)
    print(f"Exported {count} rows with their existing review status. No judgments were approved by this export.")
