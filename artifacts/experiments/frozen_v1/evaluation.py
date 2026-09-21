import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns
from typing import Mapping, Sequence
from uuid import uuid4

import numpy as np

from src.bm25 import BM25_PARAMETERS, build_bm25
from src.config import ROOT, load_config
from src.indexing import PACKAGES, IndexStore, SearchIndex, build_tfidf
from src.search import SEARCH_METHODS, resolve_methods, search


CODE_FILES = (
    "config.py", "ingestion.py", "chunking.py", "indexing.py",
    "model.py", "search.py", "evaluation.py", "bm25.py",
)


@dataclass(frozen=True)
class Query:
    query_id: str
    query: str
    split: str
    query_type: str
    topic: str
    intent_group: str
    relevant_chunk_ids: tuple[str, ...]
    review_status: str


def corpus_fingerprint(index: SearchIndex) -> str:
    value = json.dumps(sorted(chunk.chunk_id for chunk in index.chunks), separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]], relevance: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    if not rankings or set(rankings) != set(relevance):
        raise ValueError("Upiti i oznake relevantnosti moraju biti neprazni i uskladjeni.")
    per_query = []
    for query_id, ranking in rankings.items():
        if not relevance[query_id]:
            raise ValueError("Glavne metrike zahtevaju upite sa poznatim odgovorom.")
        if len(set(ranking)) != len(ranking):
            raise ValueError("Rangiranje sadrzi duple odlomke.")
        relevant = set(relevance[query_id])
        first = next((position for position, chunk_id in enumerate(ranking[:5], start=1)
                      if chunk_id in relevant), None)
        per_query.append({
            "query_id": query_id,
            "first_relevant_rank": first,
            "hit_at_5": int(first is not None),
            "rr_at_5": 0.0 if first is None else 1.0 / first,
        })
    return {
        "query_count": len(per_query),
        "hit_at_5": sum(row["hit_at_5"] for row in per_query) / len(per_query),
        "mrr_at_5": sum(row["rr_at_5"] for row in per_query) / len(per_query),
        "per_query": per_query,
    }


def validate_queries(
    payload: object, index: SearchIndex, *, allow_draft: bool = False,
    allow_ai_reviewed: bool = False,
) -> tuple[Query, ...]:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Nepodrzan evaluacioni format.")
    if (payload.get("corpus_fingerprint") != corpus_fingerprint(index)
            or payload.get("config_fingerprint") != index.config.fingerprint()):
        raise ValueError("Evaluacioni upiti nisu vezani za ovaj korpus i konfiguraciju.")
    known_statuses = {"draft", "reviewed", "ai_reviewed"}
    status = payload.get("annotation_status")
    if not isinstance(status, str) or status not in known_statuses:
        raise ValueError("Nedostaje status anotacija.")
    allowed_statuses = {"reviewed"}
    if allow_ai_reviewed:
        allowed_statuses.add("ai_reviewed")
    if allow_draft:
        allowed_statuses.update(known_statuses)
    if status not in allowed_statuses:
        if status == "ai_reviewed":
            raise ValueError("Oznake je pregledao AI, ne covek. Koristite --allow-ai-reviewed.")
        raise ValueError("Anotacije su nacrt. Potrebna je provera pre zavrsnog merenja.")
    if status == "reviewed" and payload.get("human_review_complete") is False:
        raise ValueError("Status reviewed je u sukobu sa nepotvrdjenim ljudskim pregledom.")
    entries = payload.get("queries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Evaluacioni skup nema upite.")
    chunks = {chunk.chunk_id: chunk for chunk in index.chunks}
    queries = []
    ids = set()
    texts = set()
    groups: dict[str, str] = {}
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("Neispravan zapis upita.")
        for field in ("query_id", "query", "split", "query_type", "topic", "intent_group",
                      "review_status", "answer_summary"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"Upit nema ispravno polje {field}.")
        normalized = " ".join(item["query"].casefold().split())
        if item["query_id"] in ids or normalized in texts:
            raise ValueError("Dupli identifikator ili tekst upita.")
        ids.add(item["query_id"])
        texts.add(normalized)
        split = item["split"]
        if split not in {"dev", "test", "no_answer"}:
            raise ValueError("Nepodrzana podela upita.")
        group = item["intent_group"].casefold().strip()
        if group in groups and groups[group] != split:
            raise ValueError("Ista informaciona potreba prelazi granicu skupova.")
        groups[group] = split
        if item["review_status"] not in known_statuses:
            raise ValueError("Neispravan status provere upita.")
        if item["review_status"] not in allowed_statuses:
            raise ValueError("Nisu svi upiti pregledani.")
        if status == "reviewed" and item["review_status"] != "reviewed":
            raise ValueError("Dataset reviewed ne sme sadrzati neproverene upite.")
        if status == "ai_reviewed" and item["review_status"] == "draft":
            raise ValueError("AI pregled dataset-a nije dovrsen.")
        relevant = item.get("relevant_chunk_ids")
        evidence = item.get("evidence")
        if (not isinstance(relevant, list) or any(not isinstance(value, str) for value in relevant)
                or len(set(relevant)) != len(relevant) or not isinstance(evidence, list)):
            raise ValueError("Neispravne oznake relevantnosti.")
        if any(value not in chunks for value in relevant):
            raise ValueError("Oznaka upucuje na nepostojeci odlomak.")
        if split == "no_answer":
            if relevant or evidence or item["query_type"] != "out_of_scope":
                raise ValueError("Upiti bez odgovora ne ulaze u glavni skup relevantnosti.")
        elif not relevant or item["query_type"] not in {"direct", "synonym", "paraphrase"}:
            raise ValueError("Upit sa odgovorom mora imati relevantan odlomak i tip.")
        supported = set()
        for support in evidence:
            if not isinstance(support, dict) or support.get("chunk_id") not in relevant:
                raise ValueError("Dokaz nije povezan sa relevantnim odlomkom.")
            quote = support.get("quote")
            if not isinstance(quote, str) or not quote.strip():
                raise ValueError("Prazan dokaz relevantnosti.")
            original = " ".join(chunks[support["chunk_id"]].text.split())
            if " ".join(quote.split()) not in original:
                raise ValueError(f"Citat nije pronadjen u odlomku za {item['query_id']}.")
            supported.add(support["chunk_id"])
        if supported != set(relevant):
            raise ValueError("Svaki relevantni odlomak mora imati dokaz.")
        queries.append(Query(item["query_id"], item["query"], split, item["query_type"],
                             item["topic"], group, tuple(relevant), item["review_status"]))
    return tuple(queries)


def load_queries(
    path: Path, index: SearchIndex, *, allow_draft: bool = False,
    allow_ai_reviewed: bool = False,
) -> tuple[Query, ...]:
    return validate_queries(
        json.loads(path.read_text(encoding="utf-8")), index, allow_draft=allow_draft,
        allow_ai_reviewed=allow_ai_reviewed,
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Nema redova za izvestaj.")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_method(args) -> None:
    import psutil

    config = load_config()
    index = IndexStore().load(config)
    if args.method == "bm25" and index.bm25_matrix is None:
        raise ValueError("Indeks nema BM25. Prvo pokrenite src.cli upgrade-bm25.")
    dataset_bytes = None
    annotation_mode = "performance_smoke"
    if args.profile_only:
        selected = [
            Query("smoke-001", "How can regularization reduce overfitting?",
                  "dev", "paraphrase", "regularization", "regularization-purpose", (), "not_applicable"),
            Query("smoke-002", "Why should we use separate data for testing?",
                  "dev", "paraphrase", "evaluation", "held-out-data", (), "not_applicable"),
        ]
    else:
        if args.dataset is None:
            raise ValueError("Navedite evaluacioni skup ili koristite --profile-only.")
        dataset_bytes = args.dataset.read_bytes()
        dataset_payload = json.loads(dataset_bytes.decode("utf-8"))
        queries = validate_queries(dataset_payload, index, allow_draft=args.allow_draft,
                                   allow_ai_reviewed=args.allow_ai_reviewed)
        annotation_mode = {
            "draft": "exploratory_draft",
            "ai_reviewed": "ai_source_reviewed",
            "reviewed": "reviewed",
        }[dataset_payload["annotation_status"]]
        counts = Counter(query.split for query in queries)
        if not args.allow_draft and (counts["dev"] != 10 or counts["test"] != 40):
            raise ValueError("Zavrsni protokol zahteva 10 razvojnih i 40 test upita.")
        selected = [query for query in queries if query.split == args.split]
    if not selected or args.repeats < 1 or args.warmup < 0:
        raise ValueError("Neispravan broj upita, ponavljanja ili zagrevanja.")
    args.output.mkdir(parents=True, exist_ok=False)
    if dataset_bytes is not None:
        (args.output / "dataset.json").write_bytes(dataset_bytes)
    summary = {
        "status": "running", "method": args.method,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "dev_smoke" if args.profile_only else args.split,
        "versions": {package: version(package) for package in (*PACKAGES, "psutil")},
        "code_sha256": {
            str(Path("src") / name): hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
            for name in CODE_FILES
        },
        "annotation_mode": annotation_mode,
        "human_review_complete": annotation_mode == "reviewed",
        "corpus_fingerprint": corpus_fingerprint(index),
        "dataset_sha256": (
            hashlib.sha256(dataset_bytes).hexdigest() if dataset_bytes is not None else None
        ),
        "dataset_snapshot": "dataset.json" if dataset_bytes is not None else None,
        "config": asdict(config), "generation": index.generation,
        "bm25_parameters": dict(BM25_PARAMETERS),
        "query_count": len(selected), "repeats": args.repeats, "warmup": args.warmup,
        "seed": args.seed,
        "memory_protocol": (
            "Fresh process per method. Shared index bundle is loaded for every method; "
            "the neural model is loaded only for semantic search. RSS is sampled at query "
            "boundaries; Windows peak working set, if available, covers process lifetime."
        ),
    }
    _write_json(args.output / "summary.json", summary)
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    summary["hardware"] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "python_version": platform.python_version(),
    }
    encoder = None
    model_loading_ms = 0.0
    if args.method == "semantic":
        started = perf_counter_ns()
        from src.model import SemanticEncoder

        encoder = SemanticEncoder(config)
        model_loading_ms = (perf_counter_ns() - started) / 1e6
    started = perf_counter_ns()
    texts = [chunk.text for chunk in index.chunks]
    if encoder is not None:
        index = replace(index, embeddings=encoder.encode(texts))
    elif args.method == "tfidf":
        vectorizer, matrix = build_tfidf(texts)
        index = replace(index, vectorizer=vectorizer, tfidf_matrix=matrix)
    else:
        vectorizer, matrix = build_bm25(
            texts, index.vectorizer, k1=BM25_PARAMETERS["k1"], b=BM25_PARAMETERS["b"],
        )
        index = replace(index, bm25_vectorizer=vectorizer, bm25_matrix=matrix)
    indexing_ms = (perf_counter_ns() - started) / 1e6
    loaded_rss = process.memory_info().rss
    for number in range(args.warmup):
        search(index, selected[number % len(selected)].query, args.method, encoder)
    timings = []
    rankings = {}
    ranked_rows = []
    for repetition in range(args.repeats):
        order = list(selected)
        random.Random(args.seed + repetition).shuffle(order)
        for query in order:
            started = perf_counter_ns()
            result = search(index, query.query, args.method, encoder)
            elapsed = (perf_counter_ns() - started) / 1e6
            timings.append({"query_id": query.query_id, "repetition": repetition,
                            "elapsed_ms": elapsed, "rss_bytes": process.memory_info().rss})
            if repetition == 0:
                rankings[query.query_id] = [hit.chunk_id for hit in result.hits]
                for hit in result.hits:
                    ranked_rows.append({"query_id": query.query_id, "rank": hit.rank,
                                        "chunk_id": hit.chunk_id, "score": hit.score})
    values = [row["elapsed_ms"] for row in timings]
    memory = process.memory_info()
    summary.update({
        "indexing_representation_ms": indexing_ms,
        "indexing_scope": "Representation build only; extraction, chunking and serialization excluded.",
        "model_loading_ms": model_loading_ms,
        "model_loading_scope": "Library import and model constructor; zero for lexical methods.",
        "latency_median_ms": float(np.median(values)),
        "latency_p95_ms": float(np.percentile(values, 95, method="linear")),
        "timing_samples": len(values), "percentile_method": "linear",
        "baseline_rss_bytes": baseline_rss, "loaded_rss_bytes": loaded_rss,
        "maximum_sampled_rss_bytes": max([baseline_rss, loaded_rss]
                                          + [row["rss_bytes"] for row in timings]),
        "windows_peak_working_set_bytes": getattr(memory, "peak_wset", None),
    })
    if args.profile_only:
        summary["quality_note"] = (
            "Performance smoke on two previously seen development queries. "
            "No relevance metrics and no final-test performance claims."
        )
    elif args.split != "no_answer":
        metrics = evaluate_rankings(rankings, {
            query.query_id: query.relevant_chunk_ids for query in selected
        })
        _write_csv(args.output / "per_query.csv", metrics.pop("per_query"))
        summary.update(metrics)
    else:
        summary["quality_note"] = "No-answer queries are qualitative, not part of Hit@5/MRR@5."
    _write_csv(args.output / "timings.csv", timings)
    _write_csv(args.output / "rankings.csv", ranked_rows)
    summary["status"] = "completed"
    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ponovljiva evaluacija pretrage.")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--method", choices=[*SEARCH_METHODS, "both", "all"], default="all",
                        help="all: tri metode; both: prethodni semanticki/TF-IDF par.")
    parser.add_argument("--split", choices=["dev", "test", "no_answer"], default="test")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-draft", action="store_true",
                        help="Samo istrazivacki rezultati; ne oznacava anotacije kao pregledane.")
    parser.add_argument("--allow-ai-reviewed", action="store_true",
                        help="Dozvoli eksplicitno oznacene AI-pregledane anotacije, bez tvrdnje o ljudskom pregledu.")
    parser.add_argument("--profile-only", action="store_true",
                        help="Pocetna merenja na dva poznata razvojna upita, bez metrika relevantnosti.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = ROOT / "artifacts" / "experiments" / f"{stamp}-{uuid4().hex[:8]}"
    if args.method not in {"both", "all"}:
        run_method(args)
        return
    dataset_bytes = None
    index = IndexStore().load(load_config())
    methods = resolve_methods(args.method)
    if "bm25" in methods and index.bm25_matrix is None:
        raise ValueError("Indeks nema BM25. Prvo pokrenite src.cli upgrade-bm25.")
    if not args.profile_only:
        if args.dataset is None:
            raise ValueError("Navedite evaluacioni skup ili koristite --profile-only.")
        dataset_bytes = args.dataset.read_bytes()
        validate_queries(json.loads(dataset_bytes.decode("utf-8")),
                         index, allow_draft=args.allow_draft,
                         allow_ai_reviewed=args.allow_ai_reviewed)
    args.output.mkdir(parents=True, exist_ok=False)
    if dataset_bytes is not None:
        snapshot = args.output / "dataset.json"
        snapshot.write_bytes(dataset_bytes)
        args.dataset = snapshot
    summaries = []
    for method in (name for name in ("tfidf", "bm25", "semantic") if name in methods):
        command = [
            sys.executable, "-m", "src.evaluation",
            "--method", method, "--split", args.split, "--repeats", str(args.repeats),
            "--warmup", str(args.warmup), "--seed", str(args.seed),
            "--output", str((args.output / method).resolve()),
        ]
        if args.dataset is not None:
            command.extend(["--dataset", str(args.dataset.resolve())])
        if args.profile_only:
            command.append("--profile-only")
        if args.allow_draft:
            command.append("--allow-draft")
        if args.allow_ai_reviewed:
            command.append("--allow-ai-reviewed")
        subprocess.run(command, cwd=ROOT, check=True)
        summaries.append(json.loads((args.output / method / "summary.json").read_text(encoding="utf-8")))
    shared_fields = ("dataset_sha256", "corpus_fingerprint", "config",
                     "code_sha256", "generation", "bm25_parameters")
    if any(summary.get(field) != summaries[0].get(field)
           for summary in summaries[1:] for field in shared_fields):
        raise ValueError("Metode nisu evaluirane sa istim podacima i konfiguracijom.")
    _write_json(args.output / "comparison.json", summaries)


if __name__ == "__main__":
    main()
