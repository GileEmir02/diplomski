"""Verify an untouched release directory or ZIP without extracting or loading models."""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "release_manifest.json"
CORPUS_MANIFEST = "data/full_corpus_manifest.csv"
OMISSIONS = [
    "Secrets, .env files, credentials and editor-local settings",
    "Virtual environments, model weights/cache, indexes and serialized pickles",
    "Personal thesis files, Word templates, presentations and thesis tooling/tests",
    "Source HTML snapshots, HTML preparation scripts/tests and processed corpus copies",
    "Historical preparation logs/reports with machine-local paths and draft/v1 experiment runs",
]
STATIC_FILES = set("""
.gitignore .python-version .streamlit/config.toml
README.md pyproject.toml uv.lock requirements-runtime.txt requirements-test.txt
app.py assets/app.css
config/search.json config/model_provenance.json config/bm25_experiment.json config/evaluation_v2.json
src/__init__.py src/bm25.py src/chunking.py src/cli.py src/config.py src/evaluation.py
src/indexing.py src/ingestion.py src/model.py src/presentation.py src/provenance.py src/search.py
scripts/__init__.py scripts/analyze_evaluation.py scripts/artifact_paths.py
scripts/check_environment.py scripts/evaluation_review.py scripts/export_evaluation_review.py
scripts/apply_relevance_review.py scripts/build_release.py scripts/verify_release.py
tests/__init__.py tests/helpers.py
tests/unit/test_analysis.py tests/unit/test_app.py tests/unit/test_artifact_paths.py
tests/unit/test_bm25.py tests/unit/test_chunking.py tests/unit/test_config.py
tests/unit/test_evaluation.py tests/unit/test_evaluation_review.py
tests/unit/test_indexing_search.py tests/unit/test_ingestion.py tests/unit/test_manifest_inputs.py
tests/unit/test_model.py tests/unit/test_presentation.py tests/unit/test_provenance.py
tests/unit/test_review_export.py tests/unit/test_release.py
data/corpus_manifest.csv data/full_corpus_manifest.csv data/LICENSES.txt data/full_corpus/NOTICE.txt
data/evaluation/queries_v1.json data/evaluation/queries_v2.json
data/evaluation/queries_draft.json data/evaluation/new_questions_v2_draft.json
data/evaluation/review_audit_v1.json
docs/DATASET.md docs/PUBLISHING.md docs/error_cases_v2.json
docs/images/comparison.png docs/images/metrics_v2.png docs/images/source_result.png
artifacts/experiments/frozen_v1/evaluation.py
""".split())
for name in ("judgments.csv", "pool_manifest.json", "protocol.json", "queries_draft.json", "questions.csv"):
    STATIC_FILES.add(f"data/evaluation/v2_review_01/{name}")
    STATIC_FILES.add(f"artifacts/evaluation_review/v2_assistant/input_snapshot/{name}")
for name in (
    "adjudication_collected", "adjudication_resolved", "batches_manifest",
    "csv_schema_migration", "decisions_applied", "parent_comparison", "resolutions",
):
    STATIC_FILES.add(f"artifacts/evaluation_review/v2_assistant/{name}.json")
for number in range(1, 5):
    for kind in ("input", "decisions"):
        STATIC_FILES.add(f"artifacts/evaluation_review/v2_assistant/batch_{number:02}_{kind}.json")
for run in ("test_three_methods_v2", "no_answer_three_methods_v2"):
    prefix = f"artifacts/experiments/{run}"
    for name in ("comparison.json", "dataset.json", "protocol.json"):
        STATIC_FILES.add(f"{prefix}/{name}")
    for method in ("tfidf", "bm25", "semantic"):
        names = ["dataset.json", "protocol.json", "rankings.csv", "summary.json", "timings.csv"]
        if run == "test_three_methods_v2":
            names.append("per_query.csv")
        STATIC_FILES.update(f"{prefix}/{method}/{name}" for name in names)
STATIC_FILES.add("artifacts/experiments/test_three_methods_v2/analysis.json")

DATASET_HASHES = {
    "data/evaluation/queries_v1.json": "699a9b4a79461b280d391a735e74b2f5ba09e58f3661ddf3cce39be1d40c15b2",
    "data/evaluation/queries_v2.json": "5c312d014a988e1396518d95546ca486e919d9a4ddc3b4b4150b9a24bbd243bf",
}


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def safe_name(name: str) -> str:
    """Require one portable, unambiguous relative name (ZIP uses slash separators)."""
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"Unsafe package path: {name!r}")
    for part in name.split("/"):
        if (not part or part in (".", "..") or part.endswith((".", " "))
                or re.search(r'[\x00-\x1f<>:"|?*]', part)
                or part.split(".")[0].upper() in {
                    "CON", "PRN", "AUX", "NUL",
                    *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10)),
                }):
            raise ValueError(f"Unsafe package path: {name!r}")
    return name


def validate_version(version: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", version) or version.endswith("."):
        raise ValueError("Version must be a simple nonempty filename component.")
    safe_name(version)
    return version


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(content: bytes):
    return json.loads(content, object_pairs_hook=unique_object)


def corpus_entries(content: bytes) -> dict[str, dict]:
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    if len(rows) != 20:
        raise ValueError("Frozen corpus must contain exactly 20 documents.")
    entries = {}
    for row in rows:
        name = safe_name(row["relative_path"].replace("\\", "/"))
        if (name.rsplit("/", 1)[0] not in ("data/raw", "data/full_corpus/raw")
                or Path(name).suffix.lower() not in (".pdf", ".txt")
                or name.split("/")[-1] != row["file_name"]
                or name.casefold() in {key.casefold() for key in entries}):
            raise ValueError(f"Invalid or duplicate corpus path: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError(f"Invalid corpus checksum: {name}")
        entries[name] = row
    if sum(name.endswith(".pdf") for name in entries) != 3:
        raise ValueError("Frozen corpus must contain three PDF and seventeen TXT files.")
    return entries


def source_file(root: Path, name: str) -> Path:
    path = root
    for part in safe_name(name).split("/"):
        path = path / part
        if path.is_symlink() or path.is_junction():
            raise ValueError(f"Links are not allowed: {name}")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Missing or external file: {name}")
    return path


def allowed_files(read) -> set[str]:
    return STATIC_FILES | set(corpus_entries(read(CORPUS_MANIFEST)))


def verify_evidence(read) -> None:
    for name, expected in DATASET_HASHES.items():
        if sha256(read(name)) != expected:
            raise ValueError(f"Canonical dataset changed: {name}")
    for name, row in corpus_entries(read(CORPUS_MANIFEST)).items():
        content = read(name)
        if sha256(content) != row["sha256"] or len(content) != int(row["file_size_bytes"]):
            raise ValueError(f"Corpus bytes changed: {name}")
    dataset = read("data/evaluation/queries_v2.json")
    protocol = read("config/evaluation_v2.json")
    labels = load_json(dataset)["label_review"]
    for filename, key in (
        ("pool_manifest.json", "pool_manifest_sha256"),
        ("questions.csv", "questions_csv_sha256"), ("judgments.csv", "judgments_csv_sha256"),
    ):
        if sha256(read(f"data/evaluation/v2_review_01/{filename}")) != labels[key]:
            raise ValueError(f"Review evidence changed: {filename}")
    for run in ("test_three_methods_v2", "no_answer_three_methods_v2"):
        prefix = f"artifacts/experiments/{run}"
        for folder in (prefix, *(f"{prefix}/{method}" for method in ("tfidf", "bm25", "semantic"))):
            if read(f"{folder}/dataset.json") != dataset or read(f"{folder}/protocol.json") != protocol:
                raise ValueError(f"Frozen snapshot mismatch: {folder}")
        for method in ("tfidf", "bm25", "semantic"):
            summary = load_json(read(f"{prefix}/{method}/summary.json"))
            if (summary["dataset_sha256"] != sha256(dataset)
                    or summary["protocol_sha256"] != sha256(protocol)):
                raise ValueError(f"Frozen summary mismatch: {prefix}/{method}")
            for name, checksum in summary["code_sha256"].items():
                if sha256(read(safe_name(name.replace("\\", "/")))) != checksum:
                    raise ValueError(f"Frozen retrieval/evaluation code changed: {name}")


def verify_contents(names: list[str], read) -> dict:
    normalized = [safe_name(name) for name in names]
    if len({name.casefold() for name in normalized}) != len(normalized):
        raise ValueError("Duplicate or case-colliding package paths.")
    if MANIFEST not in names:
        raise ValueError(f"Missing {MANIFEST}; verify a built release, not the development folder.")
    manifest = load_json(read(MANIFEST))
    if manifest.get("schema_version") != 1 or manifest.get("omissions") != OMISSIONS:
        raise ValueError("Unsupported manifest or changed omission policy.")
    validate_version(manifest["version"])
    expected = allowed_files(read)
    records = manifest["files"]
    if set(records) != expected or set(names) != expected | {MANIFEST}:
        raise ValueError("Package allowlist mismatch: missing files or forbidden/unlisted additions.")
    for name, record in records.items():
        content = read(name)
        if record != {"sha256": sha256(content), "size": len(content)}:
            raise ValueError(f"Integrity mismatch: {name}")
    verify_evidence(read)
    return {"version": manifest["version"], "files": len(records), "corpus_files": 20}


def verify_release(path: Path) -> dict:
    path = Path(path)
    if path.is_dir():
        if path.is_symlink() or path.is_junction():
            raise ValueError("Release root must not be a link.")
        names = []
        directories = []
        for directory, dirs, files in os.walk(path, followlinks=False):
            for name in dirs + files:
                item = Path(directory) / name
                if item.is_symlink() or item.is_junction():
                    raise ValueError(f"Links are not allowed: {item}")
            directories.extend((Path(directory) / name).relative_to(path).as_posix() for name in dirs)
            names.extend((Path(directory) / name).relative_to(path).as_posix() for name in files)
        for directory in directories:
            safe_name(directory)
            if not any(name.startswith(directory + "/") for name in names):
                raise ValueError(f"Unlisted empty directory: {directory}")
        return verify_contents(names, lambda name: source_file(path, name).read_bytes())
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        for info in infos:
            safe_name(info.filename)
            mode = info.external_attr >> 16
            if info.is_dir() or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                raise ValueError(f"ZIP must contain regular files only: {info.filename}")
        if sum(info.file_size for info in infos) > 512 * 1024 * 1024:
            raise ValueError("Release exceeds the 512 MiB verification limit.")
        return verify_contents([info.filename for info in infos], archive.read)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=ROOT, help="Untouched stage or ZIP")
    args = parser.parse_args()
    try:
        result = verify_release(args.path)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Release verification failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
