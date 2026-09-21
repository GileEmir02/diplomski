import csv
import hashlib
from pathlib import Path

import pytest

import src.ingestion as ingestion


def make_manifest(root: Path, rows):
    path = root / "manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file_name", "relative_path", "sha256"])
        writer.writerows(rows)
    return path


def test_public_manifest_loader_checks_hash_and_preserves_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion, "PROJECT_ROOT", tmp_path)
    content = b"Training data and validation data."
    source = tmp_path / "notes.txt"
    source.write_bytes(content)
    manifest = make_manifest(tmp_path, [("notes.txt", "notes.txt", hashlib.sha256(content).hexdigest())])
    inputs = ingestion.load_manifest_inputs(manifest)
    assert inputs == (ingestion.DocumentInput("notes.txt", content),)
    assert source.read_bytes() == content


def test_public_manifest_loader_rejects_hash_mismatch_and_empty_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion, "PROJECT_ROOT", tmp_path)
    (tmp_path / "notes.txt").write_bytes(b"changed")
    manifest = make_manifest(tmp_path, [("notes.txt", "notes.txt", "a" * 64)])
    with pytest.raises(ingestion.BatchImportError):
        ingestion.load_manifest_inputs(manifest)
    make_manifest(tmp_path, [])
    with pytest.raises(ingestion.BatchImportError, match="empty_manifest"):
        ingestion.load_manifest_inputs(manifest)
