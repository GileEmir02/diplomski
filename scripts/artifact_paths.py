"""Resolve historical artifact locations without rewriting frozen evidence."""

from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
PATH_RENAMES = {
    r"data\evaluation\queries_ai_reviewed.json": r"data\evaluation\queries_v1.json",
    r"data\evaluation\queries_ai_reviewed.csv": r"data\evaluation\queries_v1.csv",
    r"data\evaluation\ai_review_audit.json": r"data\evaluation\review_audit_v1.json",
    r"scripts\apply_ai_review.py": r"scripts\apply_relevance_review.py",
    r"artifacts\experiments\test_ai_reviewed_v1": r"artifacts\experiments\test_two_methods_v1",
    r"artifacts\experiments\no_answer_ai_reviewed_v1": r"artifacts\experiments\no_answer_two_methods_v1",
    r"artifacts\experiments\test_bm25_ai_reviewed_v1": r"artifacts\experiments\test_three_methods_v1",
    r"artifacts\experiments\no_answer_bm25_ai_reviewed_v1": r"artifacts\experiments\no_answer_three_methods_v1",
}


def relocated_path(value: str | Path, *, root: Path = ROOT) -> Path:
    root = root.resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        return path
    relative = path.relative_to(root)
    for old, new in PATH_RENAMES.items():
        old_path = Path(old)
        if relative.is_relative_to(old_path):
            return root / new / relative.relative_to(old_path)
    return path


def relocated_hashes(recorded: Mapping[str, str], *, root: Path = ROOT) -> dict[str, str]:
    root = root.resolve()
    result = {}
    for name, checksum in recorded.items():
        path = relocated_path(name, root=root)
        key = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        if key in result:
            raise ValueError(f"Duplicate evidence path after relocation: {key}")
        result[key] = checksum
    return result
