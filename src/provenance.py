"""Recognize the exact archived v1 evaluator while keeping retrieval checks strict."""

import hashlib
from pathlib import Path
from typing import Mapping

from src.config import ROOT


V1_DATASET_SHA256 = "699a9b4a79461b280d391a735e74b2f5ba09e58f3661ddf3cce39be1d40c15b2"
V1_EVALUATOR_SHA256 = "de0e530e8042735859eb7bfc4ecef60ea8f8abfbc6ecf041992c14b2fa7f5f05"
V1_EVALUATOR = ROOT / "artifacts" / "experiments" / "frozen_v1" / "evaluation.py"
EVALUATOR_KEY = str(Path("src") / "evaluation.py")


def code_provenance_profile(
    recorded: object, current: Mapping[str, str], dataset_sha256: str,
) -> str | None:
    if recorded == current:
        return "current"
    if (not isinstance(recorded, dict) or set(recorded) != set(current)
            or dataset_sha256 != V1_DATASET_SHA256
            or recorded.get(EVALUATOR_KEY) != V1_EVALUATOR_SHA256
            or any(recorded[key] != value for key, value in current.items() if key != EVALUATOR_KEY)):
        return None
    if (not V1_EVALUATOR.is_file()
            or hashlib.sha256(V1_EVALUATOR.read_bytes()).hexdigest() != V1_EVALUATOR_SHA256):
        raise ValueError("The frozen v1 evaluator snapshot is missing or changed.")
    return "frozen_v1"
