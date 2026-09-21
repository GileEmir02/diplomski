import importlib
import json
import os
import platform
import re
import sys
import tomllib
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts" / "preparation" / "environment.json"
MODULES = {
    "filelock": "filelock",
    "numpy": "numpy",
    "psutil": "psutil",
    "pypdf": "pypdf",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "sentence-transformers": "sentence_transformers",
    "streamlit": "streamlit",
    "torch": "torch",
}


def save_report(report: dict[str, object]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    with (ROOT / "pyproject.toml").open("rb") as source:
        project = tomllib.load(source)["project"]
    requested_python = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    expected_version = tuple(int(part) for part in requested_python.split("."))
    required = []
    for requirement in project["dependencies"]:
        match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
        if not match:
            raise ValueError(f"Unsupported dependency declaration: {requirement}")
        name = match.group().lower().replace("_", "-")
        if name not in MODULES:
            raise ValueError(f"Add an explicit import check for dependency: {name}")
        required.append(name)
    installed = {
        dist.name.lower().replace("_", "-"): dist.version
        for dist in distributions()
    }
    missing = [name for name in required if name not in installed]
    isolated = sys.prefix != sys.base_prefix
    report: dict[str, object] = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "isolated_venv": isolated,
        "required_python": project["requires-python"],
        "requested_python": requested_python,
        "installed_versions": {
            name: installed[name] for name in required if name in installed
        },
        "missing_dependencies": missing,
        "imports_verified": False,
        "cpu_only_torch_verified": False,
        "ready": False,
    }
    save_report(report)
    if not isolated:
        print("NOT READY: run with the project's .venv Python.", file=sys.stderr)
        return 1
    if sys.version_info[:len(expected_version)] != expected_version:
        print(f"NOT READY: this environment expects Python {requested_python}.",
              file=sys.stderr)
        return 1
    if missing:
        print("NOT READY: missing " + ", ".join(missing), file=sys.stderr)
        print("Retry: uv sync --no-build --system-certs", file=sys.stderr)
        return 1
    loaded = {name: importlib.import_module(MODULES[name]) for name in required}
    if (loaded["torch"].version.cuda is not None
            or loaded["torch"].version.hip is not None):
        raise RuntimeError("Expected the CPU-only PyTorch build.")
    report["imports_verified"] = True
    report["cpu_only_torch_verified"] = True
    report["ready"] = True
    save_report(report)
    print("READY: all required imports passed and PyTorch is CPU-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
