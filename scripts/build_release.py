"""Build a new allowlisted local release; never overwrite an existing version."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

from scripts.verify_release import (
    MANIFEST, OMISSIONS, ROOT, allowed_files, sha256, source_file,
    validate_version, verify_evidence, verify_release,
)


def build_release(source: Path, output: Path, version: str) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    validate_version(version)
    name = f"local-search-{version}"
    stage, archive, checksum = output / name, output / f"{name}.zip", output / f"{name}.zip.sha256"
    for path in (stage, archive, checksum):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing existing release output: {path}")
    read = lambda name: source_file(source, name).read_bytes()
    names = sorted(allowed_files(read))
    for filename in names:
        source_file(source, filename)
    verify_evidence(read)
    output.mkdir(parents=True, exist_ok=True)
    stage.mkdir()
    records = {}
    for filename in names:
        content = read(filename)
        target = stage.joinpath(*filename.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
        records[filename] = {"sha256": sha256(content), "size": len(content)}
    manifest = {
        "schema_version": 1,
        "version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "omissions": OMISSIONS,
        "files": records,
    }
    with (stage / MANIFEST).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    result = verify_release(stage)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for filename in sorted([*names, MANIFEST]):
            bundle.write(stage.joinpath(*filename.split("/")), arcname=filename)
    verify_release(archive)
    with checksum.open("x", encoding="ascii") as stream:
        stream.write(f"{sha256(archive.read_bytes())}  {archive.name}\n")
    return {**result, "stage": str(stage), "zip": str(archive), "checksum": str(checksum)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        result = build_release(
            args.source, args.output_dir or args.source / "artifacts" / "releases", args.version,
        )
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Release build failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
