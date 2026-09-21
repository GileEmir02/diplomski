import csv
import io
import json
from pathlib import Path
import stat
import zipfile

import pytest

from scripts.build_release import build_release
from scripts import verify_release as release


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build_release(release.ROOT, tmp_path_factory.mktemp("release"), "test-1")


@pytest.fixture(scope="module")
def contents(built):
    with zipfile.ZipFile(built["zip"]) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def altered(contents, name, content, *, update_manifest=False):
    result = {**contents, name: content}
    if update_manifest:
        manifest = release.load_json(contents[release.MANIFEST])
        manifest["files"][name] = {"sha256": release.sha256(content), "size": len(content)}
        result[release.MANIFEST] = json.dumps(manifest).encode()
    return result


def check(contents):
    return release.verify_contents(list(contents), contents.__getitem__)


def test_build_and_verify_directory_zip_and_frozen_source_bytes(built, contents):
    assert release.verify_release(Path(built["stage"])) == release.verify_release(Path(built["zip"]))
    assert check(contents)["corpus_files"] == 20
    for name, content in contents.items():
        if name != release.MANIFEST:
            assert content == release.source_file(release.ROOT, name).read_bytes()
    archive = Path(built["zip"])
    assert Path(built["checksum"]).read_text().split()[0] == release.sha256(archive.read_bytes())
    assert len(release.corpus_entries(contents[release.CORPUS_MANIFEST])) == 20
    assert not any(name.endswith((".html", ".pkl", ".docx")) for name in contents)
    assert "tests/unit/test_preparation.py" not in contents
    assert "scripts/prepare_sample_corpus.py" not in contents


@pytest.mark.parametrize("suffix", ["", ".zip", ".zip.sha256"])
def test_builder_refuses_any_existing_output_before_writing(tmp_path, suffix):
    target = tmp_path / f"local-search-test-1{suffix}"
    if suffix:
        target.write_bytes(b"keep")
    else:
        target.mkdir()
        (target / "keep").write_bytes(b"keep")
    before = list(tmp_path.iterdir())
    with pytest.raises(FileExistsError, match="Refusing"):
        build_release(release.ROOT, tmp_path, "test-1")
    assert list(tmp_path.iterdir()) == before
    assert (target if suffix else target / "keep").read_bytes() == b"keep"


@pytest.mark.parametrize("name", [
    "../escape", "/absolute", "C:/escape", "C:escape", r"folder\escape",
    "a/../escape", "a//b", "./app.py", "a/", "a/CON.txt",
    "a/trailing.", "a/trailing ", "a/name:stream", "a/\x00bad",
])
def test_path_validation_rejects_ambiguous_or_traversing_names(name):
    with pytest.raises(ValueError, match="Unsafe"):
        release.safe_name(name)


@pytest.mark.parametrize("version", ["../bad", "", "v1/", "v1\\x", "CON", "v1."])
def test_builder_rejects_unsafe_version(tmp_path, version):
    with pytest.raises(ValueError):
        build_release(release.ROOT, tmp_path, version)
    assert not list(tmp_path.iterdir())


def test_verifier_rejects_missing_manifest(contents):
    with pytest.raises(ValueError, match="Missing release_manifest"):
        check({key: value for key, value in contents.items() if key != release.MANIFEST})


def test_verifier_rejects_missing_payload(contents):
    with pytest.raises(ValueError, match="allowlist"):
        check({key: value for key, value in contents.items() if key != "app.py"})


@pytest.mark.parametrize("name", [
    ".env", ".streamlit/secrets.toml", ".venv/secret", ".cache/model.bin",
    "artifacts/indexes/saved.pkl", "rad/thesis.docx", "PREZENTACIJA/slides.pptx",
    "data/raw/source.html", "scripts/prepare_sample_corpus.py",
])
def test_verifier_rejects_forbidden_even_if_added_to_integrity_manifest(contents, name):
    with pytest.raises(ValueError, match="allowlist"):
        check(altered(contents, name, b"not allowed", update_manifest=True))


def test_verifier_rejects_payload_tampering(contents):
    with pytest.raises(ValueError, match="Integrity mismatch"):
        check(altered(contents, "app.py", contents["app.py"] + b"\n"))


@pytest.mark.parametrize("name,pattern", [
    ("data/evaluation/queries_v2.json", "Canonical dataset"),
    ("src/search.py", "retrieval/evaluation code"),
    ("data/raw/01_mit_introduction_classification.pdf", "Corpus bytes"),
    ("data/evaluation/v2_review_01/questions.csv", "Review evidence"),
    ("artifacts/experiments/test_three_methods_v2/tfidf/dataset.json", "snapshot mismatch"),
])
def test_frozen_checks_do_not_trust_rewritten_integrity_manifest(contents, name, pattern):
    with pytest.raises(ValueError, match=pattern):
        check(altered(contents, name, contents[name] + b"\n", update_manifest=True))


def test_corpus_manifest_rejects_traversal_and_wrong_count(contents):
    rows = list(csv.DictReader(io.StringIO(contents[release.CORPUS_MANIFEST].decode())))
    for bad_rows, pattern in ((rows[:-1], "exactly 20"), (rows, "Unsafe")):
        if len(bad_rows) == 20:
            bad_rows[0]["relative_path"] = r"data\raw\..\escape.pdf"
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(bad_rows)
        with pytest.raises(ValueError, match=pattern):
            release.corpus_entries(stream.getvalue().encode())


@pytest.mark.parametrize("name", ["../escape", "/escape", r"..\escape", "C:/escape"])
def test_zip_validation_never_extracts_traversing_members(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(name, b"bad")
    with pytest.raises(ValueError, match="Unsafe"):
        release.verify_release(archive)
    assert list(tmp_path.iterdir()) == [archive]


def test_zip_rejects_links_and_duplicate_names(tmp_path):
    archive = tmp_path / "bad.zip"
    link = zipfile.ZipInfo("app.py")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(link, "outside")
    with pytest.raises(ValueError, match="regular files"):
        release.verify_release(archive)
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("app.py", "one")
        bundle.writestr("APP.py", "two")
    with pytest.raises(ValueError, match="case-colliding"):
        release.verify_release(archive)


def test_json_rejects_duplicate_integrity_keys():
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        release.load_json(b'{"files": {}, "files": {}}')


def test_source_file_rejects_links(tmp_path):
    original = tmp_path / "original.txt"
    original.write_text("original")
    link = tmp_path / "linked.txt"
    try:
        link.symlink_to(original)
    except OSError:
        pytest.skip("Creating symbolic links requires Windows permission.")
    with pytest.raises(ValueError, match="Links"):
        release.source_file(tmp_path, "linked.txt")
