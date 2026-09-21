import csv
import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, asdict, fields, replace
from io import BytesIO
from pathlib import Path, PureWindowsPath
from threading import Barrier

import pytest
from pypdf import PageObject, PdfReader, PdfWriter
from pypdf.errors import PdfReadError
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

import src
from src import ingestion
from src.ingestion import (
    PREPROCESSING_VERSION,
    BatchImportError,
    BatchResult,
    DocumentInput,
    DocumentReadError,
    ImportIssue,
    IngestedDocument,
    PageRecord,
    clean_text,
    ingest_batch,
    read_document,
    write_pages_jsonl,
)


def pdf_bytes(texts=("First page",), *, password=None, image_only=False, nonzero_xref=False):
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=300, height=300)
        if text is None and not image_only:
            continue
        content = DecodedStreamObject()
        if image_only:
            content.set_data(
                b"q 10 0 0 10 20 20 cm BI /W 1 /H 1 /CS /RGB /BPC 8 "
                b"ID \xff\xff\xff\nEI Q"
            )
        else:
            font = DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
                NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
            })
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
            })
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            content.set_data(f"BT /F1 12 Tf 20 200 Td ({escaped}) Tj ET".encode("ascii"))
        page.replace_contents(content)
    if password is not None:
        writer.encrypt(password, algorithm="RC4-128")
    with BytesIO() as stream:
        writer.write(stream)
        content = stream.getvalue()
    writer.close()
    if nonzero_xref:
        # Omit the unused object-zero entry; all preceding object offsets stay valid.
        pattern = rb"xref\n0 (\d+)\n0000000000 65535 f \n"
        content, count = re.subn(
            pattern,
            lambda match: b"xref\n1 " + str(int(match[1]) - 1).encode("ascii") + b"\n",
            content,
            count=1,
        )
        assert count == 1
    return content


def test_shared_api_and_frozen_dataclasses():
    for name in src.__all__:
        assert getattr(src, name) is getattr(ingestion, name)
    with pytest.raises(AttributeError):
        getattr(src, "missing_ingestion_api")
    source = DocumentInput("notes.txt", b"text")
    document = read_document(source)
    issue = ImportIssue("bad.txt", "invalid_utf8", "TXT nije UTF-8.")
    result = ingest_batch((source,))
    for instance in (source, document, document.pages[0], issue, result):
        assert instance.__dataclass_params__.frozen
        with pytest.raises(FrozenInstanceError):
            setattr(instance, fields(instance)[0].name, "changed")
    assert isinstance(document, IngestedDocument)
    assert isinstance(document.pages[0], PageRecord)
    assert isinstance(result, BatchResult)
    error = DocumentReadError(issue)
    assert error.issue is issue
    assert "bad.txt" in str(error) and "invalid_utf8" in str(error)
    group_error = BatchImportError([issue])
    assert group_error.issues == (issue,)
    assert "cele grupe" in str(group_error)
    with pytest.raises(ValueError, match="bar jedan"):
        BatchImportError(())


@pytest.mark.parametrize("bom", [b"", b"\xef\xbb\xbf"])
def test_utf8_bom_metadata_and_raw_text(bom):
    text = "  Caf\u00e9 \u010d\u0107\u0161\u0111\u017e\t x\u00b2 \u2264 3.\r\n\r\nSecond  paragraph.  "
    source = DocumentInput(
        "caf\u00e9.TXT", bom + text.encode("utf-8"),
        title="Original title", source_url="https://example.test/text", license_name="CC-BY-4.0",
    )
    document = read_document(source)
    digest = hashlib.sha256(source.content).hexdigest()
    assert document.document_id == "doc_" + digest
    assert len(document.document_id) == 68
    assert document.sha256 == digest and document.byte_size == len(source.content)
    assert document.format == "txt" and document.page_count is None
    assert document.empty_pages == () and document.warnings == ()
    assert document.title == source.title
    assert document.source_url == source.source_url and document.license_name == source.license_name
    assert len(document.pages) == 1
    page = document.pages[0]
    assert page.page_number is None
    assert page.raw_text == text
    assert page.clean_text == clean_text(text)
    assert page.source_hash == digest and page.document_id == document.document_id
    assert page.file_name == source.file_name
    assert page.preprocessing_version == PREPROCESSING_VERSION


def test_cleaning_is_conservative_and_idempotent():
    raw = (
        " \tMixed  CASE,\tCaf\u00e9; x\u00b2 \u2264 3!\r\n"
        "  hyphen-\rcontinuation  \r\n \t\r\n\r\n"
        "\ufb01 \uff21 \u2163\u00a0\u00a0math \u2211 \u03b1.  "
    )
    expected = (
        "Mixed CASE, Caf\u00e9; x\u00b2 \u2264 3!\n"
        "hyphen-\ncontinuation\n\n\ufb01 \uff21 \u2163 math \u2211 \u03b1."
    )
    assert clean_text(raw) == expected
    assert clean_text(expected) == expected
    assert clean_text("a\nb\n\nc\n\n\n\n d") == "a\nb\n\nc\n\nd"
    assert clean_text("a\u2028b\u2029c\vd\fe") == "a\u2028b\u2029c\vd\fe"
    assert clean_text(" \t\r\n\r\n") == ""


@pytest.mark.parametrize(
    ("name", "content", "code"),
    [
        ("empty.txt", b"", "empty_file"),
        ("empty.pdf", b"", "empty_file"),
        ("space.txt", b" \t\r\n", "empty_text"),
        ("bom.txt", b"\xef\xbb\xbf", "empty_text"),
        ("word.docx", b"text", "unsupported_format"),
        ("book.pdf.exe", b"%PDF", "unsupported_format"),
        ("no_suffix", b"text", "unsupported_format"),
        ("binary.txt", b"text\x00more", "binary_text"),
        ("bad.txt", b"caf\xe9", "invalid_utf8"),
        ("truncated.txt", b"text\xe2\x82", "invalid_utf8"),
        ("utf16.txt", b"\xff\xfea\x00", "invalid_utf8"),
    ],
)
def test_invalid_documents(name, content, code):
    with pytest.raises(DocumentReadError) as caught:
        read_document(DocumentInput(name, content))
    assert caught.value.issue.file_name == name
    assert caught.value.issue.code == code
    assert caught.value.issue.message
    if code == "invalid_utf8":
        assert isinstance(caught.value.__cause__, UnicodeDecodeError)
        assert "bajtu" in str(caught.value)


@pytest.mark.parametrize(
    "name",
    [
        "", " ", ".", "..", "..\\notes.txt", "../notes.txt", "folder\\notes.txt",
        "folder/notes.txt", "\\notes.txt", "/notes.txt", "C:\\notes.txt", "C:notes.txt",
        "a\n.txt", "a\t.txt", "a\x00.txt", "a\x7f.txt", "a\u0085.txt",
        "a\u202e.txt", "a\ud800.txt", "NUL.txt", "CON.pdf", "LPT1.txt",
        "bad?.txt", "bad:stream.txt", "notes.txt ", "notes.txt.",
    ],
)
def test_filename_validation(name):
    with pytest.raises(DocumentReadError) as caught:
        read_document(DocumentInput(name, b"text"))
    assert caught.value.issue.code == "invalid_file_name"
    assert caught.value.issue.file_name == name


def test_wrong_argument_types_are_programming_errors():
    with pytest.raises(TypeError, match="Ime fajla"):
        read_document(DocumentInput(None, b"text"))
    with pytest.raises(TypeError, match="bytes"):
        read_document(DocumentInput("notes.txt", bytearray(b"text")))


def test_pdf_page_mapping_and_strict_reader(monkeypatch):
    calls = []

    def reader(stream, *, strict):
        assert isinstance(stream, BytesIO)
        calls.append(strict)
        return PdfReader(stream, strict=strict)

    monkeypatch.setattr(ingestion, "PdfReader", reader)
    source = DocumentInput("lecture.PDF", pdf_bytes(("Alpha", "Beta")))
    document = read_document(source)
    assert calls == [True]
    assert document.format == "pdf" and document.page_count == 2
    assert document.empty_pages == () and document.warnings == ()
    assert [page.page_number for page in document.pages] == [1, 2]
    assert [page.raw_text for page in document.pages] == ["Alpha", "Beta"]
    for page in document.pages:
        assert page.document_id == document.document_id
        assert page.file_name == "lecture.PDF"
        assert page.source_hash == hashlib.sha256(source.content).hexdigest()
        assert page.clean_text == page.raw_text


def test_blank_page_keeps_physical_page_number():
    document = read_document(DocumentInput("mixed.pdf", pdf_bytes((None, "Readable", None))))
    assert document.page_count == 3 and document.empty_pages == (1, 3)
    assert len(document.pages) == 1
    assert document.pages[0].page_number == 2
    assert document.pages[0].clean_text == "Readable"
    assert any("Strana 1" in warning for warning in document.warnings)
    assert any("Strana 3" in warning for warning in document.warnings)
    assert all("OCR nije pokrenut" in warning for warning in document.warnings)


@pytest.mark.parametrize("texts", [(), (None,), (None, None), ("   ",)])
def test_pdf_without_readable_text_is_rejected(texts):
    with pytest.raises(DocumentReadError) as caught:
        read_document(DocumentInput("blank.pdf", pdf_bytes(texts)))
    assert caught.value.issue.code == "pdf_no_text"


def test_image_only_pdf_is_rejected_without_ocr():
    with pytest.raises(DocumentReadError) as caught:
        read_document(DocumentInput("scan.pdf", pdf_bytes((None,), image_only=True)))
    assert caught.value.issue.code == "pdf_no_text"
    assert "OCR" in str(caught.value)


@pytest.mark.parametrize("content", [b"not a PDF", b"%PDF-1.7\ntruncated", b"%PDF-1.7\n%%EOF"])
def test_damaged_pdf_is_rejected(content):
    with pytest.raises(DocumentReadError) as caught:
        read_document(DocumentInput("damaged.pdf", content))
    assert caught.value.issue.code == "pdf_read_error"
    assert isinstance(caught.value.__cause__, PdfReadError)


@pytest.mark.parametrize("password", ["secret", ""])
def test_encrypted_pdf_is_rejected_even_with_empty_password(password, monkeypatch):
    content = pdf_bytes(password=password)

    def unexpected_extract(self):
        pytest.fail("Encrypted pages must not be extracted")

    monkeypatch.setattr(PageObject, "extract_text", unexpected_extract)
    with pytest.raises(DocumentReadError) as caught:
        read_document(DocumentInput("encrypted.pdf", content))
    assert caught.value.issue.code == "encrypted_pdf"


@pytest.mark.parametrize("value", [None, 123, b"text", ["text"]])
def test_non_string_pdf_extraction_is_an_explicit_failure(value, monkeypatch):
    monkeypatch.setattr(PageObject, "extract_text", lambda self: value)
    with pytest.raises(DocumentReadError) as caught:
        read_document(DocumentInput("broken-extractor.pdf", pdf_bytes()))
    assert caught.value.issue.code == "invalid_pdf_text"
    assert "strane 1" in caught.value.issue.message


def test_fatal_later_page_rejects_whole_pdf(monkeypatch):
    extracted = iter(("First page", PdfReadError("fatal page")))

    def extract(self):
        value = next(extracted)
        if isinstance(value, PdfReadError):
            raise value
        return value

    monkeypatch.setattr(PageObject, "extract_text", extract)
    with pytest.raises(DocumentReadError) as caught:
        read_document(DocumentInput("bad-page.pdf", pdf_bytes(("First", "Second"))))
    assert caught.value.issue.code == "pdf_read_error"


@pytest.mark.parametrize("error", [RuntimeError("bug"), KeyError("bug"), TypeError("bug")])
def test_unexpected_extraction_exceptions_propagate(error, monkeypatch):
    def extract(self):
        raise error

    monkeypatch.setattr(PageObject, "extract_text", extract)
    with pytest.raises(type(error)) as caught:
        ingest_batch((DocumentInput("bug.pdf", pdf_bytes()),))
    assert caught.value is error


def test_replacement_characters_are_retained_and_counted_per_page(monkeypatch):
    texts = iter(("First \ufffd\ufffd", "Second \ufffd"))
    monkeypatch.setattr(PageObject, "extract_text", lambda self: next(texts))
    document = read_document(DocumentInput("replacement.pdf", pdf_bytes(("one", "two"))))
    assert document.pages[0].raw_text == document.pages[0].clean_text == "First \ufffd\ufffd"
    assert document.pages[1].raw_text == document.pages[1].clean_text == "Second \ufffd"
    assert "Strana 1" in document.pages[0].warnings[0]
    assert "U+FFFD: 2" in document.pages[0].warnings[0]
    assert "Strana 2" in document.pages[1].warnings[0]
    assert "U+FFFD: 1" in document.pages[1].warnings[0]
    assert document.warnings == document.pages[0].warnings + document.pages[1].warnings
    txt = read_document(DocumentInput("replacement.txt", "text \ufffd".encode("utf-8")))
    assert txt.pages[0].raw_text.endswith("\ufffd")
    assert "TXT zapis" in txt.warnings[0] and "U+FFFD: 1" in txt.warnings[0]


def logger_state(logger):
    return logger.level, logger.propagate, logger.disabled, tuple(logger.handlers), tuple(logger.filters)


def test_nonfatal_xref_warning_is_persisted_without_changing_logging(caplog):
    loggers = (logging.getLogger(), logging.getLogger("pypdf"), logging.getLogger("pypdf._reader"))
    with caplog.at_level(logging.WARNING, logger="pypdf"):
        before = tuple(logger_state(logger) for logger in loggers)
        document = read_document(DocumentInput("xref.pdf", pdf_bytes(nonzero_xref=True)))
        assert tuple(logger_state(logger) for logger in loggers) == before
    assert document.pages[0].clean_text == "First page"
    assert any("Xref table not zero-indexed" in warning for warning in document.warnings)
    assert any("Xref table not zero-indexed" in record.message for record in caplog.records)


def test_concurrent_readers_do_not_share_warnings_or_logger_state(monkeypatch):
    barrier = Barrier(2)
    actual_reader = ingestion.PdfReader
    loggers = (logging.getLogger("pypdf"), logging.getLogger("pypdf._reader"))
    before = tuple(logger_state(logger) for logger in loggers)

    def synchronized_reader(stream, *, strict):
        barrier.wait(timeout=10)
        return actual_reader(stream, strict=strict)

    monkeypatch.setattr(ingestion, "PdfReader", synchronized_reader)
    inputs = (
        DocumentInput("warning.pdf", pdf_bytes(("Warning document",), nonzero_xref=True)),
        DocumentInput("normal.pdf", pdf_bytes(("Normal document",))),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        warned, normal = tuple(pool.map(read_document, inputs))
    assert len(warned.warnings) == 1 and "Xref table" in warned.warnings[0]
    assert normal.warnings == ()
    assert warned.pages[0].file_name == "warning.pdf"
    assert normal.pages[0].file_name == "normal.pdf"
    assert tuple(logger_state(logger) for logger in loggers) == before


def test_batch_collects_all_expected_errors_without_mutation_or_writes(tmp_path, monkeypatch):
    existing = (read_document(DocumentInput("old.txt", b"old")),)
    before = asdict(existing[0])
    sources = [
        DocumentInput("new.txt", b"valid new"),
        DocumentInput("bad.txt", b"\xff"),
        DocumentInput("empty.txt", b""),
        DocumentInput("bad.pdf", b"not a pdf"),
        DocumentInput("unsupported.doc", b"text"),
    ]
    snapshot = list(sources)

    def no_writes(*args, **kwargs):
        pytest.fail("ingest_batch must not write files")

    monkeypatch.setattr(ingestion, "_staged_file", no_writes)
    with pytest.raises(BatchImportError) as caught:
        ingest_batch(sources, existing)
    assert [issue.file_name for issue in caught.value.issues] == [
        "bad.txt", "empty.txt", "bad.pdf", "unsupported.doc",
    ]
    assert [issue.code for issue in caught.value.issues] == [
        "invalid_utf8", "empty_file", "pdf_read_error", "unsupported_format",
    ]
    assert sources == snapshot
    assert existing == (existing[0],) and asdict(existing[0]) == before
    assert list(tmp_path.iterdir()) == []


def test_append_success_preserves_existing_identity_and_order():
    old = read_document(DocumentInput("old.txt", b"old"))
    existing = (old,)
    result = ingest_batch(
        (DocumentInput("new.txt", b"new"), DocumentInput("third.txt", b"third")), existing,
    )
    assert result.documents[0] is old and existing == (old,)
    assert [document.file_name for document in result.documents] == ["old.txt", "new.txt", "third.txt"]
    assert result.added_ids == tuple(document.document_id for document in result.documents[1:])
    assert result.duplicate_file_names == ()


def test_same_hash_deduplicates_existing_and_new_with_explicit_filenames():
    old = read_document(DocumentInput("old.txt", b"old", title="Keep old metadata"))
    result = ingest_batch((
        DocumentInput("renamed.txt", b"old", title="Do not replace"),
        DocumentInput("new.txt", b"new"),
        DocumentInput("new-copy.TXT", b"new"),
        DocumentInput("another-old.txt", b"old"),
    ), (old,))
    assert result.documents[0] is old
    assert result.documents[0].title == "Keep old metadata"
    assert len(result.documents) == 2 and len(result.added_ids) == 1
    assert result.duplicate_file_names == ("renamed.txt", "new-copy.TXT", "another-old.txt")


def test_duplicate_bytes_still_require_valid_filename_format_and_content():
    old = read_document(DocumentInput("old.txt", b"old"))
    with pytest.raises(BatchImportError) as caught:
        ingest_batch((DocumentInput("duplicate.exe", b"old"),), (old,))
    assert caught.value.issues[0].code == "unsupported_format"


def test_same_filename_with_different_bytes_adds_distinct_documents():
    old = read_document(DocumentInput("same.txt", b"first"))
    result = ingest_batch((DocumentInput("same.txt", b"second"),), (old,))
    assert [document.file_name for document in result.documents] == ["same.txt", "same.txt"]
    assert result.documents[0] is old
    assert len({document.document_id for document in result.documents}) == 2
    assert [document.pages[0].raw_text for document in result.documents] == ["first", "second"]


def test_empty_batch_is_explicitly_rejected_even_with_existing_documents():
    existing = (read_document(DocumentInput("old.txt", b"old")),)
    with pytest.raises(BatchImportError) as caught:
        ingest_batch((), existing)
    assert caught.value.issues == (ImportIssue("", "empty_batch", "Grupa za uvoz je prazna."),)


def test_utf8_jsonl_keeps_source_mapping_text_and_only_nonempty_records(tmp_path):
    document = read_document(DocumentInput("caf\u00e9.txt", "Caf\u00e9 \ufffd\n\nx\u00b2".encode()))
    pdf = read_document(DocumentInput("mixed.pdf", pdf_bytes((None, "Actual second page"))))
    blank = replace(document.pages[0], clean_text=" \t")
    output = tmp_path / "nested" / "pages.jsonl"
    write_pages_jsonl(output, (replace(document, pages=(blank,) + document.pages), pdf))
    content = output.read_bytes()
    assert content.endswith(b"\n") and not content.startswith(b"\xef\xbb\xbf")
    assert "Caf\u00e9 \ufffd".encode("utf-8") in content
    records = [json.loads(line) for line in content.decode("utf-8").split("\n")[:-1]]
    expected = json.loads(json.dumps(asdict(document.pages[0])))
    assert records[0] == expected
    assert records[1]["page_number"] == 2
    assert records[1]["source_hash"] == pdf.sha256
    assert len(records) == 2
    assert list(output.parent.iterdir()) == [output]


@pytest.mark.parametrize("empty_kind", ["no_documents", "no_pages", "blank_page"])
def test_empty_jsonl_is_rejected_without_touching_old_output(tmp_path, empty_kind):
    output = tmp_path / "pages.jsonl"
    output.write_bytes(b"keep existing output")
    document = read_document(DocumentInput("notes.txt", b"text"))
    documents = {
        "no_documents": (),
        "no_pages": (replace(document, pages=()),),
        "blank_page": (replace(document, pages=(replace(document.pages[0], clean_text=" "),)),),
    }[empty_kind]
    with pytest.raises(ValueError, match="Nema nepraznih"):
        write_pages_jsonl(output, documents)
    assert output.read_bytes() == b"keep existing output"
    assert list(tmp_path.iterdir()) == [output]
    with pytest.raises(ValueError):
        write_pages_jsonl(tmp_path / "must-not-exist" / "pages.jsonl", documents)
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("failure", ["replace", "fsync", "serialize"])
def test_failed_write_keeps_old_output_and_only_cleans_own_temp(tmp_path, monkeypatch, failure):
    output = tmp_path / "pages.jsonl"
    output.write_bytes(b"original output")
    other_temp = tmp_path / ".pages.jsonl.someone-elses.tmp"
    other_temp.write_bytes(b"not ours")
    document = read_document(DocumentInput("new.txt", "new \u010d".encode()))

    def fail(*args, **kwargs):
        raise OSError("controlled write failure")

    if failure == "serialize":
        def partial_write(stream, pages):
            stream.write(b"partial")
            raise OSError("controlled write failure")
        monkeypatch.setattr(ingestion, "_write_page_records", partial_write)
    else:
        monkeypatch.setattr(ingestion.os, failure, fail)
    with pytest.raises(OSError, match="controlled"):
        write_pages_jsonl(output, (document,))
    assert output.read_bytes() == b"original output"
    assert other_temp.read_bytes() == b"not ours"
    assert set(tmp_path.iterdir()) == {output, other_temp}


def test_jsonl_replace_uses_a_closed_same_directory_temporary_file(tmp_path, monkeypatch):
    output = tmp_path / "pages.jsonl"
    document = read_document(DocumentInput("notes.txt", b"text"))
    real_replace = ingestion.os.replace
    replacements = []

    def verify_replace(source, target):
        staged, destination = Path(source), Path(target)
        assert staged.parent == destination.parent == tmp_path
        assert staged != destination and staged.suffix == ".tmp"
        assert json.loads(staged.read_text(encoding="utf-8"))["raw_text"] == "text"
        replacements.append((staged, destination))
        real_replace(source, target)

    monkeypatch.setattr(ingestion.os, "replace", verify_replace)
    write_pages_jsonl(output, (document,))
    assert len(replacements) == 1 and replacements[0][1] == output
    assert list(tmp_path.iterdir()) == [output]


_COLUMNS = ["file_name", "relative_path", "sha256", "title", "source_url", "license_or_permission"]


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(ingestion, "PROJECT_ROOT", root)
    return root


def source_row(root, relative_path, content, **metadata):
    relative = PureWindowsPath(relative_path)
    path = root.joinpath(*relative.parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "file_name": relative.name,
        "relative_path": relative_path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "title": "Sample title",
        "source_url": "https://example.test/sample",
        "license_or_permission": "CC-BY-4.0",
        **metadata,
    }


def manifest_file(root, rows, name="batch.csv"):
    path = root / "data" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def output_paths(root):
    return root / "data" / "processed" / "pages.jsonl", root / "artifacts" / "preparation" / "ingestion_report.json"


def run_cli(root, manifest):
    output, report = output_paths(root)
    return ingestion.main([
        "--manifest", str(manifest.relative_to(root)),
        "--output", str(output.relative_to(root)),
        "--report", str(report.relative_to(root)),
    ])


def test_cli_real_extraction_report_paths_versions_and_ascii_console(project, tmp_path, monkeypatch, capsys):
    rows = [
        source_row(project, "data\\raw\\mixed.pdf", pdf_bytes((None, "Text"), nonzero_xref=True)),
        source_row(project, "data\\raw\\caf\u00e9.txt", "Caf\u00e9 \ufffd".encode("utf-8")),
    ]
    manifest = manifest_file(project, rows)
    before_manifest = manifest.read_bytes()
    monkeypatch.chdir(tmp_path)
    assert run_cli(project, manifest) == 0
    console_text = capsys.readouterr().out
    assert console_text.isascii()
    console = json.loads(console_text)
    assert console["success"] is True
    output, report = output_paths(project)
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["success"] is True and payload["document_count"] == 2
    assert payload["text_record_count"] == 2 and payload["pdf_page_count"] == 2
    assert payload["replacement_character_count"] == 1
    assert payload["output_path"] == str(output.resolve())
    assert payload["report_path"] == str(report.resolve())
    assert payload["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert payload["versions"]["preprocessing"] == PREPROCESSING_VERSION
    assert payload["versions"]["pypdf"] == ingestion.PYPDF_VERSION
    pdf, txt = payload["documents"]
    assert pdf["page_count"] == 2 and pdf["empty_pages"] == [1]
    assert pdf["pages"][0]["page_number"] == 2
    assert any("Xref table not zero-indexed" in warning for warning in pdf["warnings"])
    assert txt["pages"][0]["page_number"] is None
    assert txt["replacement_character_count"] == 1
    assert txt["title"] == "Sample title" and txt["license_name"] == "CC-BY-4.0"
    assert manifest.read_bytes() == before_manifest
    for row in rows:
        raw = project.joinpath(*PureWindowsPath(row["relative_path"]).parts)
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == row["sha256"]


def test_cli_appends_restores_metadata_and_notices_duplicate_filenames(project, capsys):
    first = source_row(project, "data\\raw\\first\\same.txt", b"first", title="Keep this title")
    first_manifest = manifest_file(project, [first], "first.csv")
    assert run_cli(project, first_manifest) == 0
    output, report = output_paths(project)
    old_output = output.read_bytes()
    assert run_cli(project, first_manifest) == 0
    duplicates = json.loads(report.read_text(encoding="utf-8"))
    assert duplicates["added_ids"] == [] and duplicates["duplicate_file_names"] == ["same.txt"]
    assert output.read_bytes() == old_output
    second = source_row(project, "data\\raw\\second\\same.txt", b"second")
    old_copy = source_row(project, "data\\raw\\copy.txt", b"first", title="Do not replace old title")
    assert run_cli(project, manifest_file(project, [second, old_copy], "second.csv")) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["document_count"] == 2 and len(payload["added_ids"]) == 1
    assert payload["duplicate_file_names"] == ["copy.txt"]
    assert [document["file_name"] for document in payload["documents"]] == ["same.txt", "same.txt"]
    assert payload["documents"][0]["title"] == "Keep this title"
    assert output.read_bytes().startswith(old_output)
    assert (project / "data" / "raw" / "first" / "same.txt").read_bytes() == b"first"


def test_cli_rejection_aggregates_errors_and_preserves_existing_artifacts(project, capsys):
    old = source_row(project, "data\\raw\\old.txt", b"old")
    assert run_cli(project, manifest_file(project, [old], "old.csv")) == 0
    output, report = output_paths(project)
    before = output.read_bytes(), report.read_bytes()
    valid = source_row(project, "data\\raw\\new.txt", b"new")
    wrong_hash = source_row(project, "data\\raw\\changed.txt", b"changed", sha256="0" * 64)
    invalid_text = source_row(project, "data\\raw\\bad.txt", b"\xff")
    missing = {**valid, "file_name": "missing.txt", "relative_path": "data\\raw\\missing.txt"}
    capsys.readouterr()
    assert run_cli(project, manifest_file(project, [valid, wrong_hash, invalid_text, missing])) == 1
    failure_console = json.loads(capsys.readouterr().out)
    assert failure_console["success"] is False
    assert {issue["code"] for issue in failure_console["issues"]} == {
        "source_hash_mismatch", "invalid_utf8", "source_read_error",
    }
    assert (output.read_bytes(), report.read_bytes()) == before
    failure_path = Path(failure_console["report_path"])
    assert failure_path != report and failure_path.parent == report.parent
    assert json.loads(failure_path.read_text(encoding="utf-8"))["success"] is False
    assert not list(output.parent.glob("*.tmp"))


def test_cli_failed_initial_batch_can_be_retried_without_empty_success_output(project, capsys):
    bad = source_row(project, "data\\raw\\bad.txt", b"\xff")
    assert run_cli(project, manifest_file(project, [bad], "bad.csv")) == 1
    output, report = output_paths(project)
    assert not output.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["success"] is False
    good = source_row(project, "data\\raw\\good.txt", b"good")
    assert run_cli(project, manifest_file(project, [good], "good.csv")) == 0
    assert output.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["document_count"] == 1


@pytest.mark.parametrize(
    ("relative_path", "code"),
    [
        ("..\\outside.txt", "invalid_source_path"),
        ("C:\\outside.txt", "invalid_source_path"),
        ("C:outside.txt", "invalid_source_path"),
        ("\\outside.txt", "invalid_source_path"),
        ("data\\raw\\other.txt", "invalid_source_path"),
        ("data\\raw\\out\x00side.txt", "invalid_source_path"),
    ],
)
def test_cli_rejects_unsafe_or_mismatched_source_paths(project, relative_path, code, capsys):
    row = {
        "file_name": "outside.txt", "relative_path": relative_path, "sha256": "0" * 64,
        "title": "", "source_url": "", "license_or_permission": "",
    }
    assert run_cli(project, manifest_file(project, [row])) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["issues"][0]["code"] == code
    assert not output_paths(project)[0].exists()


def test_cli_checks_resolved_source_scope(project, tmp_path, monkeypatch, capsys):
    row = source_row(project, "data\\raw\\outside.txt", b"inside")
    manifest = manifest_file(project, [row])
    source = project / "data" / "raw" / "outside.txt"
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    resolve = Path.resolve

    def resolved(self, *args, **kwargs):
        return outside if self == source else resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolved)
    assert run_cli(project, manifest) == 1
    assert json.loads(capsys.readouterr().out)["issues"][0]["code"] == "source_outside_project"
    assert outside.read_bytes() == b"outside"


@pytest.mark.parametrize("kind", ["empty", "bad_header", "duplicate_header", "bad_row", "bad_hash"])
def test_cli_manifest_validation(project, kind, capsys):
    if kind in {"empty", "bad_hash"}:
        rows = [] if kind == "empty" else [
            source_row(project, "data\\raw\\notes.txt", b"notes", sha256="not-a-hash")
        ]
        manifest = manifest_file(project, rows)
    else:
        manifest = project / "bad.csv"
        content = {
            "bad_header": "name,path\nnotes.txt,notes.txt\n",
            "duplicate_header": "file_name,relative_path,sha256,sha256\n",
            "bad_row": "file_name,relative_path,sha256\nnotes.txt,too-short\n",
        }[kind]
        manifest.write_text(content, encoding="utf-8")
    assert run_cli(project, manifest) == 1
    payload = json.loads(capsys.readouterr().out)
    expected = {
        "empty": "empty_batch", "bad_header": "invalid_manifest",
        "duplicate_header": "invalid_manifest", "bad_row": "invalid_manifest_row",
        "bad_hash": "invalid_source_hash",
    }[kind]
    assert payload["issues"][0]["code"] == expected


@pytest.mark.parametrize("target", ["source", "manifest", "same_output_report"])
def test_cli_never_overwrites_sources_manifest_or_output_with_report(project, target, capsys):
    row = source_row(project, "data\\raw\\notes.txt", b"untouched")
    manifest = manifest_file(project, [row])
    raw = project / "data" / "raw" / "notes.txt"
    before_manifest = manifest.read_bytes()
    output, report = output_paths(project)
    if target == "source":
        output = raw
    elif target == "manifest":
        report = manifest
    else:
        report = output
    assert ingestion.main([
        "--manifest", str(manifest), "--output", str(output), "--report", str(report),
    ]) == 1
    assert raw.read_bytes() == b"untouched"
    assert manifest.read_bytes() == before_manifest
    payload = json.loads(capsys.readouterr().out)
    assert payload["issues"][0]["code"] == "unsafe_output_path"


@pytest.mark.parametrize("corruption", ["output_hash", "metadata", "missing_report", "duplicate_page"])
def test_cli_rejects_inconsistent_existing_pair_without_silent_reset(project, corruption, capsys):
    row = source_row(project, "data\\raw\\old.txt", b"old")
    manifest = manifest_file(project, [row])
    assert run_cli(project, manifest) == 0
    output, report = output_paths(project)
    if corruption == "output_hash":
        output.write_bytes(b"changed output")
    elif corruption == "missing_report":
        report.rename(report.with_name("user-preserved-report.json"))
    else:
        payload = json.loads(report.read_text(encoding="utf-8"))
        if corruption == "metadata":
            payload["documents"][0]["byte_size"] = "invalid"
        else:
            output.write_bytes(output.read_bytes() * 2)
            payload["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        report.write_text(json.dumps(payload), encoding="utf-8")
    previous_output = output.read_bytes()
    previous_report = report.read_bytes() if report.exists() else None
    capsys.readouterr()
    assert run_cli(project, manifest) == 1
    failure = json.loads(capsys.readouterr().out)
    assert failure["issues"][0]["code"] == "invalid_existing_collection"
    assert output.read_bytes() == previous_output
    if previous_report is not None:
        assert report.read_bytes() == previous_report


def test_cli_report_publish_failure_rolls_back_output(project, monkeypatch, capsys):
    old = source_row(project, "data\\raw\\old.txt", b"old")
    assert run_cli(project, manifest_file(project, [old], "old.csv")) == 0
    output, report = output_paths(project)
    before = output.read_bytes(), report.read_bytes()
    new = source_row(project, "data\\raw\\new.txt", b"new")
    real_replace = ingestion.os.replace

    def fail_report(source, target):
        if Path(target) == report:
            raise PermissionError("report locked")
        real_replace(source, target)

    monkeypatch.setattr(ingestion.os, "replace", fail_report)
    capsys.readouterr()
    assert run_cli(project, manifest_file(project, [new])) == 1
    assert (output.read_bytes(), report.read_bytes()) == before
    failure = json.loads(capsys.readouterr().out)
    assert failure["success"] is False and failure["issues"][0]["code"] == "filesystem_error"
    assert not list(output.parent.glob("*.tmp"))
    assert not list(report.parent.glob("*.tmp"))


def test_cli_failed_rollback_keeps_recoverable_old_bytes_and_reports_failure(project, monkeypatch, capsys):
    old = source_row(project, "data\\raw\\old.txt", b"old")
    assert run_cli(project, manifest_file(project, [old], "old.csv")) == 0
    output, report = output_paths(project)
    before_output, before_report = output.read_bytes(), report.read_bytes()
    new = source_row(project, "data\\raw\\new.txt", b"new")
    real_replace = ingestion.os.replace
    output_replacements = 0

    def fail_report_and_rollback(source, target):
        nonlocal output_replacements
        if Path(target) == output:
            output_replacements += 1
            if output_replacements == 2:
                raise PermissionError("rollback locked")
        if Path(target) == report:
            raise PermissionError("report locked")
        real_replace(source, target)

    monkeypatch.setattr(ingestion.os, "replace", fail_report_and_rollback)
    capsys.readouterr()
    assert run_cli(project, manifest_file(project, [new])) == 1
    failure = json.loads(capsys.readouterr().out)
    assert failure["success"] is False
    assert "rucni oporavak" in failure["issues"][0]["message"]
    backups = list(output.parent.glob("*.tmp"))
    assert len(backups) == 1 and backups[0].read_bytes() == before_output
    assert str(backups[0]) in failure["issues"][0]["message"]
    assert report.read_bytes() == before_report


@pytest.mark.parametrize("report_value", [None, [], "not a report"])
def test_cli_does_not_silently_accept_malformed_report_without_output(project, report_value, capsys):
    row = source_row(project, "data\\raw\\notes.txt", b"notes")
    manifest = manifest_file(project, [row])
    output, report = output_paths(project)
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps(report_value), encoding="utf-8")
    before = report.read_bytes()
    assert run_cli(project, manifest) == 1
    failure = json.loads(capsys.readouterr().out)
    assert failure["issues"][0]["code"] == "invalid_existing_collection"
    assert report.read_bytes() == before and not output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [("warnings", None), ("raw_text", 123), ("page_number", True), ("preprocessing_version", "")],
)
def test_cli_rejects_invalid_stored_page_types_explicitly(project, field, value, capsys):
    row = source_row(project, "data\\raw\\notes.txt", b"notes")
    manifest = manifest_file(project, [row])
    assert run_cli(project, manifest) == 0
    output, report = output_paths(project)
    page = json.loads(output.read_text(encoding="utf-8"))
    page[field] = value
    output.write_text(json.dumps(page) + "\n", encoding="utf-8")
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    report.write_text(json.dumps(payload), encoding="utf-8")
    before = output.read_bytes(), report.read_bytes()
    capsys.readouterr()
    assert run_cli(project, manifest) == 1
    failure = json.loads(capsys.readouterr().out)
    assert failure["issues"][0]["code"] == "invalid_existing_collection"
    assert (output.read_bytes(), report.read_bytes()) == before


def test_cli_preserves_unicode_line_separators_across_appends(project):
    row = source_row(project, "data\\raw\\unicode.txt", "a\u2028b\u2029c".encode("utf-8"))
    manifest = manifest_file(project, [row])
    assert run_cli(project, manifest) == 0
    assert run_cli(project, manifest) == 0
    output, report = output_paths(project)
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["raw_text"] == record["clean_text"] == "a\u2028b\u2029c"
    assert json.loads(report.read_text(encoding="utf-8"))["document_count"] == 1
