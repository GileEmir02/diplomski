"""Local, all-or-nothing PDF/TXT extraction; no OCR or application index activation.

The CLI appends using the previous JSONL and its successful report as a pair.
If a report already exists, a rejected batch gets a separate failure report.
Pypdf keeps its normal logging destination: no logger, handler, filter, or
stream is changed. Its per-reader xref diagnostic is also persisted explicitly.
"""

import argparse
import csv
import hashlib
import json
import ntpath
import os
import platform
import re
import tempfile
import unicodedata
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, fields
from io import BytesIO
from pathlib import Path, PureWindowsPath
from typing import BinaryIO
from uuid import uuid4

from pypdf import PdfReader, __version__ as PYPDF_VERSION
from pypdf.errors import FileNotDecryptedError, LimitReachedError, ParseError, PdfReadError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_VERSION = "whitespace-v1"
INGESTION_VERSION = "1"
REPORT_SCHEMA_VERSION = 1
_HORIZONTAL_SPACE = re.compile(r"[^\S\r\n\v\f\x85\u2028\u2029]+")
_HASH = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class DocumentInput:
    file_name: str
    content: bytes
    title: str | None = None
    source_url: str | None = None
    license_name: str | None = None


@dataclass(frozen=True)
class PageRecord:
    document_id: str
    file_name: str
    page_number: int | None
    raw_text: str
    clean_text: str
    source_hash: str
    preprocessing_version: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class IngestedDocument:
    document_id: str
    file_name: str
    sha256: str
    format: str
    byte_size: int
    title: str | None
    source_url: str | None
    license_name: str | None
    page_count: int | None
    empty_pages: tuple[int, ...]
    warnings: tuple[str, ...]
    pages: tuple[PageRecord, ...]


@dataclass(frozen=True)
class ImportIssue:
    file_name: str
    code: str
    message: str


class DocumentReadError(ValueError):
    def __init__(self, issue: ImportIssue) -> None:
        self.issue = issue
        super().__init__(f"{issue.file_name}: [{issue.code}] {issue.message}")


class BatchImportError(ValueError):
    def __init__(self, issues: Sequence[ImportIssue]) -> None:
        self.issues = tuple(issues)
        if not self.issues:
            raise ValueError("Greska grupe mora sadrzati bar jedan razlog.")
        details = "; ".join(
            f"{issue.file_name or 'Grupa'}: [{issue.code}] {issue.message}"
            for issue in self.issues
        )
        super().__init__(f"Uvoz cele grupe je odbijen. {details}")


@dataclass(frozen=True)
class BatchResult:
    documents: tuple[IngestedDocument, ...]
    added_ids: tuple[str, ...]
    duplicate_file_names: tuple[str, ...]


def clean_text(text: str) -> str:
    """Normalize whitespace only; preserve single line breaks and paragraphs."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_HORIZONTAL_SPACE.sub(" ", line).strip(" ") for line in normalized.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _read_error(file_name: str, code: str, message: str) -> DocumentReadError:
    return DocumentReadError(ImportIssue(file_name, code, message))


def _validate_file_name(file_name: str) -> None:
    if not isinstance(file_name, str):
        raise TypeError("Ime fajla mora biti tekst.")
    if (
        not file_name.strip()
        or file_name in {".", ".."}
        or ntpath.basename(file_name) != file_name
        or ntpath.splitdrive(file_name)[0]
        or ntpath.isreserved(file_name)
        or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in file_name)
    ):
        raise _read_error(
            file_name, "invalid_file_name",
            "Navedite samo dozvoljeno ime fajla, bez putanje ili kontrolnih znakova.",
        )


def _page_record(
    source: DocumentInput, digest: str, page_number: int | None, raw_text: str,
) -> PageRecord:
    count = raw_text.count("\ufffd")
    location = "TXT zapis" if page_number is None else f"Strana {page_number}"
    warnings = (
        (f"{location}: broj zamenskih znakova U+FFFD: {count}.",) if count else ()
    )
    return PageRecord(
        document_id="doc_" + digest,
        file_name=source.file_name,
        page_number=page_number,
        raw_text=raw_text,
        clean_text=clean_text(raw_text),
        source_hash=digest,
        preprocessing_version=PREPROCESSING_VERSION,
        warnings=warnings,
    )


def _read_pdf(
    source: DocumentInput, digest: str,
) -> tuple[int, tuple[int, ...], tuple[str, ...], tuple[PageRecord, ...]]:
    records: list[PageRecord] = []
    empty_pages: list[int] = []
    warnings: list[str] = []
    try:
        with BytesIO(source.content) as stream:
            reader = PdfReader(stream, strict=True)
            if reader.is_encrypted:
                raise _read_error(
                    source.file_name, "encrypted_pdf", "Sifrovani PDF fajlovi nisu podrzani."
                )
            # Per-reader state preserves this diagnostic without global log capture.
            if reader.xref_index:
                warnings.append(
                    "pypdf upozorenje: Xref table not zero-indexed "
                    f"(pocetni indeks: {reader.xref_index}); proveriti izdvojeni tekst."
                )
            page_count = len(reader.pages)
            for page_number in range(1, page_count + 1):
                raw_text = reader.pages[page_number - 1].extract_text()
                if not isinstance(raw_text, str):
                    raise _read_error(
                        source.file_name, "invalid_pdf_text",
                        f"Ekstrakcija strane {page_number} nije vratila tekst "
                        f"(tip: {type(raw_text).__name__}).",
                    )
                record = _page_record(source, digest, page_number, raw_text)
                warnings.extend(record.warnings)
                if record.clean_text:
                    records.append(record)
                else:
                    empty_pages.append(page_number)
                    warnings.append(
                        f"Strana {page_number} nema citljiv tekst; OCR nije pokrenut."
                    )
    except FileNotDecryptedError as error:
        raise _read_error(
            source.file_name, "encrypted_pdf", "Sifrovani PDF fajlovi nisu podrzani."
        ) from error
    except (PdfReadError, ParseError, LimitReachedError) as error:
        raise _read_error(
            source.file_name, "pdf_read_error", f"PDF nije moguce procitati: {error}"
        ) from error
    if not records:
        raise _read_error(
            source.file_name, "pdf_no_text",
            "PDF nema citljiv tekst ni na jednoj strani; OCR nije podrzan.",
        )
    return page_count, tuple(empty_pages), tuple(warnings), tuple(records)


def read_document(source: DocumentInput) -> IngestedDocument:
    """Extract one source without writing files or repairing/replacing its text."""
    _validate_file_name(source.file_name)
    suffix = Path(source.file_name).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise _read_error(
            source.file_name, "unsupported_format", "Podrzani su samo PDF i UTF-8 TXT fajlovi."
        )
    if not isinstance(source.content, bytes):
        raise TypeError("Sadrzaj dokumenta mora biti bytes.")
    if not source.content:
        raise _read_error(source.file_name, "empty_file", "Fajl je prazan.")
    digest = hashlib.sha256(source.content).hexdigest()
    page_count: int | None
    empty_pages: tuple[int, ...]
    warnings: tuple[str, ...]
    pages: tuple[PageRecord, ...]
    if suffix == ".pdf":
        page_count, empty_pages, warnings, pages = _read_pdf(source, digest)
    else:
        try:
            raw_text = source.content.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError as error:
            raise _read_error(
                source.file_name, "invalid_utf8",
                f"TXT mora biti UTF-8; neispravan zapis na bajtu {error.start}.",
            ) from error
        if "\x00" in raw_text:
            raise _read_error(
                source.file_name, "binary_text", "TXT sadrzi binarni NUL znak."
            )
        record = _page_record(source, digest, None, raw_text)
        if not record.clean_text:
            raise _read_error(source.file_name, "empty_text", "TXT nema neprazan tekst.")
        page_count, empty_pages, warnings, pages = None, (), record.warnings, (record,)
    return IngestedDocument(
        document_id="doc_" + digest,
        file_name=source.file_name,
        sha256=digest,
        format=suffix[1:],
        byte_size=len(source.content),
        title=source.title,
        source_url=source.source_url,
        license_name=source.license_name,
        page_count=page_count,
        empty_pages=empty_pages,
        warnings=warnings,
        pages=pages,
    )


def ingest_batch(
    sources: Sequence[DocumentInput], existing: Sequence[IngestedDocument] = (),
) -> BatchResult:
    """Validate every incoming file, then propose an immutable append/dedup result."""
    if not sources:
        raise BatchImportError((ImportIssue("", "empty_batch", "Grupa za uvoz je prazna."),))
    incoming: list[IngestedDocument] = []
    issues: list[ImportIssue] = []
    for source in sources:
        try:
            incoming.append(read_document(source))
        except DocumentReadError as error:
            issues.append(error.issue)
    if issues:
        raise BatchImportError(issues)
    previous = tuple(existing)
    known_hashes = {document.sha256 for document in previous}
    added: list[IngestedDocument] = []
    duplicates: list[str] = []
    for document in incoming:
        if document.sha256 in known_hashes:
            duplicates.append(document.file_name)
        else:
            known_hashes.add(document.sha256)
            added.append(document)
    return BatchResult(
        documents=previous + tuple(added),
        added_ids=tuple(document.document_id for document in added),
        duplicate_file_names=tuple(duplicates),
    )


class _PublicationRecoveryError(OSError):
    def __init__(self, recovery_path: Path, error: OSError) -> None:
        self.recovery_path = recovery_path
        super().__init__(
            "Upis izvestaja i povratak izlaza nisu uspeli. "
            f"Fajl za rucni oporavak je sacuvan: {recovery_path}. Razlog: {error}"
        )


@contextmanager
def _staged_file(path: Path, write: Callable[[BinaryIO], None]) -> Iterator[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    preserve_for_recovery = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        yield temporary
    except _PublicationRecoveryError as error:
        preserve_for_recovery = temporary == error.recovery_path
        raise
    finally:
        if temporary is not None and not preserve_for_recovery:
            temporary.unlink(missing_ok=True)


def _nonempty_pages(documents: Sequence[IngestedDocument]) -> tuple[PageRecord, ...]:
    pages = tuple(
        page for document in documents for page in document.pages if page.clean_text.strip()
    )
    if not pages:
        raise ValueError("Nema nepraznih tekstualnih zapisa za JSONL; izlaz nije izmenjen.")
    return pages


def _write_page_records(stream: BinaryIO, pages: Sequence[PageRecord]) -> None:
    for page in pages:
        stream.write((json.dumps(asdict(page), ensure_ascii=False) + "\n").encode("utf-8"))


def write_pages_jsonl(path: Path, documents: Sequence[IngestedDocument]) -> None:
    """Atomically replace UTF-8 extraction output, never an active search index."""
    pages = _nonempty_pages(documents)
    with _staged_file(path, lambda stream: _write_page_records(stream, pages)) as staged:
        os.replace(staged, path)


def _text_counts(pages: Sequence[PageRecord]) -> dict[str, int]:
    return {
        "text_record_count": len(pages),
        "raw_text_char_count": sum(len(page.raw_text) for page in pages),
        "clean_text_char_count": sum(len(page.clean_text) for page in pages),
        "replacement_character_count": sum(page.raw_text.count("\ufffd") for page in pages),
    }


def _document_report(document: IngestedDocument) -> dict[str, object]:
    result: dict[str, object] = {
        field.name: getattr(document, field.name)
        for field in fields(IngestedDocument) if field.name != "pages"
    }
    result.update(_text_counts(document.pages))
    result["pages"] = [
        {
            "page_number": page.page_number,
            "preprocessing_version": page.preprocessing_version,
            "warnings": page.warnings,
            **_text_counts((page,)),
        }
        for page in document.pages
    ]
    return result


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pypdf": PYPDF_VERSION,
        "preprocessing": PREPROCESSING_VERSION,
        "ingestion": INGESTION_VERSION,
    }


def _write_json(stream: BinaryIO, payload: dict[str, object]) -> None:
    stream.write((json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def _write_bytes(stream: BinaryIO, content: bytes) -> None:
    stream.write(content)


def _project_path(path: Path) -> Path:
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _manifest_sources(
    manifest: Path, protected: set[Path],
) -> tuple[tuple[DocumentInput, ...], tuple[ImportIssue, ...]]:
    sources: list[DocumentInput] = []
    issues: list[ImportIssue] = []
    root = PROJECT_ROOT.resolve()
    try:
        with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, [])
            if (
                not {"file_name", "relative_path", "sha256"}.issubset(header)
                or len(header) != len(set(header))
            ):
                raise BatchImportError((ImportIssue(
                    manifest.name, "invalid_manifest",
                    "Manifest mora imati jedinstvene kolone file_name, relative_path i sha256.",
                ),))
            for values in reader:
                if not values:
                    continue
                if len(values) != len(header):
                    issues.append(ImportIssue(
                        manifest.name, "invalid_manifest_row",
                        f"Red {reader.line_num} nema ocekivan broj kolona.",
                    ))
                    continue
                row = dict(zip(header, values, strict=True))
                file_name = row["file_name"]
                try:
                    _validate_file_name(file_name)
                    relative = PureWindowsPath(row["relative_path"])
                    if (
                        not row["relative_path"]
                        or relative.drive or relative.root or ".." in relative.parts
                        or relative.name != file_name
                        or any(
                            unicodedata.category(char) in {"Cc", "Cf", "Cs"}
                            for char in row["relative_path"]
                        )
                    ):
                        raise _read_error(
                            file_name, "invalid_source_path",
                            "relative_path mora biti relativna putanja do navedenog fajla, "
                            "bez izlaska iz projekta.",
                        )
                    source_path = root.joinpath(*relative.parts).resolve()
                    protected.add(source_path)
                    if not source_path.is_relative_to(root):
                        raise _read_error(
                            file_name, "source_outside_project", "Izvor je van projekta."
                        )
                    expected_hash = row["sha256"].lower()
                    if not _HASH.fullmatch(expected_hash):
                        raise _read_error(
                            file_name, "invalid_source_hash", "Manifest nema ispravan SHA-256."
                        )
                    content = source_path.read_bytes()
                    if hashlib.sha256(content).hexdigest() != expected_hash:
                        raise _read_error(
                            file_name, "source_hash_mismatch",
                            "Sadrzaj izvora se ne poklapa sa SHA-256 iz manifesta.",
                        )
                    sources.append(DocumentInput(
                        file_name=file_name,
                        content=content,
                        title=row.get("title", "").strip() or None,
                        source_url=row.get("source_url", "").strip() or None,
                        license_name=row.get("license_or_permission", "").strip() or None,
                    ))
                except DocumentReadError as error:
                    issues.append(error.issue)
                except OSError as error:
                    issues.append(ImportIssue(
                        file_name, "source_read_error", f"Izvor nije moguce procitati: {error}"
                    ))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        issues.append(ImportIssue(
            manifest.name, "manifest_read_error", f"Manifest nije moguce procitati: {error}"
        ))
    return tuple(sources), tuple(issues)


def default_manifest_path() -> Path:
    full = PROJECT_ROOT / "data" / "full_corpus_manifest.csv"
    return full if full.exists() else PROJECT_ROOT / "data" / "corpus_manifest.csv"


def load_manifest_inputs(manifest: Path) -> tuple[DocumentInput, ...]:
    """Read and verify a project manifest without importing or modifying documents."""
    resolved = _project_path(manifest)
    sources, issues = _manifest_sources(resolved, {resolved})
    if issues:
        raise BatchImportError(issues)
    if not sources:
        raise BatchImportError((ImportIssue(
            resolved.name, "empty_manifest", "Manifest nema dokumente za uvoz.",
        ),))
    return sources


class _StoredCollectionError(ValueError):
    """Invalid persisted extraction pair, not a programming error."""


def _require_stored(condition: bool, message: str) -> None:
    if not condition:
        raise _StoredCollectionError(message)


def _stored_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _StoredCollectionError("Ocekivan je JSON objekat.")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise _StoredCollectionError("Imena JSON polja moraju biti tekst.")
        result[key] = item
    return result


def _stored_text(value: object) -> str:
    if not isinstance(value, str):
        raise _StoredCollectionError("Ocekivano je tekstualno polje.")
    return value


def _stored_optional_text(value: object) -> str | None:
    return None if value is None else _stored_text(value)


def _stored_positive_int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _StoredCollectionError("Ocekivan je pozitivan ceo broj.")
    return value


def _stored_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise _StoredCollectionError("Ocekivana je JSON lista.")
    return value


def _stored_strings(value: object) -> tuple[str, ...]:
    return tuple(_stored_text(item) for item in _stored_list(value))


def _restore_page(value: object) -> PageRecord:
    value = _stored_object(value)
    _require_stored(
        set(value) == {field.name for field in fields(PageRecord)},
        "JSONL zapis nema ocekivana polja.",
    )
    number = value["page_number"]
    page = PageRecord(
        document_id=_stored_text(value["document_id"]),
        file_name=_stored_text(value["file_name"]),
        page_number=None if number is None else _stored_positive_int(number),
        raw_text=_stored_text(value["raw_text"]),
        clean_text=_stored_text(value["clean_text"]),
        source_hash=_stored_text(value["source_hash"]),
        preprocessing_version=_stored_text(value["preprocessing_version"]),
        warnings=_stored_strings(value["warnings"]),
    )
    _require_stored(
        bool(page.clean_text.strip()) and bool(page.preprocessing_version)
        and bool(_HASH.fullmatch(page.source_hash))
        and page.document_id == "doc_" + page.source_hash,
        "Prazan ili neuskladjen JSONL zapis.",
    )
    return page


def _restore_document(value: object, pages: tuple[PageRecord, ...]) -> IngestedDocument:
    value = _stored_object(value)
    required = {field.name for field in fields(IngestedDocument)} - {"pages"}
    _require_stored(required.issubset(value), "Nepotpuni metapodaci dokumenta.")
    empty_pages = tuple(_stored_positive_int(item) for item in _stored_list(value["empty_pages"]))
    document = IngestedDocument(
        document_id=_stored_text(value["document_id"]),
        file_name=_stored_text(value["file_name"]),
        sha256=_stored_text(value["sha256"]),
        format=_stored_text(value["format"]),
        byte_size=_stored_positive_int(value["byte_size"]),
        title=_stored_optional_text(value["title"]),
        source_url=_stored_optional_text(value["source_url"]),
        license_name=_stored_optional_text(value["license_name"]),
        page_count=(
            None if value["page_count"] is None else _stored_positive_int(value["page_count"])
        ),
        empty_pages=empty_pages,
        warnings=_stored_strings(value["warnings"]),
        pages=pages,
    )
    try:
        _validate_file_name(document.file_name)
    except DocumentReadError as error:
        raise _StoredCollectionError(error.issue.message) from error
    _require_stored(
        bool(pages) and bool(_HASH.fullmatch(document.sha256))
        and document.document_id == "doc_" + document.sha256
        and Path(document.file_name).suffix.lower() == "." + document.format
        and all(
            page.source_hash == document.sha256
            and page.document_id == document.document_id
            and page.file_name == document.file_name
            for page in pages
        ),
        "Metapodaci i tekst dokumenta nisu uskladjeni.",
    )
    if document.format == "pdf":
        count = document.page_count
        if count is None:
            raise _StoredCollectionError("Nedostaje broj PDF strana.")
        numbers = tuple(page.page_number for page in pages if page.page_number is not None)
        _require_stored(
            len(numbers) == len(pages)
            and numbers == tuple(sorted(set(numbers)))
            and empty_pages == tuple(sorted(set(empty_pages)))
            and sorted(numbers + empty_pages) == list(range(1, count + 1)),
            "PDF strane su ponovljene, izgubljene ili pogresno numerisane.",
        )
    else:
        _require_stored(
            document.format == "txt" and document.page_count is None
            and not empty_pages and len(pages) == 1 and pages[0].page_number is None,
            "Neispravni metapodaci TXT dokumenta.",
        )
    _require_stored(
        all(value.get(name) == count for name, count in _text_counts(pages).items()),
        "Broj tekstualnih zapisa ili karaktera se ne poklapa sa izvestajem.",
    )
    return document


def _optional_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _load_existing(output: Path, report: Path) -> tuple[tuple[IngestedDocument, ...], bytes | None]:
    output_bytes = _optional_bytes(output)
    report_bytes = _optional_bytes(report)
    try:
        payload = (
            _stored_object(json.loads(report_bytes.decode("utf-8")))
            if report_bytes is not None else None
        )
        if output_bytes is None:
            _require_stored(
                payload is None or payload.get("success") is False,
                "Postoji uspesan izvestaj, ali njegov JSONL izlaz nedostaje.",
            )
            return (), None
        if payload is None:
            raise _StoredCollectionError("Postojeci JSONL nema prateci izvestaj.")
        _require_stored(
            payload.get("success") is True
            and type(payload.get("schema_version")) is int
            and payload["schema_version"] == REPORT_SCHEMA_VERSION,
            "Postojeci JSONL zahteva odgovarajuci uspesan izvestaj.",
        )
        _require_stored(
            Path(_stored_text(payload.get("output_path"))) == output
            and payload.get("output_sha256") == hashlib.sha256(output_bytes).hexdigest(),
            "Putanja ili SHA-256 postojeceg izlaza se ne poklapa sa izvestajem.",
        )
        document_values = _stored_list(payload.get("documents"))
        lines = output_bytes.decode("utf-8").split("\n")
        if lines[-1] == "":
            lines.pop()
        pages_by_id: dict[str, list[PageRecord]] = {}
        for line in lines:
            page = _restore_page(json.loads(line))
            pages_by_id.setdefault(page.document_id, []).append(page)
        documents: list[IngestedDocument] = []
        for value in document_values:
            value = _stored_object(value)
            document_id = _stored_text(value.get("document_id"))
            pages = tuple(pages_by_id.pop(document_id, []))
            documents.append(_restore_document(value, pages))
        _require_stored(
            bool(documents) and not pages_by_id
            and len({document.sha256 for document in documents}) == len(documents)
            and payload.get("document_count") == len(documents)
            and all(
                payload.get(name) == count
                for name, count in _text_counts(
                    tuple(page for document in documents for page in document.pages)
                ).items()
            ),
            "Postojeca kolekcija je prazna, duplirana ili nepotpuna.",
        )
        return tuple(documents), output_bytes
    except (UnicodeDecodeError, json.JSONDecodeError, _StoredCollectionError) as error:
        raise BatchImportError((ImportIssue(
            output.name, "invalid_existing_collection",
            f"Postojeca kolekcija nije izmenjena: {error}",
        ),)) from error


def _publish(
    output: Path, report: Path, manifest: Path, result: BatchResult, previous_output: bytes | None,
) -> dict[str, object]:
    pages = _nonempty_pages(result.documents)
    with ExitStack() as stack:
        staged_output = stack.enter_context(
            _staged_file(output, lambda stream: _write_page_records(stream, pages))
        )
        payload: dict[str, object] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "success": True,
            "manifest_path": str(manifest),
            "output_path": str(output),
            "report_path": str(report),
            "output_sha256": hashlib.sha256(staged_output.read_bytes()).hexdigest(),
            "versions": _versions(),
            "preprocessing_versions": sorted({page.preprocessing_version for page in pages}),
            "document_count": len(result.documents),
            "pdf_page_count": sum(
                document.page_count for document in result.documents
                if document.page_count is not None
            ),
            **_text_counts(pages),
            "added_ids": result.added_ids,
            "duplicate_file_names": result.duplicate_file_names,
            "documents": [_document_report(document) for document in result.documents],
        }
        staged_report = stack.enter_context(
            _staged_file(report, lambda stream: _write_json(stream, payload))
        )
        backup = None
        if previous_output is not None:
            backup = stack.enter_context(
                _staged_file(output, lambda stream: _write_bytes(stream, previous_output))
            )
        os.replace(staged_output, output)
        try:
            os.replace(staged_report, report)
        except OSError as publish_error:
            # Two paths cannot be renamed in one filesystem transaction.
            try:
                if backup is None:
                    output.unlink()
                else:
                    os.replace(backup, output)
            except OSError as rollback_error:
                raise _PublicationRecoveryError(
                    output if backup is None else backup, rollback_error
                ) from publish_error
            raise
    return payload


def _failure_report(
    report: Path, output: Path, manifest: Path, issues: Sequence[ImportIssue], protected: set[Path],
) -> int:
    payload: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "success": False,
        "message": "Uvoz nije dovrsen. Nijedan neispravan dokument nije prihvacen.",
        "manifest_path": str(manifest),
        "output_path": str(output),
        "report_path": None,
        "versions": _versions(),
        "issues": [asdict(issue) for issue in issues],
    }
    if report in protected or report == output:
        payload["report_write_error"] = (
            "Izvestaj ne sme zameniti manifest, izvorni fajl ili JSONL izlaz."
        )
    else:
        try:
            failure_path = report
            if report.exists():
                failure_path = report.with_name(f"{report.stem}.failure-{uuid4().hex}.json")
            payload["report_path"] = str(failure_path)
            with _staged_file(
                failure_path, lambda stream: _write_json(stream, payload)
            ) as staged:
                os.replace(staged, failure_path)
        except OSError as error:
            payload["report_path"] = None
            payload["report_write_error"] = f"Izvestaj nije moguce zapisati: {error}"
    print(json.dumps(payload, ensure_ascii=True))
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lokalni grupni uvoz PDF i UTF-8 TXT fajlova.")
    parser.add_argument("--manifest", required=True, type=Path, help="CSV manifest izvora.")
    parser.add_argument("--output", required=True, type=Path, help="JSONL izdvojenog teksta.")
    parser.add_argument("--report", required=True, type=Path, help="JSON izvestaj uvoza.")
    args = parser.parse_args(argv)
    manifest = _project_path(args.manifest)
    output = _project_path(args.output)
    report = _project_path(args.report)
    protected = {manifest}
    try:
        sources, manifest_issues = _manifest_sources(manifest, protected)
        if output == report or output in protected or report in protected:
            raise BatchImportError((ImportIssue(
                "", "unsafe_output_path",
                "Izlaz i izvestaj moraju biti razliciti i ne smeju zameniti izvor ili manifest.",
            ),))
        existing, previous_output = _load_existing(output, report)
        if manifest_issues and not sources:
            raise BatchImportError(manifest_issues)
        try:
            result = ingest_batch(sources, existing)
        except BatchImportError as error:
            raise BatchImportError(manifest_issues + error.issues) from error
        if manifest_issues:
            raise BatchImportError(manifest_issues)
        payload = _publish(output, report, manifest, result, previous_output)
    except BatchImportError as error:
        return _failure_report(report, output, manifest, error.issues, protected)
    except OSError as error:
        return _failure_report(
            report, output, manifest,
            (ImportIssue("", "filesystem_error", f"Greska pristupa fajlovima: {error}"),),
            protected,
        )
    console = {key: value for key, value in payload.items() if key != "documents"}
    console["message"] = "Uvoz je uspesan. Izdvajanje teksta ne aktivira indekse aplikacije."
    console["documents"] = [
        {key: value for key, value in _document_report(document).items() if key != "pages"}
        for document in result.documents
    ]
    print(json.dumps(console, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
