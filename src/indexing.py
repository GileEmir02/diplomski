import hashlib
import json
import os
import pickle
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Protocol, Sequence
from uuid import uuid4

import numpy as np
from filelock import FileLock, Timeout
from numpy.typing import NDArray
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.bm25 import BM25_PARAMETERS, build_bm25, lexical_parameters
from src.chunking import Chunk, SourceSpan, Tokenizer, chunk_documents, count_tokens
from src.config import ROOT, SearchConfig
from src.ingestion import PREPROCESSING_VERSION, DocumentInput, ingest_batch


SCHEMA_VERSION = 2
TFIDF_PARAMETERS = {
    "lowercase": True, "strip_accents": None, "ngram_range": [1, 1],
    "norm": "l2", "smooth_idf": True, "min_df": 1, "stop_words": None,
    "dtype": "float32",
}
PACKAGES = ("numpy", "scipy", "scikit-learn", "pypdf",
            "sentence-transformers", "transformers", "torch")


class Encoder(Protocol):
    config: SearchConfig
    tokenizer: Tokenizer
    max_seq_length: int
    dimension: int

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


class IndexErrorBase(ValueError):
    pass


class NoIndexError(IndexErrorBase):
    pass


class StaleIndexError(IndexErrorBase):
    pass


class CorruptIndexError(IndexErrorBase):
    pass


class IndexBusyError(IndexErrorBase):
    pass


@dataclass(frozen=True)
class DocumentInfo:
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
    source_file: str


@dataclass(frozen=True)
class SearchIndex:
    generation: str
    config: SearchConfig
    documents: tuple[DocumentInfo, ...]
    chunks: tuple[Chunk, ...]
    embeddings: NDArray[np.float32]
    vectorizer: TfidfVectorizer
    tfidf_matrix: sparse.csr_matrix
    bm25_vectorizer: CountVectorizer | None = None
    bm25_matrix: sparse.csr_matrix | None = None


@dataclass(frozen=True)
class UpdateResult:
    generation: str
    added_ids: tuple[str, ...]
    duplicate_file_names: tuple[str, ...]
    changed: bool


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _versions() -> dict[str, str]:
    return {package: version(package) for package in PACKAGES}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CorruptIndexError(f"Nije moguce procitati indeksni fajl {path.name}.") from error


def _object(value: object) -> dict:
    if not isinstance(value, dict):
        raise CorruptIndexError("Neispravan objekat u indeksu.")
    return value


def _within(directory: Path, relative: str) -> Path:
    if not isinstance(relative, str):
        raise CorruptIndexError("Neispravna putanja u indeksu.")
    path = (directory / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(directory.resolve()):
        raise CorruptIndexError("Putanja izlazi iz indeksnog paketa.")
    return path


def _verify_vectors(values: NDArray, rows: int, dimension: int) -> None:
    if values.shape != (rows, dimension) or values.dtype != np.float32:
        raise CorruptIndexError("Neispravan oblik ili tip semantickih vektora.")
    if not np.isfinite(values).all():
        raise CorruptIndexError("Semanticki vektori sadrze nekonacne vrednosti.")
    if not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-5):
        raise CorruptIndexError("Semanticki vektori nisu normalizovani.")


def build_tfidf(texts: Sequence[str]) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    if isinstance(texts, str) or not texts:
        raise IndexErrorBase("Nema odlomaka za TF-IDF indeks.")
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents=None, ngram_range=(1, 1), norm="l2",
        smooth_idf=True, min_df=1, stop_words=None, dtype=np.float32,
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError as error:
        raise IndexErrorBase(f"TF-IDF indeks nije napravljen: {error}") from error
    return vectorizer, sparse.csr_matrix(matrix)


class IndexStore:
    """Load only trusted indexes created by this app, never uploaded pickle files."""

    def __init__(self, root: Path = ROOT / "artifacts" / "indexes") -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.active = self.root / "active.json"
        self.lock = FileLock(str(self.root / ".write.lock"), timeout=0)

    def _manifest(self, *, sources_only: bool = False) -> tuple[Path, dict, tuple[DocumentInfo, ...]]:
        if not self.active.exists():
            raise NoIndexError("Prvo dodajte i indeksirajte materijale.")
        pointer = _object(_read_json(self.active))
        generation = pointer.get("generation")
        if not isinstance(generation, str) or not re.fullmatch(r"[0-9a-f]{32}", generation):
            raise CorruptIndexError("Neispravna aktivna verzija indeksa.")
        directory = self.root / generation
        manifest_path = directory / "manifest.json"
        try:
            content = manifest_path.read_bytes()
        except OSError as error:
            raise CorruptIndexError("Nedostaje manifest aktivnog indeksa.") from error
        if _hash(content) != pointer.get("manifest_sha256"):
            raise CorruptIndexError("Kontrolni hash manifesta se ne poklapa.")
        manifest = _object(_read_json(manifest_path))
        if (type(manifest.get("schema_version")) is not int
                or manifest["schema_version"] not in {1, SCHEMA_VERSION}):
            raise CorruptIndexError("Nepodrzana verzija indeksnog formata.")
        entries = manifest.get("documents")
        if not isinstance(entries, list) or not entries:
            raise CorruptIndexError("Indeks nema dokumente.")
        documents = []
        try:
            for entry in entries:
                item = dict(_object(entry))
                item["empty_pages"] = tuple(item["empty_pages"])
                item["warnings"] = tuple(item["warnings"])
                documents.append(DocumentInfo(**item))
        except (KeyError, TypeError) as error:
            raise CorruptIndexError("Neispravni metapodaci dokumenata.") from error
        if len({doc.document_id for doc in documents}) != len(documents):
            raise CorruptIndexError("Dupli dokumenti u indeksu.")
        files = _object(manifest.get("files"))
        for document in documents:
            expected = str(Path("documents") / f"{document.document_id}.{document.format}")
            if (not re.fullmatch(r"doc_[0-9a-f]{64}", document.document_id)
                    or document.document_id != "doc_" + document.sha256
                    or document.format not in {"pdf", "txt"}
                    or document.source_file != expected
                    or files.get(expected) != document.sha256):
                raise CorruptIndexError("Neispravan identitet ili izvor dokumenta.")
        required = {"chunks.json", "embeddings.npy", "vectorizer.pkl", "tfidf.npz"}
        if manifest["schema_version"] == SCHEMA_VERSION:
            required |= {"bm25_vectorizer.pkl", "bm25.npz"}
        sources = {doc.source_file for doc in documents}
        if set(files) != required | sources:
            raise CorruptIndexError("Indeksni paket nema ocekivane fajlove.")
        for relative in sources if sources_only else files:
            path = _within(directory, relative)
            try:
                digest = _hash(path.read_bytes())
            except OSError as error:
                raise CorruptIndexError(f"Nedostaje indeksni fajl {relative}.") from error
            if digest != files[relative]:
                raise CorruptIndexError(f"Kontrolni hash se ne poklapa: {relative}.")
        return directory, manifest, tuple(documents)

    def source_inputs(self) -> tuple[DocumentInput, ...]:
        directory, _, documents = self._manifest(sources_only=True)
        return tuple(DocumentInput(
            doc.file_name, _within(directory, doc.source_file).read_bytes(),
            doc.title, doc.source_url, doc.license_name,
        ) for doc in documents)

    def load(self, config: SearchConfig) -> SearchIndex:
        directory, manifest, documents = self._manifest()
        if (manifest.get("config_fingerprint") != config.fingerprint()
                or manifest.get("config") != asdict(config)
                or manifest.get("preprocessing_version") != PREPROCESSING_VERSION
                or manifest.get("tfidf_parameters") != TFIDF_PARAMETERS
                or manifest.get("versions") != _versions()):
            raise StaleIndexError("Konfiguracija ili biblioteke su promenjene. Obnovite indeks.")
        has_bm25 = manifest["schema_version"] == SCHEMA_VERSION
        if has_bm25 and manifest.get("bm25_parameters") != BM25_PARAMETERS:
            raise StaleIndexError("BM25 podesavanja su promenjena. Obnovite indeks.")
        entries = _read_json(directory / "chunks.json")
        if not isinstance(entries, list) or not entries:
            raise CorruptIndexError("Indeks nema odlomke.")
        chunks = []
        try:
            for entry in entries:
                item = dict(_object(entry))
                item["source_spans"] = tuple(SourceSpan(**_object(span))
                                             for span in item["source_spans"])
                chunks.append(Chunk(**item))
        except (TypeError, KeyError) as error:
            raise CorruptIndexError("Neispravni metapodaci odlomaka.") from error
        document_ids = {doc.document_id for doc in documents}
        if (len({chunk.chunk_id for chunk in chunks}) != len(chunks)
                or any(chunk.document_id not in document_ids or not chunk.text.strip()
                       for chunk in chunks)
                or len(chunks) != manifest.get("chunk_count")):
            raise CorruptIndexError("Odlomci nisu povezani sa svojim dokumentima.")
        try:
            embeddings = np.load(directory / "embeddings.npy", allow_pickle=False)
            matrix = sparse.load_npz(directory / "tfidf.npz").tocsr()
            with (directory / "vectorizer.pkl").open("rb") as stream:
                vectorizer = pickle.load(stream)
        except (OSError, ValueError, EOFError, pickle.UnpicklingError) as error:
            raise CorruptIndexError("Indeksne reprezentacije nisu citljive.") from error
        _verify_vectors(embeddings, len(chunks), manifest["embedding_dimension"])
        if (not isinstance(vectorizer, TfidfVectorizer)
                or matrix.shape != (len(chunks), len(vectorizer.vocabulary_))
                or not np.isfinite(matrix.data).all()):
            raise CorruptIndexError("TF-IDF reprezentacija nije uskladjena sa odlomcima.")
        bm25_vectorizer, bm25_matrix = None, None
        if has_bm25:
            try:
                bm25_matrix = sparse.load_npz(directory / "bm25.npz").tocsr()
                with (directory / "bm25_vectorizer.pkl").open("rb") as stream:
                    bm25_vectorizer = pickle.load(stream)
            except (OSError, ValueError, EOFError, pickle.UnpicklingError) as error:
                raise CorruptIndexError("BM25 reprezentacija nije citljiva.") from error
            vocabulary = getattr(bm25_vectorizer, "vocabulary_", None)
            if (type(bm25_vectorizer) is not CountVectorizer
                    or not isinstance(vocabulary, dict) or not vocabulary
                    or any(not isinstance(term, str) or type(position) is not int
                           for term, position in vocabulary.items())
                    or set(vocabulary.values()) != set(range(len(vocabulary)))
                    or bm25_matrix.shape != (len(chunks), len(vocabulary))
                    or bm25_matrix.dtype != np.float32
                    or not np.isfinite(bm25_matrix.data).all()
                    or np.any(bm25_matrix.data < 0)
                    or any(bm25_vectorizer.get_params()[key] != value
                           for key, value in lexical_parameters(vectorizer).items())
                    or not np.array_equal(bm25_vectorizer.get_feature_names_out(),
                                          vectorizer.get_feature_names_out())):
                raise CorruptIndexError("BM25 reprezentacija nije uskladjena sa odlomcima i tokenizacijom.")
        embeddings.setflags(write=False)
        return SearchIndex(directory.name, config, documents, tuple(chunks),
                           embeddings, vectorizer, matrix, bm25_vectorizer, bm25_matrix)

    @staticmethod
    def _write_bm25(directory: Path, vectorizer: CountVectorizer, matrix: sparse.csr_matrix) -> None:
        sparse.save_npz(directory / "bm25.npz", matrix)
        with (directory / "bm25_vectorizer.pkl").open("wb") as stream:
            pickle.dump(vectorizer, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def _publish(self, directory: Path, manifest: dict) -> None:
        manifest_bytes = _json_bytes(manifest)
        (directory / "manifest.json").write_bytes(manifest_bytes)
        pointer = {"generation": directory.name, "manifest_sha256": _hash(manifest_bytes)}
        staged_pointer = self.root / f".active-{directory.name}.tmp"
        try:
            staged_pointer.write_bytes(_json_bytes(pointer))
            os.replace(staged_pointer, self.active)
        finally:
            staged_pointer.unlink(missing_ok=True)

    def upgrade_bm25(self, config: SearchConfig) -> UpdateResult:
        """Add BM25 to a legacy bundle without changing its sources or existing vectors."""
        try:
            with self.lock:
                current = self.load(config)
                if current.bm25_matrix is not None:
                    return UpdateResult(current.generation, (), (), False)
                old_directory, old_manifest, _ = self._manifest()
                if old_directory.name != current.generation:
                    raise StaleIndexError("Aktivna generacija je promenjena tokom pripreme.")
                vectorizer, matrix = build_bm25(
                    [chunk.text for chunk in current.chunks], current.vectorizer,
                    k1=BM25_PARAMETERS["k1"], b=BM25_PARAMETERS["b"],
                )
                directory = self.root / uuid4().hex
                shutil.copytree(old_directory, directory)
                for relative, expected in old_manifest["files"].items():
                    if _hash(_within(directory, relative).read_bytes()) != expected:
                        raise CorruptIndexError("Kopija prethodnog indeksa nije identicna izvoru.")
                self._write_bm25(directory, vectorizer, matrix)
                manifest = {
                    **old_manifest,
                    "schema_version": SCHEMA_VERSION,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "bm25_parameters": dict(BM25_PARAMETERS),
                    "upgraded_from_generation": current.generation,
                    "files": {
                        **old_manifest["files"],
                        **{name: _hash((directory / name).read_bytes())
                           for name in ("bm25.npz", "bm25_vectorizer.pkl")},
                    },
                }
                self._publish(directory, manifest)
                return UpdateResult(directory.name, (), (), True)
        except Timeout as error:
            raise IndexBusyError("Drugo indeksiranje je u toku. Pokusajte ponovo.") from error

    def _build(
        self, inputs: Sequence[DocumentInput], encoder: Encoder,
    ) -> str:
        result = ingest_batch(inputs)
        documents = tuple(sorted(result.documents, key=lambda doc: doc.document_id))
        chunks = chunk_documents(documents, encoder.tokenizer, encoder.config,
                                 max_seq_length=encoder.max_seq_length)
        full_counts = [
            count_tokens(encoder.tokenizer, chunk.text, add_special_tokens=True)
            for chunk in chunks
        ]
        embeddings = np.asarray(encoder.encode([chunk.text for chunk in chunks]),
                                dtype=np.float32)
        _verify_vectors(embeddings, len(chunks), encoder.dimension)
        vectorizer, matrix = build_tfidf([chunk.text for chunk in chunks])
        bm25_vectorizer, bm25_matrix = build_bm25(
            [chunk.text for chunk in chunks], vectorizer,
            k1=BM25_PARAMETERS["k1"], b=BM25_PARAMETERS["b"],
        )
        contents = {"doc_" + _hash(source.content): source.content for source in inputs}
        generation = uuid4().hex
        # Readers only follow active.json. A new directory stays inactive until that switch.
        directory = self.root / generation
        directory.mkdir()
        (directory / "documents").mkdir()
        info = []
        for document in documents:
            relative = str(Path("documents") / f"{document.document_id}.{document.format}")
            (directory / relative).write_bytes(contents[document.document_id])
            attributes = asdict(document)
            del attributes["pages"]
            info.append({**attributes, "source_file": relative})
        (directory / "chunks.json").write_bytes(_json_bytes([asdict(chunk) for chunk in chunks]))
        np.save(directory / "embeddings.npy", embeddings, allow_pickle=False)
        sparse.save_npz(directory / "tfidf.npz", matrix)
        with (directory / "vectorizer.pkl").open("wb") as stream:
            pickle.dump(vectorizer, stream, protocol=pickle.HIGHEST_PROTOCOL)
        self._write_bm25(directory, bm25_vectorizer, bm25_matrix)
        files = {str(path.relative_to(directory)): _hash(path.read_bytes())
                 for path in sorted(directory.rglob("*")) if path.is_file()}
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(encoder.config),
            "config_fingerprint": encoder.config.fingerprint(),
            "preprocessing_version": PREPROCESSING_VERSION,
            "versions": _versions(),
            "tfidf_parameters": TFIDF_PARAMETERS,
            "bm25_parameters": dict(BM25_PARAMETERS),
            "embedding_dimension": encoder.dimension,
            "chunk_count": len(chunks),
            "chunk_statistics": {
                "minimum_tokens": min(chunk.token_count for chunk in chunks),
                "median_tokens": float(np.median([chunk.token_count for chunk in chunks])),
                "maximum_tokens": max(chunk.token_count for chunk in chunks),
                "maximum_with_special_tokens": max(full_counts),
                "model_token_limit": encoder.max_seq_length,
            },
            "documents": info,
            "files": files,
        }
        self._publish(directory, manifest)
        return generation

    def add(self, sources: Sequence[DocumentInput], encoder: Encoder) -> UpdateResult:
        try:
            with self.lock:
                previous = self.source_inputs() if self.active.exists() else ()
                old_documents = ingest_batch(previous).documents if previous else ()
                proposed = ingest_batch(sources, old_documents)
                if not proposed.added_ids and self.active.exists():
                    current = self.load(encoder.config)
                    return UpdateResult(current.generation, (), proposed.duplicate_file_names, False)
                known = {doc.document_id for doc in old_documents}
                new_inputs = []
                for source in sources:
                    document_id = "doc_" + _hash(source.content)
                    if document_id not in known:
                        new_inputs.append(source)
                        known.add(document_id)
                generation = self._build((*previous, *new_inputs), encoder)
                return UpdateResult(generation, proposed.added_ids,
                                    proposed.duplicate_file_names, True)
        except Timeout as error:
            raise IndexBusyError("Drugo indeksiranje je u toku. Pokusajte ponovo.") from error

    def rebuild(self, encoder: Encoder) -> UpdateResult:
        try:
            with self.lock:
                generation = self._build(self.source_inputs(), encoder)
                return UpdateResult(generation, (), (), True)
        except Timeout as error:
            raise IndexBusyError("Drugo indeksiranje je u toku. Pokusajte ponovo.") from error
