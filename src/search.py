from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from src.indexing import Encoder, NoIndexError, SearchIndex, StaleIndexError


SearchMethod = Literal["semantic", "tfidf", "bm25"]
SEARCH_METHODS: tuple[SearchMethod, ...] = ("semantic", "tfidf", "bm25")


def resolve_methods(selection: str) -> tuple[SearchMethod, ...]:
    if selection == "all":
        return SEARCH_METHODS
    if selection == "both":
        return ("semantic", "tfidf")
    if selection == "semantic":
        return ("semantic",)
    if selection == "tfidf":
        return ("tfidf",)
    if selection == "bm25":
        return ("bm25",)
    raise ValueError("Nepodrzana metoda pretrage.")


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    document_id: str
    text: str
    file_name: str
    page_start: int | None
    page_end: int | None
    rank: int
    score: float
    method: SearchMethod


@dataclass(frozen=True)
class SearchResponse:
    query: str
    method: SearchMethod
    generation: str
    hits: tuple[SearchHit, ...]
    warnings: tuple[str, ...]


def _rank(index: SearchIndex, scores: NDArray, method: SearchMethod) -> tuple[SearchHit, ...]:
    if scores.shape != (len(index.chunks),) or not np.isfinite(scores).all():
        raise ValueError("Skorovi nisu uskladjeni sa odlomcima ili nisu konacni.")
    order = sorted(range(len(scores)),
                   key=lambda position: (-float(scores[position]), index.chunks[position].chunk_id))
    hits = []
    for rank, position in enumerate(order[:index.config.top_k], start=1):
        chunk = index.chunks[position]
        hits.append(SearchHit(chunk.chunk_id, chunk.document_id, chunk.text,
                              chunk.file_name, chunk.page_start, chunk.page_end,
                              rank, float(scores[position]), method))
    return tuple(hits)


def search(
    index: SearchIndex, query: str, method: SearchMethod, encoder: Encoder | None = None,
) -> SearchResponse:
    if not index.chunks:
        raise NoIndexError("Indeks nema odlomke za pretragu.")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Unesite neprazan upit na engleskom.")
    query = query.strip()
    warnings = []
    if method == "semantic":
        if encoder is None:
            raise ValueError("Semanticki model nije ucitan.")
        if encoder.config.fingerprint() != index.config.fingerprint():
            raise StaleIndexError("Model i indeks koriste razlicitu konfiguraciju.")
        vector = encoder.encode([query])
        if (vector.shape != (1, index.embeddings.shape[1])
                or not np.isfinite(vector).all()
                or not np.allclose(np.linalg.norm(vector, axis=1), 1.0, atol=1e-5)):
            raise ValueError("Vektor upita nije ispravan ili normalizovan.")
        scores = index.embeddings @ vector[0]
    elif method == "tfidf":
        vector = index.vectorizer.transform([query])
        if vector.nnz == 0:
            warnings.append("Nema poznatih termina u TF-IDF recniku; svi skorovi su nula.")
        scores = (index.tfidf_matrix @ vector.T).toarray().ravel()
    elif method == "bm25":
        if index.bm25_vectorizer is None or index.bm25_matrix is None:
            raise StaleIndexError("Indeks jos nema BM25. Pokrenite upgrade-bm25 ili obnovite indeks.")
        vector = index.bm25_vectorizer.transform([query]).astype(np.float32)
        vector.data.fill(1.0)
        if vector.nnz == 0:
            warnings.append("Nema poznatih termina u BM25 recniku; svi skorovi su nula.")
        scores = (index.bm25_matrix @ vector.T).toarray().ravel()
    else:
        raise ValueError("Nepodrzana metoda pretrage.")
    warnings.append("Skor je vrednost za rangiranje, ne verovatnoca tacnosti. Proverite izvor.")
    return SearchResponse(query, method, index.generation, _rank(index, scores, method),
                          tuple(warnings))
