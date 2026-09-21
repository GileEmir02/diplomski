"""Sparse BM25 with positive IDF and the same lexical analysis as TF-IDF."""

import math
from typing import Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


BM25_PARAMETERS = {
    "variant": "positive-idf-bm25",
    "k1": 1.5,
    "b": 0.75,
    "idf": "log1p((N-df+0.5)/(df+0.5))",
    "query_tf": "binary",
    "tokenization": "same-as-tfidf",
    "dtype": "float32",
}


def lexical_parameters(template: TfidfVectorizer) -> dict:
    parameters = template.get_params()
    return {
        name: parameters[name] for name in CountVectorizer().get_params()
        if name != "dtype"
    }


def build_bm25(
    texts: Sequence[str], template: TfidfVectorizer, *, k1: float = 1.5, b: float = 0.75,
) -> tuple[CountVectorizer, sparse.csr_matrix]:
    if isinstance(texts, str) or not texts:
        raise ValueError("Nema odlomaka za BM25 indeks.")
    if (type(k1) not in (int, float) or not math.isfinite(k1) or k1 <= 0
            or type(b) not in (int, float) or not math.isfinite(b) or not 0 <= b <= 1):
        raise ValueError("BM25 zahteva konacan k1 > 0 i b izmedju 0 i 1.")
    vectorizer = CountVectorizer(**lexical_parameters(template), dtype=np.int64)
    counts = vectorizer.fit_transform(texts).tocsr()
    lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    average = float(lengths.mean())
    if not math.isfinite(average) or average <= 0:
        raise ValueError("BM25 korpus nema upotrebljive termine.")
    document_frequency = np.asarray(counts.getnnz(axis=0), dtype=np.float64)
    idf = np.log1p((len(texts) - document_frequency + 0.5) / (document_frequency + 0.5))
    length_penalty = k1 * (1 - b + b * lengths / average)
    frequencies = counts.data.astype(np.float64)
    # Each stored CSR entry receives the length penalty of its document row.
    denominator = frequencies + np.repeat(length_penalty, np.diff(counts.indptr))
    weights = idf[counts.indices] * frequencies * (k1 + 1) / denominator
    matrix = sparse.csr_matrix(
        (weights.astype(np.float32), counts.indices.copy(), counts.indptr.copy()),
        shape=counts.shape,
    )
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError("BM25 tezine nisu ispravne.")
    return vectorizer, matrix
