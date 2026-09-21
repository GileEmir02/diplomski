import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from src.config import SearchConfig
from src.ingestion import IngestedDocument


class Tokenizer(Protocol):
    def __call__(
        self, text: str, *, add_special_tokens: bool = True,
        truncation: bool = False, return_offsets_mapping: bool = False,
        verbose: bool = True,
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class SourceSpan:
    page_number: int | None
    start_char: int
    end_char: int


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    file_name: str
    text: str
    page_start: int | None
    page_end: int | None
    token_count: int
    source_spans: tuple[SourceSpan, ...]
    chunking_version: str


def _ids(encoded: Mapping[str, object]) -> list[int]:
    ids = encoded.get("input_ids")
    if not isinstance(ids, list) or any(type(value) is not int for value in ids):
        raise ValueError("Tokenizer nije vratio ocekivanu listu tokena.")
    return ids


def _offsets(encoded: Mapping[str, object], length: int) -> list[tuple[int, int]]:
    values = encoded.get("offset_mapping")
    if not isinstance(values, list) or len(values) != len(_ids(encoded)):
        raise ValueError("Tokenizer nije vratio uskladjene raspone karaktera.")
    result = []
    previous_end = 0
    for pair in values:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("Neispravan raspon tokenizer-a.")
        start, end = pair
        if (type(start) is not int or type(end) is not int
                or not previous_end <= start < end <= length):
            raise ValueError("Rasponi tokenizer-a nisu ispravni ili uredjeni.")
        result.append((start, end))
        previous_end = end
    return result


def count_tokens(tokenizer: Tokenizer, text: str, *, add_special_tokens: bool) -> int:
    return len(_ids(tokenizer(
        text, add_special_tokens=add_special_tokens, truncation=False, verbose=False,
    )))


def chunk_documents(
    documents: Sequence[IngestedDocument], tokenizer: Tokenizer,
    config: SearchConfig, *, max_seq_length: int,
) -> tuple[Chunk, ...]:
    if not documents:
        raise ValueError("Kolekcija za podelu je prazna.")
    if type(max_seq_length) is not int or max_seq_length <= 0:
        raise ValueError("Neispravan token limit modela.")
    if len({document.document_id for document in documents}) != len(documents):
        raise ValueError("Kolekcija sadrzi duple identifikatore dokumenata.")
    signature = config.fingerprint()
    chunks = []
    for document in documents:
        for page in document.pages:
            text = page.clean_text
            if page.document_id != document.document_id or not text.strip():
                raise ValueError("Neispravan izvor ili prazan zapis teksta.")
            encoded = tokenizer(
                text, add_special_tokens=False, truncation=False,
                return_offsets_mapping=True, verbose=False,
            )
            offsets = _offsets(encoded, len(text))
            if not offsets:
                raise ValueError(
                    f"Tekst nema tokene: {document.file_name}, strana {page.page_number}."
                )
            start_token = 0
            while start_token < len(offsets):
                end_token = min(start_token + config.chunk_tokens, len(offsets))
                start_char = 0 if start_token == 0 else offsets[start_token][0]
                # Include gaps and ignored characters, without reconstructing text from tokens.
                end_char = len(text) if end_token == len(offsets) else offsets[end_token][0]
                excerpt = text[start_char:end_char]
                token_count = count_tokens(tokenizer, excerpt, add_special_tokens=False)
                full_count = count_tokens(tokenizer, excerpt, add_special_tokens=True)
                if not excerpt.strip() or token_count == 0:
                    raise ValueError("Podela je proizvela prazan odlomak.")
                if full_count > max_seq_length:
                    raise ValueError(
                        f"Odlomak iz {document.file_name} ima {full_count} tokena "
                        f"sa specijalnim tokenima; limit je {max_seq_length}."
                    )
                identity = json.dumps(
                    [document.document_id, page.page_number, start_char, end_char,
                     page.preprocessing_version, signature, excerpt],
                    ensure_ascii=False, separators=(",", ":"),
                )
                chunks.append(Chunk(
                    chunk_id="chunk_" + hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                    document_id=document.document_id,
                    file_name=document.file_name,
                    text=excerpt,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    token_count=token_count,
                    source_spans=(SourceSpan(page.page_number, start_char, end_char),),
                    chunking_version=signature,
                ))
                if end_token == len(offsets):
                    break
                start_token = end_token - config.overlap_tokens
    if not chunks:
        raise ValueError("Nije napravljen nijedan odlomak.")
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("Podela je proizvela duple identifikatore odlomaka.")
    return tuple(chunks)
