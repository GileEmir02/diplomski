from dataclasses import replace

import pytest

from src.chunking import chunk_documents
from src.config import load_config
from src.ingestion import DocumentInput, read_document
from tests.helpers import WordTokenizer


def test_overlap_short_tail_and_exact_source_spans():
    doc = read_document(DocumentInput("notes.txt", b"one two three four five six seven eight"))
    config = replace(load_config(), chunk_tokens=3, overlap_tokens=1)
    chunks = chunk_documents([doc], WordTokenizer(), config, max_seq_length=5)
    assert [chunk.text.strip() for chunk in chunks] == [
        "one two three", "three four five", "five six seven", "seven eight"
    ]
    assert [chunk.token_count for chunk in chunks] == [3, 3, 3, 2]
    for chunk in chunks:
        span = chunk.source_spans[0]
        assert chunk.text == doc.pages[0].clean_text[span.start_char:span.end_char]
        assert chunk.page_start is None and chunk.page_end is None
    assert chunks == chunk_documents([doc], WordTokenizer(), config, max_seq_length=5)


def test_zero_overlap_keeps_all_characters():
    doc = read_document(DocumentInput("notes.txt", b"one two\n\nthree four five"))
    config = replace(load_config(), chunk_tokens=2, overlap_tokens=0)
    chunks = chunk_documents([doc], WordTokenizer(), config, max_seq_length=4)
    assert "".join(chunk.text for chunk in chunks) == doc.pages[0].clean_text


def test_page_and_document_boundaries_are_preserved():
    doc = read_document(DocumentInput("notes.txt", b"first page"))
    first = replace(doc.pages[0], page_number=1)
    second = replace(first, page_number=2, raw_text="second page", clean_text="second page")
    pdf = replace(doc, format="pdf", page_count=2, pages=(first, second))
    other = read_document(DocumentInput("other.txt", b"another document"))
    chunks = chunk_documents([pdf, other], WordTokenizer(), load_config(), max_seq_length=512)
    assert [(chunk.document_id, chunk.page_start) for chunk in chunks] == [
        (pdf.document_id, 1), (pdf.document_id, 2), (other.document_id, None)
    ]
    assert all(chunk.page_start == chunk.page_end for chunk in chunks)


def test_identical_clean_text_from_distinct_sources_has_distinct_ids():
    first = read_document(DocumentInput("first.txt", b"same text"))
    second = read_document(DocumentInput("second.txt", b"same text\n"))
    chunks = chunk_documents([first, second], WordTokenizer(), load_config(), max_seq_length=512)
    assert chunks[0].text == chunks[1].text
    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_special_tokens_are_included_in_the_model_budget():
    doc = read_document(DocumentInput("notes.txt", b"one two three"))
    config = replace(load_config(), chunk_tokens=3, overlap_tokens=0)
    with pytest.raises(ValueError, match="limit"):
        chunk_documents([doc], WordTokenizer(), config, max_seq_length=4)


def test_empty_and_duplicate_collections_are_rejected():
    config = load_config()
    with pytest.raises(ValueError):
        chunk_documents([], WordTokenizer(), config, max_seq_length=512)
    doc = read_document(DocumentInput("notes.txt", b"one two"))
    with pytest.raises(ValueError):
        chunk_documents([doc, doc], WordTokenizer(), config, max_seq_length=512)


def test_changed_configuration_changes_chunk_ids():
    doc = read_document(DocumentInput("notes.txt", b"one two"))
    config = load_config()
    first = chunk_documents([doc], WordTokenizer(), config, max_seq_length=512)
    second = chunk_documents([doc], WordTokenizer(), replace(
        config, model_revision="a" * 40
    ), max_seq_length=512)
    assert first[0].chunk_id != second[0].chunk_id


def test_invalid_token_offsets_are_rejected():
    class BrokenTokenizer(WordTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if kwargs.get("return_offsets_mapping"):
                result["offset_mapping"] = [(2, 1)] * len(result["input_ids"])
            return result

    doc = read_document(DocumentInput("notes.txt", b"one two"))
    with pytest.raises(ValueError, match="Rasponi"):
        chunk_documents([doc], BrokenTokenizer(), load_config(), max_seq_length=512)
