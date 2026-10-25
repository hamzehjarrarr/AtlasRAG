from __future__ import annotations

from pathlib import Path

from atlasrag.search import DATA_DIR, Dataset, document_text


def test_data_directory_is_project_local():
    assert isinstance(DATA_DIR, Path)
    assert DATA_DIR.name == "data"


def test_document_text_combines_title_and_body():
    document = {
        "title": "Research Paper",
        "text": "This is the abstract.",
    }

    assert document_text(document) == (
        "Research Paper\n\nThis is the abstract."
    )


def test_document_text_without_title():
    document = {
        "title": "",
        "text": "PDF passage content.",
    }

    assert document_text(document) == (
        "PDF passage content."
    )


def test_document_text_handles_empty_fields():
    assert document_text({}) == ""


def test_dataset_stores_pdf_chunks():
    corpus = {
        "chunk-1": {
            "title": "Page 1",
            "text": "Example PDF content.",
        }
    }

    dataset = Dataset(
        name="pdf_test_collection",
        corpus=corpus,
    )

    assert dataset.name == "pdf_test_collection"
    assert dataset.corpus == corpus