
from atlasrag.ingestion.chunking import (
    CHUNKING_VERSION,
    DocumentChunk,
    _TextUnit,
    _merge_page_continuations,
    _split_long_text,
    chunk_pdf,
)

from atlasrag.ingestion.enrichment import enrich_chunk

from atlasrag.ingestion.parser import (
    PDFElement,
    PDFPage,
    ParsedPDF,
)

from atlasrag.ingestion.structure import (
    StructuredDocument,
    StructuredElement,
)

import atlasrag.ingestion.chunking as chunking_module


def make_element(
    text,
    page,
    kind="paragraph",
    level=None,
):
    return StructuredElement(
        text=text,
        page_number=page,
        element_type=kind,
        font_size=11.0,
        is_bold=(kind == "heading"),
        bbox=(20, 20, 500, 40),
        heading_level=level,
    )


def make_pdf_span(page_number):
    """
    Minimal layout element.

    Its presence ensures chunk_pdf() takes
    the structure-aware extraction path,
    rather than the raw-text fallback.
    """
    return PDFElement(
        text="Example layout text",
        page_number=page_number,
        block_number=0,
        line_number=0,
        span_number=0,
        font_size=11.0,
        font_name="Regular",
        is_bold=False,
        bbox=(20, 20, 200, 35),
    )


def make_structured_document():
    return ParsedPDF(
        filename="example.pdf",
        document_id="test-document",
        page_count=2,
        pages=[
            PDFPage(
                page_number=1,
                text="First page",
                elements=[make_pdf_span(1)],
            ),
            PDFPage(
                page_number=2,
                text="Second page",
                elements=[make_pdf_span(2)],
            ),
        ],
    )


def patch_structure(
    monkeypatch,
    elements,
):
    def fake_detect_structure(document):
        return StructuredDocument(
            document_id=document.document_id,
            filename=document.filename,
            body_font_size=11.0,
            elements=elements,
        )

    monkeypatch.setattr(
        chunking_module,
        "detect_structure",
        fake_detect_structure,
    )


def test_clear_cross_page_continuation():
    units = [
        _TextUnit(
            "The company developed a new",
            1,
        ),
        _TextUnit(
            "system that changed the market.",
            2,
        ),
    ]

    merged = _merge_page_continuations(units)

    assert len(merged) == 1
    assert merged[0].page_number == 1
    assert merged[0].end_page_number == 2
    assert (
        merged[0].text
        == "The company developed a new "
        "system that changed the market."
    )


def test_completed_sentence_not_joined():
    units = [
        _TextUnit(
            "The first event ended.",
            1,
        ),
        _TextUnit(
            "the next event began.",
            2,
        ),
    ]

    merged = _merge_page_continuations(units)

    assert len(merged) == 2


def test_capitalized_page_start_not_joined():
    units = [
        _TextUnit(
            "The company developed a new",
            1,
        ),
        _TextUnit(
            "System architecture was redesigned.",
            2,
        ),
    ]

    merged = _merge_page_continuations(units)

    assert len(merged) == 2


def test_long_text_respects_word_limit():
    text = (
        "The first sentence has several words. "
        "The second sentence also has words. "
        "The third sentence finishes the passage."
    )

    parts = _split_long_text(
        text,
        max_words=7,
    )

    assert parts
    assert all(
        len(part.split()) <= 7
        for part in parts
    )


def test_narrative_and_footnotes_separate(
    monkeypatch,
):
    document = make_structured_document()

    patch_structure(
        monkeypatch,
        [
            make_element(
                "February 2027: China Steals Agent-2",
                1,
                "heading",
                1,
            ),
            make_element(
                "The theft triggered a response.",
                1,
            ),
            make_element(
                "29 This is a supporting note.",
                1,
                "footnote",
            ),
        ],
    )

    chunks = chunk_pdf(
        document,
        chunk_size=100,
    )

    assert len(chunks) == 2

    assert (
        chunks[0].section_title
        == "February 2027: China Steals Agent-2"
    )

    assert (
        chunks[1].section_title
        == "February 2027: China Steals Agent-2"
        " — Footnotes"
    )

    assert "supporting note" not in chunks[0].text
    assert "theft triggered" not in chunks[1].text


def test_cross_page_chunk_range(
    monkeypatch,
):
    document = make_structured_document()

    patch_structure(
        monkeypatch,
        [
            make_element(
                "The company developed a new",
                1,
            ),
            make_element(
                "system that changed the market.",
                2,
            ),
        ],
    )

    chunks = chunk_pdf(
        document,
        chunk_size=100,
    )

    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].end_page_number == 2

    assert "new system" in chunks[0].text


def test_chunk_ids_are_deterministic(
    monkeypatch,
):
    document = make_structured_document()

    patch_structure(
        monkeypatch,
        [
            make_element(
                "A stable passage.",
                1,
            ),
        ],
    )

    first = chunk_pdf(document)
    second = chunk_pdf(document)

    assert first[0].chunk_id == second[0].chunk_id
    assert CHUNKING_VERSION == "structure-recursive-v5"


def test_narrative_context():
    chunk = DocumentChunk(
        chunk_id="one",
        document_id="doc",
        filename="example.pdf",
        page_number=1,
        end_page_number=2,
        chunk_index=0,
        text="Narrative passage.",
        section_title="Introduction",
    )

    enriched = enrich_chunk(
        chunk,
        document_title="Example",
    )

    assert (
        "Content type: Narrative"
        in enriched.retrieval_text
    )
    assert "Pages: 1-2" in enriched.retrieval_text
    assert enriched.text == chunk.text


def test_footnote_context():
    chunk = DocumentChunk(
        chunk_id="two",
        document_id="doc",
        filename="example.pdf",
        page_number=3,
        chunk_index=1,
        text="29 Supporting note.",
        section_title="Introduction — Footnotes",
    )

    enriched = enrich_chunk(chunk)

    assert (
        "Content type: Footnote"
        in enriched.retrieval_text
    )
    assert enriched.text == "29 Supporting note."


def test_invalid_page_range():
    chunk = DocumentChunk(
        chunk_id="bad",
        document_id="doc",
        filename="example.pdf",
        page_number=4,
        end_page_number=2,
        chunk_index=0,
        text="Invalid.",
    )

    try:
        enrich_chunk(chunk)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid page range was accepted."
        )
