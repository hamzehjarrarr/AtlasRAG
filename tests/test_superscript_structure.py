from atlasrag.ingestion.parser import (
    ParsedPDF,
    PDFElement,
    PDFPage,
)
from atlasrag.ingestion.structure import (
    _combine_line_spans,
    detect_structure,
)


def span(text, size, x, y, n):
    return PDFElement(
        text=text,
        page_number=1,
        block_number=0,
        line_number=0,
        span_number=n,
        font_size=size,
        font_name="Regular",
        is_bold=False,
        bbox=(
            x,
            y,
            x + len(text) * size * 0.5,
            y + size,
        ),
    )


def test_footnote_reference_not_joined_to_number():
    parts = [
        span(
            "priority list to 2",
            11,
            20,
            100,
            0,
        ),
        span(
            "29",
            6.6,
            110,
            99,
            1,
        ),
    ]

    lines = _combine_line_spans(parts)

    assert len(lines) == 1
    assert "2 [footnote 29]" in lines[0].text
    assert "229" not in lines[0].text


def test_normal_number_is_preserved():
    parts = [
        span(
            "The total is 229",
            11,
            20,
            100,
            0,
        )
    ]

    lines = _combine_line_spans(parts)

    assert lines[0].text == "The total is 229"


def test_small_number_on_baseline_not_misclassified():
    parts = [
        span(
            "Total ",
            11,
            20,
            100,
            0,
        ),
        span(
            "29",
            6.6,
            60,
            104.4,
            1,
        ),
    ]

    text = _combine_line_spans(parts)[0].text

    assert "[footnote" not in text
    assert "29" in text


def test_structure_public_interface():
    parts = [
        span(
            "priority list to 2",
            11,
            20,
            100,
            0,
        ),
        span(
            "29",
            6.6,
            110,
            99,
            1,
        ),
    ]

    document = ParsedPDF(
        filename="test.pdf",
        document_id="abc",
        page_count=1,
        pages=[
            PDFPage(
                page_number=1,
                text="priority list to 229",
                elements=parts,
                width=600,
                height=800,
            )
        ],
    )

    structured = detect_structure(document)

    assert structured.paragraphs
    assert (
        "[footnote 29]"
        in structured.paragraphs[0].text
    )


def test_exponent_after_ten_is_not_a_footnote():
    parts = [
        span(
            "Agent-0 was trained with 10",
            11,
            20,
            100,
            0,
        ),
        span(
            "27",
            6.6,
            170,
            99,
            1,
        ),
        span(
            " FLOP.",
            11,
            180,
            100,
            2,
        ),
    ]

    text = _combine_line_spans(parts)[0].text

    assert "10^27 FLOP" in text
    assert "[footnote" not in text


def test_exponent_after_multiplication_sign():
    parts = [
        span(
            "GPT-4 required 2 ⋅10",
            11,
            20,
            100,
            0,
        ),
        span(
            "25",
            6.6,
            140,
            99,
            1,
        ),
    ]

    text = _combine_line_spans(parts)[0].text

    assert "2 ⋅10^25" in text
    assert "[footnote" not in text