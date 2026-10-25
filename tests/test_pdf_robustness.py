
from __future__ import annotations

from pathlib import Path

import pymupdf

from atlasrag.ingestion.parser import parse_pdf
from atlasrag.ingestion.structure import detect_structure
from atlasrag.ingestion.chunking import chunk_pdf
from atlasrag.ingestion.enrichment import enrich_chunks


def _save_pdf(pdf: pymupdf.Document, path: Path) -> None:
    pdf.save(str(path))
    pdf.close()


def _run_pipeline(path: Path):
    parsed = parse_pdf(path)
    structured = detect_structure(parsed)
    chunks = chunk_pdf(parsed, chunk_size=100)
    enriched = enrich_chunks(chunks)
    return parsed, structured, chunks, enriched


def test_wrapped_paragraph(tmp_path: Path) -> None:
    path = tmp_path / "wrapped_paragraph.pdf"

    pdf = pymupdf.open()
    page = pdf.new_page()

    page.insert_text(
        (72, 72),
        "Liquidity and Capital Resources",
        fontsize=18,
    )

    lines = [
        "Operating cash flow declined significantly during 2025",
        "because customers took longer to settle their invoices.",
        "This resulted in higher trade receivables and reduced",
        "cash conversion efficiency during the reporting period.",
    ]

    for index, line in enumerate(lines):
        page.insert_text(
            (72, 110 + index * 15),
            line,
            fontsize=11,
        )

    _save_pdf(pdf, path)

    parsed, structured, chunks, enriched = _run_pipeline(path)

    assert parsed.page_count == 1
    assert chunks, "No chunks generated"

    combined = " ".join(chunk.text for chunk in chunks)

    expected_phrases = [
        "Operating cash flow declined",
        "customers took longer",
        "higher trade receivables",
        "cash conversion efficiency",
    ]

    for phrase in expected_phrases:
        assert phrase in combined, (
            f"Missing expected phrase: {phrase}"
        )


def test_two_columns(tmp_path: Path) -> None:
    path = tmp_path / "two_columns.pdf"

    pdf = pymupdf.open()
    page = pdf.new_page()

    page.insert_text(
        (72, 72),
        "Revenue Performance",
        fontsize=18,
    )

    # Left column.
    page.insert_text(
        (72, 120),
        "Revenue increased by 15%.",
        fontsize=11,
    )
    page.insert_text(
        (72, 140),
        "European demand strengthened.",
        fontsize=11,
    )

    # Right column.
    page.insert_text(
        (330, 120),
        "Operating expenses increased.",
        fontsize=11,
    )
    page.insert_text(
        (330, 140),
        "Technology investment grew.",
        fontsize=11,
    )

    _save_pdf(pdf, path)

    parsed, structured, chunks, enriched = _run_pipeline(path)

    combined = " ".join(chunk.text for chunk in chunks)

    positions = [
        combined.find("Revenue increased"),
        combined.find("European demand"),
        combined.find("Operating expenses"),
        combined.find("Technology investment"),
    ]

    assert all(position >= 0 for position in positions), (
        f"Missing column content. Positions: {positions}"
    )

    assert positions == sorted(positions), (
        "Incorrect column-major reading order: "
        f"{positions}"
    )


def test_financial_table(tmp_path: Path) -> None:
    path = tmp_path / "financial_table.pdf"

    pdf = pymupdf.open()
    page = pdf.new_page()

    page.insert_text(
        (72, 72),
        "Cash Flow Statement",
        fontsize=18,
    )

    rows = [
        ("Item", "2025", "2024"),
        ("Operating cash flow", "150", "200"),
        ("Investing cash flow", "-40", "-35"),
        ("Financing cash flow", "20", "-10"),
    ]

    for index, row in enumerate(rows):
        y = 120 + index * 20

        page.insert_text((72, y), row[0], fontsize=11)
        page.insert_text((300, y), row[1], fontsize=11)
        page.insert_text((400, y), row[2], fontsize=11)

    _save_pdf(pdf, path)

    parsed, structured, chunks, enriched = _run_pipeline(path)

    combined = " ".join(chunk.text for chunk in chunks)

    # Preservation test, not a table-reconstruction test.
    required = [
        "Operating cash flow",
        "Investing cash flow",
        "Financing cash flow",
        "2025",
        "2024",
        "150",
        "200",
    ]

    for value in required:
        assert value in combined, (
            f"Missing financial table content: {value}"
        )


def test_repeated_headers(tmp_path: Path) -> None:
    path = tmp_path / "repeated_headers.pdf"

    pdf = pymupdf.open()

    for page_number in range(1, 4):
        page = pdf.new_page()

        page.insert_text(
            (72, 40),
            "ACME Annual Report 2025",
            fontsize=10,
        )

        page.insert_text(
            (72, 100),
            f"Financial discussion on page {page_number}.",
            fontsize=11,
        )

        page.insert_text(
            (72, 800),
            f"Page {page_number}",
            fontsize=9,
        )

    _save_pdf(pdf, path)

    parsed, structured, chunks, enriched = _run_pipeline(path)

    combined = " ".join(chunk.text for chunk in chunks)

    header_count = combined.count(
        "ACME Annual Report 2025"
    )

    # Preserve the original test's threshold.
    assert header_count <= 1, (
        "Repeated headers were not filtered. "
        f"Found {header_count} occurrences."
    )

    # Also verify that useful content survives filtering.
    for page_number in range(1, 4):
        expected = (
            f"Financial discussion on page {page_number}."
        )
        assert expected in combined, (
            f"Missing body content: {expected}"
        )


def test_shifted_wrapped_paragraph(tmp_path: Path) -> None:
    """
    A paragraph should remain intact when its
    continuation shifts horizontally.
    """
    path = tmp_path / "shifted_wrapped.pdf"

    pdf = pymupdf.open()
    page = pdf.new_page()

    page.insert_text(
        (270, 100),
        "OpenBrain's latest public",
        fontsize=11,
    )

    page.insert_text(
        (72, 115),
        "model was trained with additional compute.",
        fontsize=11,
    )

    _save_pdf(pdf, path)

    parsed, structured, chunks, enriched = _run_pipeline(path)

    combined = " ".join(
        element.text
        for element in structured.paragraphs
    )

    assert (
        "OpenBrain's latest public model was trained "
        "with additional compute."
    ) in combined, (
        "Horizontally shifted continuation was not "
        "reconstructed correctly."
    )
