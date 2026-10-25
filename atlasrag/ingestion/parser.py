
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import math

import pymupdf


# ============================================================
# Configuration
# ============================================================

MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

MAX_PAGES = 200
MAX_CHARS_PER_PAGE = 100_000
MAX_TOTAL_CHARS = 2_000_000

MAX_ELEMENTS_PER_PAGE = 20_000

PDF_SIGNATURE = b"%PDF-"


# ============================================================
# Exceptions
# ============================================================

class PDFValidationError(ValueError):
    """Raised when a PDF is invalid, unsupported, or exceeds limits."""


# ============================================================
# Data models
# ============================================================

@dataclass(frozen=True)
class PDFElement:
    """
    A text span extracted from a PDF.

    This represents observed layout information, not an
    inferred heading, paragraph, or table classification.
    """

    text: str
    page_number: int

    block_number: int
    line_number: int
    span_number: int

    font_size: float
    font_name: str
    is_bold: bool

    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class PDFPage:
    """
    One PDF page.

    The original 'text' field is preserved for backward
    compatibility with the existing recursive chunker.
    """

    page_number: int
    text: str

    elements: list[PDFElement] = field(default_factory=list)

    width: float | None = None
    height: float | None = None

    image_count: int = 0

    @property
    def has_extractable_text(self) -> bool:
        return bool(self.text.strip())

    @property
    def may_be_scanned(self) -> bool:
        """
        Heuristic only.

        A page with images but no extractable text may
        require OCR. This does not prove it is scanned.
        """
        return (
            not self.has_extractable_text
            and self.image_count > 0
        )


@dataclass(frozen=True)
class ParsedPDF:
    filename: str
    document_id: str
    page_count: int
    pages: list[PDFPage]

    @property
    def total_characters(self) -> int:
        return sum(len(page.text) for page in self.pages)

    @property
    def pages_without_text(self) -> list[int]:
        return [
            page.page_number
            for page in self.pages
            if not page.has_extractable_text
        ]

    @property
    def potentially_scanned_pages(self) -> list[int]:
        return [
            page.page_number
            for page in self.pages
            if page.may_be_scanned
        ]


# ============================================================
# File validation
# ============================================================

def _read_and_validate_file(file_path: Path) -> bytes:
    if not file_path.is_file():
        raise PDFValidationError(
            f"PDF file does not exist: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise PDFValidationError(
            "Only .pdf files are supported."
        )

    file_size = file_path.stat().st_size

    if file_size == 0:
        raise PDFValidationError(
            "The PDF file is empty."
        )

    if file_size > MAX_FILE_SIZE_BYTES:
        raise PDFValidationError(
            f"PDF exceeds the {MAX_FILE_SIZE_MB} MB limit."
        )

    with file_path.open("rb") as file:
        signature = file.read(len(PDF_SIGNATURE))

        if signature != PDF_SIGNATURE:
            raise PDFValidationError(
                "Invalid PDF signature."
            )

        file.seek(0)
        file_bytes = file.read(MAX_FILE_SIZE_BYTES + 1)

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise PDFValidationError(
            f"PDF exceeds the {MAX_FILE_SIZE_MB} MB limit."
        )

    return file_bytes


# ============================================================
# Layout extraction helpers
# ============================================================

def _safe_bbox(
    bbox: object,
) -> tuple[float, float, float, float]:
    """
    Validate a PyMuPDF bounding box.

    Coordinates are kept in PDF page units.
    """

    if not isinstance(bbox, (tuple, list)):
        raise PDFValidationError(
            "Invalid text bounding box."
        )

    if len(bbox) != 4:
        raise PDFValidationError(
            "Text bounding box must have four coordinates."
        )

    try:
        values = tuple(float(value) for value in bbox)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PDFValidationError(
            "Invalid text bounding box coordinates."
        ) from exc

    if not all(math.isfinite(value) for value in values):
        raise PDFValidationError(
            "Non-finite text bounding box coordinates."
        )

    return values


def _is_bold_span(span: dict) -> bool:
    """
    Detect bold styling using font flags and font name.

    This is a heuristic, not a guarantee.
    """

    flags = int(span.get("flags", 0))
    font_name = str(span.get("font", ""))

    # PyMuPDF text font flag bit 4 indicates bold.
    flag_indicates_bold = bool(flags & (1 << 4))

    name_indicates_bold = any(
        marker in font_name.lower()
        for marker in (
            "bold",
            "black",
            "heavy",
            "semibold",
            "demibold",
        )
    )

    return flag_indicates_bold or name_indicates_bold


def _extract_layout_elements(
    layout: dict,
    page_number: int,
) -> list[PDFElement]:
    """
    Convert PyMuPDF's structured text dictionary
    into our own stable data model.

    Only text spans are stored.

    Image blocks and other non-text blocks are ignored.
    """

    elements: list[PDFElement] = []

    for block_number, block in enumerate(
        layout.get("blocks", [])
    ):
        if block.get("type") != 0:
            continue

        for line_number, line in enumerate(
            block.get("lines", [])
        ):
            for span_number, span in enumerate(
                line.get("spans", [])
            ):
                text = str(span.get("text", ""))

                if not text.strip():
                    continue

                try:
                    font_size = float(span.get("size", 0.0))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise PDFValidationError(
                        "Invalid font size in PDF."
                    ) from exc

                if not math.isfinite(font_size):
                    raise PDFValidationError(
                        "Non-finite font size in PDF."
                    )

                element = PDFElement(
                    text=text,
                    page_number=page_number,
                    block_number=block_number,
                    line_number=line_number,
                    span_number=span_number,
                    font_size=font_size,
                    font_name=str(span.get("font", "")),
                    is_bold=_is_bold_span(span),
                    bbox=_safe_bbox(
                        span.get("bbox", ())
                    ),
                )

                elements.append(element)

                if len(elements) > MAX_ELEMENTS_PER_PAGE:
                    raise PDFValidationError(
                        f"Page {page_number} exceeds the "
                        "maximum number of text elements."
                    )

    return elements


# ============================================================
# PDF parsing
# ============================================================

def parse_pdf(
    file_path: str | Path,
) -> ParsedPDF:
    """
    Safely validate and extract text/layout from a PDF.

    Features:
      - SHA-256 document identification
      - Page-aware text extraction
      - Font and position metadata
      - Image count metadata
      - Resource limits
      - Backward-compatible PDFPage.text

    Limitations:
      - No OCR
      - No table reconstruction
      - No heading classification
      - No guaranteed multi-column reading order
      - No sandbox isolation of the PDF parser
    """

    path = Path(file_path).expanduser()

    file_bytes = _read_and_validate_file(path)

    document_id = hashlib.sha256(
        file_bytes
    ).hexdigest()

    pages: list[PDFPage] = []
    total_characters = 0

    try:
        with pymupdf.open(
            stream=file_bytes,
            filetype="pdf",
        ) as document:

            if document.needs_pass:
                raise PDFValidationError(
                    "Password-protected PDFs are not supported."
                )

            page_count = len(document)

            if page_count == 0:
                raise PDFValidationError(
                    "PDF contains no pages."
                )

            if page_count > MAX_PAGES:
                raise PDFValidationError(
                    f"PDF exceeds the {MAX_PAGES}-page limit."
                )

            for page_index in range(page_count):
                page = document.load_page(page_index)

                page_number = page_index + 1

                # Preserve plain-text extraction for
                # compatibility with existing chunking.
                text = page.get_text("text").strip()

                if len(text) > MAX_CHARS_PER_PAGE:
                    raise PDFValidationError(
                        f"Page {page_number} exceeds the "
                        "maximum text length."
                    )

                total_characters += len(text)

                if total_characters > MAX_TOTAL_CHARS:
                    raise PDFValidationError(
                        "PDF exceeds the maximum total "
                        "extracted text length."
                    )

                # Extract structured layout separately.
                layout = page.get_text("dict")

                elements = _extract_layout_elements(
                    layout=layout,
                    page_number=page_number,
                )

                # Image count is a heuristic indicator.
                image_count = len(
                    page.get_images(full=True)
                )

                pages.append(
                    PDFPage(
                        page_number=page_number,
                        text=text,
                        elements=elements,
                        width=float(page.rect.width),
                        height=float(page.rect.height),
                        image_count=image_count,
                    )
                )

    except PDFValidationError:
        raise

    except (
        pymupdf.FileDataError,
        pymupdf.EmptyFileError,
    ) as exc:
        raise PDFValidationError(
            "PDF is corrupted or could not be parsed."
        ) from exc

    except Exception as exc:
        raise PDFValidationError(
            f"PDF parsing failed: {type(exc).__name__}"
        ) from exc

    if not any(page.has_extractable_text for page in pages):
        raise PDFValidationError(
            "PDF contains no extractable text. "
            "It may be scanned or image-only. "
            "OCR is not currently supported."
        )

    return ParsedPDF(
        filename=path.name,
        document_id=document_id,
        page_count=len(pages),
        pages=pages,
    )
