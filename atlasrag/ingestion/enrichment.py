
from __future__ import annotations

from dataclasses import dataclass
import re

from atlasrag.ingestion.chunking import DocumentChunk


MAX_METADATA_LENGTH = 200


@dataclass(frozen=True)
class ContextualChunk:
    """
    A chunk with separate evidence and
    retrieval representations.

    text:
        Extracted passage used for citations.

    retrieval_text:
        Context-enriched text used for
        lexical and dense retrieval.
    """

    chunk_id: str
    document_id: str
    filename: str

    page_number: int
    end_page_number: int

    chunk_index: int

    text: str
    retrieval_text: str

    section_title: str | None = None
    heading_level: int | None = None


def _clean_metadata(
    value: str | None,
) -> str | None:
    if not value:
        return None

    cleaned = re.sub(
        r"[\x00-\x1f\x7f]+",
        " ",
        value,
    )

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    ).strip()

    if not cleaned:
        return None

    if len(cleaned) > MAX_METADATA_LENGTH:
        cleaned = cleaned[
            :MAX_METADATA_LENGTH
        ].rstrip()

    return cleaned


def _is_footnote(
    chunk: DocumentChunk,
) -> bool:
    """
    Footnote chunks produced by the current
    chunker use a section title ending with
    '— Footnotes' or exactly 'Footnotes'.

    This avoids adding a new required field
    to existing serialized chunk models.
    """
    section = (
        chunk.section_title or ""
    ).strip()

    return bool(
        re.search(
            r"(?:^|[—–-]\s*)Footnotes$",
            section,
            re.IGNORECASE,
        )
    )


def enrich_chunk(
    chunk: DocumentChunk,
    document_title: str | None = None,
) -> ContextualChunk:
    """
    Add deterministic retrieval metadata
    without changing the cited evidence.

    No LLM calls or external services.
    """
    title = _clean_metadata(
        document_title
    )

    filename = (
        _clean_metadata(chunk.filename)
        or "Unknown document"
    )

    if title is None:
        title = filename

    section = _clean_metadata(
        chunk.section_title
    )

    start_page = chunk.page_number

    end_page = (
        chunk.end_page_number
        if chunk.end_page_number is not None
        else start_page
    )

    if (
        start_page < 1
        or end_page < start_page
    ):
        raise ValueError(
            "Invalid chunk page range."
        )

    content_type = (
        "Footnote"
        if _is_footnote(chunk)
        else "Narrative"
    )

    metadata_lines = [
        f"Document: {title}",
        f"Content type: {content_type}",
    ]

    if section:
        metadata_lines.append(
            f"Section: {section}"
        )

    if start_page == end_page:
        metadata_lines.append(
            f"Page: {start_page}"
        )
    else:
        metadata_lines.append(
            f"Pages: {start_page}-{end_page}"
        )

    retrieval_text = (
        "\n".join(metadata_lines)
        + "\n\n"
        + chunk.text.strip()
    )

    return ContextualChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        filename=chunk.filename,
        page_number=start_page,
        end_page_number=end_page,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        retrieval_text=retrieval_text,
        section_title=section,
        heading_level=chunk.heading_level,
    )


def enrich_chunks(
    chunks: list[DocumentChunk],
    document_title: str | None = None,
) -> list[ContextualChunk]:
    """
    Enrich chunks without changing their order.
    """
    return [
        enrich_chunk(
            chunk,
            document_title=document_title,
        )
        for chunk in chunks
    ]
