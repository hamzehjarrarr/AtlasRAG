
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from atlasrag.ingestion.parser import ParsedPDF
from atlasrag.ingestion.structure import detect_structure


CHUNKING_VERSION = "structure-recursive-v5"

SENTENCE_BOUNDARY = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z0-9“"\'(\[])'
)

FOOTNOTE_REFERENCE = re.compile(
    r"\[footnote\s+\d+\]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    end_page_number: int | None = None
    section_title: str | None = None
    heading_level: int | None = None


@dataclass(frozen=True)
class _TextUnit:
    text: str
    page_number: int
    end_page_number: int | None = None


@dataclass(frozen=True)
class _Section:
    title: str | None
    heading_level: int | None
    units: list[_TextUnit]
    kind: str = "narrative"


def _word_count(text: str) -> int:
    return len(text.split())


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in SENTENCE_BOUNDARY.split(text)
        if part.strip()
    ]


def _pack_text_parts(
    parts: list[str],
    max_words: int,
) -> list[str]:
    result: list[str] = []
    current: list[str] = []

    for part in parts:
        if _word_count(part) > max_words:
            if current:
                result.append(" ".join(current))
                current = []

            result.extend(
                _split_long_text(part, max_words)
            )
            continue

        candidate = " ".join(current + [part])

        if current and _word_count(candidate) > max_words:
            result.append(" ".join(current))
            current = [part]
        else:
            current.append(part)

    if current:
        result.append(" ".join(current))

    return result


def _split_long_text(
    text: str,
    max_words: int,
) -> list[str]:
    if max_words <= 0:
        raise ValueError("max_words must be positive.")

    text = _normalize_text(text)

    if not text:
        return []

    if _word_count(text) <= max_words:
        return [text]

    paragraphs = [
        part.strip()
        for part in re.split(r"\n\s*\n", text)
        if part.strip()
    ]

    if len(paragraphs) > 1:
        return _pack_text_parts(paragraphs, max_words)

    sentences = _split_sentences(text)

    if len(sentences) > 1:
        return _pack_text_parts(sentences, max_words)

    words = text.split()

    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


def _last_page(unit: _TextUnit) -> int:
    return (
        unit.end_page_number
        if unit.end_page_number is not None
        else unit.page_number
    )


def _ends_sentence(text: str) -> bool:
    cleaned = FOOTNOTE_REFERENCE.sub("", text).rstrip()
    cleaned = cleaned.rstrip('”"\')]} ')

    return cleaned.endswith((".", "!", "?", ":"))


def _can_continue_across_pages(
    previous: _TextUnit,
    current: _TextUnit,
) -> bool:
    """
    Detect conservative cross-page continuations.

    Supports lowercase continuations and uppercase
    acronyms, but rejects completed sentences and
    ordinary uppercase paragraph starts.
    """
    if current.page_number != _last_page(previous) + 1:
        return False

    before = previous.text.strip()
    after = current.text.strip()

    if not before or not after:
        return False

    if _ends_sentence(before):
        return False

    if before.endswith(("-", "—")):
        return False

    if re.match(r"^\[footnote\s+\d+\]", after, re.I):
        return False

    if re.match(r"^[a-z]", after):
        return True

    # A short all-capitals token can be the continuation
    # of an unfinished sentence, as with "in the CDZ".
    acronym = re.match(
        r"^([A-Z][A-Z0-9]{1,9})(?=[\s,.;:!?)]|$)",
        after,
    )

    if not acronym:
        return False

    # Require the preceding text to end with a
    # grammatical connector. This reduces accidental
    # joins with unrelated uppercase headings.
    return bool(
        re.search(
            r"\b(?:the|a|an|in|into|of|to|for|from|"
            r"at|by|with|within|on|and|or)\s*$",
            before,
            re.IGNORECASE,
        )
    )


def _merge_page_continuations(
    units: list[_TextUnit],
) -> list[_TextUnit]:
    merged: list[_TextUnit] = []

    for unit in units:
        if (
            merged
            and _can_continue_across_pages(
                merged[-1],
                unit,
            )
        ):
            previous = merged.pop()

            merged.append(
                _TextUnit(
                    text=(
                        previous.text.rstrip()
                        + " "
                        + unit.text.lstrip()
                    ),
                    page_number=previous.page_number,
                    end_page_number=_last_page(unit),
                )
            )
        else:
            merged.append(unit)

    return merged


def _pack_units(
    units: list[_TextUnit],
    max_words: int,
) -> list[tuple[str, int, int]]:
    result: list[tuple[str, int, int]] = []

    current: list[str] = []
    start_page: int | None = None
    end_page: int | None = None

    def flush() -> None:
        nonlocal current, start_page, end_page

        if not current:
            return

        assert start_page is not None
        assert end_page is not None

        result.append(
            (
                "\n\n".join(current),
                start_page,
                end_page,
            )
        )

        current = []
        start_page = None
        end_page = None

    for unit in units:
        pieces = _split_long_text(
            unit.text,
            max_words,
        )

        for piece in pieces:
            candidate = "\n\n".join(
                current + [piece]
            )

            if (
                current
                and _word_count(candidate) > max_words
            ):
                flush()

            if not current:
                start_page = unit.page_number

            current.append(piece)
            end_page = _last_page(unit)

    flush()
    return result


def _raw_page_sections(
    document: ParsedPDF,
) -> list[_Section]:
    return [
        _Section(
            title=None,
            heading_level=None,
            units=[
                _TextUnit(
                    text=page.text,
                    page_number=page.page_number,
                )
            ],
        )
        for page in document.pages
        if page.text.strip()
    ]


def _reconnect_narrative_sections(
    sections: list[_Section],
) -> list[_Section]:
    """
    Reconnect narrative across a page boundary when
    intervening footnotes interrupted the extraction.

    Footnote sections are retained separately.

    Only sections sharing the same heading are eligible.
    """
    result: list[_Section] = []

    for section in sections:
        if section.kind != "narrative" or not section.units:
            result.append(section)
            continue

        previous_index = None

        for index in range(len(result) - 1, -1, -1):
            candidate = result[index]

            if candidate.kind == "footnote":
                continue

            previous_index = index
            break

        if previous_index is None:
            result.append(section)
            continue

        previous = result[previous_index]

        if (
            previous.title != section.title
            or previous.heading_level != section.heading_level
            or not previous.units
        ):
            result.append(section)
            continue

        # The sections must be adjacent in document
        # narrative order, with at most footnotes
        # separating them.
        intervening = result[previous_index + 1:]

        if any(
            item.kind != "footnote"
            for item in intervening
        ):
            result.append(section)
            continue

        left = previous.units[-1]
        right = section.units[0]

        if not _can_continue_across_pages(left, right):
            result.append(section)
            continue

        combined = _TextUnit(
            text=left.text.rstrip() + " " + right.text.lstrip(),
            page_number=left.page_number,
            end_page_number=_last_page(right),
        )

        updated_units = (
            previous.units[:-1]
            + [combined]
            + section.units[1:]
        )

        result[previous_index] = _Section(
            title=previous.title,
            heading_level=previous.heading_level,
            units=_merge_page_continuations(updated_units),
            kind="narrative",
        )

    return result


def _build_sections(
    document: ParsedPDF,
) -> list[_Section]:
    if not any(page.elements for page in document.pages):
        return _raw_page_sections(document)

    structured = detect_structure(document)

    if not structured.elements:
        return []

    sections: list[_Section] = []

    active_heading: str | None = None
    active_level: int | None = None

    current_kind: str | None = None
    current_title: str | None = None
    current_level: int | None = None
    current_units: list[_TextUnit] = []
    current_note_page: int | None = None

    def flush() -> None:
        nonlocal current_units

        if not current_units:
            return

        units = current_units

        if current_kind == "narrative":
            units = _merge_page_continuations(units)

        sections.append(
            _Section(
                title=current_title,
                heading_level=current_level,
                units=units,
                kind=current_kind or "narrative",
            )
        )

        current_units = []

    for element in structured.elements:
        if element.element_type == "heading":
            heading = _normalize_text(element.text)

            # Repeated running headings should not
            # create a new logical section.
            if (
                heading == active_heading
                and element.heading_level == active_level
            ):
                continue

            flush()

            active_heading = heading
            active_level = element.heading_level

            current_kind = None
            current_title = None
            current_level = None
            current_note_page = None
            continue

        text = _normalize_text(element.text)

        if not text:
            continue

        kind = (
            "footnote"
            if element.element_type == "footnote"
            else "narrative"
        )

        note_page = (
            element.page_number
            if kind == "footnote"
            else None
        )

        if (
            current_kind != kind
            or (
                kind == "footnote"
                and current_note_page != note_page
            )
        ):
            flush()

            current_kind = kind
            current_note_page = note_page

            if kind == "footnote":
                current_title = (
                    f"{active_heading} — Footnotes"
                    if active_heading
                    else "Footnotes"
                )
            else:
                current_title = active_heading

            current_level = active_level

        current_units.append(
            _TextUnit(
                text=text,
                page_number=element.page_number,
            )
        )

    flush()

    return _reconnect_narrative_sections(sections)


def chunk_pdf(
    document: ParsedPDF,
    chunk_size: int = 250,
    overlap: int = 0,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap != 0:
        raise ValueError(
            "Overlap is not implemented. Use overlap=0."
        )

    sections = _build_sections(document)
    chunks: list[DocumentChunk] = []

    for section in sections:
        packed = _pack_units(
            section.units,
            chunk_size,
        )

        for text, start_page, end_page in packed:
            chunk_index = len(chunks)

            identity = "\n".join(
                [
                    document.document_id,
                    str(chunk_index),
                    str(start_page),
                    str(end_page),
                    str(chunk_size),
                    section.kind,
                    section.title or "",
                    text,
                    CHUNKING_VERSION,
                ]
            )

            chunk_id = hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    filename=document.filename,
                    page_number=start_page,
                    end_page_number=end_page,
                    chunk_index=chunk_index,
                    text=text,
                    section_title=section.title,
                    heading_level=section.heading_level,
                )
            )

    return chunks
