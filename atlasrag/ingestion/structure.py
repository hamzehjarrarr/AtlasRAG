from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median

from atlasrag.ingestion.parser import ParsedPDF, PDFElement


@dataclass(frozen=True)
class StructuredElement:
    text: str
    page_number: int
    element_type: str
    font_size: float
    is_bold: bool
    bbox: tuple[float, float, float, float]
    heading_level: int | None = None


@dataclass(frozen=True)
class StructuredDocument:
    document_id: str
    filename: str
    body_font_size: float
    elements: list[StructuredElement]

    @property
    def headings(self) -> list[StructuredElement]:
        return [
            e for e in self.elements
            if e.element_type == "heading"
        ]

    @property
    def paragraphs(self) -> list[StructuredElement]:
        return [
            e for e in self.elements
            if e.element_type == "paragraph"
        ]

    @property
    def footnotes(self) -> list[StructuredElement]:
        return [
            e for e in self.elements
            if e.element_type == "footnote"
        ]


def _bbox(
    spans: list[PDFElement],
) -> tuple[float, float, float, float]:
    return (
        min(s.bbox[0] for s in spans),
        min(s.bbox[1] for s in spans),
        max(s.bbox[2] for s in spans),
        max(s.bbox[3] for s in spans),
    )


POWER_BASE = re.compile(r"(?:^|[\s(⋅·×*])10\s*$")


def _ends_with_power_base(text: str) -> bool:
    """
    True when text ends with a standalone "10",
    as in "trained with 10" or "2 ⋅10".
    """
    return bool(POWER_BASE.search(text))


def _combine_line_spans(
    elements: list[PDFElement],
) -> list[PDFElement]:
    groups = defaultdict(list)

    for span in elements:
        key = (
            span.page_number,
            span.block_number,
            span.line_number,
        )
        groups[key].append(span)

    lines = []

    for spans in groups.values():
        spans.sort(key=lambda s: s.span_number)

        sizes = [
            s.font_size
            for s in spans
            if s.text.strip()
        ]

        base_size = max(sizes, default=0.0)

        regular = [
            s for s in spans
            if s.font_size >= base_size * 0.78
        ]

        baseline = (
            median(s.bbox[3] for s in regular)
            if regular else 0.0
        )

        parts = []
        previous_span = None

        for span in spans:
            value = span.text

            superscript = (
                bool(
                    re.fullmatch(
                        r"\d{1,3}",
                        value.strip(),
                    )
                )
                and span.font_size <= base_size * 0.78
                and span.bbox[3] <
                baseline - base_size * 0.08
            )

            if superscript and _ends_with_power_base(
                "".join(parts)
            ):
                # Scientific notation such as 10^27:
                # a raised number directly after "10"
                # is an exponent, not a footnote marker.
                parts.append(f"^{value.strip()}")

            elif superscript:
                parts.append(
                    f" [footnote {value.strip()}] "
                )
            else:
                if previous_span is not None and parts:
                    prior = previous_span.text

                    if (
                        prior
                        and value
                        and prior[-1].isalnum()
                        and value[0].isalnum()
                        and not prior[-1].isspace()
                        and not value[0].isspace()
                        and span.bbox[0] - previous_span.bbox[2]
                        >= max(0.6, base_size * 0.075)
                    ):
                        parts.append(" ")

                parts.append(value)

            previous_span = span

        text = re.sub(
            r"\s+",
            " ",
            "".join(parts),
        ).strip()

        if not text:
            continue

        representative = (
            regular[0] if regular else spans[0]
        )

        lines.append(
            PDFElement(
                text=text,
                page_number=spans[0].page_number,
                block_number=spans[0].block_number,
                line_number=spans[0].line_number,
                span_number=0,
                font_size=representative.font_size,
                font_name=representative.font_name,
                is_bold=any(
                    s.is_bold for s in regular
                ),
                bbox=_bbox(spans),
            )
        )

    return lines


def _detect_repeated_boilerplate(
    document: ParsedPDF,
    page_lines: dict[int, list[PDFElement]],
) -> set[tuple[int, int, int]]:
    if document.page_count < 3:
        return set()

    groups = defaultdict(list)

    for page in document.pages:
        if not page.height:
            continue

        for line in page_lines.get(
            page.page_number, []
        ):
            if (
                line.bbox[3]
                <= page.height * 0.09
            ):
                zone = "top"

            elif (
                line.bbox[1]
                >= page.height * 0.91
            ):
                zone = "bottom"

            else:
                continue

            label = " ".join(
                line.text.casefold().split()
            )

            label = re.sub(
                r"^(?:page|p\.)\s*\d+$",
                "page #",
                label,
            )

            bucket = round(
                (
                    line.bbox[1]
                    + line.bbox[3]
                )
                / (2 * page.height)
                / 0.02
            )

            groups[
                (zone, bucket, label)
            ].append(line)

    required = max(
        3,
        int(
            document.page_count * 0.60
            + 0.999999
        ),
    )

    return {
        (
            line.page_number,
            line.block_number,
            line.line_number,
        )
        for group in groups.values()
        if len({
            line.page_number
            for line in group
        }) >= required
        for line in group
    }


def _detect_running_section_headers(
    document: ParsedPDF,
    page_lines: dict[int, list[PDFElement]],
) -> set[tuple[int, int, int]]:
    """Remove repeated top-of-page section labels, not actual headings."""
    excluded = set()
    pages = sorted(
        document.pages,
        key=lambda page: page.page_number,
    )

    for previous_page, current_page in zip(
        pages,
        pages[1:],
        strict=False,
    ):
        if (
            not previous_page.height
            or not current_page.height
        ):
            continue

        previous_lines = [
            line
            for line in page_lines.get(
                previous_page.page_number, []
            )
            if (
                line.bbox[1]
                < previous_page.height * 0.085
                and len(line.text.split()) <= 14
            )
        ]

        current_lines = [
            line
            for line in page_lines.get(
                current_page.page_number, []
            )
            if (
                line.bbox[1]
                < current_page.height * 0.085
                and len(line.text.split()) <= 14
            )
        ]

        for left in previous_lines:
            for right in current_lines:
                if (
                    left.text.casefold().strip()
                    == right.text.casefold().strip()
                    and abs(
                        left.bbox[1] - right.bbox[1]
                    ) <= 3
                    and abs(
                        left.bbox[0] - right.bbox[0]
                    ) <= 5
                    and abs(
                        left.font_size - right.font_size
                    ) <= 0.5
                ):
                    excluded.add(
                        (
                            left.page_number,
                            left.block_number,
                            left.line_number,
                        )
                    )
                    excluded.add(
                        (
                            right.page_number,
                            right.block_number,
                            right.line_number,
                        )
                    )

    return excluded


def _looks_like_heading(
    line: PDFElement,
    body_size: float,
) -> bool:
    text = line.text.strip()

    if (
        not text
        or len(text) > 140
        or len(text.split()) > 18
    ):
        return False

    if text.endswith(
        (".", "!", "?", ";")
    ):
        return False

    if re.match(r"^\d+\s", text):
        return False

    large = (
        line.font_size >= body_size * 1.20
        and line.font_size - body_size >= 1.5
    )

    known = bool(
        re.match(
            r"^(?:appendix|race ending|"
            r"slowdown ending|introduction|"
            r"conclusion|references|"
            r"risk factors|executive summary)\b",
            text,
            re.I,
        )
    )

    return (
        large
        or (
            line.is_bold
            and (
                line.font_size > body_size
                or known
            )
        )
    )


def _split_by_region(
    lines: list[PDFElement],
    page_width: float,
) -> tuple[
    list[PDFElement],
    list[PDFElement],
]:
    if page_width <= 0:
        return lines, []

    boundary = page_width * 0.65

    right = [
        line
        for line in lines
        if line.bbox[0] >= boundary
    ]

    left = [
        line
        for line in lines
        if line.bbox[0] < boundary
    ]

    if len(right) < 3 or not left:
        return lines, []

    right_width = median(
        line.bbox[2] - line.bbox[0]
        for line in right
    )

    left_width = median(
        line.bbox[2] - line.bbox[0]
        for line in left
    )

    if (
        right_width > page_width * 0.34
        or left_width < page_width * 0.30
    ):
        return lines, []

    return left, right


def _join_text(
    previous: str,
    current: str,
) -> str:
    previous = previous.rstrip()
    current = current.lstrip()

    if not previous:
        return current

    if (
        previous.endswith("-")
        and len(previous) >= 2
        and previous[-2].isalpha()
        and current[:1].islower()
    ):
        return previous[:-1] + current

    return previous + " " + current


def _can_merge(
    previous: PDFElement,
    current: PDFElement,
) -> bool:
    if (
        previous.page_number
        != current.page_number
    ):
        return False

    if (
        abs(
            previous.font_size
            - current.font_size
        ) > 1.0
    ):
        return False

    height = max(
        previous.bbox[3]
        - previous.bbox[1],
        1.0,
    )

    gap = (
        current.bbox[1]
        - previous.bbox[3]
    )

    if (
        gap < -height * 0.5
        or gap > height * 0.9
    ):
        return False

    if (
        abs(
            current.bbox[1]
            - previous.bbox[1]
        ) < height * 0.5
    ):
        return False

    overlap = max(
        0.0,
        min(
            previous.bbox[2],
            current.bbox[2],
        )
        - max(
            previous.bbox[0],
            current.bbox[0],
        ),
    )

    width = max(
        min(
            previous.bbox[2]
            - previous.bbox[0],
            current.bbox[2]
            - current.bbox[0],
        ),
        1.0,
    )

    if overlap / width >= 0.25:
        return True

    return (
        gap <= height * 0.5
        and not previous.text.rstrip().endswith(
            (".", "!", "?", ":")
        )
    )


def _make_element(
    lines: list[PDFElement],
    kind: str,
    level: int | None = None,
) -> StructuredElement:
    text = ""

    for line in lines:
        text = _join_text(
            text,
            line.text,
        )

    return StructuredElement(
        text=text,
        page_number=lines[0].page_number,
        element_type=kind,
        font_size=median(
            line.font_size
            for line in lines
        ),
        is_bold=all(
            line.is_bold
            for line in lines
        ),
        bbox=_bbox(lines),
        heading_level=level,
    )


def _process_region(
    lines: list[PDFElement],
    body_size: float,
    *,
    region_type: str,
    cover_page: bool = False,
) -> list[StructuredElement]:
    result = []
    pending = []

    def flush() -> None:
        if pending:
            result.append(
                _make_element(
                    pending,
                    region_type,
                )
            )
            pending.clear()

    for line in lines:
        is_heading = (
            region_type == "paragraph"
            and not cover_page
            and _looks_like_heading(
                line,
                body_size,
            )
        )

        if is_heading:
            flush()

            level = (
                1
                if line.font_size
                >= body_size * 1.5
                else 2
            )

            if (
                result
                and result[-1].element_type
                == "heading"
                and result[-1].page_number
                == line.page_number
                and abs(
                    result[-1].font_size
                    - line.font_size
                ) <= 1
                and abs(
                    result[-1].bbox[0]
                    - line.bbox[0]
                ) <= 18
                and 0 <= (
                    line.bbox[1]
                    - result[-1].bbox[3]
                ) <= 15
            ):
                old = result.pop()

                result.append(
                    StructuredElement(
                        text=_join_text(
                            old.text,
                            line.text,
                        ),
                        page_number=line.page_number,
                        element_type="heading",
                        font_size=old.font_size,
                        is_bold=(
                            old.is_bold
                            and line.is_bold
                        ),
                        bbox=(
                            min(
                                old.bbox[0],
                                line.bbox[0],
                            ),
                            min(
                                old.bbox[1],
                                line.bbox[1],
                            ),
                            max(
                                old.bbox[2],
                                line.bbox[2],
                            ),
                            max(
                                old.bbox[3],
                                line.bbox[3],
                            ),
                        ),
                        heading_level=(
                            old.heading_level
                        ),
                    )
                )
            else:
                result.append(
                    _make_element(
                        [line],
                        "heading",
                        level,
                    )
                )

            continue

        if (
            pending
            and not _can_merge(
                pending[-1],
                line,
            )
        ):
            flush()

        pending.append(line)

    flush()
    return result


def detect_structure(
    document: ParsedPDF,
) -> StructuredDocument:
    page_lines = {
        page.page_number:
        _combine_line_spans(page.elements)
        for page in document.pages
    }

    excluded = (
        _detect_repeated_boilerplate(
            document,
            page_lines,
        )
        | _detect_running_section_headers(
            document,
            page_lines,
        )
    )

    cleaned = {
        page.page_number: [
            line
            for line in page_lines[
                page.page_number
            ]
            if (
                line.page_number,
                line.block_number,
                line.line_number,
            ) not in excluded
        ]
        for page in document.pages
    }

    sizes = Counter()

    for lines in cleaned.values():
        for line in lines:
            sizes[
                round(
                    line.font_size,
                    1,
                )
            ] += len(
                line.text.strip()
            )

    body_size = (
        sizes.most_common(1)[0][0]
        if sizes else 11.0
    )

    result = []

    for page in document.pages:
        main, notes = _split_by_region(
            cleaned[page.page_number],
            page.width or 0,
        )

        cover = (
            page.page_number == 1
            and sum(
                line.font_size
                >= body_size * 1.15
                and len(
                    line.text.split()
                ) <= 5
                for line in main
            ) >= 5
        )

        narrative_lines = []
        bottom_note_lines = []
        in_bottom_notes = False

        for line in main:
            small_bottom = (
                page.height is not None
                and line.bbox[1]
                >= page.height * 0.52
                and line.font_size
                <= body_size * 0.90
            )

            starts_note = bool(
                re.match(
                    r"^[*†‡]",
                    line.text.strip(),
                )
            )

            if (
                small_bottom
                and (
                    starts_note
                    or in_bottom_notes
                )
            ):
                in_bottom_notes = True
                bottom_note_lines.append(line)
            else:
                narrative_lines.append(line)

        result.extend(
            _process_region(
                narrative_lines,
                body_size,
                region_type="paragraph",
                cover_page=cover,
            )
        )

        result.extend(
            _process_region(
                bottom_note_lines,
                body_size,
                region_type="footnote",
            )
        )

        result.extend(
            _process_region(
                notes,
                body_size,
                region_type="footnote",
            )
        )

    return StructuredDocument(
        document_id=document.document_id,
        filename=document.filename,
        body_font_size=body_size,
        elements=result,
    )