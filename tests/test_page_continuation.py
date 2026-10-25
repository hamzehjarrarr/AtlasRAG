
from atlasrag.ingestion.chunking import (
    CHUNKING_VERSION,
    _TextUnit,
    _Section,
    _can_continue_across_pages,
    _reconnect_narrative_sections,
    _pack_units,
)


def test_uppercase_acronym_continuation():
    previous = _TextUnit(
        "China has 40% of its AI-relevant compute in the",
        10,
    )

    current = _TextUnit(
        "CDZ, where security was hardened.",
        11,
    )

    assert _can_continue_across_pages(
        previous,
        current,
    )


def test_uppercase_new_paragraph_not_joined():
    previous = _TextUnit(
        "The company developed a new",
        10,
    )

    current = _TextUnit(
        "System architecture was redesigned.",
        11,
    )

    assert not _can_continue_across_pages(
        previous,
        current,
    )


def test_completed_sentence_not_joined():
    previous = _TextUnit(
        "The operation was successful.",
        10,
    )

    current = _TextUnit(
        "CDZ security was upgraded.",
        11,
    )

    assert not _can_continue_across_pages(
        previous,
        current,
    )


def test_footnote_interrupted_continuation():
    sections = [
        _Section(
            title="February 2027: China Steals Agent-2",
            heading_level=1,
            kind="narrative",
            units=[
                _TextUnit(
                    "But by now China has 40% of its "
                    "AI-relevant compute in the",
                    10,
                )
            ],
        ),
        _Section(
            title=(
                "February 2027: China Steals Agent-2"
                " — Footnotes"
            ),
            heading_level=1,
            kind="footnote",
            units=[
                _TextUnit(
                    "29 Supporting footnote.",
                    10,
                )
            ],
        ),
        _Section(
            title="February 2027: China Steals Agent-2",
            heading_level=1,
            kind="narrative",
            units=[
                _TextUnit(
                    "CDZ, where security was hardened.",
                    11,
                )
            ],
        ),
    ]

    result = _reconnect_narrative_sections(sections)

    assert len(result) == 2

    narrative = result[0]
    footnotes = result[1]

    assert narrative.kind == "narrative"
    assert footnotes.kind == "footnote"

    assert len(narrative.units) == 1
    assert narrative.units[0].page_number == 10
    assert narrative.units[0].end_page_number == 11

    assert (
        "in the CDZ"
        in narrative.units[0].text
    )


def test_different_headings_not_reconnected():
    sections = [
        _Section(
            title="Section A",
            heading_level=1,
            kind="narrative",
            units=[
                _TextUnit("The work happened in the", 10)
            ],
        ),
        _Section(
            title="Section B",
            heading_level=1,
            kind="narrative",
            units=[
                _TextUnit("CDZ, where work continued.", 11)
            ],
        ),
    ]

    result = _reconnect_narrative_sections(sections)

    assert len(result) == 2


def test_reconnected_unit_preserves_page_range():
    units = [
        _TextUnit(
            "The work happened in the CDZ.",
            10,
            11,
        )
    ]

    packed = _pack_units(
        units,
        max_words=100,
    )

    assert len(packed) == 1
    assert packed[0][1] == 10
    assert packed[0][2] == 11


def test_version_updated():
    assert CHUNKING_VERSION == "structure-recursive-v5"
