from __future__ import annotations

from atlasrag.generation import (
    MAX_SOURCES,
    build_context,
    build_prompt,
    clean_answer,
)


def test_clean_answer_strips_leading_label():
    assert clean_answer("Final Answer: 40% [S1].") == "40% [S1]."


def test_clean_answer_strips_bold_label():
    assert clean_answer("**Answer:** 40% [S1].") == "40% [S1]."


def test_clean_answer_strips_closing_label():
    raw = (
        "The evidence is insufficient.\n"
        "Final Answer: The evidence is insufficient."
    )

    cleaned = clean_answer(raw)

    assert "Final Answer" not in cleaned
    assert cleaned.endswith("The evidence is insufficient.")


def test_clean_answer_removes_think_block():
    raw = "<think>hidden reasoning</think>\nIt is 40% [S1]."

    assert clean_answer(raw) == "It is 40% [S1]."


def test_clean_answer_keeps_normal_text():
    text = "Answering this requires page 12 [S1]."

    assert clean_answer(text) == text


def test_build_context_assigns_ids_in_rank_order():
    corpus = {
        "a": {"title": "Page 1", "text": "Alpha"},
        "b": {"title": "Page 2", "text": "Beta"},
    }

    context, citation_map = build_context(
        [("b", 0.9), ("a", 0.5)],
        corpus,
    )

    assert citation_map == {"S1": "b", "S2": "a"}
    assert context.index("[S1]") < context.index("Beta")


def test_build_context_respects_source_limit():
    corpus = {
        str(i): {"title": "", "text": f"Passage {i}"}
        for i in range(10)
    }

    _context, citation_map = build_context(
        [(str(i), 1.0) for i in range(10)],
        corpus,
    )

    assert len(citation_map) == MAX_SOURCES


def test_build_context_skips_empty_documents():
    corpus = {
        "empty": {"title": "", "text": "   "},
        "full": {"title": "", "text": "Content"},
    }

    _context, citation_map = build_context(
        [("empty", 1.0), ("full", 0.5)],
        corpus,
    )

    assert citation_map == {"S1": "full"}


def test_prompt_contains_question_and_evidence():
    prompt = build_prompt("What is X?", "[S1]\nX is 5.")

    assert "What is X?" in prompt
    assert "[S1]\nX is 5." in prompt
    assert "Final Answer:" in prompt  # only inside the "do not" rule
    assert prompt.rstrip().endswith("What is X?")