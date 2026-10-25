from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "eval" / "evaluate.py"
spec = importlib.util.spec_from_file_location("evaluate", SCRIPT)
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


def test_times_matches_x():
    assert evaluate.normalize("4,000 times less") == evaluate.normalize("4,000x less")


def test_superscript_exponent_matches_caret():
    assert evaluate.normalize("10²⁷ FLOP") == evaluate.normalize("10^27 FLOP")


def test_footnote_marks_are_ignored():
    assert evaluate.normalize("compute† in the CDZ [footnote 28]") == (
        evaluate.normalize("compute in the CDZ")
    )


def test_retrieval_metrics():
    metrics = evaluate.retrieval_metrics([1, 2, None, 5], k=5)

    assert metrics["hit@1"] == 0.25
    assert metrics["hit@5"] == 0.75
    assert round(metrics["mrr"], 3) == round((1 + 0.5 + 0.2) / 4, 3)