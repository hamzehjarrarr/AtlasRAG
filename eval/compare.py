"""
Compare several evaluation runs side by side.

Usage:
    python eval/compare.py eval/results-v1.json eval/results-v2.json ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def percent(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.0f}%"


def main() -> None:
    paths = [Path(arg) for arg in sys.argv[1:]]

    if not paths:
        raise SystemExit(__doc__)

    print(
        "| Run | Accuracy | Checks passed | False refusals "
        "| Correct refusals | Median latency |"
    )
    print("|---|---|---|---|---|---|")

    for path in paths:
        generation = json.loads(path.read_text(encoding="utf-8"))["generation"]

        if generation is None:
            print(f"| {path.stem} | no answers in this run (use --generate) |||||")
            continue

        name = generation.get("prompt") or path.stem

        print(
            f"| {name} "
            f"| {percent(generation['answer_accuracy'])} "
            f"| {percent(generation['checks_passed'])} "
            f"| {percent(generation['false_refusals'])} "
            f"| {percent(generation['refusal_rate_unanswerable'])} "
            f"| {generation['median_latency_s']:.1f}s |"
        )

    print("\nFailed questions per run:")

    for path in paths:
        generation = json.loads(path.read_text(encoding="utf-8"))["generation"]

        if generation is None:
            continue

        failed = [
            row["id"]
            for row in generation["per_question"]
            if row["correct"] is False
            or (not row["answerable"] and not row["refused"])
        ]

        name = generation.get("prompt") or path.stem
        print(f"  {name}: {', '.join(failed) or 'none'}")


if __name__ == "__main__":
    main()