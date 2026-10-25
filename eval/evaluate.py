"""
Evaluate AtlasRAG on a PDF with a labelled question set.

Retrieval (no LLM needed):
    For each answerable question, a passage counts as a hit
    when it contains the labelled evidence phrase. Hybrid
    retrieval alone is compared with hybrid + cross-encoder
    reranking (the configuration the app always uses).

Generation (optional, needs Ollama):
    Answer accuracy (every expected value appears in an
    answer that neither refuses nor hedges), verifier
    status, and refusal behaviour on questions the
    document cannot answer.

Usage:
    python eval/evaluate.py --pdf data/pdfs/ai-2027.pdf
    python eval/evaluate.py --pdf data/pdfs/ai-2027.pdf --generate

Compare prompts (a template with {context} and {question}):
    python eval/evaluate.py --pdf ... --generate --prompt eval/prompts/v4_minimal.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import atlasrag.generation as generation  # noqa: E402
from atlasrag.generation import DEFAULT_MODEL  # noqa: E402
from atlasrag.index import PDFIndex, PDFSource  # noqa: E402

SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def normalize(text: str) -> str:
    """Normalize text for evidence and answer matching."""
    text = re.sub(
        r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+",
        lambda m: "^" + m.group(0).translate(SUPERSCRIPTS),
        text,
    )
    text = re.sub(r"\[footnote \d+\]|[†‡*]", "", text)
    text = (
        text.replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("×", "x")
        .replace("\u2009", " ")
    )
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    text = re.sub(r"(?<=\d)\s*(?:times|-fold)\b", "x", text)
    text = re.sub(r"\s+", " ", text)
    return text.casefold().strip()


def first_hit(
    sources: list[PDFSource],
    evidence: str,
) -> int | None:
    """1-based rank of the first passage containing the evidence."""
    needle = normalize(evidence)

    for rank, source in enumerate(sources, start=1):
        if needle in normalize(source.text):
            return rank

    return None


def retrieval_metrics(ranks: list[int | None], k: int) -> dict:
    total = len(ranks)

    return {
        "hit@1": sum(r == 1 for r in ranks) / total,
        f"hit@{k}": sum(r is not None and r <= k for r in ranks) / total,
        "mrr": sum(1 / r for r in ranks if r) / total,
    }


def evaluate_retrieval(
    index: PDFIndex,
    questions: list[dict],
    k: int,
) -> dict:
    rows = []

    for item in questions:
        hybrid = index.retrieve(item["question"], k=k, rerank=False)
        reranked = index.retrieve(item["question"], k=k, rerank=True)

        rows.append(
            {
                "id": item["id"],
                "hybrid_rank": first_hit(hybrid, item["evidence"]),
                "reranked_rank": first_hit(reranked, item["evidence"]),
            }
        )

    return {
        "k": k,
        "hybrid": retrieval_metrics([r["hybrid_rank"] for r in rows], k),
        "reranked": retrieval_metrics([r["reranked_rank"] for r in rows], k),
        "per_question": rows,
    }


def evaluate_generation(
    index: PDFIndex,
    answerable: list[dict],
    unanswerable: list[dict],
    k: int,
    model: str,
) -> dict:
    rows = []
    latencies = []

    for item in answerable + unanswerable:
        start = time.perf_counter()
        result = index.answer(item["question"], k=k, model=model)
        latencies.append(time.perf_counter() - start)

        generated = result.generated
        report = generated.verification
        answer = normalize(generated.answer)
        expected = item.get("expected")

        refused = bool(report and report.is_refusal)
        hedged = bool(
            report
            and any("evidence-limitation" in w for w in report.warnings)
        )

        rows.append(
            {
                "id": item["id"],
                "answerable": expected is not None,
                "answer": generated.answer,
                # An answer only counts when it gives every expected
                # value without declining or hedging: a refusal that
                # mentions the right number is still a refusal.
                "correct": (
                    not refused
                    and not hedged
                    and all(normalize(value) in answer for value in expected)
                    if expected is not None
                    else None
                ),
                "refused": refused,
                "hedged": hedged,
                "status": report.status if report else None,
            }
        )

    answered = [r for r in rows if r["answerable"]]
    not_answerable = [r for r in rows if not r["answerable"]]

    return {
        "model": model,
        "answer_accuracy": sum(r["correct"] for r in answered) / len(answered),
        "false_refusals": sum(r["refused"] for r in answered) / len(answered),
        "checks_passed": (
            sum(r["status"] == "checks_passed" for r in answered) / len(answered)
        ),
        "refusal_rate_unanswerable": (
            sum(r["refused"] for r in not_answerable) / len(not_answerable)
            if not_answerable
            else None
        ),
        "median_latency_s": statistics.median(latencies),
        "per_question": rows,
    }


def use_prompt_template(template: str) -> None:
    """Swap the generation prompt for a template file."""
    if "{context}" not in template or "{question}" not in template:
        raise SystemExit("A prompt template needs {context} and {question}.")

    generation.build_prompt = lambda question, context: (
        template.replace("{context}", context).replace("{question}", question)
    )


def percent(value: float) -> str:
    return f"{value * 100:.0f}%"


def print_report(retrieval: dict, generation: dict | None) -> None:
    k = retrieval["k"]
    hybrid = retrieval["hybrid"]
    reranked = retrieval["reranked"]

    print("\n## Retrieval\n")
    print(f"| Configuration | Hit@1 | Hit@{k} | MRR |")
    print("|---|---|---|---|")
    print(
        f"| Hybrid (BM25S + BGE-M3, RRF) | {percent(hybrid['hit@1'])} "
        f"| {percent(hybrid[f'hit@{k}'])} | {hybrid['mrr']:.2f} |"
    )
    print(
        f"| Hybrid + cross-encoder rerank | {percent(reranked['hit@1'])} "
        f"| {percent(reranked[f'hit@{k}'])} | {reranked['mrr']:.2f} |"
    )

    misses = [
        row["id"]
        for row in retrieval["per_question"]
        if row["reranked_rank"] is None
    ]

    if misses:
        print(f"\nReranked misses: {', '.join(misses)}")

    if generation is None:
        return

    title = generation["model"]

    if generation.get("prompt"):
        title += f", prompt {generation['prompt']}"

    print(f"\n## Answers ({title})\n")
    print("| Metric | Result |")
    print("|---|---|")
    print(f"| Answer accuracy | {percent(generation['answer_accuracy'])} |")
    print(
        "| Verification checks passed (answerable) "
        f"| {percent(generation['checks_passed'])} |"
    )
    print(f"| False refusals | {percent(generation['false_refusals'])} |")

    if generation["refusal_rate_unanswerable"] is not None:
        print(
            "| Correct refusals (unanswerable) "
            f"| {percent(generation['refusal_rate_unanswerable'])} |"
        )

    print(f"| Median latency | {generation['median_latency_s']:.1f}s |")

    wrong = [
        row["id"]
        for row in generation["per_question"]
        if row["correct"] is False
        or (not row["answerable"] and not row["refused"])
    ]

    if wrong:
        print(f"\nNeeds review: {', '.join(wrong)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "eval" / "questions.json",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompt",
        type=Path,
        help="Prompt template to use instead of the app's built-in prompt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "eval" / "results.json",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    spec = json.loads(args.questions.read_text(encoding="utf-8"))

    if args.prompt:
        use_prompt_template(args.prompt.read_text(encoding="utf-8"))

    index = PDFIndex(args.pdf, display_name=spec.get("document"))
    index.build()

    try:
        retrieval = evaluate_retrieval(index, spec["answerable"], args.k)

        generation_results = (
            evaluate_generation(
                index,
                spec["answerable"],
                spec.get("unanswerable", []),
                args.k,
                args.model,
            )
            if args.generate
            else None
        )
    finally:
        index.close()

    if args.prompt and generation_results is not None:
        generation_results["prompt"] = args.prompt.name

    args.output.write_text(
        json.dumps(
            {"retrieval": retrieval, "generation": generation_results},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print_report(retrieval, generation_results)
    print(f"\nFull results: {args.output}")


if __name__ == "__main__":
    main()