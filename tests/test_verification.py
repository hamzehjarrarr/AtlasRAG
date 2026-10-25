from atlasrag.verification import verify_answer


def test_supported_claim():
    report = verify_answer(
        "The event occurred in 2027 [S1].",
        {"S1": "The event occurred in 2027."},
    )

    assert report.status == "checks_passed"


def test_unknown_citation():
    report = verify_answer(
        "The event occurred in 2027 [S9].",
        {"S1": "2027"},
    )

    assert "[S9]" in report.unknown_citations
    assert not report.passed


def test_uncited_claim():
    report = verify_answer(
        "The event occurred in 2027.",
        {"S1": "2027"},
    )

    assert len(report.uncited_claims) == 1


def test_wrong_percentage():
    report = verify_answer(
        "The company controlled 80% of capacity [S1].",
        {"S1": "40% of capacity"},
    )

    assert "80%" in report.numerical_issues[0]


def test_correct_percentage():
    report = verify_answer(
        "The company controlled 40% of capacity [S1].",
        {"S1": "40% of capacity"},
    )

    assert report.passed


def test_number_only_in_other_source():
    report = verify_answer(
        "Revenue grew by 80% [S1].",
        {
            "S1": "20%",
            "S2": "80%",
        },
    )

    assert report.numerical_issues


def test_refusal_without_citations():
    report = verify_answer(
        "The document does not contain enough "
        "information to answer this question.",
        {"S1": "AI"},
    )

    assert report.status == "insufficient_evidence"


def test_refusal_with_citations_real_response():
    answer = (
        "The retrieved evidence does not address "
        "why the sky is blue. "
        "Therefore, no claim can be supported by "
        "the provided documents. "
        "[S1], [S2], [S3], [S4], and [S5] "
        "discuss topics related to artificial "
        "intelligence, AI development, alignment, "
        "and geopolitical scenarios, but contain "
        "no information about the color of the sky. "
        "The evidence is insufficient to answer "
        "the question."
    )

    evidence = {
        f"S{i}": (
            "AI development and geopolitical scenarios"
        )
        for i in range(1, 6)
    }

    report = verify_answer(answer, evidence)

    assert report.status == "insufficient_evidence"
    assert len(report.cited_ids) == 5


def test_empty_evidence_refusal():
    report = verify_answer(
        "I could not find usable source text "
        "to answer this question.",
        {},
    )

    assert report.status == "insufficient_evidence"


def test_mixed_refusal_with_claim():
    report = verify_answer(
        "The document does not contain the answer.\n"
        "The event happened in 2027.",
        {},
    )

    assert report.status == "review_required"


def test_multiple_citations():
    report = verify_answer(
        "The event occurred in 2027 [S1][S2].",
        {
            "S1": "planning",
            "S2": "occurred in 2027",
        },
    )

    assert report.status == "checks_passed"


def test_empty_answer():
    report = verify_answer(
        "",
        {"S1": "anything"},
    )

    assert report.status == "review_required"


def test_identifiers_are_not_numbers():
    report = verify_answer(
        "Agent-2 uses GPT-4 with OC5 [S1].",
        {
            "S1": "Agent-2 uses GPT-4 with OC5."
        },
    )

    assert report.numerical_issues == []
    assert report.status == "checks_passed"


def test_real_dollar_cost_refusal():
    answer = (
        "The retrieved evidence does not specify "
        "the exact dollar cost of China's Agent-2 "
        "theft operation. "
        "[S1], [S2], and [S4] describe the technical "
        "and strategic aspects of the theft, including "
        "cyber operations, server compromises, and "
        "intelligence coordination, but none mention "
        "a financial cost. "
        "[S3] discusses a hypothetical $1B budget "
        "for an OC5-level effort, but this is not "
        "tied to the Agent-2 theft. "
        "Therefore, the exact dollar cost of the "
        "operation is not established in the evidence."
    )

    evidence = {
        "S1": "technical aspects",
        "S2": "cyber operations",
        "S3": (
            "hypothetical $1 billion budget "
            "for an OC5-level effort"
        ),
        "S4": "intelligence",
    }

    report = verify_answer(answer, evidence)

    assert report.numerical_issues == []
    assert report.status == "insufficient_evidence"


def test_citation_inherited_within_paragraph():
    answer = (
        "The theft was detected by a monitoring agent. "
        "It prompted additional security measures [S1]."
    )

    report = verify_answer(
        answer,
        {
            "S1": (
                "The theft was detected and prompted "
                "additional security."
            )
        },
    )

    assert report.status == "checks_passed"
    assert not report.uncited_claims
    assert report.inherited_citation_count == 1
    assert report.claims[0].inherited_citations == ["S1"]


def test_inherited_citation_still_checks_numbers():
    answer = (
        "OpenBrain runs 900,000 copies. "
        "They work in parallel [S1]."
    )

    report = verify_answer(
        answer,
        {"S1": "OpenBrain runs 200,000 copies in parallel."},
    )

    assert report.status == "review_required"
    assert "900000" in report.numerical_issues[0]


def test_real_agent3_answer_passes():
    answer = (
        "In March 2027, OpenBrain runs 200,000 Agent-3 "
        "copies in parallel. This workforce is equivalent "
        "to 50,000 copies of the best human coder sped "
        "up by 30x [S1]."
    )

    report = verify_answer(
        answer,
        {
            "S1": (
                "March 2027: Algorithmic Breakthroughs\n"
                "OpenBrain runs 200,000 Agent-3 copies in "
                "parallel, creating a workforce equivalent "
                "to 50,000 copies of the best human coder "
                "sped up by 30x."
            )
        },
    )

    assert report.status == "checks_passed"
    assert report.numerical_issues == []


def test_wrong_multiplier_is_flagged():
    report = verify_answer(
        "Coders are sped up by 300x [S1].",
        {"S1": "sped up by 30x."},
    )

    assert "300x" in report.numerical_issues[0]


def test_multiplier_symbol_and_word_are_equivalent():
    report = verify_answer(
        "Progress is 4× faster [S1].",
        {"S1": "progress speeds up by 4x"},
    )

    assert report.status == "checks_passed"


def test_trailing_citation_fragment_is_attached():
    report = verify_answer(
        "OpenBrain runs 200,000 copies. [S1]",
        {"S1": "OpenBrain runs 200,000 copies."},
    )

    assert report.status == "checks_passed"
    assert len(report.claims) == 1


def test_sentence_split_after_closing_quote():
    answer = (
        'The passage says "China has 40% of its '
        'AI-relevant compute in the CDZ." '
        "The share rose later [S1]."
    )

    report = verify_answer(
        answer,
        {"S1": "China has 40% of its AI-relevant compute in the CDZ."},
    )

    assert len(report.claims) == 2


def test_verbatim_quote_passes():
    report = verify_answer(
        'The document states that "China has 40% of its '
        'AI-relevant compute in the CDZ" [S1].',
        {
            "S1": (
                "But by now China has 40% of its "
                "AI-relevant compute† in the CDZ "
                "[footnote 28]."
            )
        },
    )

    assert report.quote_issues == []
    assert report.status == "checks_passed"


def test_fabricated_quote_is_flagged():
    report = verify_answer(
        'The document states that "China controls half '
        'of the world\'s compute" [S1].',
        {"S1": "China has 40% of its AI-relevant compute in the CDZ."},
    )

    assert report.quote_issues
    assert report.status == "review_required"


def test_uncited_conclusion_separate_paragraph():
    answer = (
        "The theft was detected [S1].\n\n"
        "Thus, it changed everything."
    )

    report = verify_answer(
        answer,
        {"S1": "The theft was detected."},
    )

    assert report.status == "review_required"
    assert len(report.uncited_claims) == 1


def test_real_numerical_answer_conclusion():
    answer = (
        "In February 2027, China has 40% of its "
        "AI-relevant compute in DeepCent [S4].\n\n"
        "Thus, the percentage of China's AI-relevant "
        "compute in DeepCent is 40%."
    )

    report = verify_answer(
        answer,
        {
            "S4": (
                "In February 2027 China has 40% "
                "of its AI-relevant compute in DeepCent."
            )
        },
    )

    assert report.numerical_issues == []
    assert report.status == "review_required"


def test_real_false_premise_response():
    answer = (
        "The evidence does not indicate that China "
        "stole Agent-3 in February 2027. "
        "The document states that China stole "
        "Agent-2 [S1]. "
        "There is no mention of a theft of Agent-3 "
        "occurring in February 2027."
    )

    report = verify_answer(
        answer,
        {
            "S1": (
                "China stole Agent-2 in February 2027."
            )
        },
    )

    assert report.status == "review_required"
    assert not report.numerical_issues


def test_uncited_hallucination_after_refusal():
    answer = (
        "The evidence is insufficient to answer "
        "the question. "
        "However, the company secretly spent "
        "$90 million."
    )

    report = verify_answer(
        answer,
        {"S1": "No costs provided."},
    )

    assert report.status == "review_required"
    assert report.uncited_claims


def test_currency_scale_equivalence():
    report = verify_answer(
        "The budget was $1B [S1].",
        {
            "S1": "The budget was $1 billion."
        },
    )

    assert report.status == "checks_passed"