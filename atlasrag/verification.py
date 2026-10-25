from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

CITATION_PATTERN = re.compile(r"\[(S\d+)\]")
# Split after terminal punctuation, including
# punctuation followed by a closing quote or bracket:
#   ...in the CDZ." The passage...
SENTENCE_SPLIT = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"”’)]))"
    r"\s+(?=[A-Z0-9\[\"“(])"
)
BULLET_PREFIX = re.compile(
    r"^\s*(?:[-*]\s+|\d+[.)]\s+)"
)

# Exclude model names and alphanumeric identifiers:
# Agent-2, Agent-3, GPT-4, OC5, etc.
# Multipliers such as 30x, 4×, or "50 times" are kept.
NUMBER_PATTERN = re.compile(
    r"(?<![\w.-])"
    r"(?P<currency>[$€£])?"
    r"(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"(?:\s*(?P<scale>billion|million|thousand|bn|mn|k|m|b))?"
    r"(?P<times>\s*[x×]|\s+times)?"
    r"(?P<percent>\s*%|\s+percent)?"
    r"(?![\w-])",
    re.IGNORECASE,
)

# Direct quotations long enough to be meaningful.
QUOTE_PATTERN = re.compile(
    r"[\"“]([^\"“”]{12,})[\"”]"
)

LIMITATION_PATTERNS = (
    re.compile(
        r"\b(?:retrieved |provided )?"
        r"(?:evidence|sources?|documents?|context)\b"
        r".{0,100}\b"
        r"(?:does not|doesn't|do not|don't|cannot|can't|"
        r"fails? to|is insufficient|are insufficient|"
        r"is not enough|are not enough)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:no|not enough|insufficient)\s+"
        r"(?:relevant\s+)?"
        r"(?:evidence|information|context|support|"
        r"claim|record|mention|details?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:cannot|can't|unable to|could not)\s+"
        r"(?:answer|determine|verify|establish|"
        r"confirm|find)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:not|never)\s+"
        r"(?:specified|stated|mentioned|provided|"
        r"established|supported|addressed|discussed|"
        r"recorded)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:uncertain|unknown|unclear)\s+"
        r"(?:whether|if)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:there is|there are)\s+no\s+"
        r"(?:mention|record|information|evidence|details?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno claim can be supported\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:does not|doesn't) support the claim\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bcould not find usable source text\b",
        re.IGNORECASE,
    ),
)

EVIDENCE_SUBJECT = re.compile(
    r"\b(?:evidence|documents?|sources?|passages?|"
    r"context|retrieved|provided|no claim|"
    r"there is no|there are no|it is uncertain|"
    r"it is unclear|the answer|the question|"
    r"information)\b",
    re.IGNORECASE,
)

SOURCE_DESCRIPTION = re.compile(
    r"^\s*\[S\d+\]"
    r"(?:\s*,\s*\[S\d+\])*"
    r"(?:\s*,?\s*and\s*\[S\d+\])?\s+"
    r"(?:discuss(?:es)?|describ(?:e|es)|"
    r"cover(?:s)?|concern(?:s)?|mention(?:s)?|"
    r"address(?:es)?|focus(?:es)? on)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClaimCheck:
    text: str
    citations: list[str]
    missing_numbers: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    inherited_citations: list[str] = field(default_factory=list)
    unverified_quotes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VerificationReport:
    status: str
    is_refusal: bool
    cited_ids: list[str]
    unknown_citations: list[str]
    uncited_claims: list[str]
    numerical_issues: list[str]
    claims: list[ClaimCheck]
    warnings: list[str]
    quote_issues: list[str] = field(default_factory=list)
    inherited_citation_count: int = 0

    @property
    def passed(self) -> bool:
        return self.status in {
            "checks_passed",
            "insufficient_evidence",
        }


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _numbers(text: str) -> list[str]:
    """
    Extract quantities while ignoring source IDs
    and numbers embedded in model identifiers.

    Common scales are normalized:
    $1B == $1 billion
    """
    text = CITATION_PATTERN.sub("", text)
    found = []

    scales = {
        "k": "thousand",
        "m": "million",
        "mn": "million",
        "b": "billion",
        "bn": "billion",
    }

    for match in NUMBER_PATTERN.finditer(text):
        value = match.group("number").replace(",", "")

        scale = (
            match.group("scale") or ""
        ).lower()

        scale = scales.get(scale, scale)

        if scale:
            value = f"{value}:{scale}"

        currency = match.group("currency") or ""

        times = (
            "x"
            if match.group("times")
            else ""
        )

        percent = (
            "%"
            if match.group("percent")
            else ""
        )

        found.append(
            f"{currency}{value}{times}{percent}"
        )

    return _unique(found)


FOOTNOTE_MARK = re.compile(
    r"\[footnote \d+\]|[†‡§¶*]"
)


def _normalize_quote(text: str) -> str:
    """
    Normalize text for verbatim quote matching:
    footnote marks, case, whitespace, curly quotes,
    and dashes.
    """
    text = FOOTNOTE_MARK.sub("", text)
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("—", "-")
        .replace("–", "-")
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip().strip(".,;:!?").casefold()


def _quotes(text: str) -> list[str]:
    return _unique(
        [
            match.group(1).strip()
            for match in QUOTE_PATTERN.finditer(text)
        ]
    )


def _limitation_statement(text: str) -> bool:
    """
    Identify statements explicitly describing
    limitations of the retrieved evidence.
    """
    plain = CITATION_PATTERN.sub(
        "",
        text,
    ).strip()

    if not EVIDENCE_SUBJECT.search(plain):
        return False

    return any(
        pattern.search(plain)
        for pattern in LIMITATION_PATTERNS
    )


def _source_description(text: str) -> bool:
    """
    Recognize sentences that describe the
    topics covered by cited sources.
    """
    return bool(
        SOURCE_DESCRIPTION.match(text.strip())
    )


def _paragraph_claims(
    answer: str,
) -> list[list[str]]:
    """
    Split the answer into paragraphs, then
    into individual statements.

    Paragraph boundaries are kept so an uncited
    sentence can inherit a citation from its own
    paragraph, but never from another one.
    """
    paragraphs = []

    for paragraph in re.split(
        r"\n\s*\n",
        answer,
    ):
        sentences = []

        for line in paragraph.splitlines():
            line = BULLET_PREFIX.sub(
                "",
                line,
            ).strip()

            if not line:
                continue

            for part in SENTENCE_SPLIT.split(line):
                part = part.strip()

                if not part:
                    continue

                if CITATION_PATTERN.sub(
                    "",
                    part,
                ).strip(" ,.;"):
                    sentences.append(part)

                elif sentences:
                    # A citation-only fragment such as
                    # "...sped up by 30x. [S1]" belongs to
                    # the preceding sentence.
                    sentences[-1] = (
                        f"{sentences[-1]} {part}"
                    )

        if sentences:
            paragraphs.append(sentences)

    return paragraphs


def _inherited_citations(
    paragraph: list[str],
    position: int,
    available: Mapping[str, str],
) -> list[str]:
    """
    Find the citations an uncited sentence most
    plausibly relies on: the next cited sentence in
    the same paragraph, otherwise the previous one.

    Small models often cite once at the end of a
    short paragraph:
        "OpenBrain runs 200,000 copies. This equals
         50,000 human coders sped up 30x [S1]."
    """

    def valid(sentence: str) -> list[str]:
        return [
            citation
            for citation in _unique(
                CITATION_PATTERN.findall(sentence)
            )
            if citation in available
        ]

    for sentence in paragraph[position + 1:]:
        found = valid(sentence)

        if found:
            return found

    for sentence in reversed(paragraph[:position]):
        found = valid(sentence)

        if found:
            return found

    return []


def verify_answer(
    answer: str,
    evidence: Mapping[str, str],
) -> VerificationReport:
    """
    Run deterministic verification checks.

    Evidence maps source IDs, such as S1,
    to text supplied to the generator.

    Checks:
    - Citation validity
    - Citation coverage, with citations inherited
      within a paragraph
    - Numerical consistency, including multipliers
    - Verbatim matching of quoted text
    - Insufficient-evidence classification

    A passed report is NOT proof of semantic
    entailment or factual correctness.
    """
    if not isinstance(answer, str):
        raise TypeError(
            "answer must be a string"
        )

    if not isinstance(evidence, Mapping):
        raise TypeError(
            "evidence must be a mapping"
        )

    if not answer.strip():
        return VerificationReport(
            status="review_required",
            is_refusal=False,
            cited_ids=[],
            unknown_citations=[],
            uncited_claims=[],
            numerical_issues=[],
            claims=[],
            warnings=[
                "The generated answer is empty."
            ],
        )

    available = {
        key: value
        for key, value in evidence.items()
        if (
            isinstance(key, str)
            and isinstance(value, str)
        )
    }

    mentioned = _unique(
        CITATION_PATTERN.findall(answer)
    )

    valid = [
        citation
        for citation in mentioned
        if citation in available
    ]

    unknown = [
        f"[{citation}]"
        for citation in mentioned
        if citation not in available
    ]

    paragraphs = _paragraph_claims(answer)

    statements = [
        claim
        for paragraph in paragraphs
        for claim in paragraph
    ]

    limitation_flags = [
        _limitation_statement(claim)
        for claim in statements
    ]

    description_flags = [
        _source_description(claim)
        for claim in statements
    ]

    # A refusal may describe the retrieved
    # sources, but must not include unrelated
    # affirmative factual claims.
    is_refusal = (
        bool(statements)
        and any(limitation_flags)
        and all(
            limitation or description
            for limitation, description in zip(
                limitation_flags,
                description_flags,
                strict=True,
            )
        )
    )

    normalized_evidence = {
        citation: _normalize_quote(text)
        for citation, text in available.items()
    }

    claim_checks = []
    uncited = []
    numerical_issues = []
    quote_issues = []
    inherited_count = 0

    for paragraph in paragraphs:
        for position, claim in enumerate(paragraph):
            claim_citations = _unique(
                CITATION_PATTERN.findall(claim)
            )

            valid_citations = [
                citation
                for citation in claim_citations
                if citation in available
            ]

            limitation = _limitation_statement(
                claim
            )

            issues = []
            missing_numbers = []
            inherited = []
            unverified_quotes = []

            if not valid_citations and not limitation:
                inherited = _inherited_citations(
                    paragraph,
                    position,
                    available,
                )

                if inherited:
                    inherited_count += 1
                else:
                    uncited.append(claim)
                    issues.append(
                        "No valid source citation."
                    )

            supporting = valid_citations or inherited

            if supporting and not limitation:
                cited_text = "\n".join(
                    available[citation]
                    for citation in supporting
                )

                source_numbers = set(
                    _numbers(cited_text)
                )

                missing_numbers = [
                    number
                    for number in _numbers(claim)
                    if number not in source_numbers
                ]

                if missing_numbers:
                    issue = (
                        "Numbers not found in "
                        "cited evidence: "
                        + ", ".join(missing_numbers)
                    )

                    issues.append(issue)

                    numerical_issues.append(
                        f"{claim} — {issue}"
                    )

            # Quoted text must appear verbatim in the
            # cited sources, or in any supplied source
            # when the sentence has no citation.
            quote_scope = supporting or list(available)

            for quote in _quotes(claim):
                needle = _normalize_quote(quote)

                if not any(
                    needle in normalized_evidence[citation]
                    for citation in quote_scope
                ):
                    unverified_quotes.append(quote)

            if unverified_quotes:
                issue = (
                    "Quoted text not found verbatim "
                    "in the evidence: "
                    + "; ".join(
                        f'"{quote}"'
                        for quote in unverified_quotes
                    )
                )

                issues.append(issue)
                quote_issues.append(
                    f"{claim} — {issue}"
                )

            claim_checks.append(
                ClaimCheck(
                    text=claim,
                    citations=claim_citations,
                    missing_numbers=missing_numbers,
                    issues=issues,
                    inherited_citations=inherited,
                    unverified_quotes=unverified_quotes,
                )
            )

    warnings = []

    if unknown:
        warnings.append(
            "Unknown source citations: "
            + ", ".join(unknown)
        )

    if uncited:
        warnings.append(
            f"{len(uncited)} statement(s) "
            "lack valid citations."
        )

    if numerical_issues:
        warnings.append(
            f"{len(numerical_issues)} statement(s) "
            "contain numbers not found "
            "in their cited evidence."
        )

    if quote_issues:
        warnings.append(
            f"{len(quote_issues)} statement(s) "
            "quote text that does not appear "
            "verbatim in the evidence."
        )

    if (
        any(limitation_flags)
        and not is_refusal
    ):
        warnings.append(
            "The answer mixes evidence-limitation "
            "statements with other claims."
        )

    if (
        not mentioned
        and not is_refusal
    ):
        warnings.append(
            "The answer contains no source citations."
        )

    if warnings:
        status = "review_required"

    elif is_refusal:
        status = "insufficient_evidence"

    else:
        status = "checks_passed"

    return VerificationReport(
        status=status,
        is_refusal=is_refusal,
        cited_ids=valid,
        unknown_citations=unknown,
        uncited_claims=uncited,
        numerical_issues=numerical_issues,
        claims=claim_checks,
        warnings=warnings,
        quote_issues=quote_issues,
        inherited_citation_count=inherited_count,
    )