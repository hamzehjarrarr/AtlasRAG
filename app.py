from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlasrag.search import DATA_DIR  # noqa: E402
from atlasrag.generation import DEFAULT_MODEL, MAX_SOURCES  # noqa: E402
from atlasrag.index import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    PDFIndex,
    manifest_path_for,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

UPLOAD_DIR = DATA_DIR / "pdf_uploads"
MAX_PDF_BYTES = 25 * 1024 * 1024
CHUNK_SIZE = DEFAULT_CHUNK_SIZE

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

:root {
    --ink: #1D2433;
    --ink-blue: #2B4C7E;
    --muted: #5B6475;
    --rule: #E3E5EA;
    --highlight: #FFE58A;
    --ok: #2F7D4F;
    --warn: #A86B12;
    --neutral: #5B6475;
}

.block-container {
    max-width: 760px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

footer { visibility: hidden; }

.atlas-wordmark {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 2.6rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    line-height: 1.1;
    color: var(--ink);
    margin: 0;
}

.atlas-tagline {
    color: var(--muted);
    font-size: 1.02rem;
    margin: 0.35rem 0 0 0;
}

.atlas-document {
    padding: 0.2rem 0;
}

.atlas-document-name {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--ink);
    overflow-wrap: anywhere;
}

.atlas-document-meta {
    color: var(--muted);
    font-size: 0.92rem;
}

.atlas-answer {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.2rem;
    line-height: 1.65;
    color: var(--ink);
}

.atlas-answer p { margin-bottom: 0.8rem; }

.atlas-cite {
    font-family: 'Source Sans Pro', 'Source Sans 3', sans-serif;
    font-size: 0.78em;
    font-weight: 600;
    color: var(--ink);
    background: var(--highlight);
    border-radius: 3px;
    padding: 0.05em 0.35em;
    margin-left: 0.1em;
    white-space: nowrap;
}

.atlas-status {
    display: flex;
    align-items: baseline;
    gap: 0.55rem;
    color: var(--ink);
    font-size: 0.98rem;
}

.atlas-status-dot {
    width: 0.6rem;
    height: 0.6rem;
    border-radius: 50%;
    flex: none;
    transform: translateY(0.05rem);
}

.atlas-status-detail {
    color: var(--muted);
    font-size: 0.9rem;
}

.atlas-passage {
    font-size: 0.98rem;
    line-height: 1.6;
    color: var(--ink);
    margin-bottom: 0.6rem;
}

.atlas-fn {
    color: var(--muted);
    font-size: 0.7em;
    margin-left: 0.05em;
}

.atlas-nowrap { white-space: nowrap; }

.atlas-check-summary {
    color: var(--muted);
    font-size: 0.92rem;
    margin-bottom: 0.8rem;
}

.atlas-section-title {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--ink);
    margin: 1.6rem 0 0.4rem 0;
}

.atlas-footer {
    color: var(--muted);
    font-size: 0.85rem;
    border-top: 1px solid var(--rule);
    padding-top: 1rem;
    margin-top: 3rem;
}

@media (max-width: 640px) {
    .block-container { padding-top: 1.5rem; }
    .atlas-wordmark { font-size: 2.1rem; }
    .atlas-answer { font-size: 1.1rem; }
}
</style>
"""

STATUS = {
    "checks_passed": (
        "var(--ok)",
        "Checks passed",
        "Every statement is cited, and its numbers and quotes "
        "match the cited passages.",
    ),
    "insufficient_evidence": (
        "var(--neutral)",
        "Not in the document",
        "The retrieved passages don't answer this question, "
        "so AtlasRAG declined instead of guessing.",
    ),
    "review_required": (
        "var(--warn)",
        "Review suggested",
        "The automatic checks found something to look at.",
    ),
}


st.set_page_config(
    page_title="AtlasRAG",
    page_icon="📄",
    layout="centered",
)


# --------------------------------------------------
# PDF INDEX CACHE
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_pdf_index(
    pdf_path: str,
    document_hash: str,
    chunk_size: int,
    _display_name: str,
) -> PDFIndex:
    """
    Keep one PDFIndex instance per PDF and configuration.

    This avoids opening multiple local Qdrant clients
    against the same database directory. The leading
    underscore keeps the display name out of the cache
    key, so re-uploading the same file under another
    name reuses the existing index.
    """
    index = PDFIndex(
        Path(pdf_path),
        chunk_size=chunk_size,
        display_name=_display_name,
    )

    if index.document_id != document_hash:
        raise ValueError(
            "PDF content changed since it was selected."
        )

    index.build(batch_size=16)

    return index


# --------------------------------------------------
# PDF STORAGE
# --------------------------------------------------

def save_uploaded_pdf(uploaded_file) -> tuple[Path, str]:
    """
    Validate and persist a PDF using its content hash.

    Repeated uploads of identical bytes reuse the
    same local file.
    """
    contents = uploaded_file.getvalue()

    if not contents:
        raise ValueError("The uploaded PDF is empty.")

    if len(contents) > MAX_PDF_BYTES:
        raise ValueError(
            "PDF exceeds the 25 MB upload limit."
        )

    if not contents.startswith(b"%PDF-"):
        raise ValueError(
            "The uploaded file does not have a PDF header."
        )

    document_hash = hashlib.sha256(contents).hexdigest()

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    destination = UPLOAD_DIR / f"{document_hash}.pdf"

    if destination.exists():
        if not destination.is_file():
            raise ValueError(
                "PDF storage destination is invalid."
            )

        existing_hash = hashlib.sha256(
            destination.read_bytes()
        ).hexdigest()

        if existing_hash != document_hash:
            raise ValueError(
                "Stored PDF failed its integrity check."
            )

        return destination, document_hash

    temporary = UPLOAD_DIR / (
        f".{document_hash}.{os.getpid()}.tmp"
    )

    try:
        with temporary.open("xb") as output:
            output.write(contents)

        temporary.replace(destination)

    finally:
        temporary.unlink(missing_ok=True)

    return destination, document_hash


def read_manifest(document_hash: str) -> dict | None:
    """Return a PDF's chunk manifest for the current index version."""
    manifest_path = manifest_path_for(document_hash, CHUNK_SIZE)

    if not manifest_path.is_file():
        return None

    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None

    if (
        not isinstance(manifest, dict)
        or manifest.get("document_id") != document_hash
    ):
        return None

    return manifest


def saved_pdfs() -> dict[str, tuple[Path, str, str]]:
    """
    Previously uploaded PDFs with an index for the
    current version, keyed by a display label.
    """
    results = {}

    if not UPLOAD_DIR.exists():
        return results

    for pdf_path in sorted(UPLOAD_DIR.glob("*.pdf")):
        document_hash = pdf_path.stem

        if not re.fullmatch(r"[0-9a-f]{64}", document_hash):
            continue

        manifest = read_manifest(document_hash)

        if manifest is None:
            continue

        filename = str(
            manifest.get("filename") or "document.pdf"
        )

        results[filename] = (pdf_path, document_hash, filename)

    return results


# --------------------------------------------------
# DISPLAY HELPERS
# --------------------------------------------------

def render_answer(text: str) -> str:
    """
    Escape model output, keep hyphenated names such as
    Agent-1 on one line, and turn [S1] into citation marks.
    """
    escaped = html.escape(text, quote=False)

    escaped = re.sub(
        r"(?<![\w-])(\w+(?:-\w+)+)(?![\w-])",
        r'<span class="atlas-nowrap">\1</span>',
        escaped,
    )

    return re.sub(
        r"\[(S\d+)\]",
        r'<span class="atlas-cite">\1</span>',
        escaped,
    )


def passage(text: str, section_title: str | None = None) -> None:
    """
    Show document text verbatim, without Markdown formatting.

    A leading line that repeats the section title is dropped,
    and [footnote N] markers become small superscripts.
    """
    text = text.strip()

    if section_title:
        first, _, rest = text.partition("\n")

        if first.strip() == section_title.strip() and rest.strip():
            text = rest.strip()

    escaped = html.escape(text, quote=False)
    escaped = re.sub(
        r"\s*\[footnote (\d+)\]",
        r'<sup class="atlas-fn">\1</sup>',
        escaped,
    )
    escaped = re.sub(r"\n\s*\n", "<br><br>", escaped).replace("\n", " ")

    st.markdown(
        f'<div class="atlas-passage">{escaped}</div>',
        unsafe_allow_html=True,
    )


def page_label(source) -> str:
    if source.end_page_number != source.page_number:
        return f"Pages {source.page_number}–{source.end_page_number}"

    return f"Page {source.page_number}"


def show_source(source, citation_id: str, expanded: bool) -> None:
    title = f"[{citation_id}] {page_label(source)}"

    if source.section_title:
        title += f", {source.section_title}"

    with st.expander(title, expanded=expanded):
        passage(source.text, source.section_title)

        if source.rerank_score is not None:
            st.caption(
                f"Rerank score {source.rerank_score:.2f}, "
                f"hybrid score {source.score:.4f}"
            )


def plural(count: int, singular: str, many: str | None = None) -> str:
    return f"{count} {singular if count == 1 else (many or singular + 's')}"


def show_status(report, timing: str) -> None:
    if report is None:
        st.caption(timing)
        return

    color, label, detail = STATUS.get(
        report.status,
        ("var(--warn)", "Review suggested", ""),
    )

    if report.status == "review_required" and report.warnings:
        detail = " ".join(report.warnings)

    st.markdown(
        f"""
        <div class="atlas-status">
          <span class="atlas-status-dot" style="background:{color}"></span>
          <span><strong>{label}.</strong>
          <span class="atlas-status-detail">{html.escape(detail)}</span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if report.status == "insufficient_evidence" or not report.claims:
        st.caption(timing)
        return

    with st.expander("How this answer was checked"):
        summary = ", ".join(
            [
                plural(len(report.cited_ids), "source cited", "sources cited"),
                plural(len(report.uncited_claims), "uncited statement"),
                plural(len(report.numerical_issues), "number flag"),
                plural(len(report.quote_issues), "quote flag"),
            ]
        )

        st.markdown(
            f'<div class="atlas-check-summary">{timing} '
            f"{summary[0].upper()}{summary[1:]}.</div>",
            unsafe_allow_html=True,
        )

        for number, claim in enumerate(report.claims, start=1):
            if claim.citations:
                cited = ", ".join(claim.citations)
            elif claim.inherited_citations:
                cited = (
                    ", ".join(claim.inherited_citations)
                    + " (from the same paragraph)"
                )
            else:
                cited = "none"

            st.markdown(f"**{number}.** {claim.text}")
            st.caption(f"Sources: {cited}")

            for issue in claim.issues:
                st.warning(issue)

        st.caption(
            "These are rule-based checks of citations, numbers, "
            "and quotes. They catch common mistakes but don't "
            "prove every claim is supported."
        )


def show_answer(result: dict) -> None:
    answer = result["answer"]
    generated = answer.generated

    st.markdown(
        f'<div class="atlas-answer">{render_answer(generated.answer)}</div>',
        unsafe_allow_html=True,
    )

    show_status(
        generated.verification,
        f"Answered by {generated.model} in {result['elapsed']:.1f}s.",
    )

    cited = [
        citation_id
        for citation_id in generated.cited_ids
        if citation_id in answer.citation_sources
    ]

    if cited:
        st.markdown(
            '<div class="atlas-section-title">Sources</div>',
            unsafe_allow_html=True,
        )

        for position, citation_id in enumerate(cited):
            show_source(
                answer.citation_sources[citation_id],
                citation_id,
                expanded=position == 0,
            )

    others = [
        (citation_id, source)
        for citation_id, source in answer.citation_sources.items()
        if citation_id not in cited
    ]

    if others:
        with st.expander(
            f"Other passages read ({len(others)})"
        ):
            for citation_id, source in others:
                st.markdown(f"**[{citation_id}]** {page_label(source)}")
                passage(source.text, source.section_title)


# --------------------------------------------------
# DOCUMENT SELECTION
# --------------------------------------------------

def choose_document() -> None:
    """Upload a PDF or reopen one, then store it as the active document."""
    nonce = st.session_state.setdefault("upload_nonce", 0)

    uploaded = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Text-based PDFs up to 25 MB. Scanned pages need OCR first.",
        key=f"uploader_{nonce}",
    )

    if uploaded is not None:
        try:
            pdf_path, document_hash = save_uploaded_pdf(uploaded)
        except (OSError, ValueError) as exc:
            st.error(f"Couldn't use this file: {exc}")
            return

        st.session_state["document"] = (
            pdf_path,
            document_hash,
            Path(uploaded.name).name,
        )
        st.rerun()

    saved = saved_pdfs()

    if saved:
        choice = st.selectbox(
            "Or open a document you've used before",
            options=list(saved),
            index=None,
            placeholder="Choose a document",
        )

        if choice is not None:
            st.session_state["document"] = saved[choice]
            st.rerun()


def close_document() -> None:
    st.session_state.pop("document", None)
    st.session_state.pop("result", None)
    st.session_state["upload_nonce"] = (
        st.session_state.get("upload_nonce", 0) + 1
    )


# --------------------------------------------------
# PAGE
# --------------------------------------------------

def main() -> None:
    st.markdown(STYLES, unsafe_allow_html=True)

    header, settings = st.columns([5, 1], vertical_alignment="bottom")

    header.markdown(
        '<div class="atlas-wordmark">AtlasRAG</div>'
        '<div class="atlas-tagline">Ask questions about a PDF and get '
        "answers that cite the exact pages they came from.</div>",
        unsafe_allow_html=True,
    )

    with settings.popover("Settings", width="stretch"):
        model_name = st.text_input(
            "Ollama model",
            value=DEFAULT_MODEL,
            key="model_name",
        )

        top_k = st.slider(
            "Passages given to the model",
            min_value=1,
            max_value=MAX_SOURCES,
            value=MAX_SOURCES,
            key="top_k",
        )

    st.write("")

    document = st.session_state.get("document")

    if document is None:
        choose_document()
        footer()
        return

    pdf_path, document_hash, display_name = document

    try:
        with st.spinner(
            f"Indexing {display_name}. The first run also downloads "
            "the search models (about 2.3 GB)."
        ):
            index = get_pdf_index(
                str(pdf_path),
                document_hash,
                CHUNK_SIZE,
                display_name,
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't index {display_name}: {exc}")
        st.button("Choose another document", on_click=close_document)
        return

    manifest = read_manifest(document_hash) or {}
    pages = manifest.get("page_count")
    meta = f"{len(index.metadata)} passages indexed"

    if pages:
        meta = f"{pages} pages, {meta}"

    st.divider()

    name_column, change_column = st.columns(
        [4, 1],
        vertical_alignment="center",
    )

    name_column.markdown(
        f'<div class="atlas-document">'
        f'<div class="atlas-document-name">{html.escape(display_name)}</div>'
        f'<div class="atlas-document-meta">{meta}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    change_column.button(
        "Change document",
        on_click=close_document,
        type="tertiary",
        width="stretch",
    )

    st.write("")

    with st.form("question_form", border=False):
        question = st.text_area(
            "Your question",
            placeholder="What does this document say about…",
            height=96,
            max_chars=4000,
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button("Ask", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Type a question first.")
        else:
            st.session_state.pop("result", None)

            try:
                start = time.perf_counter()

                with st.spinner("Reading the document…"):
                    answer = index.answer(
                        question.strip(),
                        k=top_k,
                        model=model_name,
                    )

                st.session_state["result"] = {
                    "document_hash": document_hash,
                    "answer": answer,
                    "elapsed": time.perf_counter() - start,
                }

            except Exception as exc:  # noqa: BLE001
                st.error(f"Couldn't answer: {exc}")

    result = st.session_state.get("result")

    if result and result["document_hash"] == document_hash:
        st.write("")
        show_answer(result)

    footer()


def footer() -> None:
    st.markdown(
        '<div class="atlas-footer">Runs entirely on this machine: '
        "BM25S and BGE-M3 hybrid search, cross-encoder reranking, "
        "and a local Ollama model. Documents never leave it.</div>",
        unsafe_allow_html=True,
    )


main()