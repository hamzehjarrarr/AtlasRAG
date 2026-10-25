from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from atlasrag.generation import (
    MAX_SOURCES,
    GeneratedAnswer,
    generate_answer,
)
from atlasrag.ingestion.chunking import chunk_pdf
from atlasrag.ingestion.enrichment import enrich_chunks
from atlasrag.ingestion.parser import parse_pdf
from atlasrag.search import (
    DATA_DIR,
    Dataset,
    RetrievalEngine,
    rerank_passages,
)

logger = logging.getLogger(__name__)

# Bump whenever parsing, chunking, or enrichment changes,
# so stale indexes are rebuilt instead of reused.
INDEX_VERSION = "pdf-v3"
DEFAULT_CHUNK_SIZE = 250
DEFAULT_RERANK_CANDIDATES = 20


def index_name_for(
    document_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """Directory name holding one PDF's indexes."""
    return (
        f"pdf_{INDEX_VERSION.replace('-', '_')}_"
        f"{document_id}_{chunk_size}"
    )


def manifest_path_for(
    document_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Path:
    return (
        DATA_DIR
        / index_name_for(document_id, chunk_size)
        / "manifest.json"
    )


@dataclass(frozen=True)
class PDFSource:
    chunk_id: str
    filename: str
    page_number: int
    end_page_number: int
    section_title: str | None
    text: str
    score: float
    rerank_score: float | None = None


@dataclass
class PDFAnswer:
    generated: GeneratedAnswer
    sources: list[PDFSource]
    citation_sources: dict[str, PDFSource]


class PDFIndex:
    """
    A local, single-document PDF retrieval index.

    Pipeline:
      PDF -> structure-aware chunks -> contextual
      enrichment -> BM25S + BGE-M3/Qdrant -> RRF
      -> cross-encoder reranking -> Ollama.

    display_name is the filename shown in citations.
    Uploaded files are stored under their content
    hash, so the original name is passed separately.
    """

    def __init__(
        self,
        pdf_path: Path,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        display_name: str | None = None,
    ):
        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.is_file():
            raise FileNotFoundError(self.pdf_path)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.chunk_size = chunk_size
        self.display_name = (
            Path(display_name).name
            if display_name
            else self.pdf_path.name
        )

        digest = hashlib.sha256()

        with self.pdf_path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)

        self.document_id = digest.hexdigest()

        self.index_name = index_name_for(
            self.document_id,
            chunk_size,
        )

        self.index_dir = DATA_DIR / self.index_name
        self.manifest_path = self.index_dir / "manifest.json"

        self.dataset: Dataset | None = None
        self.retrieval: RetrievalEngine | None = None
        self.metadata: dict[str, dict] = {}
        self.evidence_corpus: dict[str, dict] = {}

    def _load_manifest(self) -> bool:
        if not self.manifest_path.is_file():
            return False

        try:
            data = json.loads(
                self.manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, ValueError):
            return False

        if not isinstance(data, dict):
            return False

        if data.get("version") != INDEX_VERSION:
            return False

        if data.get("document_id") != self.document_id:
            return False

        if data.get("chunk_size") != self.chunk_size:
            return False

        records = data.get("chunks")

        if not isinstance(records, list) or not records:
            return False

        ids = set()

        required = (
            "chunk_id",
            "retrieval_text",
            "text",
            "page_number",
            "end_page_number",
            "filename",
        )

        for record in records:
            if not isinstance(record, dict):
                return False

            if any(key not in record for key in required):
                return False

            chunk_id = record["chunk_id"]

            if (
                not isinstance(chunk_id, str)
                or not chunk_id
                or chunk_id in ids
            ):
                return False

            ids.add(chunk_id)

        self._make_dataset(records)
        return True

    def _make_dataset(
        self,
        records: list[dict],
    ) -> None:
        corpus = {}
        evidence = {}
        metadata = {}

        for record in records:
            chunk_id = record["chunk_id"]

            # Enriched text for BM25S and BGE-M3.
            corpus[chunk_id] = {
                "title": "",
                "text": record["retrieval_text"],
            }

            # Original passage text for generation.
            pages = str(record["page_number"])

            if (
                record["end_page_number"]
                != record["page_number"]
            ):
                pages += f"-{record['end_page_number']}"

            section = record.get("section_title")

            heading = (
                f"Document: {record['filename']}\n"
                f"Page(s): {pages}"
            )

            if section:
                heading += f"\nSection: {section}"

            evidence[chunk_id] = {
                "title": heading,
                "text": record["text"],
            }

            metadata[chunk_id] = record

        self.dataset = Dataset(
            name=self.index_name,
            corpus=corpus,
        )

        self.evidence_corpus = evidence
        self.metadata = metadata

    def _create_manifest(self) -> None:
        parsed = dataclasses.replace(
            parse_pdf(self.pdf_path),
            filename=self.display_name,
        )

        chunks = chunk_pdf(
            parsed,
            chunk_size=self.chunk_size,
        )

        if not chunks:
            raise ValueError(
                "The PDF produced no searchable chunks."
            )

        enriched = enrich_chunks(
            chunks,
            document_title=Path(self.display_name).stem,
        )

        records = [
            {
                "chunk_id": chunk.chunk_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "end_page_number": (
                    chunk.end_page_number
                    or chunk.page_number
                ),
                "chunk_index": chunk.chunk_index,
                "section_title": chunk.section_title,
                "heading_level": chunk.heading_level,
                "text": chunk.text,
                "retrieval_text": chunk.retrieval_text,
            }
            for chunk in enriched
        ]

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest = {
            "version": INDEX_VERSION,
            "document_id": self.document_id,
            "chunk_size": self.chunk_size,
            "filename": self.display_name,
            "page_count": parsed.page_count,
            "chunk_count": len(records),
            "chunks": records,
        }

        temporary = self.manifest_path.with_suffix(
            ".json.tmp"
        )

        temporary.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary.replace(self.manifest_path)

        self._make_dataset(records)

    def build(
        self,
        rebuild: bool = False,
        batch_size: int = 16,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if rebuild or not self._load_manifest():
            logger.info("Parsing and chunking PDF...")
            self._create_manifest()
        else:
            logger.info("Loaded existing PDF chunk manifest.")

        if self.dataset is None:
            raise RuntimeError(
                "PDF dataset initialization failed."
            )

        if self.retrieval is not None:
            self.retrieval.close()

        self.retrieval = RetrievalEngine(self.dataset)

        self.retrieval.build_bm25(
            rebuild=rebuild
        )

        self.retrieval.build_dense(
            batch_size=batch_size,
            rebuild=rebuild,
        )

        logger.info(
            "PDF index ready: %d chunks",
            len(self.metadata),
        )

    def search(
        self,
        question: str,
        k: int = 5,
        method: str = "hybrid",
    ) -> list[PDFSource]:
        """
        First-stage retrieval without reranking.

        Scores are method-specific: BM25 scores,
        cosine similarities, or RRF fusion scores.
        """
        if self.retrieval is None:
            raise RuntimeError(
                "Build the PDF index before searching."
            )

        if not question.strip():
            raise ValueError(
                "A question is required."
            )

        if k <= 0:
            raise ValueError(
                "k must be positive."
            )

        results = self.retrieval.search(
            question,
            method=method,
            k=k,
        )

        sources = []

        for chunk_id, score in results:
            record = self.metadata[chunk_id]

            sources.append(
                PDFSource(
                    chunk_id=chunk_id,
                    filename=record["filename"],
                    page_number=record["page_number"],
                    end_page_number=record["end_page_number"],
                    section_title=record.get(
                        "section_title"
                    ),
                    text=record["text"],
                    score=score,
                )
            )

        return sources

    def retrieve(
        self,
        question: str,
        k: int = 5,
        *,
        rerank: bool = True,
        rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    ) -> list[PDFSource]:
        """
        Retrieve the evidence passages for a question.

        The application always reranks. rerank=False
        exists so the evaluation script can measure
        what the cross-encoder adds.
        """
        if k <= 0:
            raise ValueError("k must be positive.")

        if rerank_candidates <= 0:
            raise ValueError(
                "rerank_candidates must be positive."
            )

        if not rerank:
            return self.search(
                question,
                k=k,
                method="hybrid",
            )

        candidates = self.search(
            question,
            k=max(k, rerank_candidates),
            method="hybrid",
        )

        ranked = rerank_passages(
            question,
            candidates,
            batch_size=8,
        )

        return [
            dataclasses.replace(
                source,
                rerank_score=score,
            )
            for source, score in ranked[:k]
        ]

    def answer(
        self,
        question: str,
        k: int = 5,
        model: str = "qwen3:4b-instruct",
        *,
        rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    ) -> PDFAnswer:
        """
        Answer a PDF question.

          1. Retrieve hybrid (BM25S + BGE-M3) candidates.
          2. Rerank them with a cross-encoder.
          3. Generate a cited answer from the best passages.
          4. Run deterministic verification checks.
        """
        sources = self.retrieve(
            question,
            k=k,
            rerank=True,
            rerank_candidates=rerank_candidates,
        )

        results = [
            (
                source.chunk_id,
                source.rerank_score
                if source.rerank_score is not None
                else source.score,
            )
            for source in sources
        ]

        generated = generate_answer(
            question=question,
            results=results,
            corpus=self.evidence_corpus,
            model=model,
        )

        citation_sources = {
            f"S{index}": source
            for index, source in enumerate(
                sources[:MAX_SOURCES],
                start=1,
            )
        }

        return PDFAnswer(
            generated=generated,
            sources=sources,
            citation_sources=citation_sources,
        )

    def close(self) -> None:
        if self.retrieval is not None:
            self.retrieval.close()
            self.retrieval = None