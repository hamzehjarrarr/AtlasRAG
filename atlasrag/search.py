"""
Search: from a question to ranked passages.

1. BM25S lexical search and BGE-M3 dense search (stored in Qdrant)
2. Reciprocal Rank Fusion of the two rankings
3. Cross-encoder reranking of the fused candidates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import bm25s
import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024
RRF_K = 60
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

T = TypeVar("T")


# --------------------------------------------------
# DOCUMENTS
# --------------------------------------------------

@dataclass
class Dataset:
    """
    An in-memory document collection for retrieval.

    name:
        Unique index name; also the index directory.
    corpus:
        Maps document IDs to {"title": ..., "text": ...}.
    """

    name: str
    corpus: dict


def document_text(document: dict) -> str:
    """
    Combine a document's title and body into
    the text used for retrieval or generation.
    """
    title = str(document.get("title") or "").strip()
    body = str(document.get("text") or "").strip()

    return "\n\n".join(
        part for part in (title, body) if part
    )


# --------------------------------------------------
# HYBRID RETRIEVAL
# --------------------------------------------------

class RetrievalEngine:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.ids = list(dataset.corpus)
        self.texts = [
            document_text(dataset.corpus[doc_id])
            for doc_id in self.ids
        ]

        self.index_dir = DATA_DIR / dataset.name / "indexes"
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.bm25 = None
        self.embedding_model = None
        self.qdrant = None

    def build_bm25(self, rebuild: bool = False) -> None:
        index_path = self.index_dir / "bm25"

        if index_path.exists() and not rebuild:
            self.bm25 = bm25s.BM25.load(
                str(index_path),
                load_corpus=False,
            )
            logger.info("Loaded BM25 index: %s", self.dataset.name)
            return

        logger.info("Building BM25: %s", self.dataset.name)

        tokens = bm25s.tokenize(
            self.texts,
            stopwords="en",
            show_progress=False,
        )

        self.bm25 = bm25s.BM25()
        self.bm25.index(tokens, show_progress=False)
        self.bm25.save(str(index_path))

    def search_bm25(self, query: str, k: int = 10) -> list:
        if self.bm25 is None:
            self.build_bm25()

        tokens = bm25s.tokenize(
            [query],
            stopwords="en",
            show_progress=False,
        )

        results, scores = self.bm25.retrieve(
            tokens,
            k=min(k, len(self.ids)),
            show_progress=False,
        )

        return [
            (self.ids[int(index)], float(score))
            for index, score in zip(results[0], scores[0], strict=True)
        ]

    def _load_embedding_model(self):
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading %s (first run downloads ~2.3 GB)", EMBEDDING_MODEL)
            self.embedding_model = SentenceTransformer(
                EMBEDDING_MODEL,
                trust_remote_code=False,
            )

        return self.embedding_model

    def _load_qdrant(self):
        if self.qdrant is None:
            from qdrant_client import QdrantClient

            database_path = self.index_dir / "qdrant"
            self.qdrant = QdrantClient(path=str(database_path))

        return self.qdrant

    def build_dense(
        self,
        batch_size: int = 16,
        rebuild: bool = False,
    ) -> None:
        from qdrant_client.models import (
            Distance,
            PointStruct,
            VectorParams,
        )

        client = self._load_qdrant()
        collection = f"atlasrag_{self.dataset.name}"

        if client.collection_exists(collection):
            count = client.count(
                collection_name=collection,
                exact=True,
            ).count

            if count == len(self.ids) and not rebuild:
                logger.info("Loaded dense index: %s", self.dataset.name)
                return

            client.delete_collection(collection)

        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )

        model = self._load_embedding_model()

        for start in range(0, len(self.ids), batch_size):
            end = min(start + batch_size, len(self.ids))
            batch_texts = self.texts[start:end]

            vectors = model.encode(
                batch_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            points = [
                PointStruct(
                    id=start + offset + 1,
                    vector=vector.tolist(),
                    payload={
                        "doc_id": self.ids[start + offset],
                    },
                )
                for offset, vector in enumerate(vectors)
            ]

            client.upsert(
                collection_name=collection,
                points=points,
                wait=True,
            )

            logger.info(
                "%s: indexed %d/%d",
                self.dataset.name,
                end,
                len(self.ids),
            )

        logger.info("Dense index ready: %s", self.dataset.name)

    def search_dense(self, query: str, k: int = 10) -> list:
        model = self._load_embedding_model()
        client = self._load_qdrant()

        vector = model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        results = client.query_points(
            collection_name=f"atlasrag_{self.dataset.name}",
            query=vector,
            limit=k,
            with_payload=True,
        ).points

        return [
            (point.payload["doc_id"], float(point.score))
            for point in results
        ]

    @staticmethod
    def reciprocal_rank_fusion(
        rankings: list[list[tuple[str, float]]],
        k: int = RRF_K,
    ) -> list:
        scores = {}

        for ranking in rankings:
            for rank, (doc_id, _) in enumerate(ranking, start=1):
                scores[doc_id] = (
                    scores.get(doc_id, 0.0)
                    + 1.0 / (k + rank)
                )

        return sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )

    def search_hybrid(
        self,
        query: str,
        k: int = 10,
        candidate_k: int = 50,
    ) -> list:
        lexical = self.search_bm25(query, candidate_k)
        semantic = self.search_dense(query, candidate_k)

        return self.reciprocal_rank_fusion(
            [lexical, semantic]
        )[:k]

    def search(
        self,
        query: str,
        method: str = "bm25",
        k: int = 10,
    ) -> list:
        if method == "bm25":
            return self.search_bm25(query, k)

        if method == "dense":
            return self.search_dense(query, k)

        if method == "hybrid":
            return self.search_hybrid(query, k)

        raise ValueError(f"Unknown retrieval method: {method}")

    def close(self):
        if self.qdrant is not None:
            self.qdrant.close()
            self.qdrant = None


# --------------------------------------------------
# RERANKING
# --------------------------------------------------

@lru_cache(maxsize=2)
def get_reranker(
    model_name: str = DEFAULT_RERANKER,
) -> CrossEncoder:
    """Load and reuse a local cross-encoder model."""
    # Imported lazily so the package (and its tests)
    # can be used without loading torch.
    from sentence_transformers import CrossEncoder

    logger.info("Loading reranker: %s", model_name)
    return CrossEncoder(model_name)


def rerank_passages(  # noqa: UP047
    question: str,
    passages: list[T],
    *,
    model_name: str = DEFAULT_RERANKER,
    batch_size: int = 8,
) -> list[tuple[T, float]]:
    """
    Rerank passages containing a .text attribute.

    Returns (passage, cross-encoder score) pairs,
    best first. Higher scores mean the model judged
    the passage more relevant to the question.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if not passages:
        return []

    pairs = [
        (question, passage.text)
        for passage in passages
    ]

    model = get_reranker(model_name)

    scores = np.asarray(
        model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        ),
        dtype=float,
    ).reshape(-1)

    if len(scores) != len(passages):
        raise RuntimeError(
            "Reranker returned an unexpected number of scores."
        )

    if not np.all(np.isfinite(scores)):
        raise RuntimeError(
            "Reranker returned non-finite scores."
        )

    ranked_indices = sorted(
        range(len(passages)),
        key=lambda i: (-float(scores[i]), i),
    )

    return [
        (passages[i], float(scores[i]))
        for i in ranked_indices
    ]
