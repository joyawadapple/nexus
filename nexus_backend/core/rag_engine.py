"""
RAG Engine — sentence-transformers + numpy cosine similarity.
No external vector database. All embeddings held in memory at runtime.

Uses: sentence-transformers all-MiniLM-L6-v2
"""
from __future__ import annotations

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

log = structlog.get_logger("rag_engine")

_MODEL_NAME = "all-MiniLM-L6-v2"


class RAGEngine:
    """
    In-memory RAG over the NexaCloud knowledge base.

    Loaded once at startup via load_all_db(). Encodes all documents using
    sentence-transformers. At query time: encode query, filter by category,
    compute cosine similarity, return top-k.
    """

    def __init__(self, knowledge_base: list[dict]) -> None:
        log.info("rag_engine.init", model=_MODEL_NAME, docs=len(knowledge_base))
        self._model = SentenceTransformer(_MODEL_NAME)
        self._docs = knowledge_base

        # Pre-encode all documents
        texts = [doc["content"] for doc in self._docs]
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self._embeddings = np.array(embeddings, dtype=np.float32)

        log.info("rag_engine.ready", embedding_shape=self._embeddings.shape)

    def query(
        self,
        query_text: str,
        category: str | None = None,
        product: str | None = None,
        top_k: int = 3,
        min_similarity: float = 0.30,
    ) -> list[dict]:
        """
        Semantic search over the knowledge base.

        Args:
            query_text: Natural language query
            category: Filter to "diagnostic" or "resolution" only
            product: Optional product filter (NexaAuth, NexaStore, etc.)
            top_k: Number of results to return
            min_similarity: Minimum cosine similarity threshold

        Returns:
            List of dicts with keys: source, similarity, excerpt_summary, content, tags
        """
        if not query_text.strip():
            return []

        # Filter candidate docs
        candidates = []
        candidate_indices = []
        for i, doc in enumerate(self._docs):
            if category and doc.get("category") != category:
                continue
            if product and doc.get("product") not in (product, "all"):
                continue
            candidates.append(doc)
            candidate_indices.append(i)

        if not candidates:
            log.warning("rag_engine.no_candidates", category=category, product=product)
            return []

        # Encode query
        query_embedding = self._model.encode(
            [query_text], normalize_embeddings=True, show_progress_bar=False
        )
        query_vec = np.array(query_embedding[0], dtype=np.float32)

        # Cosine similarity (embeddings already normalized → dot product = cosine sim)
        candidate_embeddings = self._embeddings[candidate_indices]
        similarities = candidate_embeddings @ query_vec

        # Sort by similarity descending
        ranked_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in ranked_indices[:top_k]:
            sim = float(similarities[idx])
            if sim < min_similarity:
                continue
            doc = candidates[idx]
            results.append({
                "source": doc["title"],
                "similarity": round(sim, 4),
                "excerpt_summary": _extract_summary(doc["content"]),
                "content": doc["content"],
                "tags": doc.get("tags", []),
                "category": doc.get("category"),
                "product": doc.get("product"),
            })

        log.debug(
            "rag_engine.query",
            query=query_text[:80],
            category=category,
            results=len(results),
        )
        return results


def _extract_summary(content: str, max_chars: int = 200) -> str:
    """Return the first sentence or first max_chars characters as a summary."""
    if len(content) <= max_chars:
        return content
    # Try to break at a sentence
    period_idx = content.find(". ", 0, max_chars)
    if period_idx > 0:
        return content[: period_idx + 1]
    return content[:max_chars] + "..."
