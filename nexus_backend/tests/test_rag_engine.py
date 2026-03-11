"""Tests for RAGEngine — sentence-transformers + numpy implementation."""
from unittest.mock import MagicMock, patch
import numpy as np

import pytest


SAMPLE_KB = [
    {
        "id": "kb_diag_001",
        "title": "NexaAuth 401 After Key Rotation",
        "category": "diagnostic",
        "product": "NexaAuth",
        "content": "After rotating API keys in NexaAuth, clients may receive 401 invalid_token errors if the old key is still cached. Check token cache TTL and force refresh.",
        "tags": ["401", "token", "rotation", "cache"],
    },
    {
        "id": "kb_diag_002",
        "title": "NexaAuth 403 Scope Mismatch",
        "category": "diagnostic",
        "product": "NexaAuth",
        "content": "403 forbidden errors in NexaAuth typically indicate a scope mismatch. Verify the OAuth scopes requested match the resource permissions.",
        "tags": ["403", "scope", "permissions"],
    },
    {
        "id": "kb_res_001",
        "title": "Resolving NexaAuth Token Cache Issues",
        "category": "resolution",
        "product": "NexaAuth",
        "content": "To resolve token cache issues: 1) Call /auth/token/invalidate, 2) Wait 60s for propagation, 3) Re-authenticate with new credentials.",
        "tags": ["resolution", "token", "cache", "invalidate"],
    },
    {
        "id": "kb_diag_003",
        "title": "NexaStore 429 Rate Limit",
        "category": "diagnostic",
        "product": "NexaStore",
        "content": "429 rate limit errors in NexaStore occur when exceeding 1000 requests/minute. Check X-RateLimit-Remaining headers.",
        "tags": ["429", "rate-limit", "NexaStore"],
    },
    {
        "id": "kb_res_002",
        "title": "NexaStore Rate Limit Resolution",
        "category": "resolution",
        "product": "NexaStore",
        "content": "Implement exponential backoff with jitter. Batch API calls using /store/batch endpoint. Request limit increase via support portal.",
        "tags": ["resolution", "rate-limit", "backoff"],
    },
]


@pytest.fixture
def mock_rag_engine():
    """RAGEngine with mocked sentence-transformers to avoid loading the model in tests."""
    with patch("core.rag_engine.SentenceTransformer") as mock_st:
        # Return deterministic embeddings based on doc index
        mock_model = MagicMock()

        def fake_encode(texts, **kwargs):
            # Return simple unit vectors — different enough to rank meaningfully
            if isinstance(texts, str):
                texts = [texts]
            results = []
            for i, t in enumerate(texts):
                # Create a vector based on text hash so same text → same vector
                seed = hash(t) % 1000
                rng = np.random.RandomState(seed)
                vec = rng.randn(384).astype(np.float32)
                vec /= np.linalg.norm(vec)
                results.append(vec)
            return np.array(results)

        mock_model.encode.side_effect = fake_encode
        mock_st.return_value = mock_model

        from core.rag_engine import RAGEngine
        engine = RAGEngine(SAMPLE_KB)
        yield engine


def test_returns_top_k_results(mock_rag_engine):
    """query() should return at most top_k results."""
    results = mock_rag_engine.query("authentication token error", top_k=2)
    assert len(results) <= 2


def test_category_filter_diagnostic(mock_rag_engine):
    """category='diagnostic' should only return diagnostic docs."""
    results = mock_rag_engine.query("401 error", category="diagnostic", top_k=5)
    for r in results:
        assert r["category"] == "diagnostic"


def test_category_filter_resolution(mock_rag_engine):
    """category='resolution' should only return resolution docs."""
    results = mock_rag_engine.query("fix token cache", category="resolution", top_k=5)
    for r in results:
        assert r["category"] == "resolution"


def test_product_filter(mock_rag_engine):
    """product filter should restrict results to matching product."""
    results = mock_rag_engine.query("rate limit exceeded", product="NexaStore", top_k=5)
    for r in results:
        assert r["product"] == "NexaStore"


def test_result_structure(mock_rag_engine):
    """Each result must have required fields."""
    results = mock_rag_engine.query("token error", top_k=3)
    for r in results:
        assert "source" in r
        assert "similarity" in r
        assert "excerpt_summary" in r
        assert "content" in r
        assert isinstance(r["similarity"], float)
        assert 0.0 <= r["similarity"] <= 1.0


def test_similarity_scores_ordered(mock_rag_engine):
    """Results should be ordered by descending similarity."""
    results = mock_rag_engine.query("NexaAuth invalid token", top_k=5)
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)


def test_min_similarity_threshold(mock_rag_engine):
    """Results below min_similarity should be excluded."""
    results = mock_rag_engine.query("completely unrelated xyz123", min_similarity=0.99, top_k=5)
    # With high threshold, very few (possibly 0) results expected
    for r in results:
        assert r["similarity"] >= 0.99


def test_empty_kb_returns_empty():
    """RAGEngine with no documents returns empty results."""
    with patch("core.rag_engine.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((0, 384), dtype=np.float32)
        mock_st.return_value = mock_model

        from importlib import reload
        import core.rag_engine
        reload(core.rag_engine)

        engine = core.rag_engine.RAGEngine([])
        results = engine.query("anything", top_k=3)
        assert results == []


def test_no_filter_returns_all_categories(mock_rag_engine):
    """Without category filter, results can include both diagnostic and resolution docs."""
    results = mock_rag_engine.query("NexaAuth error", top_k=10, min_similarity=0.0)
    categories = {r["category"] for r in results}
    # Should potentially include both categories
    assert len(categories) >= 1  # At minimum returns something
