"""
ProductIdentifier — semantic product matching using sentence-transformers embeddings.

Replaces regex + heuristics in TriageAgent.reason().
Handles any natural language description, synonym, or non-English input.
Reuses the RAGEngine's already-loaded SentenceTransformer model — no second model instance.
"""
from __future__ import annotations

import re
import numpy as np
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger("product_identifier")

# Rich product descriptions — more text = better embedding coverage.
# Include common error codes, synonyms, and distinguishing phrases per product.
PRODUCT_PROFILES: dict[str, str] = {
    "NexaAuth": (
        "NexaAuth authentication and authorization API. Handles JWT tokens, OAuth2, API keys, "
        "service accounts, token validation, login flows, session management, CORS configuration. "
        "Common errors: 401 unauthorized, invalid_token, token_expired, invalid_api_key, "
        "authentication failed, access denied, forbidden credentials, login error, bearer token, "
        "auth layer, identity service, SSO, single sign-on, permission denied."
    ),
    "NexaStore": (
        "NexaStore object storage and file management API. Handles file uploads, downloads, "
        "blob storage, CDN delivery, data ingestion, S3-compatible storage, buckets, objects. "
        "Common errors: 403 forbidden, 404 object not found, upload failed, download error, "
        "storage quota exceeded, bucket policy, presigned URL, multipart upload, "
        "data pipeline, ETL, file transfer, content delivery, static assets."
    ),
    "NexaMsg": (
        "NexaMsg messaging and webhook delivery API. Handles webhooks, event notifications, "
        "message queues, pub/sub, delivery callbacks, retry logic, event streaming. "
        "Common errors: webhook not received, delivery failure, 400 bad request, "
        "signature validation failed, event not firing, subscription missing, "
        "topic not found, message broker, event bus, notification service, callback URL."
    ),
    "NexaPay": (
        "NexaPay payments and financial routing API. Handles payment processing, reconciliation, "
        "invoicing, refunds, transaction routing, billing, charge capture, payout. "
        "Common errors: 409 duplicate payment, payment failed, invalid card, insufficient funds, "
        "idempotency key conflict, reconciliation mismatch, refund error, billing cycle, "
        "financial transaction, charge, invoice generation, revenue recognition."
    ),
}


@dataclass
class ProductMatch:
    product: str | None
    confidence: float
    needs_clarification: bool
    inferred: bool = False
    all_scores: dict[str, float] = field(default_factory=dict)


class ProductIdentifier:
    """
    Matches any free-text product description to a NexaCloud product using
    cosine similarity on sentence-transformer embeddings.

    Reuses rag_engine._model — no second SentenceTransformer instance loaded.
    """

    def __init__(self, rag_engine) -> None:
        self._model = rag_engine._model
        self._product_embeddings: dict[str, np.ndarray] = {}

        descriptions = list(PRODUCT_PROFILES.values())
        names = list(PRODUCT_PROFILES.keys())
        embeddings = self._model.encode(descriptions, normalize_embeddings=True)
        for name, emb in zip(names, embeddings):
            self._product_embeddings[name] = np.array(emb, dtype=np.float32)

        log.info("product_identifier.ready", products=names)

    def identify(self, raw_input: str, threshold: float = 0.45) -> ProductMatch:
        """
        Match raw_input to a NexaCloud product.

        Returns:
          - product + needs_clarification=False  when confidence >= threshold
          - product + needs_clarification=True   when 0.30 <= confidence < threshold
          - product=None + needs_clarification=True  when confidence < 0.30
        """
        if not raw_input or len(raw_input.strip()) < 2:
            return ProductMatch(product=None, confidence=0.0, needs_clarification=True)

        # Exact name match — short-circuit embeddings when user explicitly names a product
        _normalized = raw_input.lower()
        for product_name in PRODUCT_PROFILES:
            if re.search(r'\b' + re.escape(product_name.lower()) + r'\b', _normalized):
                return ProductMatch(
                    product=product_name,
                    confidence=1.0,
                    inferred=False,
                    needs_clarification=False,
                    all_scores={product_name: 1.0},
                )

        query_emb = self._model.encode([raw_input], normalize_embeddings=True)[0]
        query_emb = np.array(query_emb, dtype=np.float32)

        if not self._product_embeddings:
            return ProductMatch(product=None, confidence=0.0, needs_clarification=True)

        scores: dict[str, float] = {
            name: float(np.dot(query_emb, emb))
            for name, emb in self._product_embeddings.items()
        }

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best]

        # Check ambiguity — if top two are very close, ask for clarification
        sorted_scores = sorted(scores.values(), reverse=True)
        ambiguous = len(sorted_scores) >= 2 and (sorted_scores[0] - sorted_scores[1]) < 0.10

        HIGH_CONFIDENCE = 0.80  # commit silently — client clearly named the product
        if best_score >= HIGH_CONFIDENCE and not ambiguous:
            return ProductMatch(product=best, confidence=best_score,
                                inferred=False, needs_clarification=False, all_scores=scores)

        # 0.45–0.79: plausible match but not certain — use as hypothesis, ask to confirm
        if best_score >= 0.45:
            return ProductMatch(product=best, confidence=best_score,
                                inferred=True, needs_clarification=True, all_scores=scores)

        if best_score >= 0.30:
            return ProductMatch(product=best, confidence=best_score,
                                inferred=True, needs_clarification=True, all_scores=scores)

        return ProductMatch(product=None, confidence=best_score,
                            inferred=True, needs_clarification=True)

    def build_clarification_question(self, match: ProductMatch) -> dict:
        products = list(PRODUCT_PROFILES.keys())
        others = ", ".join(p for p in products if p != match.product)

        if match.product is None or match.confidence < 0.30:
            question = (
                "I want to make sure I route this to the right team. "
                "Which NexaCloud product are you working with — "
                f"{', '.join(products[:-1])}, or {products[-1]}?"
            )
        elif match.confidence < 0.60:
            question = (
                f"It sounds like this might be related to {match.product} — is that correct? "
                f"Or is it one of our other products: {others}?"
            )
        else:
            question = f"Just to confirm — are you working with {match.product}?"

        return {
            "field": "product",
            "question": question,
            "blocking": False,
            "priority": "medium",
            "is_confirmation": True,
        }
