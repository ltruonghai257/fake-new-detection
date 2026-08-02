"""BM25+PhoBERT CLS embedding reranker for evidence selection.

Combines keyword-based BM25 scoring with semantic similarity from PhoBERT
CLS embeddings. Greedily selects evidence to fit within 256 tokens while
maximizing relevance. Falls back to BM25-only if PhoBERT is unavailable.
"""

from __future__ import annotations

import logging
import os

from .config import settings
from .state import FactCheckState

logger = logging.getLogger(__name__)


def reranker(state: FactCheckState) -> dict:
    """Rerank and select evidence using BM25 + PhoBERT CLS embeddings.

    Pool: evidence_real + evidence_fake + evidence_social.
    Returns: {"evidence": selected, "consistency_score": float}
    Returns {} if pool is empty (D-05).
    """
    # Build evidence pool
    pool = (
        (state.get("evidence_real") or [])
        + (state.get("evidence_fake") or [])
        + (state.get("evidence_social") or [])
    )

    if not pool:
        return {}

    # Try BM25 scoring
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning(
            "rank_bm25 not installed; returning full pool with consistency_score=0.1"
        )
        return {"evidence": pool, "consistency_score": 0.1}

    # Build corpus for BM25
    corpus = [e.get("snippet", "").split() for e in pool]
    if not corpus or all(not tokens for tokens in corpus):
        logger.warning("Empty corpus; returning full pool with consistency_score=0.1")
        return {"evidence": pool, "consistency_score": 0.1}

    try:
        bm25 = BM25Okapi(corpus)
        claim_tokens = state.get("statement", "").split()
        bm25_scores = bm25.get_scores(claim_tokens)
    except Exception as exc:
        logger.warning(
            f"BM25 scoring failed: {exc}; returning full pool with consistency_score=0.1"
        )
        return {"evidence": pool, "consistency_score": 0.1}

    # Normalize BM25 scores (D-02)
    max_score = max(bm25_scores) if bm25_scores.size > 0 else 1.0
    bm25_norm = [score / (max_score or 1.0) for score in bm25_scores]

    # Try PhoBERT embedding scoring (D-01)
    embed_scores = None
    phobert_fallback = False
    snippet_embeddings = []
    try:
        # Only try PhoBERT if explicitly enabled via environment variable
        if not os.getenv("FACTCHECK_ENABLE_PHOBERT_RERANKER"):
            raise Exception(
                "PhoBERT reranker not enabled via FACTCHECK_ENABLE_PHOBERT_RERANKER"
            )

        from .agents.verify_agent import _phobert

        phobert_checker = _phobert()
        # Check if already loaded to avoid slow first-time load
        if not phobert_checker._loaded and not phobert_checker.load():
            raise Exception("PhoBERT load failed")

        import torch

        claim_text = state.get("statement", "")
        snippets = [e.get("snippet", "") for e in pool]

        # Encode claim to CLS embedding
        claim_enc = phobert_checker._tokenizer(
            claim_text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        claim_enc = {k: v.to(phobert_checker._device) for k, v in claim_enc.items()}
        with torch.no_grad():
            claim_out = phobert_checker._model.backbone(
                claim_enc["input_ids"], claim_enc["attention_mask"]
            )
            claim_embedding = claim_out.last_hidden_state[:, 0, :].cpu()

        # Encode snippets to CLS embeddings
        for snippet in snippets:
            snippet_enc = phobert_checker._tokenizer(
                snippet,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            snippet_enc = {
                k: v.to(phobert_checker._device) for k, v in snippet_enc.items()
            }
            with torch.no_grad():
                snippet_out = phobert_checker._model.backbone(
                    snippet_enc["input_ids"], snippet_enc["attention_mask"]
                )
                snippet_embedding = snippet_out.last_hidden_state[:, 0, :].cpu()
            snippet_embeddings.append(snippet_embedding)

        # Compute cosine similarity
        import torch.nn.functional as F

        embed_scores = []
        for emb in snippet_embeddings:
            sim = F.cosine_similarity(claim_embedding, emb, dim=-1).item()
            embed_scores.append(sim)

    except Exception as exc:
        logger.debug(f"PhoBERT embedding scoring failed: {exc}; using BM25-only")
        phobert_fallback = True
        embed_scores = [0.0] * len(pool)

    # Combine scores (D-02)
    if phobert_fallback:
        final_scores = bm25_norm
    else:
        final_scores = [
            0.5 * bm + 0.5 * emb for bm, emb in zip(bm25_norm, embed_scores)
        ]

    # Sort by final score descending
    scored_pool = list(zip(pool, final_scores))
    scored_pool.sort(key=lambda x: x[1], reverse=True)

    # Greedy fill to 256 tokens (D-03)
    selected = []
    total_tokens = 0

    # Always include the first item
    if scored_pool:
        first_item, _ = scored_pool[0]
        selected.append(first_item)
        # Estimate tokens via word count as fallback
        total_tokens = len(first_item.get("snippet", "").split())

    for item, score in scored_pool[1:]:
        if not item:
            continue
        # Estimate token count via word count (fallback from PhoBERT tokenizer)
        snippet_tokens = len(item.get("snippet", "").split())
        if total_tokens + snippet_tokens <= 256:
            selected.append(item)
            total_tokens += snippet_tokens

    # Compute consistency_score (D-06)
    if phobert_fallback:
        consistency_score = 0.1  # D-07
    else:
        # Mean cosine similarity of selected evidence to claim
        if selected and snippet_embeddings:
            selected_indices = [pool.index(item) for item in selected]
            selected_sims = [
                embed_scores[i] for i in selected_indices if i < len(embed_scores)
            ]
            consistency_score = (
                sum(selected_sims) / len(selected_sims) if selected_sims else 0.1
            )
        else:
            consistency_score = 0.1

    return {"evidence": selected, "consistency_score": consistency_score}
