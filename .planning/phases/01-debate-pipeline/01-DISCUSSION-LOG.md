# Phase 1: Debate Pipeline - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-03
**Phase:** 01-debate-pipeline
**Areas discussed:** Reranker embedding backend, consistency_score definition, Social loop targets + reuse, M2 graph topology (evidence flow)

---

## Reranker Embedding Backend

| Option | Description | Selected |
|--------|-------------|----------|
| BM25 + TF-IDF cosine | Use sklearn TF-IDF vectorizer — no new deps, fast, but no semantic understanding of Vietnamese | |
| BM25 + PhoBERT embeddings | Reuse already-loaded PhoBERT model (CLS token pooling) — zero new deps, strong Vietnamese semantics | ✓ |
| BM25 + sentence-transformers | New dep ~500MB; best semantic quality but heavy | |

**User's choice:** BM25 + PhoBERT embeddings
**Notes:** No new model deps. PhoBERT already loaded via lru_cache in verify_agent.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Linear combo (you decide weights) | α × bm25_norm + (1-α) × embed_sim; Claude picks alpha | ✓ |
| BM25 first, embed to break ties | Sort by BM25, use embedding cosine only for equal-score items | |
| Reciprocal Rank Fusion (RRF) | 1/(k+rank_bm25) + 1/(k+rank_embed); no hyperparameter | |

**User's choice:** Linear combo — deferred alpha choice to Claude (α=0.5)

---

| Option | Description | Selected |
|--------|-------------|----------|
| Greedy fill by token count | Add snippets in rank order until 256-token budget would be exceeded | ✓ |
| Fixed k (e.g. top-5) | Always take top-5 regardless of token count, then truncate | |

**User's choice:** Greedy fill by token count

---

| Option | Description | Selected |
|--------|-------------|----------|
| Always run BM25 fallback | If PhoBERT unavailable, use BM25-only reranking. Reranker never fails pipeline | ✓ |
| Skip reranker on unavailable | If PhoBERT unavailable, pass evidence through unranked (tier-sorted only) | |

**User's choice:** Always run BM25 fallback

---

## consistency_score Definition

| Option | Description | Selected |
|--------|-------------|----------|
| Evidence direction agreement | 1.0 - (conflicts / total): measure if real/fake evidence lists point the right direction | |
| Claim–evidence semantic overlap | Average cosine similarity between claim embedding and top evidence embeddings | |
| Inter-evidence overlap ratio | Fraction of evidence snippets sharing key entity/phrase with another snippet | |
| You decide | Deferred to Claude | ✓ |

**User's choice:** "You decide"
**Notes:** Claude chose mean cosine similarity of top-k evidence embeddings to claim embedding (PhoBERT CLS), as a side-effect of the reranker — no extra inference needed.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed 0.5 (neutral) | Assume medium credibility when unavailable | |
| Floor value 0.1 (pessimistic) | Set to AGREE-02 floor when unavailable. Don't inflate credibility | ✓ |

**User's choice:** Floor value 0.1 when PhoBERT unavailable

---

## Social Loop Targets + Reuse

| Option | Description | Selected |
|--------|-------------|----------|
| Flagged domains only | Query flagged_domains + tingia.gov.vn; skip TikTok (unreliable) | |
| tiktok.com + flagged domains | Add tiktok.com to include_domains alongside flagged | |
| Separate from existing social_search_agent | Define target list independently | |
| You decide | Deferred to Claude | ✓ |

**User's choice:** "You decide"
**Notes:** Claude chose tiktok.com + settings.flagged_domains — separate from M1 social_search_agent (twitter/facebook).

---

| Option | Description | Selected |
|--------|-------------|----------|
| Appended to evidence_real | Social treated as supplemental credible evidence | |
| Appended to evidence_fake | Social targets flagged domains, goes to evidence_fake | |
| New field evidence_social | Third list; keeps the three cleanly separated | ✓ |
| Split by source_tier | classify_domain() directs each result to real or fake | |

**User's choice:** New field `evidence_social: List[Evidence]`

---

| Option | Description | Selected |
|--------|-------------|----------|
| Include in rerank pool | Reranker processes evidence_real ∪ evidence_fake ∪ evidence_social | |
| Append after real+fake | Social appended only if token budget remains | |
| You decide (social is neutral) | User said: "social should not be considered fake or real, it may claim statement" | ✓ |

**User's choice:** Unified rerank pool — social evidence is neutral, ranked purely by relevance to claim, source_tier preserved.

---

## M2 Graph Topology (Evidence Flow)

| Option | Description | Selected |
|--------|-------------|----------|
| Replace search_agent | M2 skips M1 search_agent; uses real_source + fake_source only | ✓ |
| Keep search_agent, add on top | search_agent still runs, real/fake add new fields | |
| search_agent only as fallback | M2 skips by default; fallback if both lists empty | |

**User's choice:** Replace search_agent — M2 starts with parallel real_source + fake_source fan-out.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Reranker writes back to `evidence` | verify_agent unchanged — still reads `evidence` | ✓ |
| Reranker writes to `evidence_ranked` | New field; verify_agent updated to read from it | |

**User's choice:** Reranker writes back to `evidence`

---

| Option | Description | Selected |
|--------|-------------|----------|
| Parallel (LangGraph fan-out) | Both agents run concurrently | |
| Sequential (real then fake) | Simpler to debug, slightly slower | |
| Debate until agree | User initially suggested this | |

**User's choice:** Parallel fan-out
**Notes:** User initially said "they should run parallel and they should debate together (not stop until they agree one decision)." After clarification, confirmed this refers to debate_node behavior (already scoped with fixed max_rounds), not source agent behavior.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Removed from M2 graph | M2 uses social_loop_agent (tiktok + flagged). M1 social_search_agent stays in build_graph() only | ✓ |
| Kept alongside social_loop | twitter/facebook + tiktok/flagged both run in M2 | |

**User's choice:** M1 social_search_agent removed from build_debate_graph()

---

## Claude's Discretion

- **Reranker alpha=0.5**: User deferred blend weight to Claude; chose equal weighting.
- **consistency_score formula**: User deferred; Claude chose mean cosine sim to claim embedding (PhoBERT side-effect).
- **social_loop domains**: User deferred; Claude chose tiktok.com + settings.flagged_domains.

## Deferred Ideas

- DEBATE-EXT-01 (adaptive convergence / Wald-SPRT) — deferred to v4+
- Twitter/Facebook social search in M2 — stays in M1 build_graph() only
