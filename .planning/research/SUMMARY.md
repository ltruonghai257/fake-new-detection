# Research Summary — v2.0 Evidence-Graph Vietnamese Pipeline

## Stack Additions

| Addition | Why | Notes |
|----------|-----|-------|
| `networkx` | In-memory evidence DiGraph (entities, snippets, source tiers, relations) | Pure Python, no external process; standard for NLP fact-checking graphs (EVOCA, ClaimVer) |

No other new deps required. `networkx` is the only addition to `factcheck_agents/requirements.txt`.

## Architecture: Key Findings

**TradingAgents mirrors exactly our target shape:**
- Sequential nodes sharing one TypedDict state (not goroutine-style parallelism)
- Each node writes its own slice; downstream reads it
- `ConditionalLogic.should_continue_debate` pattern → our `route_after_verify` function
- "Build once, query downstream" — analysts don't re-fetch; graph context travels in state

**LangGraph conditional edge (exact pattern):**
```python
g.add_conditional_edges("verify", route_after_verify, {"social_search": ..., "conclusion": ...})
```
- `route_after_verify(state)` reads `state["reliability_signal"]` synchronously
- Reliability signal must be SET inside the verify node (not async in the router)

**ViFactCheck evidence format:**
- Input: `(statement, evidence_text)` tokenized as sentence pair, `truncation="only_second"`
- In v2.0, `build_evidence_text()` should prioritize trusted-tier snippets from the evidence graph

**Evidence graph:**
- `networkx.DiGraph` — nodes: statement + evidence_snippet nodes; edges: `supports`/`contradicts`/`mentions` with `tier` attribute
- Built once in the Evidence/Search agent, stored in `FactCheckState.evidence_graph`
- Verify and Conclusion agents query it via `G.nodes` / `G.edges` — no re-fetching

## Feature Table Stakes

| Feature | Milestone |
|---------|-----------|
| Source-tier tagging (trusted/flagged/unknown) | v2.0 |
| Evidence graph built once, queried downstream | v2.0 |
| Reliability signal from fused model outputs | v2.0 |
| Conditional social-search on reliability | v2.0 |
| Binary verdict (REAL/FAKE) + `verdict_label_vi` | v2.0 |
| Vietnamese rationale + citations | v2.0 |
| Additive output (no breaking change to callers) | v2.0 |

## Watch Out For

1. **LangGraph router pitfall**: path function must return a key present in `path_map`; test all branches explicitly — compile() doesn't catch unmapped keys, only invoke() does
2. **PhoBERT evidence text**: `build_evidence_text()` must keep statement-level context; truncation=`only_second` means evidence is what gets cut, not the statement — preserve this
3. **Binary mapping UNVERIFIED→FAKE**: document clearly in rationale so users understand why UNVERIFIED is reported as Giả (not a model failure, a design decision)
4. **Concurrent model calls**: `asyncio.gather` or `concurrent.futures.ThreadPoolExecutor` for PhoBERT + COOLANT; but LangGraph nodes are sync by default — use `ThreadPoolExecutor` inside the verify node, not async
5. **Social search rate**: site-restricted queries add 2-4 more API calls per invocation; keep max_results low (2-3) for social tier to avoid Tavily rate limits
6. **Vietnamese prompts**: LLM must be told explicitly to output Vietnamese; system prompt must include language instruction — not just a translation of existing prompts
7. **networkx not installed** in current `factcheck_agents/requirements.txt` — must add; confirm it doesn't conflict with existing deps (it won't, pure Python)
