# Phase 1 Debate Pipeline — Research

**Produced:** 2026-08-03  
**Purpose:** Concrete code-level answers for the planner. Covers all 10 required topics.

---

## 1. Existing Patterns — New Files and Their Mimics

Each new file maps to one existing file to copy style, structure, and guard patterns from.

### `reranker.py` → mimic `factcheck_agents/agents/verify_agent.py`

Same pattern: `lru_cache(maxsize=1)` singleton for the shared PhoBERT handle, `ThreadPoolExecutor` for parallel score computation, inner `_run_*` helpers that never raise (return a safe fallback instead). BM25 replaces the model runner in the fallback branch.

```python
# verify_agent.py style — reuse directly
from functools import lru_cache
from ..models import PhoBERTChecker

@lru_cache(maxsize=1)
def _phobert() -> PhoBERTChecker:
    return PhoBERTChecker()
```

The reranker calls `_phobert()` — it does NOT create a second PhoBERTChecker. The same instance already loaded by `verify_agent` is returned from the module-level cache.

### `real_source_agent.py` → mimic `factcheck_agents/agents/search_agent.py`

Copy the exact `web_search(q, max_results=N, include_domains=[...])` call signature, the `seen: set` dedup loop, the `classify_domain(url)` call, and the `_fetch_evidence_image(url)` call per result. Return dict with a new state key (`evidence_real`) instead of `evidence`.

### `fake_source_agent.py` → mimic `factcheck_agents/agents/search_agent.py`

Same structure as `real_source_agent.py`. Adds an optional Google Fact Check API call (stubbed with `[]` when `settings.google_factcheck_api_key` is `None`) before the `web_search` pass. Returns `evidence_fake`.

### `social_loop_agent.py` → mimic `factcheck_agents/agents/social_search_agent.py`

Copy the `include_domains` web_search call, the `seen` set seeded from `eg.graph.nodes`, the `eg.add_node` / `eg.add_edge` merge loop, and the `source_tier="social"` tag. Swap the target domains to `["tiktok.com"] + settings.flagged_domains.split(",")`. Write results to `evidence_social` (not into the evidence_graph).

### `agreement_gate.py` → mimic `route_after_verify` in `factcheck_agents/graph.py`

Pure routing function: `def route_after_agreement(state: FactCheckState) -> str`. Reads `state.get("agreement_score", 0.0)` against `settings.agreement_threshold`. Returns one of two string node names. No side effects, no imports beyond `FactCheckState` and `settings`.

```python
# graph.py pattern
def route_after_verify(state: FactCheckState) -> str:
    if state.get("reliability_signal"):
        return "social_search"
    return "conclusion"
```

### `debate_node.py` → mimic `factcheck_agents/agents/conclusion_agent.py`

Use the `[("system", PROMPT), ("user", user_text)]` LLM invoke pattern. Wrap in `try/except` that catches any `Exception` and appends a fallback turn dict. Accumulate rounds in a list and append to `state["debate_turns"]`. Return `{"debate_turns": [...], "debate_exit_reason": str, "messages": [...]}`.

### `judge_agent.py` → mimic `factcheck_agents/agents/conclusion_agent.py`

Identical overall skeleton: call `get_llm()`, fall back to `_fallback_verdict()` when `None`, build `Verdict` TypedDict from LLM JSON response, return `{"verdict": verdict, "messages": [...]}`. The only addition: read `state.get("debate_turns")` and `state.get("weight_breakdown")` to augment the user prompt.

---

## 2. State Fields Audit

### `FactCheckState` — current fields (all `total=False` via class declaration)

| Field | Type | Writer |
|---|---|---|
| `statement` | `str` | caller |
| `image_path` | `Optional[str]` | caller |
| `language` | `str` | caller |
| `search_queries` | `List[str]` | `search_agent` |
| `evidence` | `List[Evidence]` | `search_agent`, `reranker` |
| `model_results` | `List[ModelResult]` | `verify_agent` |
| `evidence_graph` | `Optional[Any]` | `search_agent`, `social_search_agent` |
| `reliability_signal` | `Optional[bool]` | `verify_agent` |
| `verdict` | `Verdict` | `conclusion_agent` / `judge_agent` |
| `messages` | `Annotated[list, add_messages]` | every agent |
| `errors` | `List[str]` | any agent |
| `meta` | `dict[str, Any]` | caller / any agent |

### `Evidence` — current fields (all `total=False`)

`title`, `url`, `snippet`, `content`, `source`, `score`, `source_tier`, `image_path`, `image_caption`

No new fields needed on `Evidence`.

### New fields to add to `FactCheckState` — all need `total=False`

Because `FactCheckState` already declares `class FactCheckState(TypedDict, total=False)`, every field in the class body is implicitly optional. Simply append the new field names to the class body; no per-field `total=False` annotation is needed.

```python
# append to FactCheckState body in state.py
evidence_real: List[Evidence]       # written by real_source_agent
evidence_fake: List[Evidence]       # written by fake_source_agent
evidence_social: List[Evidence]     # written by social_loop_agent
social_loop_fired: bool             # written by agreement_gate / social_loop_agent
request_id: str                     # set in initial_state(); UUID4
consistency_score: float            # set by reranker (side-effect of embedding)
agreement_score: float              # set by agreement_gate
debate_turns: List[dict]            # set by debate_node
debate_exit_reason: str             # set by debate_node
weight_breakdown: dict              # set by judge_agent
```

All 10 fields go into the same `FactCheckState(TypedDict, total=False)` class — no separate subclass or wrapper needed.

---

## 3. Config.py Pattern

Every setting follows this exact form:

```python
field_name: type = field(
    default_factory=lambda: os.getenv("ENV_VAR_NAME", "default")
)
```

For numeric types the `os.getenv` return is wrapped:

```python
# float
reliability_threshold: float = field(
    default_factory=lambda: float(
        os.getenv("FACTCHECK_RELIABILITY_THRESHOLD", "0.5")
    )
)

# int
max_results: int = field(
    default_factory=lambda: int(os.getenv("FACTCHECK_MAX_RESULTS", "6"))
)

# Optional[str] — no wrapper, None when unset
openai_api_key: Optional[str] = field(
    default_factory=lambda: os.getenv("OPENAI_API_KEY")
)
```

New fields to add (append inside `Settings` dataclass, after `debate_rounds`):

```python
# ── Debate pipeline (Phase 1 / M2) ───────────────────────────────────────
agreement_threshold: float = field(
    default_factory=lambda: float(os.getenv("FACTCHECK_AGREEMENT_THRESHOLD", "0.7"))
)
max_debate_rounds: int = field(
    default_factory=lambda: int(os.getenv("FACTCHECK_MAX_DEBATE_ROUNDS", "2"))
)
google_factcheck_api_key: Optional[str] = field(
    default_factory=lambda: os.getenv("GOOGLE_FACTCHECK_API_KEY")
)
social_loop_min_count: int = field(
    default_factory=lambda: int(os.getenv("FACTCHECK_SOCIAL_LOOP_MIN_COUNT", "3"))
)
social_loop_min_credibility: float = field(
    default_factory=lambda: float(
        os.getenv("FACTCHECK_SOCIAL_LOOP_MIN_CREDIBILITY", "0.6")
    )
)
```

---

## 4. LangGraph Fan-Out Pattern

### Mechanism

LangGraph's `StateGraph` natively supports static parallel fan-out: add `add_edge(upstream, nodeA)` and `add_edge(upstream, nodeB)` from the same source. LangGraph runs both nodes concurrently and creates an implicit barrier — no downstream node executes until all branches that feed it have completed.

**`Send` API is not needed here.** `Send` is for dynamic fan-out where the branch count is unknown at graph-build time (e.g., map over a variable-length list). The Phase 1 fan-out is always exactly 2 fixed nodes.

### State merge

LangGraph merges branch outputs by field. Fields with no reducer (plain `List`, `str`, `bool`) use last-writer-wins. Since `real_source_agent` writes only `evidence_real` and `fake_source_agent` writes only `evidence_fake`, there is zero field conflict. The `messages` field uses the `add_messages` reducer (`Annotated[list, add_messages]` in `state.py`) and correctly concatenates both agents' messages.

### EVRET-04 insertion — the NEI gate node

The fan-out barrier is created by having both source nodes edge into a single next node. Insert a lightweight `nei_gate` node between the fan-out and the reranker:

```python
# in build_debate_graph()
g.add_node("real_source", real_source_agent)
g.add_node("fake_source", fake_source_agent)
g.add_node("nei_gate", lambda state: {})   # passthrough; routing happens on edge
g.add_node("reranker", reranker)
# ...

g.add_edge(START, "real_source")
g.add_edge(START, "fake_source")
g.add_edge("real_source", "nei_gate")      # both feed into barrier node
g.add_edge("fake_source", "nei_gate")      # LangGraph waits for both
g.add_conditional_edges(
    "nei_gate",
    route_nei_check,
    {"reranker": "reranker", "judge": "judge"},
)
```

`route_nei_check` mirrors the `route_after_verify` style:

```python
def route_nei_check(state: FactCheckState) -> str:
    if not (state.get("evidence_real") or []) and not (state.get("evidence_fake") or []):
        return "judge"   # EVRET-04: NEI short-circuit
    return "reranker"
```

When `"judge"` is selected via the NEI short-circuit, `judge_agent` must detect the empty-evidence condition and produce `label="NEI"` / `verdict_binary="FAKE"` without attempting debate.

---

## 5. Import Path Conventions

All agent files live in `factcheck_agents/agents/`. Their imports follow this pattern (from `search_agent.py`, `verify_agent.py`, `social_search_agent.py`, `conclusion_agent.py`):

```python
from ..config import settings                   # Settings singleton
from ..state import Evidence, FactCheckState    # TypedDicts
from ..tools.web_search import web_search       # search tool
from ..source_tier import classify_domain       # URL → tier string
from ..helpers import _fetch_evidence_image     # image fetch utility
from .llm import get_llm, parse_json            # LLM helpers (same package)
from ..graph_utils import EvidenceGraph         # if writing to evidence_graph
from ..models import PhoBERTChecker             # if loading model directly
from ..models.phobert_checker import build_evidence_text  # if building text
```

`reranker.py` lives in `factcheck_agents/agents/` and references the PhoBERT singleton already in `verify_agent`:

```python
# reranker.py
from .verify_agent import _phobert   # reuse cached singleton — DO NOT create new
```

No new top-level package is needed. All 7 new files are `factcheck_agents/agents/<name>.py`.

---

## 6. Test Patterns

### Patching `web_search`

Target is the name as imported in the module under test — not the original definition location.

```python
# test_search_agent.py — exact pattern
@patch("factcheck_agents.agents.search_agent.web_search")
def test_three_passes(mock_ws):
    mock_ws.return_value = []
    ...

# For real_source_agent.py (mirroring search_agent):
@patch("factcheck_agents.agents.real_source_agent.web_search")
def test_real_source_calls_web_search(mock_ws): ...

# For fake_source_agent.py:
@patch("factcheck_agents.agents.fake_source_agent.web_search")
def test_fake_source_calls_web_search(mock_ws): ...
```

### Patching `get_llm`

```python
# test_conclusion_agent.py — two forms
@patch("factcheck_agents.agents.conclusion_agent.get_llm", return_value=None)
def test_fallback_no_llm(mock_llm): ...

@patch("factcheck_agents.agents.conclusion_agent.get_llm")
def test_with_llm(mock_llm):
    mock_llm.return_value = MagicMock(
        invoke=lambda _msgs: MagicMock(content=json.dumps({...}))
    )
    ...

# For judge_agent.py (same form, different module path):
@patch("factcheck_agents.agents.judge_agent.get_llm", return_value=None)
def test_judge_fallback(mock_llm): ...
```

### Patching model singletons (`_phobert`, `_coolant`)

```python
# test_verify_agent.py — exact pattern
@patch("factcheck_agents.agents.verify_agent._run_coolant")
@patch("factcheck_agents.agents.verify_agent._run_phobert")
def test_verify_both_models(mock_phobert, mock_coolant):
    mock_phobert.return_value = ModelResult(model="phobert_vifactcheck", available=False, note="x")
    mock_coolant.return_value = ModelResult(model="coolant", available=False, note="x")
    ...

# For reranker.py — patch the singleton getter directly:
@patch("factcheck_agents.agents.reranker._phobert")
def test_reranker_bm25_fallback(mock_ph):
    mock_ph.return_value = MagicMock(...)  # or raise to trigger BM25-only
    ...
```

### Patching `_fetch_evidence_image`

```python
# test_search_agent.py — autouse fixture pattern
@pytest.fixture(autouse=True)
def _stub_fetch_evidence_image():
    with patch(
        "factcheck_agents.helpers._fetch_evidence_image",
        return_value=(None, None),
    ):
        yield

# Or per-test (test_social_search_agent.py pattern):
@patch(
    "factcheck_agents.agents.social_search_agent._fetch_evidence_image",
    return_value=(None, None),
)
@patch("factcheck_agents.agents.social_search_agent.web_search")
def test_something(mock_ws, mock_img): ...
```

Note: social_search_agent imports `_fetch_evidence_image` directly (`from ..helpers import _fetch_evidence_image`), so the patch target is `factcheck_agents.agents.<module>._fetch_evidence_image`, not `factcheck_agents.helpers._fetch_evidence_image`. Both work; use the module-local path for precision.

---

## 7. Atomic File I/O

No existing precedent in the codebase — `helpers.py` only writes image cache files. Use the following prescribed pattern for debate/verdict logs.

### Directory creation

```python
from pathlib import Path

_DEBATE_LOG_DIR = Path("logs") / "debates"
_VERDICT_LOG_DIR = Path("logs") / "verdicts"

_DEBATE_LOG_DIR.mkdir(parents=True, exist_ok=True)
_VERDICT_LOG_DIR.mkdir(parents=True, exist_ok=True)
```

Call `mkdir` once at module import time or at the top of the agent function — `exist_ok=True` makes it idempotent.

### JSONL append (debate turns)

```python
import json
from pathlib import Path

def _append_debate_turn(request_id: str, turn: dict) -> None:
    path = Path("logs") / "debates" / f"{request_id}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(turn, ensure_ascii=False) + "\n")
```

### JSON write (verdict)

```python
def _write_verdict_log(request_id: str, verdict: dict) -> None:
    path = Path("logs") / "verdicts" / f"{request_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)
```

Both functions must be wrapped in `try/except Exception: pass` inside the agent — log writes must never crash the pipeline.

### `request_id` generation

Add to `initial_state()` in `graph.py`:

```python
import uuid

def initial_state(...) -> FactCheckState:
    return FactCheckState(
        ...
        request_id=str(uuid.uuid4()),
        ...
    )
```

---

## 8. Dependency Gap

`rank_bm25>=0.2.2` is **not present** in `pyproject.toml`.

Current `agents` extra (lines 57–64):

```toml
agents = [
    "langgraph>=0.2.28",
    "langchain-core>=0.3.0",
    "langchain-openai>=0.2.0",
    "mcp>=1.2.0",
    "networkx>=3.0",
    "openai>=1.40.0",
]
```

Add `rank_bm25>=0.2.2` as the last entry:

```toml
agents = [
    "langgraph>=0.2.28",
    "langchain-core>=0.3.0",
    "langchain-openai>=0.2.0",
    "mcp>=1.2.0",
    "networkx>=3.0",
    "openai>=1.40.0",
    "rank_bm25>=0.2.2",
]
```

Exact insertion: after line 63 (`"openai>=1.40.0",`), before line 64 (`]`).

No other dependency is missing. `networkx` (already present) is sufficient for `EvidenceGraph`. `numpy` (present in main deps) covers the cosine-similarity computation in the reranker.

---

## 9. NEI Gate (EVRET-04) — Insertion Point in `build_debate_graph()`

### Where

Insert the NEI gate between the fan-out barrier and the reranker. The barrier is the `nei_gate` node that both `real_source` and `fake_source` edge into (described in §4). The conditional edge comes off `nei_gate`.

### Full topology skeleton for `build_debate_graph()`

```python
def build_debate_graph(checkpointer=None):
    g = StateGraph(FactCheckState)

    # nodes
    g.add_node("real_source", real_source_agent)
    g.add_node("fake_source", fake_source_agent)
    g.add_node("nei_gate", lambda state: {})        # ← EVRET-04 insertion point
    g.add_node("reranker", reranker)
    g.add_node("social_loop", social_loop_agent)
    g.add_node("verify", verify_agent)
    g.add_node("agreement_gate_node", lambda state: {})  # passthrough; routing on edge
    g.add_node("debate", debate_node)
    g.add_node("judge", judge_agent)

    # fan-out from START
    g.add_edge(START, "real_source")
    g.add_edge(START, "fake_source")

    # fan-out barrier + NEI check
    g.add_edge("real_source", "nei_gate")
    g.add_edge("fake_source", "nei_gate")
    g.add_conditional_edges(
        "nei_gate",
        route_nei_check,                            # ← returns "reranker" or "judge"
        {"reranker": "reranker", "judge": "judge"},
    )

    # main path
    g.add_edge("reranker", "social_loop")           # or conditional if social optional
    g.add_edge("social_loop", "verify")
    g.add_edge("verify", "agreement_gate_node")
    g.add_conditional_edges(
        "agreement_gate_node",
        route_after_agreement,
        {"debate": "debate", "judge": "judge"},
    )
    g.add_edge("debate", "judge")
    g.add_edge("judge", END)

    return g.compile(checkpointer=checkpointer)
```

`route_nei_check` (defined at module level, like `route_after_verify`):

```python
def route_nei_check(state: FactCheckState) -> str:
    real = state.get("evidence_real") or []
    fake = state.get("evidence_fake") or []
    if not real and not fake:
        return "judge"
    return "reranker"
```

When NEI short-circuit fires, `judge_agent` receives a state with empty `evidence_real`/`evidence_fake`. It must detect this and produce a fixed NEI verdict without calling the LLM debate path.

---

## 10. Edge Case Inventory by Wave

### Wave 1 — State + Config additions

No runtime edge cases (pure TypedDict/dataclass additions). The only risk: forgetting `total=False` — mitigated since `FactCheckState` already declares `total=False` at class level; all new fields inherit it.

### Wave 2 — real_source_agent, fake_source_agent, reranker

| Situation | Required behaviour |
|---|---|
| `web_search()` raises exception | Catch in `_run_search()` helper, return `[]`; log to `state["errors"]` |
| Both source agents return `[]` | `route_nei_check` fires, skip to `judge` with NEI verdict |
| Only `evidence_real` is empty | Pool is `evidence_fake ∪ evidence_social`; reranker proceeds |
| PhoBERT checkpoint missing (BM25 fallback) | `_phobert().load()` returns `False`; reranker skips embedding, uses BM25 scores only; `consistency_score = 0.1` |
| `rank_bm25` not installed | `ImportError` → catch, return evidence unchanged (no reranking); log warning |
| All evidence snippets empty strings | BM25 tokenizer produces empty lists; no crash, scores all zero, preserve original order |
| Single evidence item in pool | BM25 max-score normalization divides by itself → 1.0; no division-by-zero guard needed except when list is empty (guard with `if not corpus: return evidence`) |
| Token budget already exceeded by first item | Return that first item alone (single item always fits if max_length=256; truncation handled by PhoBERT tokenizer) |
| `evidence_fake` is empty list, `evidence_real` non-empty | Only real evidence goes into reranker pool; `fake_source_agent` stub returns `[]` silently |

### Wave 3 — social_loop_agent, agreement_gate

| Situation | Required behaviour |
|---|---|
| Social search returns 0 results | `evidence_social = []`; `social_loop_fired = True` (it ran); pipeline continues |
| Social search raises exception | Catch, `evidence_social = []`, append to `errors` |
| PhoBERT unavailable during agreement gate | `consistency_score` already set to `0.1` by reranker (D-07); gate uses that floor value |
| `agreement_score >= settings.agreement_threshold` | Route directly to `judge`; skip `debate` entirely |
| LLM unavailable but `agreement_score < threshold` | Route to `debate_node`; debate_node checks `get_llm() is None` and sets `debate_exit_reason = "no_llm"`, `debate_turns = []` |
| `settings.agreement_threshold` not set | Default `0.7` from config; no crash |

### Wave 4 — debate_node

| Situation | Required behaviour |
|---|---|
| `get_llm()` returns `None` | Return `{"debate_turns": [], "debate_exit_reason": "no_llm", "messages": [...]}` |
| `evidence_real` empty (real advocate has nothing) | Real advocate prompt includes "(no supporting evidence available)"; still produces argument from statement alone |
| `evidence_fake` empty (fake advocate has nothing) | Same stub argument pattern |
| LLM raises mid-round | Catch, append error turn `{"round": n, "role": "error", "content": str(exc)}`, set `debate_exit_reason = "llm_error"`, stop |
| Round 1 LLM call succeeds but round 2 fails | Keep round 1 turns, mark exit_reason = "llm_error" at round 2 boundary |
| `max_debate_rounds` reached | Normal exit, `debate_exit_reason = "max_rounds"` |
| Debate result JSON unparseable | `parse_json()` returns `None`; append raw content as `{"role": "...", "content": raw, "parse_error": True}` |

### Wave 5 — judge_agent

| Situation | Required behaviour |
|---|---|
| NEI short-circuit path (evidence_real and evidence_fake both empty) | Return `Verdict(label="NEI", verdict_binary="FAKE", verdict_label_vi="Giả", confidence=0.1, rationale="No evidence retrieved from either source.")` without LLM call |
| `get_llm()` returns `None` | Call `_fallback_verdict()` (copy from `conclusion_agent.py`); do not crash |
| `debate_turns` is empty | Build user prompt without debate section; normal LLM verdict path |
| `weight_breakdown` missing from state | Use equal weights in prompt; do not key-error |
| LLM raises | Fall through to `_fallback_verdict()` with `rationale += f" (LLM error: {exc})"` — exact pattern from `conclusion_agent.py` line 168–169 |
| Verdict JSON missing required keys | Default each field: `label` → `"UNVERIFIED"`, `confidence` → `0.0`, `rationale` → `""`, `citations` → evidence URLs |
| Atomic file write fails (disk full, permission) | Wrap in `try/except Exception: pass`; verdict still returned to caller |

---

*Research produced by gsd-phase-researcher. Source files read: state.py, config.py, graph.py, agents/__init__.py, agents/search_agent.py, agents/verify_agent.py, agents/evaluate_agent.py, agents/social_search_agent.py, agents/conclusion_agent.py, agents/llm.py, source_tier.py, helpers.py, tools/web_search.py, graph_utils.py, models/phobert_checker.py, pyproject.toml, tests/test_verify_agent.py, tests/test_search_agent.py, tests/test_social_search_agent.py, tests/test_conclusion_agent.py, 01-CONTEXT.md.*
