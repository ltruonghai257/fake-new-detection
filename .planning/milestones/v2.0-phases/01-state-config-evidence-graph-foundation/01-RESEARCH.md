# Phase 1: Research — State, Config & Evidence Graph Foundation

**Phase:** 01 — State, Config & Evidence Graph Foundation
**Requirements:** EVGRAPH-01, EVGRAPH-02, EVGRAPH-03, CONFIG-01, CONFIG-02, CONFIG-03
**Researched:** 2026-07-19

---

## Summary

Phase 1 is pure data contracts and utility code — no LLM calls, no LangGraph wiring, no HTTP. All changes are additive (TypedDict `total=False`, optional fields, new modules). The primary planning risk is ensuring `networkx` is available and that `state.py` stays import-clean (no heavy imports). Test scaffolding must be created because `tests/` currently has no `factcheck_agents/` subdirectory.

---

## Codebase Findings

### `factcheck_agents/state.py` (current)

```
Evidence(TypedDict, total=False): title, url, snippet, source, score
ModelResult(TypedDict, total=False): model, available, label, label_id, probabilities, confidence, note
Verdict(TypedDict, total=False): label, confidence, rationale, citations, recommendation
FactCheckState(TypedDict, total=False): statement, image_path, language, search_queries,
    evidence, model_results, verdict, messages, errors, meta
```

**Key observations:**
- All TypedDicts use `total=False` — every new field is automatically optional for existing callers.
- `state.py` imports only `Annotated`, `Any`, `List`, `Optional`, `TypedDict` from `typing` plus `add_messages` from `langgraph`. It is intentionally import-cheap.
- `Optional[Any]` for `evidence_graph` follows the `meta: dict[str, Any]` precedent — same pattern already in use.
- `from __future__ import annotations` is present — `Literal` import must be explicit.

**Changes needed:**
1. Add `Literal` to the `typing` import.
2. Add `source_tier: Literal["trusted", "flagged", "social", "unknown"]` to `Evidence`.
3. Add `evidence_graph: Optional[Any]` to `FactCheckState`.
4. Add `reliability_signal: Optional[bool]` to `FactCheckState`.

### `factcheck_agents/config.py` (current)

```python
@dataclass
class Settings:
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    llm_model: str = field(default_factory=lambda: os.getenv("FACTCHECK_LLM_MODEL", "gpt-4o-mini"))
    max_results: int = field(default_factory=lambda: int(os.getenv("FACTCHECK_MAX_RESULTS", "6")))
    ...

settings = Settings()
```

**Key observations:**
- Module-level `settings = Settings()` singleton used by all agents via `from .config import settings`.
- Three field type patterns: `Optional[str]` (no default), `str` (with default), `int` (with int() conversion).
- New fields follow the exact same `field(default_factory=lambda: ...)` pattern.
- `FACTCHECK_*` namespace is consistent — all new env vars should follow it.

**Changes needed:**
1. Add `trusted_domains: str` → `FACTCHECK_TRUSTED_DOMAINS`, default `"vnexpress.net,thanhnien.vn,dantri.com.vn,tuoitre.vn"`.
2. Add `flagged_domains: str` → `FACTCHECK_FLAGGED_DOMAINS`, default `"kenh14.vn"`.
3. Add `reliability_threshold: float` → `FACTCHECK_RELIABILITY_THRESHOLD`, default `"0.5"` (parse with `float()`).

### `pyproject.toml` (current `agents` extra)

```toml
agents = [
    "langgraph>=0.2.28",
    "langchain-core>=0.3.0",
    "langchain-openai>=0.2.0",
    "openai>=1.40.0",
    "mcp>=1.2.0",
]
```

**Change needed:** Add `"networkx>=3.0"` to `agents` extra.

**Note:** `networkx` is widely available and has no binary dependencies — safe to add with a loose `>=3.0` bound. `networkx` 3.x is the stable API series; `DiGraph` and its methods are stable across all 3.x versions.

### `tests/` directory

No `factcheck_agents/` subdirectory exists. Existing test subdirectories:
- `tests/crawler/`, `tests/helpers/`, `tests/processing/`
- `tests/conftest.py` (empty)

**Implication:** Plan 01-02 must create `tests/factcheck_agents/__init__.py` before writing tests, following the pattern of other subdirectories.

### `factcheck_agents/` module structure

New files to create:
- `factcheck_agents/graph_utils.py` — `EvidenceGraph` class
- `factcheck_agents/source_tier.py` — `classify_domain()` function

Neither is imported by existing code, so they can be created cleanly without circular import risk.

---

## networkx.DiGraph API Reference

Relevant API for `EvidenceGraph` wrapper:

```python
import networkx as nx

g = nx.DiGraph()
g.add_node("statement", text="...", node_type="statement")
g.add_node("url_or_hash", title="...", snippet="...", source_tier="trusted", node_type="evidence")
g.add_edge("statement", "url_or_hash", type="supports")
g.add_edge("url_or_hash", "statement", type="contradicts")

# Access
list(g.nodes(data=True))
list(g.edges(data=True))
g.in_edges("statement", data=True)   # Phase 5 will use this
```

**DiGraph construction notes:**
- `add_node(id, **attrs)` — id is any hashable; keyword attrs stored on the node.
- `add_edge(src, dst, **attrs)` — directed edge; keyword attrs stored on the edge.
- `DiGraph` allows duplicate `add_node` calls (updates attrs, doesn't error).
- `to_evidence_list()` needs to reconstruct `Evidence` dicts from nodes: iterate `g.nodes(data=True)`, filter `node_type == "evidence"`, yield `{"title": d["title"], "snippet": d["snippet"], "source_tier": d["source_tier"], "url": node_id}`.

---

## `EvidenceGraph` Design (D-03, D-04, D-05)

```python
class EvidenceGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    @classmethod
    def build_from_evidence(cls, evidence_list: list) -> "EvidenceGraph":
        eg = cls()
        eg.graph.add_node("statement", node_type="statement")
        for ev in evidence_list:
            node_id = ev.get("url") or hash(ev.get("snippet", ""))
            eg.graph.add_node(node_id,
                node_type="evidence",
                title=ev.get("title", ""),
                snippet=ev.get("snippet", ""),
                source_tier=ev.get("source_tier", "unknown"))
            eg.graph.add_edge("statement", node_id, type="mentions")
        return eg

    def add_node(self, id, attrs: dict):
        self.graph.add_node(id, **attrs)

    def add_edge(self, src, dst, type: str, attrs: dict = None):
        self.graph.add_edge(src, dst, type=type, **(attrs or {}))

    def to_evidence_list(self) -> list:
        result = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") == "evidence":
                result.append({"url": node_id, ...})
        return result
```

**Critical note:** `type` is a Python builtin — safe to use as an edge attribute keyword since it's passed via `**kwargs` to networkx, not as a Python expression. No shadowing issue.

---

## `classify_domain()` Design (D-07, D-08)

```python
from .config import settings
from urllib.parse import urlparse

def classify_domain(url: str) -> str:
    hostname = urlparse(url).hostname or ""
    domain = hostname.removeprefix("www.")
    trusted = [d.strip() for d in settings.trusted_domains.split(",") if d.strip()]
    flagged = [d.strip() for d in settings.flagged_domains.split(",") if d.strip()]
    if any(domain == t or domain.endswith("." + t) for t in trusted):
        return "trusted"
    if any(domain == f or domain.endswith("." + f) for f in flagged):
        return "flagged"
    return "unknown"
```

**Key decisions:**
- `urllib.parse.urlparse` is stdlib — no new imports needed.
- Reads `settings.trusted_domains` / `settings.flagged_domains` at call-time, not module-import time → safe if env vars change between runs.
- Subdomain support: `domain.endswith("." + t)` handles `www.vnexpress.net` → `"trusted"`.
- `"social"` tier is not returned by `classify_domain()` — it is set explicitly by `social_search_agent` (Phase 4). `classify_domain()` only returns `"trusted" | "flagged" | "unknown"`.

---

## Test Design (TEST-01, TEST-02)

### `tests/factcheck_agents/test_source_tier.py`

```python
def test_trusted_domain(): assert classify_domain("https://vnexpress.net/...") == "trusted"
def test_flagged_domain(): assert classify_domain("https://kenh14.vn/...") == "flagged"
def test_unknown_domain(): assert classify_domain("https://example.com/...") == "unknown"
def test_www_prefix(): assert classify_domain("https://www.tuoitre.vn/...") == "trusted"
def test_subdomain(): assert classify_domain("https://news.dantri.com.vn/...") == "trusted"
def test_empty_url(): assert classify_domain("") == "unknown"
```

### `tests/factcheck_agents/test_evidence_graph.py`

```python
def test_build_from_evidence_creates_nodes(): ...
def test_build_from_evidence_statement_node(): g.graph.nodes["statement"]["node_type"] == "statement"
def test_add_edge_type_attribute(): edge data has type="supports"
def test_to_evidence_list_round_trip(): evidence list reconstructed correctly
def test_empty_evidence_list(): EvidenceGraph.build_from_evidence([]) creates only statement node
```

---

## Validation Architecture

### Dimension 8 — Test Infrastructure

- `tests/factcheck_agents/` directory must be created.
- Tests use `pytest` (existing `pytest.ini` has `testpaths = ["tests"]`).
- `conftest.py` is empty — no fixtures needed for Phase 1 tests.
- No heavy imports (no torch, no LangGraph) in Phase 1 tests → fast test suite.

### Verification Criteria

1. `factcheck_agents/state.py` contains `source_tier`, `evidence_graph`, `reliability_signal` fields.
2. `factcheck_agents/config.py` contains `trusted_domains`, `flagged_domains`, `reliability_threshold` fields.
3. `factcheck_agents/graph_utils.py` exists with `EvidenceGraph` class implementing 4 methods + `.graph` property.
4. `factcheck_agents/source_tier.py` exists with `classify_domain(url: str) -> str`.
5. `pyproject.toml` `agents` extra includes `networkx`.
6. `tests/factcheck_agents/test_source_tier.py` passes.
7. `tests/factcheck_agents/test_evidence_graph.py` passes.
8. `pytest tests/factcheck_agents/` exits 0.
9. Existing tests (`pytest tests/`) still pass (no regressions).

---

## Risk & Edge Cases

| Risk | Mitigation |
|------|-----------|
| `import networkx` fails if `networkx` not installed | Add to `agents` extra in `pyproject.toml`; plans note `uv sync --extra agents` |
| `classify_domain("")` crashes on empty URL | `urlparse("").hostname` returns `None`; guard with `or ""` |
| Hash collision in `build_from_evidence` (two snippets, same URL) | `DiGraph.add_node` with same id updates attrs — last write wins; acceptable for Phase 1 |
| Circular import: `source_tier.py` imports `config.py`, `config.py` has no circular deps | Safe — `config.py` imports only stdlib |
| `state.py` stays import-clean | `evidence_graph: Optional[Any]` — no `import networkx` needed in `state.py` |

---

## ## RESEARCH COMPLETE

Phase 1 is a straightforward data-contract phase. All changes are additive, no existing behavior is modified, and the two new modules have no circular import risk. Primary planning focus: ensure `networkx` dependency is captured in `pyproject.toml` and test directory structure is created.
