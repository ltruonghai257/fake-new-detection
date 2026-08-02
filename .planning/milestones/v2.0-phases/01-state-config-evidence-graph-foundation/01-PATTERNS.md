# Phase 1 — Pattern Map

**Phase:** 01 — State, Config & Evidence Graph Foundation
**Generated:** 2026-07-19

---

## Files to Create/Modify

| File | Role | Data Flow |
|------|------|-----------|
| `factcheck_agents/state.py` | Modify — add 3 new fields | Defines contract for all agents |
| `factcheck_agents/config.py` | Modify — add 3 new env-var fields | Settings singleton imported by all agents |
| `pyproject.toml` | Modify — add `networkx` to `agents` extra | Dependency declaration |
| `factcheck_agents/graph_utils.py` | Create — `EvidenceGraph` class | Built by search agent (Phase 2), read by verify + conclusion |
| `factcheck_agents/source_tier.py` | Create — `classify_domain()` | Called by search agent (Phase 2) |
| `tests/factcheck_agents/__init__.py` | Create — empty init | Test package setup |
| `tests/factcheck_agents/test_source_tier.py` | Create — TEST-01 | Verifies `classify_domain()` |
| `tests/factcheck_agents/test_evidence_graph.py` | Create — TEST-02 | Verifies `EvidenceGraph` |

---

## Pattern: TypedDict Field Extension (`state.py`)

**Analog:** `Evidence(TypedDict, total=False)` existing fields.

```python
# factcheck_agents/state.py (lines 14–21) — existing pattern
class Evidence(TypedDict, total=False):
    """A single retrieved web result used as a truth source."""
    title: str
    url: str
    snippet: str
    source: str          # search provider that returned it (tavily/google_cse)
    score: float         # provider relevance score, if any
```

**New field follows same pattern:**
```python
# Add to Evidence
source_tier: Literal["trusted", "flagged", "social", "unknown"]
```

**Import change required:** Add `Literal` to the `from typing import ...` line (line 9).

```python
# factcheck_agents/state.py (line 9) — current
from typing import Annotated, Any, List, Optional, TypedDict
# → new
from typing import Annotated, Any, List, Literal, Optional, TypedDict
```

**New `FactCheckState` fields follow existing `Optional[Any]` precedent (`meta: dict[str, Any]`):**
```python
# factcheck_agents/state.py (lines 46–67) — existing FactCheckState
class FactCheckState(TypedDict, total=False):
    ...
    meta: dict[str, Any]
# → new fields (append before closing)
    evidence_graph: Optional[Any]
    reliability_signal: Optional[bool]
```

---

## Pattern: Settings Dataclass Field (`config.py`)

**Analog:** Every existing field in `Settings`. Three type patterns:

```python
# factcheck_agents/config.py — existing patterns
# str with default:
llm_model: str = field(
    default_factory=lambda: os.getenv("FACTCHECK_LLM_MODEL", "gpt-4o-mini")
)
# int with default:
max_results: int = field(
    default_factory=lambda: int(os.getenv("FACTCHECK_MAX_RESULTS", "6"))
)
# Optional[str] no default:
openai_api_key: Optional[str] = field(
    default_factory=lambda: os.getenv("OPENAI_API_KEY")
)
```

**New fields use the `str` and `float` patterns:**
```python
# Append to Settings dataclass (inside the "Web search providers" or new "Source tier" section)
trusted_domains: str = field(
    default_factory=lambda: os.getenv(
        "FACTCHECK_TRUSTED_DOMAINS",
        "vnexpress.net,thanhnien.vn,dantri.com.vn,tuoitre.vn"
    )
)
flagged_domains: str = field(
    default_factory=lambda: os.getenv("FACTCHECK_FLAGGED_DOMAINS", "kenh14.vn")
)
reliability_threshold: float = field(
    default_factory=lambda: float(os.getenv("FACTCHECK_RELIABILITY_THRESHOLD", "0.5"))
)
```

---

## Pattern: pyproject.toml `agents` Extra

```toml
# pyproject.toml (lines 57–63) — current
agents = [
    "langgraph>=0.2.28",
    "langchain-core>=0.3.0",
    "langchain-openai>=0.2.0",
    "openai>=1.40.0",
    "mcp>=1.2.0",
]
```

**Add `networkx` in alphabetical position (after `mcp`, before closing bracket):**
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

---

## Pattern: New Module (`graph_utils.py`)

**Analog:** No existing analog in `factcheck_agents/` — closest in spirit is the TypedDict-only design of `state.py` (plain data, no heavy deps). `graph_utils.py` is a plain Python class wrapping networkx.

**Import pattern:** Top-level `import networkx as nx` (not lazy — module is only loaded when explicitly imported by agents, no startup cost).

**Module structure:**
```python
"""In-memory evidence graph backed by networkx.DiGraph."""
from __future__ import annotations
import networkx as nx
from typing import Any, Dict, List, Optional


class EvidenceGraph:
    """Thin wrapper around networkx.DiGraph for per-check-run evidence graphs."""
    
    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
    
    @classmethod
    def build_from_evidence(cls, evidence_list: List[Dict[str, Any]]) -> "EvidenceGraph":
        ...
    
    def add_node(self, id: Any, attrs: Dict[str, Any]) -> None:
        ...
    
    def add_edge(self, src: Any, dst: Any, type: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        ...
    
    def to_evidence_list(self) -> List[Dict[str, Any]]:
        ...
```

---

## Pattern: New Module (`source_tier.py`)

**Analog:** No existing analog. Closest pattern is the module-level `settings` singleton import used throughout `factcheck_agents/agents/`.

```python
# factcheck_agents/agents/search_agent.py — how agents import settings
from ..config import settings
```

`source_tier.py` imports `settings` the same way:
```python
from .config import settings
```

**stdlib-only:** Uses `urllib.parse.urlparse` (no new deps).

---

## Pattern: Test File

**Analog:** `tests/helpers/test_json_helper.py` — closest existing test.

```python
# tests/helpers/test_json_helper.py — pattern to follow
import pytest
# stdlib + project imports only
# parametrize for table-driven tests
```

**New tests follow the same pattern:** `pytest.mark.parametrize` for domain classification table.

---

## ## PATTERN MAPPING COMPLETE
