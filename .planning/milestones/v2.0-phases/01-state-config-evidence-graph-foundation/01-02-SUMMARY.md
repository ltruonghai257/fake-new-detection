# Plan 01-02 Summary: EvidenceGraph, Source Tier, and Tests

**Status:** Complete  
**Commit:** b970eb6

## What was built

### `factcheck_agents/graph_utils.py`
- `EvidenceGraph` class backed by `networkx.DiGraph`
- `build_from_evidence(cls, evidence_list)` — classmethod; creates `"statement"` node + evidence nodes with `mentions` edges
- `add_node`, `add_edge`, `to_evidence_list` — full API per spec
- `.graph` attribute exposes the `nx.DiGraph` directly

### `factcheck_agents/source_tier.py`
- `classify_domain(url: str) -> str` — returns `"trusted"`, `"flagged"`, or `"unknown"`
- Reads `settings.trusted_domains` / `settings.flagged_domains` (comma-separated)
- Handles `www.` prefix via `removeprefix`, subdomain matching via `endswith("." + t)`
- Empty/invalid URL falls through to `"unknown"` naturally

### Tests
- `tests/__init__.py` — added to fix pytest "prepend" import mode shadowing the source `factcheck_agents` package
- `tests/factcheck_agents/__init__.py` — empty package marker
- `tests/factcheck_agents/test_source_tier.py` — 10 parametrized cases (TEST-01)
- `tests/factcheck_agents/test_evidence_graph.py` — 6 tests (TEST-02)

## Verification

- `pytest tests/factcheck_agents/ -v` → 16 passed (6 graph + 10 source_tier)
- `pytest tests/ -q --ignore=tests/processing/coolant` → 46 passed (16 new + 30 existing)
- Pre-existing broken test at `tests/processing/coolant/test_pair_extractor.py` is unrelated to this phase

## Notable fix
pytest's default "prepend" import mode interprets `tests/factcheck_agents/__init__.py` (with no `tests/__init__.py`) as a top-level `factcheck_agents` package, shadowing the source. Adding `tests/__init__.py` forces pytest to treat the whole `tests/` directory as a package rooted at the project root, resolving the conflict.
