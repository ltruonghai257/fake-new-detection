# Plan 01-01 Summary: State, Config & Dependency Declarations

**Status:** Complete  
**Commit:** 260d138

## What was built

Purely additive changes to three existing files — no behavior changes, no new modules.

### `factcheck_agents/state.py`
- Added `Literal` to the `from typing import` line
- Added `source_tier: Literal["trusted", "flagged", "social", "unknown"]` to `Evidence`
- Added `evidence_graph: Optional[Any]` and `reliability_signal: Optional[bool]` to `FactCheckState` under a new `# evidence graph` block

### `factcheck_agents/config.py`
- Added `trusted_domains`, `flagged_domains`, `reliability_threshold` fields to `Settings` dataclass under a new `# Source tier & reliability` section
- All three fields read from `FACTCHECK_TRUSTED_DOMAINS`, `FACTCHECK_FLAGGED_DOMAINS`, `FACTCHECK_RELIABILITY_THRESHOLD` env vars with documented defaults

### `pyproject.toml`
- Added `"networkx>=3.0"` to the `agents` optional extra (alphabetically between `mcp` and `openai`)
- Added `dev` optional extra with `pytest>=8.0` (pytest was missing from the project's dependency declarations)

## Verification

- `python -c "from factcheck_agents.state import Evidence, FactCheckState; e = Evidence(title='t', source_tier='trusted'); print('ok')"` → ok
- `python -c "from factcheck_agents.config import settings; assert settings.reliability_threshold == 0.5; assert 'vnexpress.net' in settings.trusted_domains; print('ok')"` → ok
- `pytest tests/ -q --ignore=tests/factcheck_agents --ignore=tests/processing/coolant` → 30 passed
- Pre-existing `tests/processing/coolant/test_pair_extractor.py` collection error is unrelated (references archived `src/processing/coolant/pair_extractor.py`)
