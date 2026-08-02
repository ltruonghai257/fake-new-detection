# Phase 3: Verify Agent - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-19
**Phase:** 3-Verify Agent
**Areas discussed:** reliability_signal fusion, build_evidence_text() upgrade, COOLANT image fallback

---

## reliability_signal fusion

| Option | Description | Selected |
|--------|-------------|----------|
| True (OR logic) | At least one confident model → True. Social search runs to break the tie. Aligns with REQUIREMENTS 'at least one model' wording. | |
| False (conflict → skip) | Models disagree → unreliable signal → skip social search, go straight to Conclusion. | |
| Weighted confidence magnitudes | 0.6 × PhoBERT conf + 0.4 × COOLANT conf ≥ threshold → True. Label direction ignored. | ✓ |

**User's choice:** Weighted 0.6/0.4 because "it's better for context for case 2 of them are both important"

**Follow-up — weight normalization when COOLANT unavailable:**

| Option | Description | Selected |
|--------|-------------|----------|
| Normalize to 1.0 | Available model carries full weight. PhoBERT conf 0.8 → True (0.8 ≥ 0.5). | ✓ (Claude's decision) |
| Weight stays 0.6 | PhoBERT conf 0.8 → weighted 0.48 < 0.5 → False. More conservative. | |

**User's choice:** "You decide for me" — Claude chose normalize to 1.0 (threshold stays meaningful regardless of models available)

**Follow-up — NEI label handling:**

| Option | Description | Selected |
|--------|-------------|----------|
| NEI always → False | If any available model returns NEI, signal = False regardless of confidence. | ✓ |
| NEI treated like any label | NEI confidence counts toward weighted score. | |

**User's choice:** NEI always → False

---

## build_evidence_text() upgrade

| Option | Description | Selected |
|--------|-------------|----------|
| Sort inside the function | Add tier-sort to build_evidence_text() before iterating. Defensive: correct regardless of input order. | ✓ |
| Rely on pre-sorted input | Phase 2 already tier-sorts state["evidence"]. No code change to build_evidence_text(). | |

**User's choice:** Sort inside the function

**Follow-up — truncation behavior:**

| Option | Description | Selected |
|--------|-------------|----------|
| Just sort, then truncate | Sort by tier first, then fill to max_chars in order. Simple, consistent with existing logic. | ✓ |
| Per-tier budget | Reserve minimum chars for trusted items. More complex. | |

**User's choice:** Just sort, then truncate

**Notes:** User explicitly said "just baseline first, this milestone I expect it run well" — preference for minimal, correct implementations.

---

## COOLANT image fallback

| Option | Description | Selected |
|--------|-------------|----------|
| Keep the fallback | verify_agent scans evidence list for first image_path when state["image_path"] is None. Preserves v1.0 behavior. | ✓ |
| Strict: explicit only | Use only state.get("image_path"). Simpler. COOLANT unavailable if no explicit image. | |

**User's choice:** Keep the fallback

---

## Claude's Discretion

- **Weight normalization when COOLANT unavailable:** User said "you decide" → Claude chose normalize to 1.0 (keeps threshold consistent)
- **ThreadPoolExecutor pattern:** Per-call executor with `max_workers=2`; no persistent executor needed
- **Exception isolation:** Catch per-model inside futures → unavailable ModelResult (consistent with graceful-degrade)
- **evaluate_agent.py lifecycle:** Leave intact through Phase 3; Phase 6 swaps in verify_agent via graph.py

## Deferred Ideas

None — discussion stayed within phase scope.
