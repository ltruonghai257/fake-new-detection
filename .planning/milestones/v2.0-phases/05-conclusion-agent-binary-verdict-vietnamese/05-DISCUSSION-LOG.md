# Phase 5: Conclusion Agent (Binary Verdict + Vietnamese) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-19
**Phase:** 05-conclusion-agent-binary-verdict-vietnamese
**Areas discussed:** Conflict detection scope, LLM JSON response format, Fallback rationale language, Evidence tier visibility to LLM

---

## Conflict Detection Scope

### Q1: Which contradicts edges flip the verdict to FAKE?

| Option | Description | Selected |
|--------|-------------|----------|
| Trusted-tier only (per CONCL-01) | Only contradicts edges from a trusted-tier node to the statement node trigger FAKE | ✓ |
| Any tier | Any contradicts edge regardless of source tier counts | |
| Trusted required, flagged amplifies | Trusted sufficient alone; flagged alone is not | |

**User's choice:** Trusted-tier only (per CONCL-01)
**Notes:** Aligns directly with REQUIREMENTS.md CONCL-01 wording.

### Q2: What happens when evidence_graph is None?

| Option | Description | Selected |
|--------|-------------|----------|
| Skip graph check, fall through to LLM/fallback | If evidence_graph is None, skip conflict detection entirely | ✓ |
| Treat as no conflict (REAL bias) | None graph = no contradictions, proceed as consistent | |
| Force FAKE (conservative) | Can't verify without graph → default to FAKE | |

**User's choice:** Yes, skip (recommended)
**Notes:** Consistent with graceful-degrade pattern. Missing input = skip step, not force verdict.

### Q3: When trusted contradiction found — short-circuit or two-pass?

| Option | Description | Selected |
|--------|-------------|----------|
| Hard rule: short-circuit to FAKE (skip LLM) | Skip LLM call entirely on conflict detection | |
| Soft signal: pass to LLM as context | Let LLM decide binary verdict | |
| Two-pass: rule overrides LLM output | LLM runs for rationale → agent code overrides verdict_binary | ✓ |

**User's choice:** Yes, two-pass (recommended)
**Notes:** LLM generates Vietnamese rationale always. Agent code applies deterministic binary override.

---

## LLM JSON Response Format

### Q1: Who applies the binary mapping?

| Option | Description | Selected |
|--------|-------------|----------|
| Agent code (recommended) | LLM returns 4-class label; agent maps to verdict_binary | ✓ |
| LLM emits verdict_binary directly | Prompt asks LLM to return REAL\|FAKE | |
| LLM emits both label and verdict_binary | LLM returns both; agent validates/overrides | |

**User's choice:** Agent code (recommended)
**Notes:** Deterministic, testable (TEST-04), consistent with two-pass conflict override.

### Q2: Does the prompt change expected JSON keys?

| Option | Description | Selected |
|--------|-------------|----------|
| No — same keys (recommended) | label, confidence, rationale, citations, recommendation unchanged | ✓ |
| Yes — remove label, add verdict_binary | Replace label with verdict_binary in prompt | |

**User's choice:** No — same keys (recommended)
**Notes:** Minimal prompt change: add Vietnamese rationale instruction only. No parse_json breakage risk.

---

## Fallback Rationale Language

### Q1: Vietnamese for _fallback_verdict strings?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — Vietnamese (recommended) | CONCL-05: all user-facing text in Vietnamese | ✓ |
| No — English acceptable for fallback | Fallback is edge case; English rationale ok | |

**User's choice:** Yes — Vietnamese (recommended)
**Notes:** CONCL-05 is explicit; fallback path is most likely to reach the user (no LLM configured).

---

## Evidence Tier Visibility to LLM

### Q1: Should LLM context include tier annotations?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — English tier labels [TRUSTED]/[FLAGGED]/[SOCIAL] | Prepend to each evidence item in _format_evidence() | ✓ |
| Yes — Vietnamese tier labels | Use Vietnamese tier brackets | |
| No — tier info for rule-based detection only | Keep _format_evidence() unchanged | |

**User's choice:** Yes — English tier labels (recommended)
**Notes:** Technical context labels; not user-facing output. LLM can weigh trusted sources appropriately.

### Q2: Where does tier annotation come from?

| Option | Description | Selected |
|--------|-------------|----------|
| From state["evidence"] flat list (recommended) | e.get("source_tier", "unknown") — no graph traversal | ✓ |
| From evidence_graph node attributes | Traverse graph for tier per node | |

**User's choice:** From state["evidence"] flat list (recommended)
**Notes:** Evidence and graph have the same tier data; flat list is simpler.

---

## Claude's Discretion

- Exact Vietnamese phrasing for fallback rationale/recommendation strings (grammatically correct Vietnamese, functional meaning specified in CONTEXT.md specifics)
- Graph traversal implementation details (edge iteration pattern using networkx API)
- Helper function name for `verdict_binary`/`verdict_label_vi` assignment (to avoid duplication between LLM and fallback paths)

## Deferred Ideas

None — discussion stayed within phase scope.
