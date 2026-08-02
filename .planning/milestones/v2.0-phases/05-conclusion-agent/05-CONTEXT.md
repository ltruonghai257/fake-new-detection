# Phase 5: Conclusion Agent (Binary Verdict + Vietnamese) - Context

**Gathered:** 2026-07-19
**Status:** Ready for execution

<domain>
## Phase Boundary

Extend the conclusion agent to emit a binary verdict (`REAL`/`FAKE`) and a Vietnamese user-facing label (`Thật`/`Giả`), while preserving the internal 4-class signal inside the rationale. The agent reads `state["evidence_graph"]` to detect cross-source conflict (trusted vs flagged/social tiers) and applies the rule: **any conflict ⇒ Giả**. If no conflict, the 4-class label maps as SUPPORTED/REAL/TRUE → REAL, everything else → FAKE. The system prompt is updated to request Vietnamese rationale/recommendation and to document the binary mapping.

</domain>

<decisions>
## Implementation Decisions

### Binary Verdict Mapping
- **D-01:** Real labels: `{SUPPORTED, REAL, TRUE}` → `verdict_binary="REAL"`, `verdict_label_vi="Thật"`.
- **D-02:** Fake labels: `{REFUTED, FAKE, FALSE, MISLEADING, UNVERIFIED, NEI}` → `verdict_binary="FAKE"`, `verdict_label_vi="Giả"`.
- **D-03:** Cross-source conflict overrides any label to FAKE/`Giả`. Conflict is defined as the evidence graph containing at least one `trusted`-tier evidence node and at least one `flagged` or `social`-tier evidence node.
- **D-04:** Missing `evidence_graph` means no conflict; fallback to label-based mapping.

### Prompt Updates
- **D-05:** `CONCLUSION_SYSTEM_PROMPT` requests `rationale` and `recommendation` in Vietnamese.
- **D-06:** The prompt documents the binary rule so the LLM knows the final user-facing label is either REAL/Thật or FAKE/Giả.
- **D-07:** The prompt instructs the LLM to preserve the original 4-class label inside the rationale for nuance (CONCL-06).

### Backward Compatibility
- **D-08:** Existing `label`, `confidence`, `rationale`, `citations`, `recommendation` keys remain unchanged. New keys `verdict_binary` and `verdict_label_vi` are additive.

</decisions>

<canonical_refs>
## Canonical References

- `.planning/REQUIREMENTS.md` §Conclusion Agent (CONCL-01..CONCL-06)
- `.planning/phases/03-verify-agent/03-CONTEXT.md` for `reliability_signal` and `model_results` shape
- `.planning/phases/02-search-evidence-agent/02-CONTEXT.md` for `evidence_graph` structure and `source_tier` values
- `factcheck_agents/graph_utils.py` — `EvidenceGraph` API
- `factcheck_agents/state.py` — `Verdict` TypedDict
</canonical_refs>
