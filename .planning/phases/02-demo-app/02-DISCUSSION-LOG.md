# Phase 2: Demo App - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-03
**Phase:** 02-demo-app
**Areas discussed:** Streaming bridge, Image input method, Evidence panel layout, Score badge timing, Pipeline entry point, Stage progress display, Error state UX

---

## Streaming Bridge

| Option | Description | Selected |
|--------|-------------|----------|
| Re-chunk buffered text | Run debate_node sync in thread, buffer full turn, re-emit ~8 chars/20ms | ✓ |
| Real LLM streaming | Change debate_node to llm.stream(), emit tokens live | |
| LangGraph astream_events | Use graph.astream_events() for node events | |

**User's choice:** Re-chunk buffered text

**asyncio.Queue bridge sub-question:**

| Option | Description | Selected |
|--------|-------------|----------|
| asyncio.Queue bridge | Pipeline thread posts to queue via loop.call_soon_threadsafe() | You decide |
| Callback injection | Pass callback into build_debate_graph()/debate_node | |

**User's choice:** Delegated to Claude → asyncio.Queue bridge chosen.

---

## Image Input Method

| Option | Description | Selected |
|--------|-------------|----------|
| File upload (multipart) | Standard file input; backend saves to temp path | |
| URL input field | User types/pastes image URL | |
| Optional — skip | Make image input future enhancement | |
| Both | File upload + URL field; URL takes priority | ✓ |

**User's choice:** Both (freeform "both") — URL takes priority when both provided.

---

## Evidence Panel Layout

**Timing:**

| Option | Description | Selected |
|--------|-------------|----------|
| Revealed with verdict | Evidence panel hidden; shown all at once with verdict card | ✓ |
| Live during streaming | Evidence panel populates as evidence retrieval stage completes | |

**Organization:**

| Option | Description | Selected |
|--------|-------------|----------|
| Real/Fake tabs | "Nguồn ủng hộ" / "Nguồn phản bác" | |
| Unified tier-colored list | All evidence merged with tier badges | |
| Both: tabs + tier badges | Real/Fake tabs with tier badges per item within tabs | ✓ |

**User's choice:** Revealed with verdict; Real/Fake tabs with tier badges.

---

## Score Badge Timing

**Timing:**

| Option | Description | Selected |
|--------|-------------|----------|
| Appear at verdict reveal | Badges hidden during streaming; all appear when verdict event arrives | ✓ |
| Animate retroactively | Badges fade-in sequentially after verdict arrives | |

**Badge detail:**

| Option | Description | Selected |
|--------|-------------|----------|
| Single aggregate score | One badge showing average of 3 dimensions | |
| Three separate scores | Three small dimension badges per bubble | ✓ |

**User's choice:** Appear at verdict reveal; three separate dimension badges.

---

## Pipeline Entry Point

| Option | Description | Selected |
|--------|-------------|----------|
| New run_debate_fact_check() in __init__.py | Add convenience function; follows existing pattern | |
| Direct import in streaming.py | Import build_debate_graph + initial_state directly | You decide |

**User's choice:** Delegated to Claude → direct import in streaming.py chosen (demo app self-contained).

---

## Stage Progress Display

**Display type:**

| Option | Description | Selected |
|--------|-------------|----------|
| Step indicator with Vietnamese labels | Horizontal stepper driven by stage_start events | ✓ |
| Status text only | Simple status text updated per stage_start | |
| Progress bar + status | Indeterminate bar + status text | |

**Stage names:**

| Option | Description | Selected |
|--------|-------------|----------|
| You decide — suggest them | Claude picks Vietnamese stage display names | ✓ |
| I'll specify | User provides specific text | |

**User's choice:** Horizontal stepper; Vietnamese labels delegated to Claude.
**Labels decided:** "Tìm bằng chứng" → "Xếp hạng bằng chứng" → "Kiểm định mô hình" → "Tranh luận" → "Phán quyết"

---

## Error State UX

| Option | Description | Selected |
|--------|-------------|----------|
| Error card + retry button | Red error card with "Thử lại" button | You decide |
| Partial results + error banner | Keep rendered turns; show "Lỗi xảy ra" banner | |
| Simple error text | Plain text with no retry button | |

**User's choice:** Delegated to Claude → error card + retry button chosen.

---

## Claude's Discretion

- **asyncio.Queue bridge** — standard pattern specified in ROADMAP; no alternative considered.
- **Pipeline entry point** — direct import in streaming.py chosen to keep demo app self-contained.
- **Error state UX** — error card + retry button chosen for clearest UX.
- **Stage names** — Vietnamese labels: "Tìm bằng chứng" → "Xếp hạng bằng chứng" → "Kiểm định mô hình" → "Tranh luận" → "Phán quyết".
- **Confidence gauge style** — not discussed; Claude to decide (radial arc or horizontal bar).

## Deferred Ideas

- Real LLM token streaming (llm.stream()) — deferred; re-chunk sufficient for thesis demo.
- WebSockets — out of scope per REQUIREMENTS.md.
- Auth, public deployment — explicitly out of scope.
