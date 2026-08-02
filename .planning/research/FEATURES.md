# FEATURES.md — M2: Debate-Based Verification Pipeline and Demo App

## Agreement Gate

### State-of-the-Art
- **TRIDENT** (MDPI 2025): dual-threshold arbiter — individual confidence (τ_con) + heterogeneous consistency (τ_agree)
- **AgtOpen Consensus Engine**: weighted supermajority 66% as standard baseline
- Dissent ratio across heterogeneous vendors catches blind spots single-model gates miss

### Recommended Implementation

Signal composition (mirrors judge weights for consistency):
```
agreement_score = 0.30 * phobert_confidence + 0.30 * coolant_confidence + 0.40 * evidence_credibility
```

Threshold values:
- ≥ 0.75: skip debate → log "debate_skipped: high_agreement"
- 0.50–0.74: enter debate
- < 0.50: enter debate (conflicting signals)

Edge cases:
- Model unavailable → treat confidence as 0.0; weight redistributes proportionally across available signals
- NEI from any model → forces agreement_score to 0.0 (existing D-03 rule)
- Empty evidence → evidence_credibility = 0.0; rely entirely on model confidences

### Complexity: Medium
Dependencies: `verify_agent` model results, `search_agent` evidence lists

### Anti-Features
- ❌ Bayesian belief updating (over-engineering)
- ❌ Dynamic threshold adjustment (instability)
- ❌ Per-signal calibration curves (requires training data we don't have)

---

## Debate Loop

### State-of-the-Art
- **TradingAgents Bull/Bear**: mandatory opposing positions → fixed round rebuttals → Research Manager synthesis; adversarial framing prevents confirmation bias
- **Debate-to-Detect (D2D)**: 5-stage: Opening → Rebuttal → Free Debate → Closing → Judgment; multi-dimensional evaluation (factuality, source reliability, reasoning quality)
- **Adaptive termination**: Wald-SPRT log-likelihood, Debate Fatigue Q(T)=a·log(T+c)−b·T, HCP-MAD pair-agent convergence detection

### Recommended Structure

```
Round 0: real_advocate opening  (REAL, cites evidence_real)
Round 1: fake_advocate rebuttal (FAKE, cites evidence_fake + contradictions)
Round 2: real_advocate counter-rebuttal
Round 3: fake_advocate closing (final FAKE argument)
```

Termination conditions beyond `max_rounds`:
1. Early convergence: both advocates agree on verdict + quality scores within 0.2 for 2 consecutive rounds
2. Evidence exhaustion: no new evidence cited in last 2 rounds
3. Quality degradation: argument quality scores decline for 2 consecutive rounds (debate fatigue)
4. Max rounds hard cap (default 2, configurable)

Rebuttal rules:
- Must quote + counter opponent's previous argument explicitly
- Citations must come from advocate's assigned tier (real → trusted, fake → flagged/unknown)
- Judge scores each turn: factuality + rebuttal specificity + evidence grounding

### Complexity: Complex
New state fields: `debate_history`, `current_round`, `debate_exit_reason`  
New agents: `real_advocate`, `fake_advocate`  
Dependencies: dual-source evidence separation, argument quality scoring

### Anti-Features
- ❌ Free-form unstructured debate (hard to evaluate)
- ❌ Audience/participant agents (noise)
- ❌ Cross-lingual debate (Vietnamese-only for M2)

---

## Evidence-Credibility Score

### State-of-the-Art
- **Truth Mapper**: Source Quality 30% + Evidence Strength 30% + Fallacy Penalty 30% + Coherence 10%
- **NATO Admiralty Code**: source reliability (A–F) × information credibility (1–6) — independent dimensions
- **Vericore**: `source_credibility`, `conviction`, `narrative_momentum` as continuous floats

### Recommended Formula

```
evidence_credibility = 0.40 * tier_score + 0.30 * count_score + 0.30 * consistency_score
```

Components:
- **tier_score**: trusted=1.0, flagged=0.5, unknown=0.3 (NOT 0/1); weighted average over all evidence items
- **count_score**: `min(1.0, log2(1 + trusted_count) / log2(6))` — logarithmic, ~0.43 at 1 item, 1.0 at 4+ items
- **consistency_score**: % evidence directionally aligned with majority model verdict; diversity bonus +0.2 if ≥ 3 domains

Anti-binary design: floor at 0.1, no item is ever exactly 0.0 or 1.0; log all 3 components to verdict JSON.

### Complexity: Medium
Dependencies: evidence_graph traversal, source_tier.py classification

### Anti-Features
- ❌ Temporal recency weighting (Vietnamese news lacks reliable timestamps)
- ❌ Author reputation scoring (requires external database)
- ❌ Semantic similarity clustering (over-engineering for M2)

---

## Argument Quality Scoring (1–5)

### State-of-the-Art
- **Debatrix**: multi-dimensional judge — argument, language, clash dimensions; iterative chronological analysis
- **DebateFlow**: clash engagement, burden fulfillment, rebuttal quality, argument extension
- **LLM-as-Judge**: moderate correlation with human experts (κ=0.493); Bradley-Terry pairwise comparison

### Recommended Implementation

Three dimensions (1-5 each):

| Dimension | 5 | 3 | 1 |
|-----------|---|---|---|
| Factuality | All claims grounded in cited evidence | Most grounded, minor unsupported | Significant hallucination |
| Rebuttal Engagement | Directly addresses opponent's specific points with counter-evidence | Acknowledges opponent, generic counter | Ignores opponent entirely |
| Evidence Grounding | Multiple citations from appropriate tier | Single citation or mixed-tier | No citations |

LLM judge prompt (after each turn):
```
Rate argument on 3 dimensions (1-5). Respond as JSON:
{"factuality": N, "engagement": N, "grounding": N, "overall": N}
Argument: {argument_text}
Opponent's previous: {opponent_text}
Available evidence: {evidence_list}
```

Store in state: `debate_history[{round, agent, argument, scores}]`  
Use `overall` for termination condition; individual scores for judge's final weights.

### Complexity: Medium
Adds one LLM call per debate turn (latency/cost tradeoff).

### Anti-Features
- ❌ More than 3 scoring dimensions (diminishing returns)
- ❌ Human-in-the-loop calibration (breaks automation)
- ❌ Ensemble of multiple judges (single LLM sufficient)

---

## Streaming Debate UI

### State-of-the-Art
- **DebateAI**: SSE preferred over WebSockets (simpler, CDN-friendly); 8-char/20ms flush for "live" feel
- **AI Debate Arena**: card-based strategic UI — Move Type (Attack/Defense/Refute), Power Level, Judge Evaluation per turn
- **PowerArchi/ai-debate**: Planner → Advocates → Judge layout; localStorage for last run; `type` field on each message (analysis/argument/rebuttal)
- **ludovic/debat-ia-agents**: LangGraph.js + Express + React + Tailwind; `POST /api/debate/stream`

### Recommended Implementation

SSE event types:
```
event: stage_start  data: {"stage": "retrieval"|"debate"|"judge"|"verdict", ...}
event: turn_start   data: {"agent": "real_advocate"|"fake_advocate", "round": N}
event: chunk        data: {"content": "...text fragment..."}
event: turn_end     data: {"agent": "...", "scores": {"factuality": N, ...}}
event: heartbeat    data: {}   (every 5s)
event: verdict      data: {"verdict": "...", "confidence": ..., "weight_breakdown": {...}}
```

UX:
1. Alternating chat bubbles — blue (real_advocate) / red (fake_advocate)
2. Each bubble: agent avatar, streaming text, evidence citations, quality score badge
3. Round counter + progress bar
4. Evidence sidebar with tier badges [TRUSTED] / [FLAGGED]
5. Final verdict card: label, confidence gauge, 30/30/40 weight bar, log download button

### Complexity: Complex
Dependencies: FastAPI + sse-starlette, React + native EventSource, debate pipeline

### Anti-Features
- ❌ WebSockets (SSE is sufficient and simpler)
- ❌ Video/audio streaming
- ❌ User participation in debate

---

## Integration with Existing Components

| Existing | How Extended for M2 |
|----------|---------------------|
| `verify_agent.py` | Add agreement_score computation after model results |
| `search_agent.py` | Split to `real_source_agent` + `fake_source_agent`; reranker hook before PhoBERT truncation |
| `conclusion_agent.py` | Replaced by `debate_judge_agent` for final verdict (existing kept for backward compat) |
| `graph.py` | Add agreement gate edge, debate loop nodes, social loop edge |
| `state.py` | Add: `evidence_real`, `evidence_fake`, `agreement_score`, `debate_history`, `current_round`, `debate_exit_reason`, `weight_breakdown` |

## Implementation Priority

1. State + config + evidence agents (EVRET, RERANK, SOCLOOP)
2. Agreement gate + evidence-credibility scoring (AGREE)
3. Debate loop + argument quality (DEBATE, JUDGE)
4. Demo web app (DEMO)

---
*Research completed: 2026-08-02*
