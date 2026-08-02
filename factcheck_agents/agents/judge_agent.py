"""Judge agent: weighted judge with 1-5 dimension scoring and NEI short-circuit.

Evaluates model results, evidence, and debate turns to produce a final verdict.
If no evidence is retrieved from either source, returns NEI without LLM call (EVRET-04).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

from ..state import Evidence, FactCheckState, ModelResult, Verdict
from .llm import get_llm, parse_json


_BINARY_REAL_LABELS = {"SUPPORTED", "REAL", "TRUE"}
_BINARY_FAKE_LABELS = {"REFUTED", "FAKE", "FALSE", "MISLEADING", "UNVERIFIED", "NEI"}


def _has_cross_source_conflict(evidence_graph: Optional[Any]) -> bool:
    if evidence_graph is None:
        return False
    tiers = {
        data.get("source_tier")
        for _, data in evidence_graph.graph.nodes(data=True)
        if data.get("node_type") == "evidence"
    }
    return "trusted" in tiers and bool(tiers & {"flagged", "social"})


def _map_to_binary(label: str, conflict: bool) -> Tuple[str, str]:
    if conflict:
        return "FAKE", "Giả"
    label_upper = str(label).upper()
    if label_upper in _BINARY_REAL_LABELS:
        return "REAL", "Thật"
    return "FAKE", "Giả"


def _fallback_verdict(
    model_results: List[ModelResult],
    evidence: List[Evidence],
    evidence_graph: Optional[Any] = None,
) -> Verdict:
    """Rule-based fallback when LLM is unavailable or fails."""
    avail = [m for m in model_results if m.get("available")]
    citations = [e.get("url", "") for e in evidence if e.get("url")][:5]
    if not avail:
        return Verdict(
            label="UNVERIFIED",
            verdict_binary="FAKE",
            verdict_label_vi="Giả",
            confidence=0.2,
            rationale="No trained model was available and no LLM was configured to weigh evidence.",
            citations=citations,
            recommendation="Configure model checkpoints and/or an LLM key, or review evidence manually.",
        )
    top = max(avail, key=lambda m: m.get("confidence", 0.0))
    label_map = {
        "REFUTED": "FALSE",
        "FAKE": "FALSE",
        "SUPPORTED": "TRUE",
        "REAL": "TRUE",
        "NEI": "UNVERIFIED",
    }
    label = label_map.get(top.get("label", ""), "UNVERIFIED")
    conflict = _has_cross_source_conflict(evidence_graph)
    binary, label_vi = _map_to_binary(label, conflict)
    return Verdict(
        label=label,
        verdict_binary=binary,
        verdict_label_vi=label_vi,
        confidence=round(float(top.get("confidence", 0.0)) * 0.7, 3),
        rationale=f"Rule-based fallback from {top['model']} ({top.get('label')}).",
        citations=citations,
        recommendation="Heuristic verdict; enable an LLM for evidence-weighted reasoning.",
    )


def _format_models(model_results: List[ModelResult]) -> str:
    lines = []
    for m in model_results:
        if m.get("available"):
            lines.append(
                f"- {m['model']}: {m.get('label')} (conf={m.get('confidence')}, probs={m.get('probabilities')})"
            )
        else:
            lines.append(f"- {m['model']}: unavailable ({m.get('note')})")
    return "\n".join(lines)


def _format_evidence(evidence: List[Evidence]) -> str:
    if not evidence:
        return "(no web evidence retrieved)"
    lines = []
    for i, e in enumerate(evidence, 1):
        lines.append(f"[{i}] {e.get('title')} — {e.get('url')}\n    {e.get('snippet')}")
    return "\n".join(lines)


def _format_debate_turns(debate_turns: List[dict]) -> str:
    if not debate_turns:
        return "(no debate turns)"
    lines = []
    for turn in debate_turns:
        agent = turn.get("agent", "unknown")
        round_num = turn.get("round", 0)
        text = turn.get("text", "")
        error = turn.get("error")
        if error:
            lines.append(f"- {agent} (round {round_num}): ERROR: {error}")
        else:
            lines.append(f"- {agent} (round {round_num}): {text[:200]}...")
    return "\n".join(lines)


JUDGE_SYSTEM_PROMPT = (
    "You are the final judge in a fact-checking debate. You are given a claim, "
    "model predictions, web evidence from trusted and flagged sources, and "
    "debate turns between a REAL advocate and a FAKE advocate. "
    "\n\n"
    "Your task:\n"
    "1. Score each debate turn on three dimensions (1-5 scale):\n"
    "   - Factuality: How factual is the argument?\n"
    "   - Rebuttal Engagement: How well does it address opposing arguments?\n"
    "   - Evidence Grounding: How well is it grounded in the provided evidence?\n"
    "2. Produce a final verdict with:\n"
    "   - label: one of TRUE, FALSE, MISLEADING, UNVERIFIED\n"
    "   - confidence: 0.0-1.0\n"
    "   - rationale: explanation in Vietnamese\n"
    "   - citations: list of URLs backing the verdict\n"
    "   - recommendation: action in Vietnamese\n"
    "\n"
    "Respond ONLY as JSON with keys:\n"
    "{\n"
    '  "turn_scores": [{"agent": "real_advocate", "round": 0, "factuality": 4, "rebuttal_engagement": 3, "evidence_grounding": 5}, ...],\n'
    '  "verdict": {"label": "TRUE", "confidence": 0.85, "rationale": "...", "citations": ["..."], "recommendation": "..."}\n'
    "}\n"
)


def _write_verdict_log(request_id: str, data: dict) -> None:
    """Write verdict log to logs/verdicts/<request_id>.json.

    Wrapped in try/except to never block the pipeline (JUDGE-03).
    """
    try:
        log_dir = Path("logs/verdicts")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{request_id}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def judge_agent(state: FactCheckState) -> dict:
    """Weighted judge with dimension scoring and NEI short-circuit.

    Returns:
        dict with keys:
            - verdict: Verdict TypedDict
            - weight_breakdown: dict with phobert, coolant, evidence, argument_scores
            - messages: list of message tuples
    """
    statement = state["statement"]
    model_results = state.get("model_results", []) or []
    evidence_real = state.get("evidence_real") or []
    evidence_fake = state.get("evidence_fake") or []
    evidence = evidence_real + evidence_fake
    evidence_graph = state.get("evidence_graph")
    debate_turns = state.get("debate_turns") or []
    request_id = state.get("request_id", "unknown")

    # NEI short-circuit (EVRET-04)
    if not evidence_real and not evidence_fake:
        nei_verdict = Verdict(
            label="NEI",
            verdict_binary="FAKE",
            verdict_label_vi="Giả",
            confidence=0.1,
            rationale="No evidence retrieved from either source.",
            citations=[],
            recommendation="Unable to verify without evidence sources.",
        )
        return {
            "verdict": nei_verdict,
            "weight_breakdown": {},
            "messages": [("assistant", "[Judge] NEI — no evidence")],
        }

    # Dir creation (JUDGE-03)
    Path("logs/verdicts").mkdir(parents=True, exist_ok=True)

    # Fallback if no LLM
    llm = get_llm()
    if llm is None:
        verdict = _fallback_verdict(model_results, evidence, evidence_graph)
        weight_breakdown = {
            "phobert": 0.0,
            "coolant": 0.0,
            "evidence": 0.0,
            "argument_scores": [],
        }
        _write_verdict_log(request_id, {"verdict": verdict, "weight_breakdown": weight_breakdown})
        return {
            "verdict": verdict,
            "weight_breakdown": weight_breakdown,
            "messages": [("assistant", f"[Judge] {verdict['label']} (fallback)")],
        }

    # Build user prompt (JUDGE-01)
    user = (
        f"CLAIM:\n{statement}\n\n"
        f"MODEL PREDICTIONS:\n{_format_models(model_results)}\n\n"
        f"EVIDENCE (TRUSTED):\n{_format_evidence(evidence_real)}\n\n"
        f"EVIDENCE (FLAGGED/FACT-CHECK):\n{_format_evidence(evidence_fake)}\n\n"
        f"DEBATE TURNS:\n{_format_debate_turns(debate_turns)}\n"
    )

    # Call LLM
    try:
        resp = llm.invoke([("system", JUDGE_SYSTEM_PROMPT), ("user", user)])
        data = parse_json(getattr(resp, "content", "") or "") or {}
    except Exception as exc:
        verdict = _fallback_verdict(model_results, evidence, evidence_graph)
        verdict["rationale"] += f" (LLM error: {exc})"
        weight_breakdown = {
            "phobert": 0.0,
            "coolant": 0.0,
            "evidence": 0.0,
            "argument_scores": [],
        }
        _write_verdict_log(request_id, {"verdict": verdict, "weight_breakdown": weight_breakdown})
        return {
            "verdict": verdict,
            "weight_breakdown": weight_breakdown,
            "messages": [("assistant", f"[Judge] {verdict['label']} (fallback)")],
        }

    # Extract turn scores (JUDGE-01)
    turn_scores = data.get("turn_scores", [])

    # Extract verdict data
    verdict_data = data.get("verdict", {})
    label = str(verdict_data.get("label", "UNVERIFIED")).upper()
    conflict = _has_cross_source_conflict(evidence_graph)
    binary, label_vi = _map_to_binary(label, conflict)

    # Weight computation (JUDGE-02)
    ph_conf = 0.0
    co_conf = 0.0
    for m in model_results:
        if m.get("model") == "phobert_vifactcheck" and m.get("available"):
            ph_conf = m.get("confidence", 0.0)
        elif m.get("model") == "coolant" and m.get("available"):
            co_conf = m.get("confidence", 0.0)

    ev_cred = state.get("weight_breakdown", {}).get("evidence", 0.1)
    confidence = ph_conf * 0.30 + co_conf * 0.30 + ev_cred * 0.40

    # Cap confidence if no debate turns (JUDGE-02)
    if not debate_turns:
        confidence = min(confidence, 0.7)

    verdict = Verdict(
        label=label,
        verdict_binary=binary,
        verdict_label_vi=label_vi,
        confidence=float(verdict_data.get("confidence", confidence)),
        rationale=str(verdict_data.get("rationale", "")),
        citations=list(
            verdict_data.get("citations", [])
            or [e.get("url", "") for e in evidence if e.get("url")][:5]
        ),
        recommendation=str(verdict_data.get("recommendation", "")),
    )

    weight_breakdown = {
        "phobert": ph_conf,
        "coolant": co_conf,
        "evidence": ev_cred,
        "argument_scores": turn_scores,
    }

    # Log write (JUDGE-03)
    _write_verdict_log(request_id, {"verdict": verdict, "weight_breakdown": weight_breakdown})

    return {
        "verdict": verdict,
        "weight_breakdown": weight_breakdown,
        "messages": [("assistant", f"[Judge] {verdict['label']} ({verdict['confidence']:.2f})")],
    }