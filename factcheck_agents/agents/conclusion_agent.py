"""Conclusion agent: fuse model verdicts + evidence into a final decision.

Uses the LLM to weigh the trained-model signals against the retrieved
evidence. If no LLM is configured it falls back to a deterministic rule based
on the available model outputs so the pipeline still produces a verdict.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional, Tuple

from ..state import Evidence, FactCheckState, ModelResult, Verdict
from .llm import get_llm, parse_json
from ..prompts import CONCLUSION_SYSTEM_PROMPT


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


def conclusion_agent(state: FactCheckState) -> dict:
    statement = state["statement"]
    model_results = state.get("model_results", []) or []
    evidence = state.get("evidence", []) or []
    evidence_graph = state.get("evidence_graph")

    llm = get_llm()
    if llm is None:
        verdict = _fallback_verdict(model_results, evidence, evidence_graph)
        return {
            "verdict": verdict,
            "messages": [("assistant", f"[Conclusion] {verdict['label']} (fallback)")],
        }

    user = (
        f"CLAIM:\n{statement}\n\n"
        f"MODEL PREDICTIONS:\n{_format_models(model_results)}\n\n"
        f"WEB EVIDENCE:\n{_format_evidence(evidence)}\n"
    )
    try:
        resp = llm.invoke([("system", CONCLUSION_SYSTEM_PROMPT), ("user", user)])
        data = parse_json(getattr(resp, "content", "") or "") or {}
    except Exception as exc:
        verdict = _fallback_verdict(model_results, evidence, evidence_graph)
        verdict["rationale"] += f" (LLM error: {exc})"
        return {
            "verdict": verdict,
            "messages": [("assistant", f"[Conclusion] {verdict['label']} (fallback)")],
        }

    label = str(data.get("label", "UNVERIFIED")).upper()
    conflict = _has_cross_source_conflict(evidence_graph)
    binary, label_vi = _map_to_binary(label, conflict)
    verdict = Verdict(
        label=label,
        verdict_binary=binary,
        verdict_label_vi=label_vi,
        confidence=float(data.get("confidence", 0.0) or 0.0),
        rationale=str(data.get("rationale", "")),
        citations=list(
            data.get("citations", [])
            or [e.get("url", "") for e in evidence if e.get("url")][:5]
        ),
        recommendation=str(data.get("recommendation", "")),
    )
    return {
        "verdict": verdict,
        "messages": [
            (
                "assistant",
                f"[Conclusion] {verdict['label']} ({verdict['confidence']:.2f})",
            )
        ],
    }
