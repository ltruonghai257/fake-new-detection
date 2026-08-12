"""Shared state definitions for the fact-checking graph.

Mirrors the TradingAgents `AgentState` idea: a single typed dict is threaded
through every node, and each node writes its own slice.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, TypedDict

from langgraph.graph.message import add_messages


# ── Canonical label vocabulary ──────────────────────────────────────────────
# Single source of truth for all agent <-> verdict conversions.
# Model outputs:  PhoBERT → SUPPORTED|REFUTED|NEI,  COOLANT → REAL|FAKE
# Verdict output: binary → REAL|FAKE|NEI,  label_vi → Thật|Giả|Chưa xác thực
# 4-class label:  TRUE|FALSE|MISLEADING|UNVERIFIED

REAL_MODEL_LABELS = frozenset({"SUPPORTED", "REAL", "TRUE"})
FAKE_MODEL_LABELS = frozenset({"REFUTED", "FAKE", "FALSE", "MISLEADING"})
NEI_MODEL_LABELS = frozenset({"NEI", "UNVERIFIED"})

_BINARY_SYNONYMS: Dict[str, str] = {
    "SUPPORTED": "REAL", "REAL": "REAL", "TRUE": "REAL",
    "REFUTED": "FAKE", "FAKE": "FAKE", "FALSE": "FAKE", "MISLEADING": "FAKE",
    "NEI": "NEI", "UNVERIFIED": "NEI",
}
_LABEL_VI: Dict[str, str] = {
    "REAL": "Thật", "FAKE": "Giả", "NEI": "Chưa xác thực",
}
_4CLASS_SYNONYMS: Dict[str, str] = {
    "SUPPORTED": "TRUE", "REAL": "TRUE", "TRUE": "TRUE",
    "REFUTED": "FALSE", "FAKE": "FALSE", "FALSE": "FALSE",
    "MISLEADING": "MISLEADING",
    "NEI": "UNVERIFIED", "UNVERIFIED": "UNVERIFIED",
}


def canonicalize_binary(label: str) -> str:
    """Normalize any model/LLM label to REAL | FAKE | NEI."""
    return _BINARY_SYNONYMS.get(str(label).strip().upper(), "NEI")


def binary_to_vi(binary: str) -> str:
    return _LABEL_VI.get(binary, "Chưa xác thực")


def canonicalize_4class(label: str) -> str:
    """Normalize any label to TRUE | FALSE | MISLEADING | UNVERIFIED."""
    return _4CLASS_SYNONYMS.get(str(label).strip().upper(), "UNVERIFIED")


def label_to_binary_vi(label: str) -> Tuple[str, str]:
    """Canonicalize a flexible label → (binary, label_vi)."""
    binary = canonicalize_binary(label)
    return binary, binary_to_vi(binary)


class Evidence(TypedDict, total=False):
    """A single retrieved web result used as a truth source."""

    title: str
    url: str
    snippet: str
    content: str  # Full content from search API
    source: str  # search provider that returned it (tavily/google_cse)
    score: float  # provider relevance score, if any
    source_tier: Literal["trusted", "flagged", "social", "unknown"]
    image_path: Optional[str]  # local path to downloaded page image, if any
    image_caption: Optional[str]  # caption / alt text for the page image, if any


class ModelResult(TypedDict, total=False):
    """Output from one trained model on the statement."""

    model: str  # "phobert_vifactcheck" | "coolant"
    available: bool  # False if checkpoint missing / model skipped
    label: str  # human label, e.g. SUPPORTED / REFUTED / NEI
    label_id: int
    probabilities: dict  # {label: prob}
    confidence: float
    note: str  # why it was skipped, or extra context
    evidence_text: Optional[str]  # full evidence text used as context (PhoBERT)
    workflow_steps: Optional[List[dict]]  # [{step, description, input, output}]


class Verdict(TypedDict, total=False):
    """Final synthesized decision from the conclusion agent."""

    label: str  # TRUE | FALSE | MISLEADING | UNVERIFIED
    verdict_binary: Literal["REAL", "FAKE", "NEI"]
    verdict_label_vi: Literal["Thật", "Giả", "Chưa xác thực"]
    confidence: float  # 0..1
    rationale: str
    citations: List[str]  # URLs backing the verdict
    recommendation: str
    # structured explanation sections (judge_agent)
    explanation: dict  # {model_summary, debate_winner, evidence_summary, confidence_breakdown}
    debate_transcript: List[dict]  # full debate turns
    model_detail: dict  # {phobert: ModelResult, coolant: ModelResult}


class FactCheckState(TypedDict, total=False):
    """Full graph state threaded Search -> Evaluate -> Conclusion."""

    # inputs
    statement: str
    image_path: Optional[str]
    language: str  # "vi" | "en" | "auto"
    use_phobert: bool  # ablation toggle — skip PhoBERT when False
    use_coolant: bool  # ablation toggle — skip COOLANT when False
    use_evidence: bool  # ablation toggle — skip web search when False

    # search agent
    search_queries: List[str]
    claim_variants: List[str]  # LLM-generated variant claims for deeper search
    evidence: List[Evidence]

    # evaluate agent
    model_results: List[ModelResult]

    # evidence graph
    evidence_graph: Optional[Any]
    reliability_signal: Optional[bool]

    # conclusion agent
    verdict: Verdict

    # trace / debugging
    messages: Annotated[list, add_messages]
    errors: List[str]
    meta: dict[str, Any]

    # M2 debate pipeline fields
    evidence_real: List[Evidence]
    evidence_fake: List[Evidence]
    evidence_social: List[Evidence]
    social_loop_fired: bool
    request_id: str
    consistency_score: float
    agreement_score: float
    debate_turns: List[dict]
    debate_exit_reason: str
    debate_converged: bool  # True when both advocates agreed on same verdict
    debate_agreed_verdict: Optional[str]  # "REAL" | "FAKE" | None
    weight_breakdown: dict
    evidence_workflow_steps: Optional[
        List[dict]
    ]  # [{step, description, count, filtered}]
