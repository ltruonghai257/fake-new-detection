"""Shared state definitions for the fact-checking graph.

Mirrors the TradingAgents `AgentState` idea: a single typed dict is threaded
through every node, and each node writes its own slice.
"""

from __future__ import annotations

from typing import Annotated, Any, List, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages


class Evidence(TypedDict, total=False):
    """A single retrieved web result used as a truth source."""

    title: str
    url: str
    snippet: str
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


class Verdict(TypedDict, total=False):
    """Final synthesized decision from the conclusion agent."""

    label: str  # TRUE | FALSE | MISLEADING | UNVERIFIED
    confidence: float  # 0..1
    rationale: str
    citations: List[str]  # URLs backing the verdict
    recommendation: str


class FactCheckState(TypedDict, total=False):
    """Full graph state threaded Search -> Evaluate -> Conclusion."""

    # inputs
    statement: str
    image_path: Optional[str]
    language: str  # "vi" | "en" | "auto"

    # search agent
    search_queries: List[str]
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
