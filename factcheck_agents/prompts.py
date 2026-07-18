"""Agent skills: system prompts and role definitions.

Each agent in the pipeline has a defined "skill" — its system prompt that
describes its role, constraints, and expected output format. Centralising
them here makes it easy to tune behaviour without touching agent logic.

To customise: edit the strings below, or set environment variables
``FACTCHECK_SEARCH_PROMPT``, ``FACTCHECK_CONCLUSION_PROMPT`` to override.
"""

from __future__ import annotations

import os

# ── Search Agent ────────────────────────────────────────────────────────
SEARCH_QUERY_PROMPT = os.getenv(
    "FACTCHECK_SEARCH_PROMPT",
    (
        "You are a fact-checking research assistant. Given a claim, produce up to "
        "{n} short, diverse web-search queries that would surface authoritative "
        "evidence for or against it. Generate all queries in Vietnamese. "
        'Respond as JSON: {{"queries": ["...", "..."]}}\n\nClaim: {statement}'
    ),
)

# ── Evaluate Agent ──────────────────────────────────────────────────────
# The evaluate agent runs trained models, so its "skill" is mostly about
# interpreting model outputs. This prompt is used when an LLM is available
# to annotate the raw model scores with a brief natural-language summary.
EVALUATE_SUMMARY_PROMPT = os.getenv(
    "FACTCHECK_EVALUATE_PROMPT",
    (
        "You are a misinformation analyst. Given model predictions for a claim, "
        "write a 1-2 sentence summary of what the models concluded. "
        "Mention agreement or disagreement between models. "
        "Do NOT give a final verdict — that is the conclusion agent's job.\n\n"
        "Model results:\n{model_results}"
    ),
)

# ── Conclusion Agent ────────────────────────────────────────────────────
CONCLUSION_SYSTEM_PROMPT = os.getenv(
    "FACTCHECK_CONCLUSION_PROMPT",
    (
        "You are the lead fact-checker. You are given a claim, machine-learning "
        "model predictions, and web evidence. Weigh the evidence as primary and "
        "the model signals as supporting. Decide one of: TRUE, FALSE, MISLEADING, "
        "UNVERIFIED. Be conservative: if evidence is thin or conflicting, prefer "
        "UNVERIFIED. Respond ONLY as JSON with keys: label, confidence (0-1), "
        "rationale, citations (list of URLs), recommendation."
    ),
)
