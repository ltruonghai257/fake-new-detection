"""
factcheck_agents — a standalone multi-agent fact-checking module.

A statement (optionally with an image) is checked by three agents wired with
LangGraph, mirroring the TradingAgents shared-state pattern:

    Search  -> Evaluate -> Conclusion

- Search:     drafts queries and gathers evidence from the open web
              (truth sources) via Tavily / Google Custom Search.
- Evaluate:   runs the project's two trained models
              (PhoBERT ViFactCheck text model + optional COOLANT multimodal).
- Conclusion: an LLM fuses model verdicts with the retrieved evidence into a
              final label, confidence, rationale, and citations.

The module is intentionally decoupled from the training pipeline. Model
checkpoints are loaded lazily and every model failure degrades gracefully so
the pipeline stays runnable while the models are still being validated.
"""

from .state import FactCheckState, Verdict, Evidence, ModelResult

__all__ = ["FactCheckState", "Verdict", "Evidence", "ModelResult", "run_fact_check"]

__version__ = "0.1.0"


def run_fact_check(statement: str, image_path: str | None = None, language: str = "auto"):
    """Convenience one-shot entrypoint. Builds the graph and runs it once."""
    from .graph import build_graph, initial_state

    graph = build_graph()
    state = initial_state(statement, image_path=image_path, language=language)
    return graph.invoke(state)
