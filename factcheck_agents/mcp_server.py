"""MCP server exposing the fact-checking pipeline as tools.

Run (stdio transport):

    python -m factcheck_agents.mcp_server

Register in an MCP client (e.g. Windsurf `.mcp.json`):

    {
      "mcpServers": {
        "factcheck": {
          "command": "python",
          "args": ["-m", "factcheck_agents.mcp_server"],
          "cwd": "/Users/haila/My File/projects/fake-new-detection"
        }
      }
    }

Tools:
    - fact_check(statement, image_path?)      -> full verdict + trace
    - search_evidence(statement)              -> evidence items only
    - evaluate_statement(statement, image_path?, evidence_text?) -> raw model outputs
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from .agents.evaluate_agent import _coolant, _phobert
from .agents.search_agent import search_agent
from .graph import build_graph, initial_state
from .state import FactCheckState

mcp = FastMCP("factcheck")


@mcp.tool()
def fact_check(statement: str, image_path: Optional[str] = None, language: str = "auto") -> dict:
    """Run the full Search -> Evaluate -> Conclusion pipeline on a claim."""
    graph = build_graph()
    result = graph.invoke(initial_state(statement, image_path=image_path, language=language))
    return {
        "statement": statement,
        "verdict": result.get("verdict", {}),
        "model_results": result.get("model_results", []),
        "evidence": result.get("evidence", []),
        "search_queries": result.get("search_queries", []),
    }


@mcp.tool()
def search_evidence(statement: str) -> dict:
    """Search open truth sources for evidence about the claim (no model inference)."""
    out = search_agent(FactCheckState(statement=statement))
    return {"queries": out.get("search_queries", []), "evidence": out.get("evidence", [])}


@mcp.tool()
def evaluate_statement(
    statement: str, image_path: Optional[str] = None, evidence_text: str = ""
) -> dict:
    """Run the two trained models directly on a claim (+optional evidence/image)."""
    return {
        "results": [
            _phobert().predict(statement, evidence_text),
            _coolant().predict(statement, image_path),
        ]
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
