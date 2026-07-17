"""Command-line entrypoint.

    python -m factcheck_agents.cli "Your claim here"
    python -m factcheck_agents.cli "Claim with a picture" --image /path/to.jpg --json
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import settings
from .graph import build_graph, initial_state


def _print_human(result: dict) -> None:
    v = result.get("verdict", {}) or {}
    print("\n" + "=" * 60)
    print(f"VERDICT: {v.get('label', 'UNVERIFIED')}  (confidence {v.get('confidence', 0):.2f})")
    print("=" * 60)
    if v.get("rationale"):
        print(f"\nRationale:\n{v['rationale']}")
    print("\nModel predictions:")
    for m in result.get("model_results", []) or []:
        if m.get("available"):
            print(f"  - {m['model']}: {m.get('label')} (conf={m.get('confidence')})")
        else:
            print(f"  - {m['model']}: unavailable — {m.get('note')}")
    cites = v.get("citations") or []
    if cites:
        print("\nCitations:")
        for c in cites:
            print(f"  - {c}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-agent fake-news fact checker")
    parser.add_argument("statement", help="the claim/statement to fact-check")
    parser.add_argument("--image", default=None, help="optional image path (enables COOLANT)")
    parser.add_argument("--language", default="auto", help="vi | en | auto")
    parser.add_argument("--json", action="store_true", help="print full result as JSON")
    args = parser.parse_args(argv)

    if not settings.has_llm():
        print("[warn] OPENAI_API_KEY not set — using rule-based fallback conclusion.", file=sys.stderr)
    if not settings.has_search():
        print("[warn] No search provider (Tavily/Google CSE) — evidence step will be empty.", file=sys.stderr)

    graph = build_graph()
    result = graph.invoke(initial_state(args.statement, image_path=args.image, language=args.language))

    if args.json:
        printable = {k: v for k, v in result.items() if k != "messages"}
        print(json.dumps(printable, ensure_ascii=False, indent=2, default=str))
    else:
        _print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
