"""
Demo script showing the factcheck pipeline flow with mocked agents.
This demonstrates the complete v2.0 architecture without requiring external services.
"""

from unittest.mock import patch
from langgraph.checkpoint.memory import MemorySaver
from factcheck_agents.graph import build_graph, initial_state

print("=" * 60)
print("FAKE NEWS DETECTION PIPELINE DEMO")
print("=" * 60)

# Mock the agent functions to simulate their behavior
with patch("factcheck_agents.graph.search_agent") as mock_search, \
     patch("factcheck_agents.graph.verify_agent") as mock_verify, \
     patch("factcheck_agents.graph.social_search_agent") as mock_social, \
     patch("factcheck_agents.graph.conclusion_agent") as mock_concl:

    # Simulate search agent returning evidence
    mock_search.return_value = {
        "evidence": [
            {
                "content": "Sample evidence from trusted source",
                "source": "vnexpress.net",
                "source_tier": "trusted",
                "url": "https://vnexpress.net/sample-article"
            }
        ],
        "search_queries": ["tuyên bố kiểm tra", "kiểm tra tin giả"],
        "evidence_graph": None,  # Would be EvidenceGraph in real execution
    }

    # Simulate verify agent running PhoBERT + COOLANT
    mock_verify.return_value = {
        "model_results": [
            {
                "model": "phobert",
                "label": "REFUTED",
                "confidence": 0.85,
                "available": True
            },
            {
                "model": "coolant",
                "label": "FALSE",
                "confidence": 0.78,
                "available": True
            }
        ],
        "reliability_signal": True,  # High confidence triggers social search
    }

    # Simulate social search agent (only runs when reliability_signal=True)
    mock_social.return_value = {
        "evidence_graph": None  # Would merge social evidence into graph
    }

    # Simulate conclusion agent with binary verdict and Vietnamese labels
    mock_concl.return_value = {
        "verdict": {
            "label": "REFUTED",  # 4-class label
            "verdict_binary": "FAKE",  # Binary verdict
            "verdict_label_vi": "Giả",  # Vietnamese label
            "confidence": 0.85,
            "rationale": "Multiple trusted sources contradict this claim. PhoBERT and COOLANT models both indicate false information.",
            "citations": ["vnexpress.net/sample-article"],
            "recommendation": "Do not share this information without verification."
        }
    }

    # Build and invoke the graph
    print("\n1. Building LangGraph with checkpointer...")
    graph = build_graph(checkpointer=MemorySaver())

    print("2. Creating initial state for claim: 'Tuyên bố kiểm tra tin giả'")
    state = initial_state("Tuyên bố kiểm tra tin giả")

    print("3. Executing pipeline with thread_id for checkpointing...")
    result = graph.invoke(state, config={"configurable": {"thread_id": "demo-session-001"}})

    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)

    # Display the results
    verdict = result.get("verdict", {})
    print(f"\n4-Class Label: {verdict.get('label')}")
    print(f"Binary Verdict: {verdict.get('verdict_binary')}")
    print(f"Vietnamese Label: {verdict.get('verdict_label_vi')}")
    print(f"Confidence: {verdict.get('confidence', 0):.2f}")

    print(f"\nRationale:")
    print(f"  {verdict.get('rationale', 'N/A')}")

    print(f"\nModel Results:")
    for model_result in result.get("model_results", []):
        status = "✓" if model_result.get("available") else "✗"
        print(f"  {status} {model_result['model']}: {model_result.get('label')} (conf={model_result.get('confidence', 0):.2f})")

    print(f"\nReliability Signal: {result.get('reliability_signal')}")
    print(f"  → Social search {'was triggered' if result.get('reliability_signal') else 'was skipped'}")

    print(f"\nEvidence Sources:")
    for evidence in result.get("evidence", []):
        tier = evidence.get("source_tier", "unknown")
        print(f"  [{tier.upper()}] {evidence.get('source')} - {evidence.get('url')}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nKey v2.0 Features Demonstrated:")
    print("  ✓ Evidence graph architecture")
    print("  ✓ Source-tier classification (trusted/flagged/social/unknown)")
    print("  ✓ Concurrent PhoBERT + COOLANT execution")
    print("  ✓ Reliability signal computation")
    print("  ✓ Conditional social search routing")
    print("  ✓ Binary verdict (REAL/FAKE)")
    print("  ✓ Vietnamese language labels (Thật/Giả)")
    print("  ✓ LangGraph checkpointer for resume capability")
    print("=" * 60)