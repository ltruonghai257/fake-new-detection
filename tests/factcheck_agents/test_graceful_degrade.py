from unittest.mock import patch

from langgraph.checkpoint.memory import MemorySaver

from factcheck_agents.graph import build_graph, initial_state


@patch("factcheck_agents.graph.conclusion_agent")
@patch("factcheck_agents.graph.social_search_agent")
@patch("factcheck_agents.graph.verify_agent")
@patch("factcheck_agents.graph.search_agent")
def test_graceful_degrade_no_crash(mock_search, mock_verify, mock_social, mock_concl):
    mock_search.return_value = {
        "evidence": [],
        "search_queries": [],
        "evidence_graph": None,
    }
    mock_verify.return_value = {
        "model_results": [],
        "reliability_signal": False,
    }
    mock_social.return_value = {}
    mock_concl.return_value = {
        "verdict": {
            "label": "UNVERIFIED",
            "confidence": 0.0,
            "verdict_binary": "FAKE",
            "verdict_label_vi": "Giả",
            "rationale": "",
            "citations": [],
            "recommendation": "",
        }
    }

    graph = build_graph(checkpointer=MemorySaver())
    state = initial_state("Tuyên bố kiểm tra")
    result = graph.invoke(
        state, config={"configurable": {"thread_id": "test-graceful-01"}}
    )
    assert isinstance(
        result.get("verdict"), dict
    ), f"verdict missing or not dict: {result.get('verdict')!r}"
