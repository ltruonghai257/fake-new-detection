from unittest.mock import patch, MagicMock
import pytest


@patch("factcheck_agents.agents.evaluate_agent._phobert")
@patch("factcheck_agents.agents.evaluate_agent._coolant")
def test_coolant_receives_image_path_from_state(mock_coolant, mock_phobert):
    """COOLANT must receive the image_path carried in FactCheckState."""
    from factcheck_agents.agents.evaluate_agent import evaluate_agent

    mock_phobert.return_value.predict.return_value = {
        "model": "phobert_vifactcheck",
        "available": True,
        "label": "NEI",
        "confidence": 0.5,
    }
    mock_coolant.return_value.predict.return_value = {
        "model": "coolant",
        "available": True,
        "label": "REAL",
        "confidence": 0.9,
    }

    state = {
        "statement": "claim",
        "evidence": [],
        "image_path": "/tmp/test.jpg",
    }

    evaluate_agent(state)

    mock_coolant.return_value.predict.assert_called_once_with("claim", "/tmp/test.jpg")
