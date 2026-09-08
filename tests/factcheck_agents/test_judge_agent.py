import importlib


def test_judge_prompt_override(monkeypatch):
    """FACTCHECK_JUDGE_PROMPT override is picked up at judge_agent module load."""
    from factcheck_agents.config import settings

    monkeypatch.setattr(settings, "judge_prompt", "CUSTOM JUDGE PROMPT")
    judge_agent = importlib.import_module("factcheck_agents.agents.judge_agent")
    judge_agent = importlib.reload(judge_agent)
    assert judge_agent.JUDGE_SYSTEM_PROMPT == "CUSTOM JUDGE PROMPT"
