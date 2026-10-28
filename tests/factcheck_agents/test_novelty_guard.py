import json

from factcheck_agents.agents.debate_utils import (
    _extract_prior_claims,
    _invoke_advocate_with_novelty,
    _repeat_report,
)

OLD = "Nguồn VnExpress xác nhận sự việc đã xảy ra vào sáng nay tại Hà Nội."
NEW = "Ảnh hiện trường có metadata chụp ngày khác nên không khớp với thời điểm đưa tin."


class _Resp:
    def __init__(self, arg):
        self.content = json.dumps(
            {"verdict": "REAL", "confidence": 0.6, "argument": arg, "concession": None}
        )


class _LLM:
    def __init__(self, args):
        self.args, self.calls = list(args), []

    def invoke(self, messages):
        self.calls.append(messages)
        return _Resp(self.args.pop(0))


def test_repeat_report_detects_copy():
    overlap, rep = _repeat_report(OLD, [{"argument": OLD}])
    assert overlap > 0.9 and rep


def test_repeat_report_new_content_low_overlap():
    overlap, rep = _repeat_report(NEW, [{"argument": OLD}])
    assert overlap < 0.35 and not rep


def test_prior_claims_include_all_sentences():
    claims = _extract_prior_claims([{"agent": "a", "argument": f"{OLD} {NEW}"}])
    assert len(claims) == 2


def test_guard_retries_until_novel():
    turns = [{"agent": "fake_advocate", "argument": OLD}]
    llm = _LLM([OLD, NEW])
    content, data, attempts = _invoke_advocate_with_novelty(llm, "sys", "usr", turns)
    assert attempts == 2 and data["argument"] == NEW
    assert "LẶP LẠI" in llm.calls[1][1][1]


def test_guard_first_turn_no_retry():
    llm = _LLM([OLD])
    _, _, attempts = _invoke_advocate_with_novelty(llm, "sys", "usr", [])
    assert attempts == 1


def test_guard_gives_least_repetitive_after_retries():
    turns = [{"agent": "x", "argument": OLD}]
    llm = _LLM([OLD, OLD, OLD])
    _, data, attempts = _invoke_advocate_with_novelty(llm, "sys", "usr", turns)
    assert attempts == 3 and data["argument"] == OLD
