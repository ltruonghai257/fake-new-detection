from factcheck_agents.agents.debate_utils import (
    _assess_source_unanimity,
    _build_advocate_user_message,
)


def _ev(url, snippet, tier="trusted"):
    return {"title": "t", "url": url, "snippet": snippet, "source_tier": tier}


def test_no_evidence_no_warning():
    assert _assess_source_unanimity([], []) is None


def test_balanced_diverse_sources_no_warning():
    real = [_ev("https://vnexpress.net/a", "Tin A")]
    fake = [_ev("https://kenh14.vn/b", "Tin B", "flagged")]
    assert _assess_source_unanimity(real, fake) is None


def test_one_sided_sources_raise_doubt():
    real = [_ev("https://vnexpress.net/a", "A"), _ev("https://tuoitre.vn/b", "B")]
    warn = _assess_source_unanimity(real, [])
    assert warn and "NGHI NGỜ" in warn and "TẤT CẢ" in warn


def test_same_domain_and_copied_snippets_raise_doubt():
    real = [_ev("https://vnexpress.net/a", "Cùng một tin")]
    fake = [_ev("https://www.vnexpress.net/b", "Cùng một tin", "flagged")]
    warn = _assess_source_unanimity(real, fake)
    assert "cùng một domain" in warn and "trùng nhau" in warn


def test_advocate_message_splits_truth_and_fake_sources():
    real = [_ev("https://vnexpress.net/a", "A")]
    msg = _build_advocate_user_message(
        "claim", [], real, [], role="real", evidence_real=real, evidence_fake=[]
    )
    assert "TRUTH SOURCES" in msg and "FAKE SOURCES" in msg and "NGHI NGỜ" in msg
