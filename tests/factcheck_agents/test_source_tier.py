import pytest
from factcheck_agents.source_tier import classify_domain


@pytest.mark.parametrize("url,expected", [
    ("https://vnexpress.net/story", "trusted"),
    ("https://thanhnien.vn/story", "trusted"),
    ("https://dantri.com.vn/story", "trusted"),
    ("https://tuoitre.vn/story", "trusted"),
    ("https://www.vnexpress.net/story", "trusted"),    # www prefix
    ("https://news.dantri.com.vn/story", "trusted"),  # subdomain
    ("https://kenh14.vn/story", "flagged"),
    ("https://example.com/story", "unknown"),
    ("https://google.com/search", "unknown"),
    ("", "unknown"),
])
def test_classify_domain(url, expected):
    assert classify_domain(url) == expected
