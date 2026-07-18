"""URL-to-source-tier classifier. Returns 'trusted', 'flagged', or 'unknown'."""
from __future__ import annotations

from urllib.parse import urlparse

from .config import settings


def classify_domain(url: str) -> str:
    hostname = urlparse(url).hostname or ""
    domain = hostname.removeprefix("www.")
    trusted = [d.strip() for d in settings.trusted_domains.split(",") if d.strip()]
    flagged = [d.strip() for d in settings.flagged_domains.split(",") if d.strip()]
    if any(domain == t or domain.endswith("." + t) for t in trusted):
        return "trusted"
    if any(domain == f or domain.endswith("." + f) for f in flagged):
        return "flagged"
    return "unknown"
