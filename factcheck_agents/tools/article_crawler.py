"""Article crawler: fetch full content and extract publish date.

Supports major Vietnamese news domains. Uses newspaper3k for content extraction
and dateutil for date parsing. Returns full article text + parsed date.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
from dateutil import parser as date_parser


def _extract_date_from_html(html: str, url: str) -> Optional[datetime]:
    """Extract publish date from HTML using common patterns."""
    # Try meta tags first
    date_patterns = [
        r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+property=["\']og:published_time["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+name=["\']date["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+name=["\']pubdate["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+name=["\']DC.date["\'][^>]+content=["\']([^"\']+)["\']',
        r'<time[^>]+datetime=["\']([^"\']+)["\']',
    ]

    for pattern in date_patterns:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            try:
                date_str = match.group(1)
                return date_parser.parse(date_str)
            except Exception:
                continue

    # Try to extract from URL pattern (YYYY/MM/DD or YYYY-MM-DD)
    url_date_patterns = [
        r'/(\d{4})/(\d{2})/(\d{2})/',
        r'/(\d{4})-(\d{2})-(\d{2})/',
    ]
    for pattern in url_date_patterns:
        match = re.search(pattern, url)
        if match:
            try:
                year, month, day = map(int, match.groups())
                return datetime(year, month, day, tzinfo=timezone.utc)
            except Exception:
                continue

    return None


def _extract_content_newspaper3k(url: str) -> Tuple[Optional[str], Optional[datetime]]:
    """Extract article content using newspaper3k library."""
    try:
        from newspaper import Article
    except ImportError:
        return None, None

    try:
        article = Article(url, language='vi')
        article.download()
        article.parse()

        content = article.text
        date = article.publish_date

        # If newspaper3k didn't find date, try manual extraction
        if date is None:
            date = _extract_date_from_html(article.html, url)

        return content, date
    except Exception:
        return None, None


def _extract_content_fallback(url: str) -> Tuple[Optional[str], Optional[datetime]]:
    """Fallback: fetch HTML and extract paragraphs + date."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        html = resp.text

        # Extract date
        date = _extract_date_from_html(html, url)

        # Extract paragraphs from article content
        # Common Vietnamese news site patterns
        content_patterns = [
            r'<div[^>]*class="[^"]*article[^"]*"[^>]*>(.*?)</div>',
            r'<article[^>]*>(.*?)</article>',
            r'<div[^>]*class="[^"]*content[^"]*"[^>]*>(.*?)</div>',
        ]

        for pattern in content_patterns:
            match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
            if match:
                # Extract text from paragraphs
                text = re.sub(r'<[^>]+>', '\n', match.group(1))
                text = re.sub(r'\n+', '\n', text).strip()
                if len(text) > 200:  # Only if substantial content
                    return text, date

        # Fallback: extract all paragraphs
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html, re.DOTALL)
        text = '\n'.join([re.sub(r'<[^>]+>', '', p).strip() for p in paragraphs if p.strip()])
        if len(text) > 200:
            return text, date

        return None, date
    except Exception:
        return None, None


def crawl_article(url: str) -> Tuple[Optional[str], Optional[datetime]]:
    """Crawl article URL and return (full_content, publish_date).

    Tries newspaper3k first, then falls back to manual extraction.
    Returns (None, None) on failure.
    """
    # Try newspaper3k
    content, date = _extract_content_newspaper3k(url)
    if content:
        return content, date

    # Fallback to manual extraction
    return _extract_content_fallback(url)


def is_within_days(date: Optional[datetime], days: int = 7) -> bool:
    """Check if date is within the last N days from now."""
    if date is None:
        return True  # If no date found, include it (conservative)

    # Ensure date is timezone-aware
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    return date >= cutoff
