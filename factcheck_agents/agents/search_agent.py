"""Search agent: gather evidence from open truth sources.

Drafts a few focused search queries for the statement (LLM if available, else
a heuristic fallback), runs them through the web-search tool, and de-duplicates
the results into the shared state.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests

from ..config import settings
from ..graph_utils import EvidenceGraph
from ..source_tier import classify_domain
from ..state import Evidence, FactCheckState
from ..tools.web_search import web_search
from .llm import get_llm, parse_json
from ..prompts import SEARCH_QUERY_PROMPT


def _draft_queries(statement: str) -> List[str]:
    llm = get_llm()
    if llm is not None:
        try:
            resp = llm.invoke(
                SEARCH_QUERY_PROMPT.format(n=settings.max_queries, statement=statement)
            )
            data = parse_json(getattr(resp, "content", "") or "")
            if data and isinstance(data.get("queries"), list):
                qs = [q.strip() for q in data["queries"] if q and q.strip()]
                if qs:
                    return qs[: settings.max_queries]
        except Exception:
            pass
    # heuristic fallback: use the statement itself
    return [statement.strip()]


def _fetch_evidence_image(
    url: str,
) -> tuple[Optional[str], Optional[str]]:
    """Best-effort: fetch the HTML page and download its primary image + caption.

    Returns (local_image_path, image_caption). Either may be None.
    Uses BeautifulSoup if available; falls back to regex.
    """
    if not url:
        return None, None
    try:
        cache_dir = settings.data_root / "cache" / "evidence_images"
        cache_dir.mkdir(parents=True, exist_ok=True)
        url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()

        # Try an existing cached file first.
        cached = sorted(cache_dir.glob(f"{url_hash}.*"))
        if cached:
            ext = cached[0].suffix
            caption_path = cached[0].with_suffix(f"{ext}.caption")
            caption = caption_path.read_text() if caption_path.exists() else None
            return str(cached[0]), caption

        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return None, None

        headers = {"User-Agent": "Mozilla/5.0 (compatible; FakeNewBot/0.1)"}
        html_resp = requests.get(url, timeout=15, headers=headers)
        html_resp.raise_for_status()
        text = html_resp.text

        image_url: Optional[str] = None
        image_caption: Optional[str] = None

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(text, "html.parser")
            og = soup.find("meta", property="og:image")
            if og and og.get("content"):
                image_url = og["content"]
            else:
                tw = soup.find("meta", attrs={"name": "twitter:image"})
                if tw and tw.get("content"):
                    image_url = tw["content"]

            if image_url:
                og_caption = soup.find("meta", property="og:image:alt")
                if og_caption and og_caption.get("content"):
                    image_caption = og_caption["content"]

            if not image_url:
                figure = soup.find("figure")
                if figure:
                    img = figure.find("img")
                    if img:
                        image_url = img.get("src") or img.get("data-src")
                        image_caption = (
                            figure.find("figcaption").get_text(strip=True)
                            if figure.find("figcaption")
                            else None
                        )
                        if not image_caption and img:
                            image_caption = img.get("alt") or img.get("title")
                if not image_url:
                    img = soup.find("img")
                    if img:
                        image_url = img.get("src") or img.get("data-src")
                        image_caption = img.get("alt") or img.get("title")
        except Exception:
            # Fallback: regex extraction.
            og_match = re.search(
                r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
                text,
                re.IGNORECASE,
            )
            if og_match:
                image_url = og_match.group(1)
            else:
                tw_match = re.search(
                    r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
                    text,
                    re.IGNORECASE,
                )
                if tw_match:
                    image_url = tw_match.group(1)
                else:
                    img_match = re.search(
                        r'<img[^>]+src=["\']([^"\']+)["\']', text, re.IGNORECASE
                    )
                    if img_match:
                        image_url = img_match.group(1)

        if not image_url:
            return None, None

        image_url = urljoin(url, image_url)
        ext = Path(urlparse(image_url).path).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
            ext = ".jpg"
        image_path = cache_dir / f"{url_hash}{ext}"

        img_resp = requests.get(image_url, timeout=15, headers=headers)
        img_resp.raise_for_status()
        image_path.write_bytes(img_resp.content)

        if image_caption:
            caption_path = image_path.with_suffix(f"{ext}.caption")
            caption_path.write_text(image_caption, encoding="utf-8")

        return str(image_path), image_caption
    except Exception:
        return None, None


def search_agent(state: FactCheckState) -> dict:
    statement = state["statement"]
    queries = _draft_queries(statement)

    trusted_list = [d.strip() for d in settings.trusted_domains.split(",") if d.strip()]
    flagged_list = [d.strip() for d in settings.flagged_domains.split(",") if d.strip()]

    seen: set = set()
    raw: List[Evidence] = []

    for include_domains in (trusted_list, flagged_list, None):
        for q in queries:
            results = web_search(q, include_domains=include_domains)
            for e in results:
                url = e.get("url", "")
                if url and url in seen:
                    continue
                if url:
                    seen.add(url)
                e["source_tier"] = classify_domain(url) if url else "unknown"
                e["image_path"], e["image_caption"] = (
                    _fetch_evidence_image(url) if url else (None, None)
                )
                raw.append(e)

    _tier_priority = {"trusted": 0, "flagged": 1, "social": 2, "unknown": 3}
    raw.sort(
        key=lambda e: (
            _tier_priority.get(e.get("source_tier", "unknown"), 3),
            -e.get("score", 0.0),
        )
    )
    cap = 2 * settings.max_results
    evidence = raw[:cap]

    evidence_graph = EvidenceGraph.build_from_evidence(evidence)

    msg = f"[Search] {len(queries)} queries x 3 passes -> {len(evidence)} evidence items (capped at {cap})"
    if not settings.has_search():
        msg += " (no search provider configured; continuing on model output only)"

    return {
        "search_queries": queries,
        "evidence": evidence,
        "evidence_graph": evidence_graph,
        "messages": [("assistant", msg)],
    }
