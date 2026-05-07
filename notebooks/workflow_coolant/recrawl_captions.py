#!/usr/bin/env python3
"""
Re-crawl article pages to recover image captions using improved extraction.

Reads existing JSON files, re-fetches each article's HTML page asynchronously,
re-extracts captions with fallback chain (figcaption -> alt -> title -> default),
and saves to NEW JSON files (does not overwrite originals).

Usage:
    python recrawl_captions.py
    python recrawl_captions.py --splits train
    python recrawl_captions.py --concurrent 20 --timeout 15
"""

import asyncio
import json
import argparse
import os
import sys
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm_asyncio

DEFAULT_CAPTION = "Không được đề cập"

# Site-specific image selectors (matching the crawler implementations)
SITE_SELECTORS = {
    "tienphong.vn": {
        "container": ["table.picture", "figure"],
        "caption": ["td.caption", "figcaption"],
        "img_attr": "src",
    },
    "thanhnien.vn": {
        "container": ["figure"],
        "caption": ["figcaption"],
        "img_attr": "src",
    },
    "default": {
        "container": ["figure"],
        "caption": ["figcaption"],
        "img_attr": "src",
    },
}


def get_site_selector(url: str) -> dict:
    """Get the appropriate selector config for a URL's domain."""
    try:
        domain = urlparse(url).netloc.lower()
    except Exception:
        return SITE_SELECTORS["default"]

    for site_key, selector in SITE_SELECTORS.items():
        if site_key in domain:
            return selector
    return SITE_SELECTORS["default"]


def extract_captions_from_html(html: str, url: str) -> list[dict]:
    """
    Extract image captions from HTML using the improved fallback chain.

    Returns list of {src_url, caption} dicts.
    """
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    selector = get_site_selector(url)
    images = []

    # Find image containers
    containers = []
    for css in selector["container"]:
        containers.extend(soup.select(css))

    for container in containers:
        img_tag = container.find("img")
        if not img_tag:
            continue

        # Get image src
        img_attr = selector["img_attr"]
        src = img_tag.get(img_attr) or img_tag.get("data-src") or img_tag.get("src")
        if not src:
            continue

        # Caption fallback chain
        caption = ""

        # 1. Try caption selectors (figcaption, td.caption, etc.)
        for cap_sel in selector["caption"]:
            cap_el = container.select_one(cap_sel)
            if cap_el:
                caption = cap_el.get_text(strip=True)
                if caption:
                    break

        # 2. Fallback: alt text
        if not caption:
            caption = img_tag.get("alt", "").strip()

        # 3. Fallback: title attribute
        if not caption:
            caption = img_tag.get("title", "").strip()

        # 4. Final fallback: default
        if not caption:
            caption = DEFAULT_CAPTION

        images.append({"src_url": src, "caption": caption})

    return images


async def fetch_page(client: httpx.AsyncClient, url: str, timeout: float = 10) -> str:
    """Fetch a single page, return HTML or empty string on failure."""
    try:
        resp = await client.get(url, timeout=timeout, follow_redirects=True)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return ""


async def recrawl_split(
    articles: list[dict],
    concurrent: int = 10,
    timeout: float = 10,
) -> list[dict]:
    """
    Re-crawl articles to recover captions. Returns updated articles.

    Only re-fetches articles that have images with empty/default captions.
    """
    # Identify articles that need re-crawling
    needs_recrawl = []
    for idx, article in enumerate(articles):
        has_empty = any(
            not (img.get("caption") or "").strip()
            or (img.get("caption") or "").strip() == DEFAULT_CAPTION
            for img in article.get("images", [])
        )
        if has_empty and article.get("source_url"):
            needs_recrawl.append((idx, article["source_url"]))

    if not needs_recrawl:
        print("  All articles already have captions, nothing to re-crawl")
        return articles

    print(f"  Re-crawling {len(needs_recrawl)} articles with missing captions...")

    # Async fetch with semaphore for concurrency control
    sem = asyncio.Semaphore(concurrent)
    results = {}

    async def fetch_with_sem(idx, url):
        async with sem:
            html = await fetch_page(client, url, timeout)
            if html:
                new_captions = extract_captions_from_html(html, url)
                results[idx] = new_captions

    async with httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0 (compatible; research bot)"},
        verify=False,
    ) as client:
        tasks = [fetch_with_sem(idx, url) for idx, url in needs_recrawl]
        await tqdm_asyncio.gather(*tasks, desc="  Fetching pages")

    # Merge new captions into articles
    updated = 0
    for idx, new_imgs in results.items():
        article = articles[idx]
        old_imgs = article.get("images", [])

        # Match by src_url
        new_by_src = {}
        for img in new_imgs:
            # Normalize: extract filename part for matching
            src = img["src_url"]
            new_by_src[src] = img["caption"]

        for old_img in old_imgs:
            old_caption = (old_img.get("caption") or "").strip()
            if not old_caption or old_caption == DEFAULT_CAPTION:
                # Try to find matching new caption
                old_src = old_img.get("src_url", "")
                new_caption = new_by_src.get(old_src, "")

                if not new_caption:
                    # Try partial match on filename
                    old_filename = old_src.split("/")[-1] if old_src else ""
                    for new_src, cap in new_by_src.items():
                        if old_filename and old_filename in new_src:
                            new_caption = cap
                            break

                if new_caption and new_caption != DEFAULT_CAPTION:
                    old_img["caption"] = new_caption
                    updated += 1
                elif not old_caption:
                    old_img["caption"] = DEFAULT_CAPTION

    return articles, updated


async def main():
    parser = argparse.ArgumentParser(description="Re-crawl captions for ViFactCheck images")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--input-dir", default="../data/json", help="Input JSON directory")
    parser.add_argument("--output-dir", default="../data/json/recrawled", help="Output directory for new JSONs")
    parser.add_argument("--concurrent", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--timeout", type=float, default=15, help="Request timeout (seconds)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_dir.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Concurrent: {args.concurrent}, Timeout: {args.timeout}s")
    print()

    for split in args.splits:
        # Find input file
        input_path = input_dir / f"news_data_vifactcheck_{split}_cleaned.json"
        if not input_path.exists():
            input_path = input_dir / f"news_data_vifactcheck_{split}.json"
        if not input_path.exists():
            print(f"{split}: input not found, skipping")
            continue

        print(f"=== {split} ===")
        with open(input_path, "r", encoding="utf-8") as f:
            articles = json.load(f)

        # Count before
        total_imgs = sum(len(a.get("images", [])) for a in articles)
        empty_before = sum(
            1 for a in articles for img in a.get("images", [])
            if not (img.get("caption") or "").strip()
            or (img.get("caption") or "").strip() == DEFAULT_CAPTION
        )
        print(f"  Articles: {len(articles)}, Images: {total_imgs}, Empty captions: {empty_before}")

        # Re-crawl
        articles, n_updated = await recrawl_split(
            articles, concurrent=args.concurrent, timeout=args.timeout
        )

        # Count after
        real_captions = sum(
            1 for a in articles for img in a.get("images", [])
            if (img.get("caption") or "").strip()
            and (img.get("caption") or "").strip() != DEFAULT_CAPTION
        )
        default_captions = sum(
            1 for a in articles for img in a.get("images", [])
            if (img.get("caption") or "").strip() == DEFAULT_CAPTION
        )
        pct = real_captions / max(total_imgs, 1) * 100

        print(f"  Updated: {n_updated} captions recovered")
        print(f"  Real captions: {real_captions}/{total_imgs} ({pct:.1f}%)")
        print(f"  Default captions: {default_captions}")

        # Ensure 100% coverage
        for a in articles:
            for img in a.get("images", []):
                if not (img.get("caption") or "").strip():
                    img["caption"] = DEFAULT_CAPTION

        # Save to new file
        output_path = output_dir / f"news_data_vifactcheck_{split}_recrawled.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {output_path.name}")
        print()

    print("Done. New JSON files in:", output_dir.resolve())


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())
