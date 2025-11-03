# Copilot / AI Agent instructions for this repository

This repo contains crawlers and helpers used to build a fake-news dataset. The notes below focus on actionable, repo-specific knowledge an AI coding agent needs to be productive immediately.

1) Big picture
- Purpose: crawl Vietnamese news sites, extract title/content/images, and save JSON + image files for downstream dataset work.
- Where to look:
  - crawling core: `src/crawler/base_crawler.py` (generic crawling/parsing/saving flow)
  - crawler registry / orchestrator: `src/crawler/base.py` (CRAWLER_MAPPING and `get_crawler`)
  - per-site implementations: `src/crawler/news/real/*.py` (one class per site)
  - HTTP wrapper: `src/helpers/httpx_client.py` (async httpx client with custom SSL settings)
  - utilities: `src/helpers/string_handle.py` and `src/helpers/file_handler/*` (filenames, sanitization, persistence)

2) Important patterns & contracts (copy these when editing or adding crawlers)
- Crawler subclass contract (see `BaseCrawler`): implement properties `title_selector`, `image_selector`, `link_selector`, `url_prefix`, `content_selector` (optional `url_suffix`). Selectors accept dicts with keys `css_selector` (list) and/or `xpath_selector` (list).
- Image selector supports keys: `image_tag_selector` (default `img`), `caption_tag_selector` (default `figcaption`), and `image_tag_attr` (default `src`). Use these keys when extracting images.
- All network code is async. Use `await crawler.arun(...)` or `await crawler.simple_crawling(...)`. The project uses `BaseClient` (httpx.AsyncClient) — keep async patterns and error handling.
- When adding a new site crawler:
  1. Create `src/crawler/news/real/<SiteName>Crawler.py` implementing the BaseCrawler interface.
  2. Import the new class in `src/crawler/base.py` and add it to `CRAWLER_MAPPING` keyed by a domain substring (e.g. `"vnexpress.net": VnExpressCrawler`).

3) Data flow & file layout
- Crawled JSON is written through `helpers.file_handler.FileHandler`; images are saved under the image format folder. File naming uses `StringHandler.sanitize_filename` and `class_name_to_snake_case`.
- The `arun` method returns a list of `CrawlResult` objects (`src/crawler/crawl_result.py`) and optionally writes output; default `.json` output is `{crawler_name}.json`.

4) Helpful implementation details / gotchas
- Selector resolution: `_get_elements` in `BaseCrawler` checks `css_selector` first, then `xpath_selector`. Provide both where needed; prefer css selectors for simple pages.
- Links filtering: `BaseCrawler` currently filters links by `url_prefix` and `url_suffix` — match these correctly for each site (see `url_suffix` default `.html`).
- SSL and legacy servers: `BaseClient` uses a custom SSL context and low security level to handle older/misconfigured servers. Avoid removing this unless you confirm target hosts are modern.
- Network failures: `_execute_single_crawl` catches `httpx.RequestError` and wraps them into `CrawlResult(success=False)` — follow that pattern when adding error handling.

5) Developer workflows (commands & environment)
- Create environment (preferred):
  - conda: `conda env create -f environment.yml` (conda env name `fake_news` in file)
  - or pip: `pip install -r requirements.txt` (requirements exist but environment.yml is canonical)
- Run tests: `python -m pytest -q` (there is a `pytest.ini` present; tests are small/partial).
- Run a single crawler (example snippet to run interactively):
  - Example (async snippet you can run from a small script or notebook):
    ```py
    import asyncio
    from src.crawler.base import get_crawler

    async def main():
        crawler = get_crawler('https://vnexpress.net/some-article')
        results = await crawler.arun(url='https://vnexpress.net/some-article')
        print(results[0].title)

    asyncio.run(main())
    ```

6) Tests & quick checks
- Unit tests live under `tests/`. Run pytest and fix failures locally. There are some empty test files — treat them as placeholders.

7) What to change/where when editing
- If you change selector formats or CrawlResult fields, update `BaseCrawler._parse_html_content`, `crawl_result.py`, and any code that serializes results (file handlers).
- If you change how images are saved, update `FileHandler` implementations in `src/helpers/file_handler/*`.

8) Minimal examples (quick references)
- Adding crawler: add class file `src/crawler/news/real/MySiteCrawler.py` and add mapping in `src/crawler/base.py`.
- Use selectors like:
  - title_selector = {"css_selector": ["h1.title"]}
  - image_selector = {"css_selector": ["figure.article-image"], "image_tag_attr": "data-src"}

If anything below is unclear or you want more detail in a particular area (e.g., file-handler internals, test setup, or how image folders are organized), tell me which part to expand and I'll iterate.
