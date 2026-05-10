# Phase 1 Plan: Data Crawling Notebook

**Phase:** 1 — Data Crawling Notebook  
**Goal:** Create `notebooks/01_data_crawling.ipynb` — a clean, resumable, production-ready crawling notebook that replaces the scattered experimental crawling notebooks.  
**Requirements:** CRAWL-01, CRAWL-02, CRAWL-03, CRAWL-04, NB-01, NB-02, NB-03  
**Planned:** 2026-05-10  
**Status:** Ready to execute

---

## Context

### Existing Notebooks to Replace/Supersede

- `notebooks/1_crawl_only.ipynb` — basic ViFactCheck crawler, no progress display, clears cache on start (breaks resumability)
- `notebooks/crawl_and_preprocess_final.ipynb` — mixed crawl + preprocess, not clean separation
- Several other `crawl_and_preprocess_*.ipynb` variants — experimental, not clean

### Key Existing Infrastructure (reuse, don't rewrite)

- `src/crawler/crawler_factory.py` — `CrawlerFactory` handles domain routing, async crawl, checkpoint caching (`crawling_status.json` pattern), progress via `tqdm`
- `src/crawler/base_crawler.py` — `BaseCrawler.arun()` + `_save_images()` + resume logic
- `src/crawler/output_formats.py` — `OutputFormatter` with `custom`, `ml_training`, `research`, etc. formats
- `src/crawler/news/real/` — 9 site-specific crawlers (VnExpress, DanTri, TuoiTre, etc.)
- `src/processing/dataset_handler.py` — `DatasetHandler` loads ViFactCheck HuggingFace dataset URLs

### Existing Bugs / Issues to Fix in New Notebook

- Old `1_crawl_only.ipynb` calls `crawler_factory.clear_cache()` before crawling — this destroys resume state. Fix: only load cache, never clear it unless explicitly requested.
- Output path in `crawl_and_save_all` is hardcoded to `data/json/` — must be config-driven.
- No human-readable progress summary after crawl completes.

### Output Format Decision

Use `ml_training` format (produces `id`, `features`, `metadata`, `media` fields — structured for ML downstream use). This is richer than `custom` and matches thesis preprocessing needs.

---

## Tasks

### Task 1 — Create notebook skeleton with config cell

**File:** `notebooks/01_data_crawling.ipynb`  
**Action:** Create new notebook

**Cell 0 — Markdown header:**
```markdown
# Data Crawling — Vietnamese Fake News Detection

Automated, resumable crawling of Vietnamese news articles from the ViFactCheck dataset.

**Sources:** ViFactCheck HuggingFace dataset (`tranthaihoa/vifactcheck`)  
**Output:** Structured JSON files per split (train/dev/test) under `OUTPUT_DIR`  
**Resume:** Re-run this notebook at any time — already-crawled URLs are skipped automatically.
```

**Cell 1 — Config cell (CRAWL-01, NB-01, NB-02):**
```python
# ============================================================
# CONFIGURATION — edit this cell only
# ============================================================
from pathlib import Path

PROJECT_ROOT = Path("..").resolve()  # adjust if running from different cwd

CONFIG = {
    # Data source
    "dataset_name": "tranthaihoa/vifactcheck",
    "url_column": "Url",
    "splits": ["train", "dev", "test"],
    "url_limit": None,           # None = all URLs; set int to limit (e.g. 100 for testing)

    # Output
    "output_dir": PROJECT_ROOT / "data" / "json",
    "cache_dir": PROJECT_ROOT / "data" / "caches",
    "output_format": "ml_training",  # options: custom, ml_training, research, detailed

    # Crawl settings
    "max_concurrent": 10,        # parallel requests per split
    "save_interval": 50,         # checkpoint every N completed URLs
    "retry_failed": False,       # set True to retry previously failed URLs
}
```

**Verify:** Cell exists with all CONFIG keys, uses `PROJECT_ROOT` relative paths, no hardcoded absolute paths.

---

### Task 2 — Setup cell (environment + imports)

**Cell 2 — Install dependencies:**
```python
import subprocess, sys

_PACKAGES = ["loguru", "tqdm", "beautifulsoup4", "lxml", "httpx",
             "nest-asyncio", "datasets", "Pillow"]

for pkg in _PACKAGES:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
        print(f"Installed {pkg}")
print("Dependencies ready.")
```

**Cell 3 — Setup paths + imports:**
```python
import sys, os
import nest_asyncio
nest_asyncio.apply()

# Add project root to path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Create output directories
CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
CONFIG["cache_dir"].mkdir(parents=True, exist_ok=True)

from src.crawler.crawler_factory import CrawlerFactory
from src.processing.dataset_handler import DatasetHandler

print(f"Project root: {PROJECT_ROOT}")
print(f"Output dir:   {CONFIG['output_dir']}")
print(f"Cache dir:    {CONFIG['cache_dir']}")
```

**Verify:** Imports succeed, directories created, no `clear_cache()` called.

---

### Task 3 — URL loading cell with counts display (CRAWL-02 prerequisite)

**Cell 4 — Markdown section header (NB-03):**
```markdown
## Step 1: Load URLs from ViFactCheck Dataset
```

**Cell 5 — Load URLs per split:**
```python
dataset_handler = DatasetHandler(CONFIG["dataset_name"])

split_urls = {}
for split in CONFIG["splits"]:
    urls = dataset_handler.get_urls_from_split(
        split=split,
        url_column=CONFIG["url_column"],
        limit=CONFIG["url_limit"],
    )
    split_urls[split] = urls
    print(f"  {split:6s}: {len(urls):,} URLs")

total = sum(len(v) for v in split_urls.values())
print(f"\nTotal URLs to process: {total:,}")
```

**Verify:** Prints URL counts per split. Total is > 0.

---

### Task 4 — Resume status display cell (CRAWL-03)

**Cell 6 — Markdown section header (NB-03):**
```markdown
## Step 2: Check Resume State
```

**Cell 7 — Show existing cache state:**
```python
import json

for split in CONFIG["splits"]:
    cache_file = CONFIG["cache_dir"] / f"crawling_status_{split}.json"
    failed_file = CONFIG["cache_dir"] / f"failed_urls_{split}.json"

    completed = 0
    failed = 0

    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
            completed = data.get("length", len(data.get("urls", [])))

    if failed_file.exists():
        with open(failed_file) as f:
            failed = len(json.load(f))

    total = len(split_urls[split])
    remaining = total - completed
    print(f"  {split:6s}: {completed:,} done / {failed:,} failed / {remaining:,} remaining")

print("\nRe-run this notebook to resume. Already-crawled URLs are skipped automatically.")
```

**Verify:** Shows per-split resume status. No cache files cleared.

---

### Task 5 — Crawl execution cells (CRAWL-02, CRAWL-03, CRAWL-04)

**Cell 8 — Markdown section header (NB-03):**
```markdown
## Step 3: Crawl Articles

Progress is shown per split. Checkpoints are saved every `save_interval` URLs.
Re-run this cell at any time to resume from where crawling stopped.
```

**Cell 9 — Crawl function:**
```python
async def crawl_split(split: str, urls: list) -> dict:
    """Crawl one split, return summary dict."""
    cache_file = str(CONFIG["cache_dir"] / f"crawling_status_{split}.json")
    failed_file = str(CONFIG["cache_dir"] / f"failed_urls_{split}.json")
    output_file = f"news_data_vifactcheck_{split}.json"

    print(f"\n{'='*50}")
    print(f"Crawling split: {split} ({len(urls):,} URLs)")
    print(f"{'='*50}")

    factory = CrawlerFactory(
        cache_filename=cache_file,
        failed_log_filename=failed_file,
    )

    await factory.crawl_and_save_all(
        urls=urls,
        output_filename=output_file,
        format_name=CONFIG["output_format"],
        max_concurrent=CONFIG["max_concurrent"],
        retry_failed=CONFIG["retry_failed"],
        save_interval=CONFIG["save_interval"],
    )

    # Return summary
    completed = 0
    if Path(cache_file).exists():
        with open(cache_file) as f:
            data = json.load(f)
            completed = data.get("length", 0)

    failed = 0
    if Path(failed_file).exists():
        with open(failed_file) as f:
            failed = len(json.load(f))

    return {"split": split, "total": len(urls), "completed": completed, "failed": failed}
```

**Cell 10 — Execute crawl:**
```python
import asyncio

summaries = []
for split in CONFIG["splits"]:
    summary = await crawl_split(split, split_urls[split])
    summaries.append(summary)

print("\nAll splits processed.")
```

**Verify:** `tqdm` progress bar shown per split. Cache files written to `CONFIG["cache_dir"]`. No exception on re-run (resume works).

---

### Task 6 — Results summary cell (CRAWL-02, CRAWL-04, NB-03)

**Cell 11 — Markdown section header:**
```markdown
## Step 4: Results Summary
```

**Cell 12 — Display crawl results:**
```python
print(f"\n{'Split':<8} {'Total':>8} {'Completed':>10} {'Failed':>8} {'Rate':>8}")
print("-" * 50)

for s in summaries:
    rate = f"{s['completed'] / s['total'] * 100:.1f}%" if s['total'] else "—"
    print(f"{s['split']:<8} {s['total']:>8,} {s['completed']:>10,} {s['failed']:>8,} {rate:>8}")

print("-" * 50)
total_urls = sum(s["total"] for s in summaries)
total_done = sum(s["completed"] for s in summaries)
total_fail = sum(s["failed"] for s in summaries)
overall_rate = f"{total_done / total_urls * 100:.1f}%" if total_urls else "—"
print(f"{'TOTAL':<8} {total_urls:>8,} {total_done:>10,} {total_fail:>8,} {overall_rate:>8}")
```

**Cell 13 — Verify output files:**
```python
print("\nOutput files:")
for split in CONFIG["splits"]:
    out_file = CONFIG["output_dir"] / f"news_data_vifactcheck_{split}.json"
    if out_file.exists():
        with open(out_file) as f:
            count = len(json.load(f))
        size_mb = out_file.stat().st_size / 1024 / 1024
        print(f"  {split:6s}: {out_file.name} — {count:,} articles ({size_mb:.1f} MB)")
    else:
        print(f"  {split:6s}: not yet created")

print("\nNext step: Run notebooks/02_preprocessing.ipynb")
```

**Verify:** Table printed. File sizes shown. Articles count > 0 for crawled splits.

---

### Task 7 — Fix output path bug in CrawlerFactory

**File:** `src/crawler/crawler_factory.py`  
**Issue:** Line 203 hardcodes `os.path.join("data", "json", output_filename)` — ignores the caller's working directory. The factory has no concept of a configurable output path.

**Fix approach:** Pass `output_dir` as an optional parameter to `crawl_and_save_all`. Default to existing behavior (`data/json/`) for backward compatibility.

```python
# In crawl_and_save_all signature:
async def crawl_and_save_all(
    self,
    urls: List[str],
    output_filename: str,
    format_name: str = "default",
    max_concurrent: int = 15,
    retry_failed: bool = False,
    save_interval: int = 50,
    output_dir: Optional[str] = None,   # NEW
):

# Replace line 203:
# OLD: output_path = os.path.join("data", "json", output_filename)
# NEW:
    if output_dir:
        output_path = os.path.join(output_dir, output_filename)
    else:
        output_path = os.path.join("data", "json", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
```

**Update notebook Task 5** — pass `output_dir=str(CONFIG["output_dir"])` to `crawl_and_save_all`.

**Verify:** Notebook writes JSON to `PROJECT_ROOT/data/json/`, not to `notebooks/data/json/`.

---

## Verification Checklist

- [ ] `notebooks/01_data_crawling.ipynb` exists
- [ ] Cell 1 is a pure config cell — all paths use `PROJECT_ROOT`, no hardcoded strings
- [ ] Notebook runs top-to-bottom without error on a fresh kernel
- [ ] Resume works: run once (partial), kill, run again → previously crawled URLs skipped
- [ ] Progress table prints after crawl: split, total, completed, failed, rate
- [ ] Output JSON files appear at `PROJECT_ROOT/data/json/news_data_vifactcheck_{split}.json`
- [ ] Each JSON article has at minimum: `id`, `features.title`, `features.text`, `metadata.source_url`
- [ ] `src/crawler/crawler_factory.py` `crawl_and_save_all` accepts `output_dir` param
- [ ] Old experimental notebooks (`1_crawl_only.ipynb` etc.) are NOT deleted — just superseded

---

## Notes

- `CrawlerFactory` already handles async + `asyncio.Semaphore` — no concurrency changes needed
- `DatasetHandler` loads HuggingFace dataset — requires internet access on first run (cached after)
- Resume state files: `crawling_status_{split}.json` and `failed_urls_{split}.json` under `CONFIG["cache_dir"]`
- The notebook must work on vast.ai GPU instances (Linux, conda env) as well as local MPS/CPU
- `nest_asyncio.apply()` required for Jupyter async `await` in cells
- Do NOT call `crawler_factory.clear_cache()` anywhere in the notebook

---

*Phase 1 Plan — created 2026-05-10*
