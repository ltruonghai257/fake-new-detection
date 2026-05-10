---
status: testing
phase: 01-data-crawling-notebook
source: [01-SUMMARY.md]
started: 2026-05-10T23:06:00+07:00
updated: 2026-05-10T23:06:00+07:00
---

## Current Test
<!-- OVERWRITE each test - shows where we are -->

number: 1
name: Notebook File Location
expected: |
  Open the project root and navigate to `notebooks/pipeline/`. The file `01_data_crawling.ipynb` should exist here. The `notebooks/` root should contain only the `pipeline/` folder (for official notebooks), `all_stage_final/` (archive), and runtime data directories (data/, logs/, mlruns/, etc.). No old experimental notebooks should remain in the root.
awaiting: user response

## Tests

### 1. Notebook File Location
expected: Open the project root and navigate to `notebooks/pipeline/`. The file `01_data_crawling.ipynb` should exist here. The `notebooks/` root should contain only the `pipeline/` folder (for official notebooks), `all_stage_final/` (archive), and runtime data directories (data/, logs/, mlruns/, etc.). No old experimental notebooks should remain in the root.
result: pending

### 2. Config Cell Structure
expected: Open `notebooks/pipeline/01_data_crawling.ipynb` in Jupyter. Cell 1 should be a pure config cell with a CONFIG dictionary containing: dataset_name, url_column, splits, url_limit, output_dir, cache_dir, output_format, max_concurrent, save_interval, retry_failed. All paths should use `PROJECT_ROOT` (a Path object), not hardcoded absolute strings.
result: pending

### 3. Dependency Installation
expected: Run Cell 2 (dependency installation). It should check for packages (loguru, tqdm, beautifulsoup4, lxml, httpx, nest-asyncio, datasets, Pillow) and install any missing ones. Output should show "Dependencies ready." without errors.
result: pending

### 4. Setup and Imports
expected: Run Cell 3 (setup). It should apply nest_asyncio, add project root to sys.path, create output and cache directories, and import CrawlerFactory and DatasetHandler. Output should print project root, output dir, and cache dir paths without ImportError.
result: pending

### 5. URL Loading
expected: Run Cell 5 (URL loading). It should load URLs from the ViFactCheck HuggingFace dataset for each split (train, dev, test) and print the count per split. Total URLs should be > 0. No authentication or network errors should occur.
result: pending

### 6. Resume State Display
expected: Run Cell 7 (resume status). It should check for cache files and print per-split status showing completed/failed/remaining counts. On first run, all should show 0 done / 0 failed / N remaining. The cell should NOT clear any cache files.
result: pending

### 7. Crawl Execution with output_dir
expected: Inspect Cell 9 (crawl_split function). The call to `factory.crawl_and_save_all()` should include `output_dir=str(CONFIG["output_dir"])` as a parameter. This ensures the notebook passes the configured output path to the factory instead of relying on the hardcoded default.
result: pending

### 8. Output Path Verification (source code)
expected: Open `src/crawler/crawler_factory.py` and find line 204-208. The code should check `if output_dir:` and use `os.path.join(output_dir, output_filename)` when provided, otherwise fall back to `"data/json"`. The `os.makedirs(os.path.dirname(output_path), exist_ok=True)` line should follow.
result: pending

### 9. FileHandler.write Path Check
expected: In `src/crawler/crawler_factory.py` line 217, `file_handler.write(format_name="json", data=existing_data, file_name=output_filename)` is called. Verify that FileHandler.write() respects the output_path variable set earlier in the function, or that it accepts an output_dir parameter. If FileHandler ignores output_path and uses its own hardcoded path, this is a bug.
result: pending

### 10. Small End-to-End Test
expected: Set `url_limit: 10` in the config cell for testing. Run the crawl cells (9 and 10) with this limit. Verify that output JSON appears at `PROJECT_ROOT/data/json/news_data_vifactcheck_{split}.json` (not in `notebooks/data/json/`). Check that the JSON contains articles with at least `id`, `features.title`, `features.text`, and `metadata.source_url` fields.
result: pending

### 11. Resume Functionality
expected: After the small crawl completes, run Cell 7 (resume status) again. It should show completed URLs matching the crawl count. Re-run Cell 10 (crawl execution) — it should skip the already-crawled URLs and complete immediately without re-fetching. The cache file should NOT be cleared.
result: pending

### 12. Results Summary
expected: Run Cell 12 (results summary). It should print a formatted table showing split, total, completed, failed, and success rate. After the small crawl, success rate should be > 0%.
result: pending

### 13. Old Notebooks Archived
expected: Navigate to `notebooks/all_stage_final/`. It should contain the old experimental notebooks (1_crawl_only_EXPERIMENTAL_broken_resume.ipynb, crawl_and_preprocess*.ipynb, test_*.ipynb, train_*.ipynb) and workflow folders. The `notebooks/` root should NOT contain any of these files directly.
result: pending

## Summary

total: 13
passed: 0
issues: 0
pending: 13
skipped: 0

## Gaps

[none yet]
