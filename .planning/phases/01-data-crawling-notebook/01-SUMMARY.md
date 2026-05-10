# Phase 1 Summary: Data Crawling Notebook

**Phase:** 1 — Data Crawling Notebook  
**Completed:** 2026-05-10  
**Status:** Implementation complete, pending verification

---

## Accomplishments

### Notebook Created
- **File:** `notebooks/pipeline/01_data_crawling.ipynb`
- **Features:**
  - Single config cell with all parameters (dataset name, splits, output paths, crawl settings)
  - Config-driven paths using `PROJECT_ROOT` — no hardcoded absolute paths
  - Dependency installation cell (auto-installs missing packages)
  - Setup cell creates output/cache directories automatically
  - URL loading from ViFactCheck HuggingFace dataset with per-split counts
  - Resume state display (shows completed/failed/remaining per split)
  - Async crawl execution with progress bars via `tqdm`
  - Results summary table (split / total / completed / failed / success rate)
  - Output file verification (article counts + file sizes)
  - **Critical fix:** Never calls `clear_cache()` — resume works on re-run
  - Uses `ml_training` output format (richer fields for ML downstream)

### Source Code Fixed
- **File:** `src/crawler/crawler_factory.py`
- **Change:** Added optional `output_dir` parameter to `crawl_and_save_all()`
- **Impact:** Notebook can now control output path via config instead of hardcoded `data/json/`

### Project Organization
- **Folder created:** `notebooks/pipeline/` — official pipeline notebooks live here
- **Folder created:** `notebooks/all_stage_final/` — old experimental notebooks archived
- **Old notebooks moved:** 14 experimental notebooks + 3 workflow folders + scripts moved to `all_stage_final/`
- **ROADMAP.md updated:** All notebook paths now reference `notebooks/pipeline/`

---

## Files Modified

| File | Change |
|------|--------|
| `notebooks/pipeline/01_data_crawling.ipynb` | Created |
| `src/crawler/crawler_factory.py` | Added `output_dir` parameter to `crawl_and_save_all()` |
| `.planning/ROADMAP.md` | Updated notebook paths to `notebooks/pipeline/` |

---

## Files Moved (Archived)

| From | To |
|------|-----|
| `notebooks/1_crawl_only.ipynb` | `notebooks/all_stage_final/1_crawl_only_EXPERIMENTAL_broken_resume.ipynb` |
| `notebooks/crawl_and_preprocess*.ipynb` (7 files) | `notebooks/all_stage_final/` |
| `notebooks/test_*.ipynb` (3 files) | `notebooks/all_stage_final/` |
| `notebooks/train_*.ipynb` (2 files) | `notebooks/all_stage_final/` |
| `notebooks/workflow/` | `notebooks/all_stage_final/workflow/` |
| `notebooks/workflow_coolant/` | `notebooks/all_stage_final/workflow_coolant/` |
| `notebooks/workflow_coolant_adabelief/` | `notebooks/all_stage_final/workflow_coolant_adabelief/` |
| `notebooks/research/` | `notebooks/all_stage_final/research/` |
| `notebooks/*.py` (3 scripts) | `notebooks/all_stage_final/` |

---

## Verification Checklist

- [ ] Notebook opens without error
- [ ] Config cell sets `PROJECT_ROOT` correctly
- [ ] Dependency installation cell succeeds
- [ ] Setup cell creates directories and imports modules successfully
- [ ] URL loading cell displays ViFactCheck dataset counts
- [ ] Resume state cell shows cache status (may be empty on first run)
- [ ] Crawl execution cell accepts `output_dir` parameter
- [ ] Output JSON files write to `PROJECT_ROOT/data/json/` (not `notebooks/data/json/`)
- [ ] Results summary table prints correctly
- [ ] Notebook can be re-run without clearing cache (resume works)

---

## Known Issues / Pending

- **FileHandler.write() verification needed:** Line 217 in `crawler_factory.py` calls `file_handler.write()` which may still ignore the `output_dir` parameter and use its own hardcoded path. This should be verified during UAT.
- **No actual crawl performed yet:** Notebook structure is complete but hasn't been tested with a real crawl. UAT should include a small crawl test (e.g., `url_limit: 10`) to verify end-to-end.
- **Git commit pending:** Changes not yet committed.

---

## Next Steps

1. Run UAT to verify notebook works end-to-end
2. Commit changes (notebook + crawler_factory.py + ROADMAP.md)
3. Update STATE.md to mark Phase 1 complete
4. Begin Phase 2: Preprocessing Notebook

---

*Phase 1 Summary — created 2026-05-10*
