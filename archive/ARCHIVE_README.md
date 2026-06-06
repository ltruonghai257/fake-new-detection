# Archive Documentation

This directory contains archived files that are no longer part of the active codebase but are preserved for reference.

## Main Pipeline (Active - NOT ARCHIVED)

The main feature pipeline is preserved in `../notebooks/pipeline/`:

- **01_data_crawling.ipynb** - Stage 1: Data acquisition from Vietnamese news sources
- **02_preprocessing.ipynb** - Stage 2: Text and image preprocessing (PhoBERT + ResNet)
- **03_coolant_training.ipynb** - Stage 3: COOLANT model training
- **04_mm_vifactcheck_integration.ipynb** - Stage 4: Multimodal ViFactCheck integration
- **pipeline_config.py** - Shared configuration across all pipeline stages

---

## Archived Notebooks

### `notebooks/experimental/` (15 items)
Experimental notebooks that were used during development but superseded by the main pipeline:

| File | Purpose | Reason for Archive |
|------|---------|-------------------|
| `1_crawl_only_EXPERIMENTAL_broken_resume.ipynb` | Early crawler experiment | Superseded by 01_data_crawling.ipynb |
| `2_preprocess_only.ipynb` | Standalone preprocessing | Integrated into 02_preprocessing.ipynb |
| `3_load_dataset_and_train.ipynb` | Dataset loading + training | Split into separate pipeline stages |
| `4_train_model.ipynb` | Model training (legacy) | Superseded by 03_coolant_training.ipynb |
| `test_crawler.ipynb` | Crawler unit tests | Moved to tests/ directory |
| `test_deep.ipynb` | Deep crawling tests | Experimental, not in main pipeline |
| `test_vnexpress_crawler.ipynb` | VnExpress-specific tests | Domain-specific testing |

**Research folder:** Contains exploratory analysis notebooks.

### `notebooks/legacy_workflows/` (7 items)
Older workflow attempts that were consolidated into the main pipeline:

| File/Folder | Purpose | Status |
|-------------|---------|--------|
| `crawl_and_preprocess.ipynb` | Combined crawl+preprocess | Split into separate stages |
| `crawl_and_preprocess_final.ipynb` | "Final" version attempt | Superseded by pipeline |
| `crawl_and_preprocess_final copy.ipynb` | Backup copy | Duplicate |
| `crawl_and_preprocess_fixed.ipynb` | Bug fixes version | Integrated into main |
| `crawl_and_preprocess_simple.ipynb` | Simplified version | Integrated into main |
| `workflow/` folder | Workflow experiments | Superseded |
| `workflow_coolant/` folder | COOLANT-specific workflows | Integrated into 03_coolant_training.ipynb |
| `workflow_coolant_adabelief/` folder | AdaBelief optimizer experiments | Alternative optimizer testing |

### `notebooks/dataset_crawling/` (2 items)
Alternative dataset crawling approaches:

| File | Purpose |
|------|---------|
| `crawl_data.ipynb` | General data crawling |
| `crawl_vifactcheck.py` | ViFactCheck-specific crawler script |

### `notebooks/training_variants/` (4 items)
Different training approaches and model variants:

| File | Purpose |
|------|---------|
| `model.ipynb` | Minimal model notebook (244 bytes - nearly empty) |
| `train_phased_coolant.ipynb` | Phased training approach |
| `train_vietnamese_coolant.ipynb` | Vietnamese-specific training |

---

## Archived Scripts

### `scripts/data_conversion/` (3 items)

| File | Purpose |
|------|---------|
| `convert_npz_to_hdf5.py` | Convert NPZ to HDF5 format (also in legacy_workflows) |
| `recover_labels.py` | Label recovery utility |

### `scripts/preprocessing_init/` (2 items)

| File | Purpose |
|------|---------|
| `preprocess_vietnamese_data.py` | Quick-start preprocessing script |
| `init_vietnamese_preprocessing.py` | Vietnamese text preprocessing initializer |

### `scripts/testing/` (5 items)

| File | Purpose |
|------|---------|
| `test_multimodal_vietnamese.py` | Multimodal preprocessing tests |
| `test_existing_splits.py` | Test existing dataset splits |
| `test_src_preprocessing.py` | Source preprocessing tests |
| `check_json.py` | JSON structure checker (hardcoded path) |
| `verify_fix.py` | Verification script for fixes |

### `scripts/visualization/` (1 item)

| File | Purpose |
|------|---------|
| `visualize_coolant.py` | COOLANT training visualization |

### `scripts/notebook_fixes/` (2 items)

| File | Purpose |
|------|---------|
| `fix_train_notebook.py` | Script to regenerate 4_train_model.ipynb |
| `gen_train_notebook.py` | Training notebook generator |

---

## Archived Tests

### `tests/` (5 items)

| File | Original Location | Purpose |
|------|-------------------|---------|
| `test_crawler.ipynb` | src/ | Crawler testing notebook |
| `test_crawler.py` | src/ | Crawler test script |
| `agent_test.py` | src/ | Agent testing |
| `test_scripts.py` | src/ | Script testing |
| `test_pkl.ipynb` | root/ | Pickle testing |

---

## Other Archived Items

### `mlruns/` (1 item)
MLflow experiment logs and tracking data.

---

## Summary Statistics

| Category | Items Archived |
|----------|---------------|
| Experimental Notebooks | 15 |
| Legacy Workflows | 7 folders + notebooks |
| Dataset Crawling | 2 |
| Training Variants | 4 |
| Data Conversion Scripts | 3 |
| Preprocessing Scripts | 2 |
| Testing Scripts | 5 |
| Visualization Scripts | 1 |
| Notebook Fix Scripts | 2 |
| Test Files | 5 |
| MLflow Logs | 1 folder |
| **TOTAL** | **~47 items** |

---

## Active Codebase (Post-Cleanup)

### Preserved Notebooks
- `notebooks/pipeline/` - 4 main pipeline notebooks + config (6 items)

### Preserved Root Files
- `config.json` - Configuration
- `environment.yml` - Conda environment
- `requirements.txt` - Python dependencies
- `pytest.ini` - Test configuration
- `README_SIMPLE_MODULES.md` - Simple modules documentation
- `README_preprocessing.md` - Preprocessing documentation
- `news_data.json` - Main dataset
- `crawling_status.json` - Crawling progress tracking
- `GEMINI.md` - Gemini documentation
- `thesis_plan.md` - Thesis planning
- `check_json.py` → **DELETED** (hardcoded path)
- `convert_npz_to_hdf5.py` → **ARCHIVED** (duplicate)
- `preprocess_vietnamese_data.py` → **ARCHIVED** (alternative)
- `init_vietnamese_preprocessing.py` → **ARCHIVED** (initializer)
- `fix_train_notebook.py` → **ARCHIVED** (fix script)
- `test_*.py` → **ARCHIVED** (all test scripts)
- `verify_fix.py` → **ARCHIVED** (verification)
- `visualize_coolant.py` → **ARCHIVED** (visualization)

### Preserved Directories
- `src/` - Source code (84 items)
- `tests/` - Test suite (8 items)
- `examples/` - Example scripts (2 items)
- `docs/` - Documentation (5 items)
- `diagrams/` - Diagrams (1 item)
- `vastai/` - VastAI deployment (6 items)

---

## Cleanup Date
Archive created: 2024

## Note
These files are preserved for reference but are not maintained. The active development should focus on the `notebooks/pipeline/` directory.
