# CONCERNS.md — Technical Concerns & Risk Areas

## 🔴 High Priority

### 1. Minimal Test Coverage
- `tests/crawler/test_simple_crawler.py` is **empty**
- No tests for any model code (`COOLANT`, `CLIP`, `SENet`, `ModelFactory`)
- No tests for preprocessing (`text_preprocessing.py`, `image_preprocessing.py`)
- `tests/conftest.py` is empty — no shared fixtures
- Risk: regressions go undetected; crawler and model changes are unverified

### 2. Broken Import Path in `test_pair_extractor.py`
- Line 12: `src / "processing" / "coolant" / "pair_extractor.py"` — hardcoded path uses `processing/` but actual directory is `preprocessing/`
- The test will fail with `FileNotFoundError` unless this is corrected
- This likely means the test suite is currently broken for this file

### 3. `sys.path.insert` Cross-Package Hacks
- `src/preprocessing/text_preprocessing.py` line 21: `sys.path.insert(0, str(Path(__file__).parent.parent))` to import `utils.device`
- Fragile — breaks if file moves; should use proper package imports or `pythonpath` config

## 🟡 Medium Priority

### 4. No Model Training Tests or Smoke Tests
- No test that even instantiates a model — any breaking change to `coolant.py`, `factory.py`, or `config.py` is invisible until training time
- `examples/train_coolant_official.py` is an example script, not a test

### 5. Data Path Management Complexity
- `DATA_ROOT` env var drives all data resolution; if unset, falls back to repo root
- Multiple `.env.*` files for 4 environments (mac, vastai, colab, windows) — easy to misconfigure or forget to export before launching Jupyter
- `migrate_data_root.py` is a one-shot migration script with no undo

### 6. SSL Global Override Side Effect
- `OPENSSL_CONF` is set as a process-level env var in `main.py` — affects ALL subsequent network calls in the process, not just the crawler
- Could cause unexpected SSL behavior if other libs make TLS connections

### 7. Pinned `protobuf==3.20.3`
- Hard-pinned old version to work around MLflow compatibility bug
- Will conflict with any library that requires `protobuf>=4.x` (e.g. newer TensorFlow, gRPC)
- Needs revisiting when upgrading MLflow

### 8. `archive/` and `data_archived_20260607/` in Repo
- Archived code and old cache data committed to repo — adds noise to searches and git history
- Should be moved out or properly documented

## 🟢 Low Priority / Observations

### 9. Two COOLANT Implementations
- `src/models/coolant.py` (primary, modular)
- `src/models/coolant_official.py` (paper-faithful variant, 17.9 KB)
- No clear documentation on which to use when; risk of divergence

### 10. `base.py` in `src/crawler/` is Empty
- `src/crawler/base.py` (0 bytes) — likely an unintentional leftover alongside `base_crawler.py`

### 11. Large Binary Files Expected at Runtime
- `data/json/`, `data/jpg/`, `processed_data/hdf5/`, `training/checkpoints*/` are all git-ignored
- New contributors must download via Google Drive / rclone — no automated setup script beyond `vastai/setup_vastai.sh`

### 12. `failed_urls_test_agent_test.json` Committed to `src/`
- Test artifact (`src/failed_urls_test_agent_test.json`, 277 bytes) committed to `src/` — should be in `tests/` or gitignored

### 13. Roadmap Items Not Tracked
- Future features (Social Media crawling, OCR, GNN, FastAPI) mentioned in `src/OVERVIEW.md` but no issues, milestones, or planning directory to track them
