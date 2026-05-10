---
phase: 2
plan: 02-PLAN
type: execute
wave: 1
depends_on:
  - Phase 1: Data Crawling Notebook
files_modified:
  - notebooks/pipeline/02_preprocessing.ipynb
  - src/processing/coolant/pair_extractor.py
  - tests/processing/coolant/test_pair_extractor.py
autonomous: true
requirements:
  - PREP-01
  - PREP-02
  - PREP-03
  - PREP-04
  - PREP-05
  - NB-01
  - NB-02
  - NB-03
---

# Phase 2 Plan: Preprocessing Notebook

**Phase:** 2 — Preprocessing Notebook  
**Goal:** Build `notebooks/pipeline/02_preprocessing.ipynb`, a unified preprocessing notebook that converts raw ViFactCheck/crawled JSON into COOLANT-compatible HDF5 with PhoBERT-base-v2 text token embeddings, ResNet50 image features, train/dev/test splits, and dataset statistics.  
**Requirements:** PREP-01, PREP-02, PREP-03, PREP-04, PREP-05, NB-01, NB-02, NB-03  
**Planned:** 2026-05-10  
**Status:** Ready to execute

---

<objective>
Create a reproducible Phase 2 preprocessing workflow that reuses existing `src/` utilities, preserves COOLANT HDF5 compatibility, skips invalid pairs, and produces verifiable dataset statistics for Phase 3 training.
</objective>

---

## Source Truths and Must-Haves

<must_haves>

### Roadmap and Requirement Truths

- **PREP-01:** Notebook loads ViFactCheck JSON into a unified sample format with train/dev/test splits.
- **PREP-02:** Text preprocessing uses Vietnamese normalization and PhoBERT-base-v2 tokenization/embedding.
- **PREP-03:** Image preprocessing uses ResNet50 feature extraction with 2048-dimensional outputs.
- **PREP-04:** Text and image features are saved to HDF5 for efficient DataLoader access.
- **PREP-05:** Notebook reports class balance, split sizes, and missing-image count.
- **NB-01:** Notebook has one top config cell for tunable parameters and paths.
- **NB-02:** Notebook uses relative/config-driven paths only.
- **NB-03:** Notebook has clear markdown section headers.

### CONTEXT.md Decision Coverage

- **D-01, D-02, D-03, D-04:** `DATA_SOURCE` supports `vifactcheck`, `crawled`, and `merged`; default root labeled JSON paths are `notebooks/data/json/news_data_vifactcheck_{split}_labeled.json`; `merged` preserves ViFactCheck dev/test splits and adds extra crawled data to train only.
- **D-05, D-06, D-07, D-08:** Phase 2 stores valid matched image-text pairs only; fake/unmatched pairs are not materialized and are created dynamically in Phase 3 via `make_detection_batch()` / image rolling.
- **D-09, D-10, D-11:** Pair text is `title + image caption`; full article content is not used by default; default `MAX_LENGTH=128`.
- **D-12, D-13, D-14:** `LABEL_VARIANT` supports `root`, `nei_as_real`, and `three_class`; source labels are metadata/statistics, not COOLANT dynamic labels.
- **D-15, D-16, D-17, D-18, D-19:** Write separate repo-level `processed_data/hdf5/coolant_train.h5`, `coolant_dev.h5`, `coolant_test.h5` files with `caption_features`, `image_features`, and `article_ids`.
- **D-20, D-21, D-22, D-23:** Text features are PhoBERT token embeddings `(N, 128, 768)` and image features are ResNet50 `(N, 2048)`; do not switch to pooled `[CLS]`.
- **D-24, D-25, D-26:** Missing/broken images and invalid captions are skipped and counted; reuse `PairExtractor` caption filtering.
- **D-27, D-28:** Stats include raw articles, generated pairs, skipped counts, final HDF5 rows, label counts, feature shapes, file sizes, and are saved to `processed_data/hdf5/preprocessing_stats.json`.
- **D-29, D-30, D-31, D-32:** Dependency checks target conda env `fake_news`; the notebook checks for missing `h5py`; `AUTO_INSTALL_DEPS=False` by default; missing deps print exact install commands and stop clearly.
- **D-33, D-34:** Existing outputs are skipped unless `FORCE_REBUILD=True`.
- **D-35, D-36, D-37, D-38:** Device auto-detection priority is `cuda > mps > cpu`; default `BATCH_SIZE=32`; clear CUDA memory after batches; use `num_workers=0` for HDF5 verification.

</must_haves>

---

<threat_model>

## Security and Data-Safety Considerations

- **External dependency installation:** The notebook must not mutate the environment unless `AUTO_INSTALL_DEPS=True`. Default behavior is fail-fast with explicit `conda install -n fake_news h5py -c conda-forge` and equivalent pip instructions.
- **Path safety:** The notebook must not contain hardcoded absolute local paths. All file operations must resolve under `PROJECT_ROOT` or user-configured relative directories.
- **Data leakage:** In `merged` mode, preserve ViFactCheck dev/test splits and append extra crawled data only to train.
- **Accidental expensive overwrite:** Existing HDF5 outputs must be skipped by default unless `FORCE_REBUILD=True`.

</threat_model>

---

<tasks>

## Task 1 — Add pair extraction metadata and stats support

**Type:** tdd  
**Files:** `src/processing/coolant/pair_extractor.py`, `tests/processing/coolant/test_pair_extractor.py`  
**Requirements:** PREP-01, PREP-05

<read_first>

- `.planning/phases/02-preprocessing-notebook/02-CONTEXT.md`
- `.planning/phases/02-preprocessing-notebook/02-RESEARCH.md`
- `.planning/phases/02-preprocessing-notebook/02-PATTERNS.md`
- `src/processing/coolant/pair_extractor.py`
- `src/processing/coolant/pair_dataset.py`
- `tests/crawler/test_simple_crawler.py`
- `tests/helpers/test_json_helper.py`

</read_first>

<action>

1. Create `tests/processing/coolant/test_pair_extractor.py` with fixture JSON and tiny local image files.
2. Add tests proving `PairExtractor.extract_from_json(str(json_path))` remains backward compatible and returns a `list` of pair dicts.
3. Add tests for an opt-in stats mode, for example `PairExtractor.extract_from_json(str(json_path), return_stats=True)`, returning `(pairs, stats)`.
4. Update `src/processing/coolant/pair_extractor.py` minimally so extracted pairs include these additional keys when source data exists:
   - `pair_text`: title plus cleaned caption, separated by one space
   - `title`
   - `source_url`
   - `source_label`
5. Add stats fields needed by the notebook:
   - `raw_articles`
   - `total_images`
   - `valid_pairs`
   - `skipped.no_caption`
   - `skipped.credit_only`
   - `skipped.too_short`
   - `skipped.no_image`
   - `source_label_counts`
6. Preserve `clean_caption()`, `is_credit_only()`, `save_pairs()`, and `load_pairs()` behavior.

</action>

<acceptance_criteria>

- `src/processing/coolant/pair_extractor.py` contains `return_stats` in the `extract_from_json` signature.
- `src/processing/coolant/pair_extractor.py` contains the string `pair_text`.
- `src/processing/coolant/pair_extractor.py` contains the string `source_label_counts`.
- `tests/processing/coolant/test_pair_extractor.py` contains `test_extract_from_json_backward_compatible`.
- `tests/processing/coolant/test_pair_extractor.py` contains `test_extract_from_json_return_stats`.
- `rtk pytest tests/processing/coolant/test_pair_extractor.py` exits 0.

</acceptance_criteria>

---

## Task 2 — Create the preprocessing notebook skeleton and dependency gate

**Type:** execute  
**Files:** `notebooks/pipeline/02_preprocessing.ipynb`  
**Requirements:** NB-01, NB-02, NB-03

<read_first>

- `.planning/phases/02-preprocessing-notebook/02-CONTEXT.md`
- `.planning/phases/01-data-crawling-notebook/01-SUMMARY.md`
- `notebooks/pipeline/01_data_crawling.ipynb`
- `environment.yml`
- `requirements.txt`

</read_first>

<action>

Create `notebooks/pipeline/02_preprocessing.ipynb` with these sections and cells:

1. Markdown title: `# Preprocessing — ViFactCheck to COOLANT HDF5`.
2. One top config cell containing exactly these required config keys:
   - `PROJECT_ROOT`
   - `DATA_SOURCE = "vifactcheck"`
   - `LABEL_VARIANT = "root"`
   - `SPLITS = ["train", "dev", "test"]`
   - `VIFACTCHECK_JSON_DIR = PROJECT_ROOT / "notebooks" / "data" / "json"`
   - `CRAWLED_JSON_DIR = PROJECT_ROOT / "data" / "json"`
   - `JPG_DIR = PROJECT_ROOT / "notebooks" / "data" / "jpg"`
   - `OUTPUT_DIR = PROJECT_ROOT / "processed_data" / "hdf5"`
   - `PAIRS_CACHE_DIR = OUTPUT_DIR / "pairs"`
   - `AUTO_INSTALL_DEPS = False`
   - `FORCE_REBUILD = False`
   - `TEXT_MODEL_NAME = "vinai/phobert-base-v2"`
   - `IMAGE_MODEL_NAME = "resnet50"`
   - `MAX_LENGTH = 128`
   - `BATCH_SIZE = 32`
   - `MAX_PAIRS_PER_SPLIT = None`
3. Dependency check cell that verifies imports for `h5py`, `torch`, `transformers`, `torchvision`, `PIL`, `numpy`, `pandas`, and `tqdm`.
4. If missing dependencies exist and `AUTO_INSTALL_DEPS=False`, raise a clear `RuntimeError` after printing:
   - `conda install -n fake_news h5py -c conda-forge`
   - `pip install h5py`
5. Setup/import cell that inserts `PROJECT_ROOT` into `sys.path`, creates `OUTPUT_DIR` and `PAIRS_CACHE_DIR`, and imports:
   - `PairExtractor`
   - `TextPreprocessor`
   - `ImagePreprocessor`
   - `CoolantPairDataset`
   - `get_device`

</action>

<acceptance_criteria>

- `notebooks/pipeline/02_preprocessing.ipynb` exists.
- Notebook text contains `DATA_SOURCE = "vifactcheck"`.
- Notebook text contains `LABEL_VARIANT = "root"`.
- Notebook text contains `AUTO_INSTALL_DEPS = False`.
- Notebook text contains `FORCE_REBUILD = False`.
- Notebook text contains `TEXT_MODEL_NAME = "vinai/phobert-base-v2"`.
- Notebook text contains `IMAGE_MODEL_NAME = "resnet50"`.
- Notebook text contains `MAX_LENGTH = 128`.
- Notebook text contains `conda install -n fake_news h5py -c conda-forge`.
- Notebook text does not contain `/Users/haila/`.

</acceptance_criteria>

---

## Task 3 — Implement input resolution, pair extraction, and split preservation

**Type:** execute  
**Files:** `notebooks/pipeline/02_preprocessing.ipynb`  
**Requirements:** PREP-01, PREP-05, NB-02, NB-03

<read_first>

- `notebooks/pipeline/02_preprocessing.ipynb`
- `.planning/phases/02-preprocessing-notebook/02-CONTEXT.md`
- `src/processing/coolant/pair_extractor.py`
- `notebooks/all_stage_final/workflow_coolant/0_extract_pairs.ipynb`

</read_first>

<action>

Add notebook sections and code to resolve source JSON paths and extract matched pairs:

1. Add markdown section `## Step 1: Resolve Input Files`.
2. Implement `resolve_json_path(split, data_source, label_variant)` with exact label variant mapping:
   - `root`: `VIFACTCHECK_JSON_DIR / f"news_data_vifactcheck_{split}_labeled.json"`
   - `nei_as_real`: `VIFACTCHECK_JSON_DIR / "labeled_nei_as_real" / f"news_data_vifactcheck_{split}_labeled.json"`
   - `three_class`: `VIFACTCHECK_JSON_DIR / "labeled_three_class" / f"news_data_vifactcheck_{split}_labeled.json"`
3. Implement `load_articles(path)` and show raw article counts for each split.
4. Add markdown section `## Step 2: Extract Valid Matched Pairs`.
5. Instantiate `PairExtractor(jpg_base_dir=str(JPG_DIR), min_caption_len=5)`.
6. For each split, call the new stats mode: `pairs, pair_stats = extractor.extract_from_json(str(json_path), return_stats=True)`.
7. Use `MAX_PAIRS_PER_SPLIT` to optionally truncate pairs for smoke tests.
8. Cache pair JSON files to `PAIRS_CACHE_DIR / f"pairs_{split}.json"`.
9. In `merged` mode, preserve `dev` and `test` from ViFactCheck only and add crawled/extra data to train only.
10. Build an in-memory `stats` dictionary keyed by split.

</action>

<acceptance_criteria>

- Notebook text contains `def resolve_json_path`.
- Notebook text contains `labeled_nei_as_real`.
- Notebook text contains `labeled_three_class`.
- Notebook text contains `PairExtractor(jpg_base_dir=str(JPG_DIR), min_caption_len=5)`.
- Notebook text contains `return_stats=True`.
- Notebook text contains `MAX_PAIRS_PER_SPLIT`.
- Notebook text contains `pairs_train.json`.
- Notebook text contains `merged` and `train` in the input-resolution logic.

</acceptance_criteria>

---

## Task 4 — Implement PhoBERT + ResNet50 feature extraction and HDF5 writing

**Type:** execute  
**Files:** `notebooks/pipeline/02_preprocessing.ipynb`  
**Requirements:** PREP-02, PREP-03, PREP-04, NB-02, NB-03

<read_first>

- `notebooks/pipeline/02_preprocessing.ipynb`
- `src/preprocessing/text_preprocessing.py`
- `src/preprocessing/image_preprocessing.py`
- `src/processing/coolant/pair_dataset.py`
- `src/processing/coolant/training_utils.py`
- `notebooks/all_stage_final/workflow_coolant/1_preprocess.ipynb`

</read_first>

<action>

Add notebook sections and code for model initialization, feature extraction, and HDF5 writing:

1. Add markdown section `## Step 3: Initialize Feature Extractors`.
2. Select device via `device = get_device()` and print it.
3. Instantiate:
   - `text_preprocessor = TextPreprocessor(model_name=TEXT_MODEL_NAME, max_length=MAX_LENGTH, language="vi", device=str(device), use_word_segmentation=True)`
   - `image_preprocessor = ImagePreprocessor(model_name=IMAGE_MODEL_NAME, device=str(device))`
4. Add markdown section `## Step 4: Extract Features and Save HDF5`.
5. For each split, skip when `OUTPUT_DIR / f"coolant_{split}.h5"` exists and `FORCE_REBUILD=False`.
6. Use pair text with fallback: `pair.get("pair_text") or f"{pair.get('title', '')} {pair.get('caption', '')}".strip()`.
7. Extract text features with `text_preprocessor.extract_token_embeddings(batch_texts)`.
8. Extract image features with `image_preprocessor.extract_features(batch_image_paths)`.
9. After each batch, run `gc.collect()` and, if CUDA is available, `torch.cuda.empty_cache()`.
10. Write HDF5 with datasets:
    - `caption_features`
    - `image_features`
    - `article_ids`
    - optional metadata datasets `source_labels`, `source_urls`, `image_paths`, `folder_paths` when available
11. Write attributes:
    - `n_samples`
    - `caption_shape`
    - `image_shape`
    - `text_model`
    - `image_model`
    - `max_length`
    - `data_source`
    - `label_variant`

</action>

<acceptance_criteria>

- Notebook text contains `TextPreprocessor(model_name=TEXT_MODEL_NAME, max_length=MAX_LENGTH, language="vi"`.
- Notebook text contains `ImagePreprocessor(model_name=IMAGE_MODEL_NAME`.
- Notebook text contains `extract_token_embeddings`.
- Notebook text contains `extract_features`.
- Notebook text contains `coolant_{split}.h5`.
- Notebook text contains `caption_features`.
- Notebook text contains `image_features`.
- Notebook text contains `article_ids`.
- Notebook text contains `torch.cuda.empty_cache()`.
- Notebook text contains `FORCE_REBUILD`.

</acceptance_criteria>

---

## Task 5 — Add statistics report and HDF5 verification cells

**Type:** execute  
**Files:** `notebooks/pipeline/02_preprocessing.ipynb`  
**Requirements:** PREP-04, PREP-05, NB-03

<read_first>

- `notebooks/pipeline/02_preprocessing.ipynb`
- `src/processing/coolant/pair_dataset.py`
- `src/processing/hdf5_dataset.py`
- `.planning/phases/02-preprocessing-notebook/02-CONTEXT.md`

</read_first>

<action>

Add final notebook sections for reporting and verification:

1. Add markdown section `## Step 5: Dataset Statistics Report`.
2. Build a `pandas.DataFrame` with one row per split and columns:
   - `split`
   - `raw_articles`
   - `valid_pairs`
   - `skipped_no_image`
   - `skipped_no_caption`
   - `skipped_credit_only`
   - `skipped_too_short`
   - `hdf5_rows`
   - `source_label_counts`
   - `caption_shape`
   - `image_shape`
   - `file_size_mb`
3. Display the dataframe.
4. Save JSON stats to `OUTPUT_DIR / "preprocessing_stats.json"`.
5. Add markdown section `## Step 6: Verify COOLANT Dataset Loading`.
6. For each output file, open with `h5py.File(path, "r")` and assert required datasets exist.
7. Instantiate `CoolantPairDataset(str(path), device="cpu")` for each split.
8. Print one sample shape per split: caption `[768, 128]`, image `[2048]`, and article id.
9. Print final line `Ready for Phase 3: COOLANT Training Notebook`.

</action>

<acceptance_criteria>

- Notebook text contains `preprocessing_stats.json`.
- Notebook text contains `source_label_counts`.
- Notebook text contains `skipped_no_image`.
- Notebook text contains `CoolantPairDataset(str(path), device="cpu")`.
- Notebook text contains `Ready for Phase 3: COOLANT Training Notebook`.
- Notebook text contains `num_workers=0` or explicitly states HDF5 loading uses `num_workers=0`.

</acceptance_criteria>

---

## Task 6 — Run verification and update planning status artifacts

**Type:** verify  
**Files:** `notebooks/pipeline/02_preprocessing.ipynb`, `src/processing/coolant/pair_extractor.py`, `tests/processing/coolant/test_pair_extractor.py`  
**Requirements:** PREP-01, PREP-02, PREP-03, PREP-04, PREP-05, NB-01, NB-02, NB-03

<read_first>

- `notebooks/pipeline/02_preprocessing.ipynb`
- `.planning/phases/02-preprocessing-notebook/02-PLAN.md`
- `.planning/phases/02-preprocessing-notebook/02-CONTEXT.md`

</read_first>

<action>

Run non-destructive verification that does not require full dataset preprocessing:

1. Run `rtk pytest tests/processing/coolant/test_pair_extractor.py`.
2. Read notebook JSON and verify required strings from Tasks 2–5 are present.
3. Verify `notebooks/pipeline/02_preprocessing.ipynb` does not contain `/Users/haila/` or `G:\`.
4. If dependencies and local data are available, perform a smoke run with `SPLITS=["dev"]`, `MAX_PAIRS_PER_SPLIT=8`, `FORCE_REBUILD=True`, and output under `processed_data/hdf5/smoke_test/`.
5. Record verification outcome in `.planning/phases/02-preprocessing-notebook/02-SUMMARY.md` after execution completes.

</action>

<acceptance_criteria>

- `rtk pytest tests/processing/coolant/test_pair_extractor.py` exits 0.
- Notebook file contains all config keys listed in Task 2.
- Notebook file contains all HDF5 dataset names listed in Task 4.
- Notebook file contains `preprocessing_stats.json`.
- Notebook file contains no hardcoded local absolute path `/Users/haila/`.
- If smoke test is run, smoke HDF5 contains `caption_features`, `image_features`, and `article_ids`.

</acceptance_criteria>

</tasks>

---

<verification>

## Plan-Level Verification

Run these checks after execution:

1. `rtk pytest tests/processing/coolant/test_pair_extractor.py`
2. Search notebook JSON for required config strings:
   - `DATA_SOURCE = "vifactcheck"`
   - `LABEL_VARIANT = "root"`
   - `TEXT_MODEL_NAME = "vinai/phobert-base-v2"`
   - `IMAGE_MODEL_NAME = "resnet50"`
   - `MAX_LENGTH = 128`
3. Search notebook JSON for required output strings:
   - `caption_features`
   - `image_features`
   - `article_ids`
   - `preprocessing_stats.json`
4. Confirm no hardcoded absolute user path appears in the notebook.
5. Optional smoke run on a small dev subset if dependencies and data are present.

</verification>

---

<success_criteria>

- `notebooks/pipeline/02_preprocessing.ipynb` exists and has a single top config cell.
- The notebook supports `DATA_SOURCE` values `vifactcheck`, `crawled`, and `merged`.
- The notebook supports `LABEL_VARIANT` values `root`, `nei_as_real`, and `three_class`.
- Pair extraction reuses `PairExtractor` and records skip statistics.
- The notebook extracts PhoBERT-base-v2 token embeddings shaped `(N, 128, 768)`.
- The notebook extracts ResNet50 image features shaped `(N, 2048)`.
- HDF5 output files use `caption_features`, `image_features`, and `article_ids`.
- `processed_data/hdf5/preprocessing_stats.json` is written by full runs.
- `CoolantPairDataset` can open generated HDF5 files.
- All Phase 2 requirements and notebook quality requirements are covered.

</success_criteria>

---

## Wave Plan

| Wave | Tasks | What it builds |
|------|-------|----------------|
| 1 | Task 1 | Reusable pair extraction metadata and stats foundation |
| 2 | Tasks 2–3 | Notebook skeleton, dependency gate, input resolver, pair extraction |
| 3 | Tasks 4–5 | Feature extraction, HDF5 writer, stats report, dataset verification |
| 4 | Task 6 | Test and verification loop |

---

## Notes for Executor

- Prefix shell commands with `rtk`.
- Do not perform a full expensive preprocessing run unless the user asks or smoke-test inputs are small.
- Prefer notebook creation/editing through Python or notebook tooling rather than hand-editing invalid JSON.
- Keep code changes surgical: only modify `PairExtractor` if needed for metadata/stats and do not refactor unrelated preprocessing modules.
- The plan intentionally does not store fake/unmatched labels in HDF5; Phase 3 uses `make_detection_batch()`.

## PLANNING COMPLETE
