# Phase 2 Research: Preprocessing Notebook

**Phase:** 2 — Preprocessing Notebook  
**Researched:** 2026-05-10  
**Status:** Complete

---

## Research Question

What do we need to know to plan `notebooks/pipeline/02_preprocessing.ipynb` well?

---

## Phase Scope

Phase 2 builds a reproducible notebook that converts ViFactCheck/crawled Vietnamese news JSON into COOLANT-compatible HDF5 feature files:

- **Input:** ViFactCheck JSON splits and optionally Phase 1 crawled JSON.
- **Pair unit:** one valid matched article/image-caption pair per HDF5 row.
- **Text feature:** PhoBERT-base-v2 token embeddings shaped `(N, 128, 768)`.
- **Image feature:** ResNet50 final features shaped `(N, 2048)`.
- **Output:** `processed_data/hdf5/coolant_train.h5`, `coolant_dev.h5`, `coolant_test.h5`, plus `preprocessing_stats.json`.

Out of scope: Stage 1 training, static unmatched/fake pair materialization, COOLANT architecture changes, and Stage 2 MM-ViFactCheck integration.

---

## Existing Assets to Reuse

### Pair Extraction

`src/processing/coolant/pair_extractor.py` already provides:

- `PairExtractor.extract_from_json(json_path)`
- `PairExtractor.extract_all_splits(json_dir, splits)`
- `PairExtractor.save_pairs(pairs, output_path)`
- `PairExtractor.load_pairs(json_path)`
- `clean_caption(text)`
- `is_credit_only(text)`

Current behavior extracts valid image-caption pairs, skips missing images and invalid captions, and prints skip counts. The plan should preserve this behavior and extend only what Phase 2 needs: source label metadata, source URL metadata, title + caption pair text, and machine-readable skip statistics.

### Feature Extraction

`src/preprocessing/text_preprocessing.py` provides `TextPreprocessor.extract_token_embeddings(texts)`, which returns token-level embeddings shaped `(batch, seq_len, hidden_size)`. This matches the legacy COOLANT HDF5 representation and should be used instead of pooled `[CLS]` features.

`src/preprocessing/image_preprocessing.py` provides `ImagePreprocessor(model_name="resnet50")`, where the registry defines ResNet50 as `feature_dim=2048` and `image_size=224`.

`src/preprocessing/combined_preprocessing.py` initializes both preprocessors but defaults are not aligned with Phase 2: `text_model_name="vinai/phobert-base"`, `image_model_name="resnet18"`, and `max_text_length=512`. The notebook must explicitly configure `vinai/phobert-base-v2`, `resnet50`, and `max_length=128`.

### COOLANT HDF5 Consumer

`src/processing/coolant/pair_dataset.py::CoolantPairDataset` expects:

- `caption_features`
- `image_features`
- optional `article_ids`

It transposes captions from `[seq_len, 768]` to `[768, seq_len]` for Conv1d. It does not read labels because matched/unmatched labels are created dynamically during training.

`src/processing/coolant/training_utils.py::make_detection_batch()` dynamically creates fake/unmatched examples by rolling images in a batch and emits labels `0=Real/matched`, `1=Fake/unmatched`. Phase 2 must not precompute fake pairs.

### Legacy Notebook Schema

`notebooks/all_stage_final/workflow_coolant/1_preprocess.ipynb` writes per-split HDF5 files with:

- `caption_features` using gzip compression level 4
- `image_features` using gzip compression level 4
- `article_ids`
- attributes `n_samples`, `caption_shape`, `image_shape`

Observed legacy output shapes:

- train caption `(3537, 128, 768)`, image `(3537, 2048)`
- dev caption `(1054, 128, 768)`, image `(1054, 2048)`
- test caption `(876, 128, 768)`, image `(876, 2048)`

---

## Implementation Strategy

1. Create `notebooks/pipeline/02_preprocessing.ipynb` with a single top config cell.
2. Add small backward-compatible enhancements to `src/processing/coolant/pair_extractor.py` so the notebook can collect title+caption pair text and stats without duplicating extraction logic.
3. Use explicit preprocessors in the notebook:
   - `TextPreprocessor(model_name="vinai/phobert-base-v2", max_length=128, language="vi")`
   - `ImagePreprocessor(model_name="resnet50")`
4. Save separate HDF5 files per split with the existing COOLANT dataset names.
5. Display and save split statistics including raw articles, generated pairs, skipped counts, row counts, label counts, feature shapes, and file sizes.
6. Verify output by opening each HDF5 with `h5py` and instantiating `CoolantPairDataset`.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Missing `h5py` in `fake_news` env | Add dependency check cell; default `AUTO_INSTALL_DEPS=False`; print exact install commands and stop clearly. |
| Existing `PairExtractor` only prints skip stats | Extend it to optionally return stats while preserving default return type. |
| Label variants may live in different folders | Implement a config-driven resolver for `root`, `nei_as_real`, and `three_class` paths. |
| Phase 1 output path may differ from historical `notebooks/data/json` | Keep paths configurable: `VIFACTCHECK_JSON_DIR`, `CRAWLED_JSON_DIR`, `JPG_DIR`. |
| Large feature arrays can exhaust memory | Process per split in batches, clear Python/CUDA memory after each batch, and skip existing output unless `FORCE_REBUILD=True`. |
| HDF5 worker/file handle issues | Use `num_workers=0` in verification and document it in notebook output. |

---

## Validation Architecture

### Static Checks

- Notebook exists at `notebooks/pipeline/02_preprocessing.ipynb`.
- Notebook contains one top config cell with `DATA_SOURCE`, `LABEL_VARIANT`, `OUTPUT_DIR`, `FORCE_REBUILD`, `AUTO_INSTALL_DEPS`, `TEXT_MODEL_NAME`, `IMAGE_MODEL_NAME`, `MAX_LENGTH`, and `BATCH_SIZE`.
- No hardcoded absolute paths appear in the notebook.
- `PairExtractor.extract_from_json` remains backward compatible when called with only `json_path`.

### Lightweight Runtime Checks

Use small config values to avoid full expensive preprocessing during verification:

- `SPLITS = ["dev"]`
- `MAX_PAIRS_PER_SPLIT = 8`
- `FORCE_REBUILD = True`
- `OUTPUT_DIR = PROJECT_ROOT / "processed_data" / "hdf5" / "smoke_test"`

Expected smoke-test outputs:

- one HDF5 file exists
- contains `caption_features`, `image_features`, and `article_ids`
- `caption_features.shape[1:] == (128, 768)`
- `image_features.shape[1:] == (2048,)`
- `CoolantPairDataset` can load the file with `num_workers=0`

### Full Acceptance Checks

- `processed_data/hdf5/coolant_train.h5`, `coolant_dev.h5`, and `coolant_test.h5` are produced for full runs.
- `processed_data/hdf5/preprocessing_stats.json` contains entries for `train`, `dev`, and `test`.
- Stats report includes raw article counts, valid pair counts, missing-image skips, invalid-caption skips, source label counts, HDF5 row counts, feature shapes, and file sizes.

---

## RESEARCH COMPLETE
