# Phase 2 Pattern Map: Preprocessing Notebook

**Phase:** 2 — Preprocessing Notebook  
**Created:** 2026-05-10  
**Status:** Complete

---

## Files to Create or Modify

| File | Role | Closest Existing Analog | Key Pattern |
|------|------|-------------------------|-------------|
| `notebooks/pipeline/02_preprocessing.ipynb` | User-facing preprocessing workflow | `notebooks/pipeline/01_data_crawling.ipynb`; `notebooks/all_stage_final/workflow_coolant/1_preprocess.ipynb` | Single config cell, relative paths, clear markdown sections, per-split HDF5 output. |
| `src/processing/coolant/pair_extractor.py` | Reusable pair extraction helper | Existing same file | Preserve extraction API; add optional stats/metadata support without breaking legacy calls. |
| `tests/processing/coolant/test_pair_extractor.py` | Regression tests for extractor changes | `tests/crawler/test_simple_crawler.py`; `tests/helpers/*` | Small fixture-driven tests; no external model downloads. |

---

## Existing Code Patterns

### Pair Extraction Pattern

Source: `src/processing/coolant/pair_extractor.py`

- Iterates over `article.get("images", [])`.
- Reads `caption` and `folder_path` from each image.
- Resolves paths against `jpg_base_dir`, with fallback strategies for `jpg/` prefixes.
- Cleans captions with `clean_caption()`.
- Skips missing image files, default captions, credit-only captions, and too-short captions.
- Produces dicts currently containing `image_path`, `caption`, `article_idx`, and `folder_path`.

Phase 2 should extend each pair with:

- `pair_text`: `title + " " + caption`
- `title`
- `source_url`
- `source_label`

The extension should be opt-in or backward-compatible so old notebooks that expect only the legacy keys still work.

### Text Feature Pattern

Source: `src/preprocessing/text_preprocessing.py`

- `TextPreprocessor.extract_token_embeddings(texts)` returns token embeddings shaped `(batch_size, seq_len, hidden_size)`.
- `TextPreprocessor.tokenize_text()` applies cleaning, optional Vietnamese word segmentation, max-length padding, and truncation.

Phase 2 concrete values:

- `model_name="vinai/phobert-base-v2"`
- `max_length=128`
- `language="vi"`
- `use_word_segmentation=True`

### Image Feature Pattern

Source: `src/preprocessing/image_preprocessing.py`

- `IMAGE_MODEL_REGISTRY["resnet50"]` has `feature_dim=2048` and `image_size=224`.
- `ImagePreprocessor(model_name="resnet50")` removes the ResNet classifier and returns final features.

Phase 2 concrete value:

- `model_name="resnet50"`

### HDF5 Writer Pattern

Source: `notebooks/all_stage_final/workflow_coolant/1_preprocess.ipynb`

Required datasets and attributes:

- `caption_features`
- `image_features`
- `article_ids`
- `attrs["n_samples"]`
- `attrs["caption_shape"]`
- `attrs["image_shape"]`

Use gzip compression level 4 and chunk size `min(64, total)`.

### HDF5 Reader Pattern

Source: `src/processing/coolant/pair_dataset.py`

`CoolantPairDataset` reads:

- `caption_features[idx]`
- `image_features[idx]`
- `article_ids[idx]` if present

It transposes captions from `[seq_len, 768]` to `[768, seq_len]`. The notebook must write captions in `[seq_len, 768]` layout.

### Dynamic Negative Pattern

Source: `src/processing/coolant/training_utils.py`

`make_detection_batch(caption, image, shift=3)` creates labels dynamically:

- matched rows label `0`
- rolled image rows label `1`

Phase 2 HDF5 should store matched rows only.

---

## Execution Invariants

- Use `PROJECT_ROOT` and config-derived paths only.
- Never write absolute machine-specific paths into the notebook.
- Default `AUTO_INSTALL_DEPS=False`.
- Default `FORCE_REBUILD=False`.
- Preserve ViFactCheck dev/test splits in `merged` mode; extra crawled data goes to train only.
- Use `num_workers=0` for HDF5 loading.
- Do not depend on model training labels in Phase 2 HDF5.

---

## PATTERN MAPPING COMPLETE
