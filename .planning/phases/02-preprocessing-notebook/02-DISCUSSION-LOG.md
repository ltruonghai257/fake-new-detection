# Phase 2: Preprocessing Notebook - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-10
**Phase:** 2-Preprocessing Notebook
**Areas discussed:** Input data source, HDF5 file layout, Feature representation, Missing image handling, Dataset statistics report, Dependency/install behavior, Artifact naming/schema, Pair generation strategy, Label handling, Caching and overwrite behavior, Batch/device behavior

---

## Input Data Source

| Option | Description | Selected |
|--------|-------------|----------|
| ViFactCheck only | Load only original ViFactCheck dataset. | |
| Phase 1 output + ViFactCheck merged | Merge Phase 1 crawled news with ViFactCheck before preprocessing. | |
| Config-switchable | Config cell has `data_source`: `vifactcheck`, `crawled`, or `merged`. | ✓ |

**User's choice:** Config-switchable.
**Notes:** User clarified that crawled/paired data follows matched/cross-image semantics. Claude decided that ViFactCheck dev/test should be preserved and extra crawled data should be train-only to avoid leakage.

---

## HDF5 File Layout

| Option | Description | Selected |
|--------|-------------|----------|
| Separate files per split | Write `coolant_train.h5`, `coolant_dev.h5`, and `coolant_test.h5`. | ✓ |
| Single file with split groups | Write one combined HDF5 with train/dev/test groups. | |

**User's choice:** Keep per-split HDF5 files.
**Notes:** User asked to inspect `notebooks/data/json/news_data_vifactcheck_dev_labeled.json` for context. Existing artifacts and old notebook also use per-split HDF5 files.

---

## Feature Representation

| Option | Description | Selected |
|--------|-------------|----------|
| Preserve existing representation | Text token embeddings `(N, max_len, 768)` and image features `(N, 2048)`. | ✓ |
| Switch text to pooled `[CLS]` | Text becomes `(N, 768)`. | |
| Store both token embeddings and pooled `[CLS]` | More flexible but larger and more complex. | |

**User's choice:** You decide.
**Notes:** Claude chose to preserve existing representation for compatibility with COOLANT training and the legacy preprocessing notebook.

---

## Text Used Per Pair

| Option | Description | Selected |
|--------|-------------|----------|
| Image caption only | Matches old pair workflow but has less context. | |
| Article title + content | Richer context but longer and less compatible with old caption workflow. | |
| Title + image caption | More context than caption alone while staying short. | ✓ |
| You decide | Claude chooses. | ✓ |

**User's choice:** You decide.
**Notes:** Claude chose title + image caption as the default text input for matched pairs.

---

## Missing Image Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Skip missing-image pairs | Do not create HDF5 rows for pairs whose image cannot be loaded. | ✓ |
| Use zero-vector image features | Keep rows with 2048-dim zero vector. | |
| Use placeholder image | Extract feature from placeholder image. | |

**User's choice:** Skip missing-image pairs.
**Notes:** Missing counts must be reported in preprocessing statistics.

---

## Dataset Statistics Report

| Option | Description | Selected |
|--------|-------------|----------|
| Complete preprocessing report | Raw articles, generated pairs, skipped counts, final rows, labels, feature shapes, file sizes. | ✓ |
| Minimal report | Split sizes, label balance, and missing count only. | |
| You decide | Claude chooses. | |

**User's choice:** Complete preprocessing report.
**Notes:** User delegated save format to Claude. Claude chose display tables plus machine-readable JSON at `processed_data/hdf5/preprocessing_stats.json`.

---

## Dependency/Install Behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Auto-install missing packages | Convenient but mutates the conda env. | |
| Check only, fail with install command | Safer but less convenient. | |
| Config-controlled install | `AUTO_INSTALL_DEPS=False` by default; install only when enabled. | ✓ |

**User's choice:** Config-controlled install.
**Notes:** User explicitly said to use conda. `fake_news` env was found, but `h5py` was missing when checked with `conda run -n fake_news`.

---

## Artifact Naming/Schema

| Option | Description | Selected |
|--------|-------------|----------|
| Keep old COOLANT names | `caption_features`, `image_features`, `article_ids`, optional metadata. | ✓ |
| Use generic names | `text_features`, `image_features`, `labels`, `article_ids`. | |
| Store both aliases | Flexible but redundant/confusing. | |

**User's choice:** You decide.
**Notes:** Claude chose old COOLANT-compatible names. User also delegated output path; Claude chose canonical repo-level `processed_data/hdf5/`.

---

## Pair Generation Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Dynamic negatives | Phase 2 stores matched pairs only; Phase 3 uses `make_detection_batch()` / image roll for fake pairs. | ✓ |
| Precomputed fake pairs | Phase 2 materializes fake pairs and labels in HDF5. | |
| Discuss tradeoff | Compare before deciding. | ✓ |

**User's choice:** Discuss tradeoff, then Dynamic negatives.
**Notes:** User asked to investigate `make_pair_sim` for consistency. Search found existing logic in `src/processing/coolant/training_utils.py` as `make_coolant_pairs()` and `make_detection_batch()`. Existing pipeline dynamically creates fake pairs during training, so Phase 2 should not materialize them.

---

## Label Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Root labeled JSON | Use `notebooks/data/json/news_data_vifactcheck_{split}_labeled.json`. | |
| `labeled_nei_as_real` | Use alternate binary label interpretation. | |
| `labeled_three_class` | Preserve three-class metadata. | |
| Config-selectable variant | `LABEL_VARIANT = root | nei_as_real | three_class`, root default. | ✓ |

**User's choice:** You recommend.
**Notes:** Claude recommended config-selectable labels with root as default because user explicitly referenced the root labeled JSON file.

---

## Caching and Overwrite Behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Skip existing by default | Skip outputs unless `FORCE_REBUILD=True`. | ✓ |
| Overwrite by default | Always rebuild outputs. | |
| Ask interactively | Prompt before overwrite. | |

**User's choice:** Skip existing by default.
**Notes:** Important because preprocessing and feature extraction are expensive.

---

## Batch/Device Behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Auto device + conservative batch | Use `cuda > mps > cpu`, batch size 16/32, clear CUDA cache. | ✓ |
| CUDA only | Fail if CUDA unavailable. | |
| Config explicit device | User sets device in config. | |

**User's choice:** Auto device + conservative batch.
**Notes:** Locked defaults: auto device priority `cuda > mps > cpu`, `BATCH_SIZE=32`, memory cleanup after batches, and `num_workers=0` for HDF5 compatibility.

---

## Claude's Discretion

- Preserve ViFactCheck dev/test and append extra crawled data to train only.
- Preserve existing token embedding representation.
- Use title + image caption as pair text.
- Save stats as JSON in addition to notebook display.
- Keep old COOLANT HDF5 dataset names.
- Use repo-level `processed_data/hdf5/` as canonical output.
- Use config-selectable label variants with root as default.

## Deferred Ideas

None.
