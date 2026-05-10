# Phase 2: Preprocessing Notebook - Context

**Gathered:** 2026-05-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Build `notebooks/pipeline/02_preprocessing.ipynb`, a reproducible preprocessing notebook that converts ViFactCheck/crawled Vietnamese news JSON into COOLANT-compatible HDF5 feature files using PhoBERT text embeddings and ResNet50 image features. The phase includes pair extraction, feature extraction, HDF5 writing, dependency checks, resumable/re-run behavior, and dataset statistics reporting.

Out of scope: changing COOLANT architecture, implementing Phase 3 training, materializing fake/unmatched pairs into HDF5, or redesigning dataset labels beyond selecting the source label variant for metadata/statistics.

</domain>

<decisions>
## Implementation Decisions

### Input Data Source
- **D-01:** The notebook must be config-switchable with `DATA_SOURCE` supporting `vifactcheck`, `crawled`, and `merged`.
- **D-02:** `vifactcheck` mode loads labeled ViFactCheck JSON files from the selected label variant.
- **D-03:** `merged` mode must preserve ViFactCheck dev/test splits. Any extra crawled/extended data should be added to train only to avoid evaluation leakage.
- **D-04:** The user-referenced root labeled JSON files are the default source: `notebooks/data/json/news_data_vifactcheck_{split}_labeled.json`.

### Pair Generation Strategy
- **D-05:** Follow the existing COOLANT approach: Phase 2 stores only valid matched image-text pairs in HDF5.
- **D-06:** Do not materialize fake/unmatched pairs in Phase 2 HDF5.
- **D-07:** Phase 3 training is responsible for dynamic negatives via `src/processing/coolant/training_utils.py::make_detection_batch()` / image rolling.
- **D-08:** Each HDF5 row represents one matched article/image-caption pair. Training labels for real/fake matched/unmatched detection are dynamic, not stored as required Phase 2 labels.

### Text Used Per Pair
- **D-09:** Use `title + image caption` as the text input for each matched pair.
- **D-10:** Do not use full article `content` by default for COOLANT Stage 1 pairs; it is longer and diverges from the existing image-caption workflow.
- **D-11:** Keep `max_length=128` as a reasonable default because the selected text is title + caption, not full article body.

### Label Handling
- **D-12:** Use config-selectable `LABEL_VARIANT` with default `root`.
- **D-13:** Supported label variants should include `root`, `nei_as_real`, and `three_class`, mapped to existing folders under `notebooks/data/json/`.
- **D-14:** Source article labels may be stored as metadata/statistics for inspection, but they are not the COOLANT matched/unmatched labels used by Stage 1 training.

### HDF5 Layout and Schema
- **D-15:** Write separate per-split HDF5 files, not one monolithic HDF5.
- **D-16:** Canonical Phase 2 output directory is repo-level `processed_data/hdf5/`.
- **D-17:** Canonical output files are `coolant_train.h5`, `coolant_dev.h5`, and `coolant_test.h5`.
- **D-18:** Preserve existing COOLANT-compatible dataset names: `caption_features`, `image_features`, and `article_ids`.
- **D-19:** Optional metadata such as `source_urls`, `image_paths`, `folder_paths`, or source labels may be added if useful, but model training must not depend on them.

### Feature Representation
- **D-20:** Preserve the existing representation from `notebooks/all_stage_final/workflow_coolant/1_preprocess.ipynb`.
- **D-21:** Text/caption features are PhoBERT token embeddings shaped `(N, max_len, 768)`.
- **D-22:** Image features are ResNet50 final features shaped `(N, 2048)`.
- **D-23:** Do not switch text to pooled `[CLS]` for Phase 2; that would likely break compatibility with the existing COOLANT text encoder expectations.

### Missing Image and Caption Handling
- **D-24:** Skip missing or broken image pairs rather than using zero vectors or placeholder images.
- **D-25:** Report skipped missing-image counts in preprocessing statistics.
- **D-26:** Reuse existing caption cleaning/filtering behavior from `src/processing/coolant/pair_extractor.py`, including credit-only and too-short caption filtering.

### Dataset Statistics Report
- **D-27:** Produce a complete preprocessing report per split: raw articles, generated matched pairs, skipped missing-image pairs, skipped invalid-caption pairs, final HDF5 rows, source label counts, feature shapes, and HDF5 file sizes.
- **D-28:** Display the report in notebook output and save machine-readable JSON to `processed_data/hdf5/preprocessing_stats.json`.

### Dependency and Environment Behavior
- **D-29:** The intended conda environment is `fake_news`, from `environment.yml`.
- **D-30:** The notebook must include dependency checks because the current `fake_news` env is missing `h5py`.
- **D-31:** Dependency installation is config-controlled: `AUTO_INSTALL_DEPS = False` by default.
- **D-32:** If dependencies are missing and auto-install is false, the notebook should print exact conda/pip install commands and stop clearly.

### Caching and Re-run Behavior
- **D-33:** Skip existing pair/HDF5 outputs by default to avoid expensive recomputation.
- **D-34:** Use `FORCE_REBUILD = False` in the config cell; when true, rebuild outputs.

### Batch and Device Behavior
- **D-35:** Use automatic device selection with priority `cuda > mps > cpu`.
- **D-36:** Use a conservative default batch size of `32`; planner may lower to `16` if memory issues are likely.
- **D-37:** Run `gc.collect()` and `torch.cuda.empty_cache()` after batches when CUDA is available.
- **D-38:** Use `num_workers=0` for HDF5 compatibility.

### Claude's Discretion
- The user delegated several choices to Claude. Locked recommendations: preserve ViFactCheck dev/test and append extra crawled data to train only; preserve existing token embedding representation; use title + caption; display tables plus save JSON stats; keep old COOLANT HDF5 dataset names; use repo-level `processed_data/hdf5/`; use config-selectable label variants with root as default.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Roadmap and project scope
- `.planning/ROADMAP.md` — Phase 2 objective, deliverables, requirements PREP-01 through PREP-05 and notebook requirements NB-01 through NB-03.
- `.planning/PROJECT.md` — project constraints: PhoBERT required, HDF5 preferred, config-driven notebooks, no hardcoded absolute paths, notebooks import shared `src/` utilities.
- `.planning/phases/01-data-crawling-notebook/01-SUMMARY.md` — Phase 1 notebook conventions and output path decisions to carry forward.

### Existing preprocessing and COOLANT pair pipeline
- `notebooks/all_stage_final/workflow_coolant/1_preprocess.ipynb` — legacy preprocessing notebook defining current HDF5 schema and feature shapes.
- `notebooks/all_stage_final/workflow_coolant/0_extract_pairs.ipynb` — legacy pair extraction notebook; useful for migration context.
- `src/processing/coolant/pair_extractor.py` — existing pair extraction, caption cleaning, image path resolution, and skip statistics.
- `src/processing/coolant/pair_dataset.py` — existing COOLANT HDF5 dataset reader using `caption_features`, `image_features`, and `article_ids`.
- `src/processing/coolant/training_utils.py` — dynamic matched/unmatched pair creation with `make_coolant_pairs()` and `make_detection_batch()`.

### Feature extraction utilities
- `src/preprocessing/text_preprocessing.py` — PhoBERT text preprocessing and token embedding extraction.
- `src/preprocessing/image_preprocessing.py` — ResNet50 image feature extraction.
- `src/preprocessing/combined_preprocessing.py` — combined text/image preprocessor used by the legacy notebook.
- `src/processing/hdf5_dataset.py` — HDF5 dataset classes; useful for HDF5 loading patterns and constraints.

### Data examples and label variants
- `notebooks/data/json/news_data_vifactcheck_dev_labeled.json` — user-referenced root labeled JSON example with fields `title`, `content`, `source_url`, `images`, and `label`.
- `notebooks/data/json/news_data_vifactcheck_train_labeled.json` — root train split input.
- `notebooks/data/json/news_data_vifactcheck_test_labeled.json` — root test split input.
- `notebooks/data/json/labeled_nei_as_real/` — alternate binary label interpretation.
- `notebooks/data/json/labeled_three_class/` — alternate three-class label interpretation.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `PairExtractor` in `src/processing/coolant/pair_extractor.py`: can load split JSON files, resolve image paths, clean captions, skip bad captions/images, print stats, and save/load pair JSON.
- `CombinedPreprocessor` in `src/preprocessing/combined_preprocessing.py`: legacy notebook used it to combine PhoBERT and ResNet50 preprocessing.
- `TextPreprocessor.extract_token_embeddings()`: produces token embeddings compatible with the legacy COOLANT HDF5 schema.
- `ImagePreprocessor.extract_features()`: produces ResNet50 image features compatible with existing COOLANT training.
- `CoolantPairDataset` in `src/processing/coolant/pair_dataset.py`: reads `caption_features` and `image_features`, transposes captions to `[768, seq_len]`, and leaves real/fake label creation to training.

### Established Patterns
- Pipeline notebooks use a single top config cell, relative/config-driven paths, and clear markdown sections.
- Generated artifacts should be path-configurable and should avoid hardcoded absolute paths.
- Existing COOLANT training dynamically creates fake/unmatched pairs by rolling images inside a batch; Phase 2 should not duplicate that responsibility.
- HDF5 access should prefer `num_workers=0` to avoid worker/file-handle issues.

### Integration Points
- Phase 2 output feeds Phase 3 COOLANT Training Notebook.
- Phase 3 should load `processed_data/hdf5/coolant_train.h5`, `coolant_dev.h5`, and `coolant_test.h5` through `CoolantPairDataset` or compatible code.
- The notebook should be located at `notebooks/pipeline/02_preprocessing.ipynb`.

</code_context>

<specifics>
## Specific Ideas

- The user clarified the conceptual pair logic: matched image/text is real; cross image/text is fake. After inspecting existing COOLANT utilities, this should be implemented dynamically in Phase 3, not materialized in Phase 2.
- The user explicitly requested consistency with the existing pair simulation logic; downstream agents should respect `make_detection_batch()` / image roll behavior.
- The user requested conda usage. Dependency checks and instructions should target conda env `fake_news`.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within Phase 2 scope.

</deferred>

---

*Phase: 2-Preprocessing Notebook*
*Context gathered: 2026-05-10*
