# Roadmap: v1.0 Full Pipeline Notebook Workflow

**Milestone:** v1.0 Full Pipeline Notebook Workflow  
**Goal:** Refactor scattered experimental notebooks into a clean, reproducible end-to-end research workflow covering data crawling, preprocessing, COOLANT training, and MM-ViFactCheck Stage 2 integration.

**Started:** 2026-05-08  
**Target:** TBD

---

## Phase 1: Data Crawling Notebook

**Objective:** Create a production-ready notebook for automated, resumable Vietnamese news crawling.

**Delivers:**
- `notebooks/pipeline/01_data_crawling.ipynb` with config cell, progress display, and resumable state
- Structured JSON output with text, image paths, and source metadata

**Requirements:**
- CRAWL-01: Config cell for crawl sources and output path
- CRAWL-02: Real-time progress display (URLs attempted/succeeded/failed)
- CRAWL-03: Resume from checkpoint without re-crawling
- CRAWL-04: Export to structured JSON
- NB-01: Single config cell for all parameters
- NB-02: Relative/config-driven paths only
- NB-03: Clear markdown section headers

**Depends on:** —

**Status:** ● Complete

---

## Phase 2: Preprocessing Notebook

**Objective:** Build unified preprocessing pipeline converting raw ViFactCheck JSON → HDF5 with PhoBERT + ResNet features.

**Delivers:**
- `notebooks/pipeline/02_preprocessing.ipynb` with text + image feature extraction
- HDF5 dataset with train/dev/test splits
- Dataset statistics report

**Requirements:**
- PREP-01: Load ViFactCheck JSON into unified format with splits
- PREP-02: Vietnamese normalization + PhoBERT-base-v2 tokenization/embedding
- PREP-03: ResNet50 feature extraction (2048-dim)
- PREP-04: Save to HDF5 for efficient DataLoader access
- PREP-05: Output dataset statistics (class balance, split sizes, missing images)
- NB-01: Single config cell
- NB-02: Relative/config-driven paths
- NB-03: Clear markdown sections

**Depends on:** Phase 1 (crawled data)

**Status:** ● Complete

---

## Phase 3: COOLANT Training Notebook (Stage 1)

**Objective:** Train PatchedCOOLANT model with MLflow tracking, checkpointing, and inline visualization.

**Delivers:**
- `notebooks/pipeline/03_coolant_training.ipynb` with config cell and MLflow integration
- Best checkpoint saved by validation accuracy
- Training curves (loss, accuracy plots)

**Requirements:**
- TRAIN-01: Config cell for model variant, hyperparameters, data paths
- TRAIN-02: MLflow logging (loss, accuracy, F1 per epoch)
- TRAIN-03: Checkpoints every N epochs with embedded config
- TRAIN-04: Inline training curves
- TRAIN-05: Best checkpoint selection by val accuracy; weights frozen for Stage 2
- NB-01: Single config cell
- NB-02: Relative/config-driven paths
- NB-03: Clear markdown sections

**Depends on:** Phase 2 (HDF5 dataset)

**Status:** ● Complete

---

## Phase 4: MM-ViFactCheck Integration Notebook (Stage 2)

**Objective:** Integrate frozen COOLANT checkpoint with ViFactCheck Stage 2 model, train, evaluate, and export results.

**Delivers:**
- `notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` with Stage 2 training and evaluation
- Evaluation report (accuracy, macro-F1, per-class metrics, confusion matrix)
- Ablation table (configs A–D)
- JSON export for thesis documentation

**Requirements:**
- MMVF-01: Load frozen COOLANT checkpoint, extract features per article
- MMVF-02: PhoBERT-base-v2 encodes [Statement; SEP; Evidence] → [CLS] (768-dim)
- MMVF-03: Gated fusion module (h_nli_proj + h_mm_proj)
- MMVF-04: Train on ViFactCheck train split; best checkpoint by val macro-F1
- MMVF-05: Evaluation report on test split
- MMVF-06: Ablation table (text-only → full MM-ViFactCheck)
- MMVF-07: Export results to JSON
- NB-01: Single config cell
- NB-02: Relative/config-driven paths
- NB-03: Clear markdown sections

**Depends on:** Phase 3 (trained COOLANT checkpoint)

**Status:** ● Complete

---

## Progress Summary

| Phase | Status | Requirements | Progress |
|-------|--------|--------------|----------|
| Phase 1: Data Crawling | ● Complete | 7 | 100% |
| Phase 2: Preprocessing | ● Complete | 8 | 100% |
| Phase 3: COOLANT Training | ● Complete | 8 | 100% |
| Phase 4: MM-ViFactCheck Integration | ● Complete | 10 | 100% |

**Overall:** [██████████] 100% (4/4 phases complete)

---

## Legend

- ○ Not started
- ◐ In progress
- ● Complete
- ✗ Blocked

---

*Last updated: 2026-05-12*
