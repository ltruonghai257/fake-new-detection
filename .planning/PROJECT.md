# Fake News Detection — Multimodal Vietnamese Pipeline

## What This Is

A Vietnamese multimodal fake news detection research pipeline for a thesis project. It processes news articles (text + images) from Vietnamese sources using PhoBERT for text encoding and ResNet/CLIP backbones for image encoding, then trains a COOLANT-based cross-modal contrastive model to classify real vs. fake news. The dataset extends ViFactCheck with newly crawled Vietnamese news sources.

## Core Value

A fully reproducible end-to-end pipeline — from raw Vietnamese news crawling to COOLANT training and ViFactCheck Stage 2 integration — that produces thesis-quality results.

## Current Milestone: v1.0 Full Pipeline Notebook Workflow

**Goal:** Refactor scattered experimental notebooks into a clean, reproducible end-to-end research workflow covering data crawling, preprocessing, COOLANT training, and ViFactCheck Stage 2 integration.

**Target features:**
- Crawling notebook: automated, resumable Vietnamese news data crawling
- Preprocessing notebook: unified text (PhoBERT) + image (ResNet/CLIP) pipeline → HDF5
- COOLANT Training notebook: config management, MLflow tracking, checkpointing
- ViFactCheck Stage 2 Integration notebook: load checkpoint → evaluate → report

## Requirements

### Validated

<!-- Shipped and confirmed valuable from existing src/ codebase. -->

- ✓ Async Vietnamese news crawler with resumable state (`crawling_status.json`) — `src/crawler/`
- ✓ PhoBERT text preprocessing pipeline — `src/preprocessing/text_preprocessing.py`
- ✓ ResNet/CLIP/SigLIP image preprocessing pipeline — `src/preprocessing/image_preprocessing.py`
- ✓ COOLANT/ResNetCOOLANT model with composite loss — `src/models/`
- ✓ ViFactCheck dataset processor and dataloader — `src/processing/`
- ✓ HDF5-backed dataset for efficient large-scale access — `src/processing/hdf5_dataset.py`

### Active

- [ ] Crawling notebook: single-entry workflow with config cell, progress display, and resume
- [ ] Preprocessing notebook: unified raw → HDF5 pipeline with dataset stats reporting
- [ ] COOLANT training notebook: config cell, MLflow logging, checkpoint management, plots
- [ ] ViFactCheck Stage 2 integration notebook: load checkpoint → eval → export JSON report

### Out of Scope

- Web UI / demo interface — not needed for thesis deliverable
- Real-time inference API — out of scope for thesis
- Multi-GPU distributed training — single-GPU sufficient for thesis scale
- COOLANT architecture changes — using existing ResNetCOOLANT; architecture research is separate
- Mocheg submodule integration — separate research codebase, not part of main pipeline
- Data annotation tooling — labels from ViFactCheck + source-based rules

## Context

- Vietnamese language NLP stack: PhoBERT (`vinai/phobert-base-v2`), underthesea normalization (optional, graceful fallback)
- Image backbones: ResNet50 (2048-dim), CLIP ViT-L/14 (1024-dim), SigLIP (768-dim)
- COOLANT has documented paper ≠ official-repo ≠ current-impl discrepancies (see `docs/COOLANT_WORKFLOW_ANALYSIS.md`); decision: use ResNetCOOLANT as-is
- Existing notebooks (1–4) are experimental and will be replaced/consolidated by this milestone
- Target execution environment: GPU instance (vast.ai) + local (MPS/CPU fallback via `src/utils/device.py`)
- All notebooks import from `src/` package for shared utilities; no code duplication

## Constraints

- **Language**: Vietnamese NLP — PhoBERT required; underthesea optional (graceful fallback)
- **Reproducibility**: Notebooks must work on fresh clone with minimal setup (conda env + data paths configured)
- **Config-driven**: All tunable parameters in a single top-of-notebook config cell; no hardcoded absolute paths
- **Thesis timeline**: Prioritize clean, readable notebooks over optimization

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use ResNetCOOLANT (PatchedCOOLANT) as primary model | Adapts for arbitrary feature dims; avoids fixed-dim constraints | — Pending |
| PhoBERT-base-v2 as default text encoder | Best Vietnamese NLP model; 768-dim compatible with COOLANT | — Pending |
| HDF5 for preprocessed features | Efficient random access for DataLoader; avoids re-computing features each epoch | ✓ Good |
| MLflow for experiment tracking | Low setup overhead; local tracking; `mlruns/` already exists in notebooks/ | — Pending |
| notebooks/ as target directory for all pipeline notebooks | Consolidates all experiment code in one place | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-08 — Milestone v1.0 initialized*
