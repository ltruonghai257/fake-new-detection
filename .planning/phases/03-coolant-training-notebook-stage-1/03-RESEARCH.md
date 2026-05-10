# Phase 3 Research: COOLANT Training Notebook (Stage 1)

**Phase:** 3 — COOLANT Training Notebook (Stage 1)  
**Researched:** 2026-05-11  
**Status:** Ready for planning

---

## Research Question

What does Phase 3 need to know to plan a reproducible COOLANT Stage 1 training notebook that satisfies TRAIN-01 through TRAIN-05 and NB-01 through NB-03?

---

## Key Findings

### 1. Existing model implementation is already the intended Stage 1 model

- `src/models/resnet_coolant.py` defines `PatchedCOOLANT` and exposes `ResNetCOOLANT = PatchedCOOLANT` for backward compatibility.
- The model is a wrapper around `COOLANT_Official` and must be dimension-patched for Phase 2 features.
- Required patch helpers already exist:
  - `patch_encoding(enc, image_dim=2048)` patches image input layers.
  - `patch_clip_projection(clip_module, target_dim, is_image=True|False)` patches CLIP projections.
  - `patch_cnn_with_dropout(m, input_dim, dropout)` patches text Conv1d blocks for `[B, 768, seq_len]` input.
- The notebook should reuse these helpers directly instead of changing architecture code.

### 2. Phase 2 COOLANT HDF5 contract uses separate split files and dynamic labels

- `src/processing/coolant/pair_dataset.py` defines `CoolantPairDataset` and `create_coolant_dataloaders()`.
- It expects HDF5 keys:
  - `caption_features`
  - `image_features`
  - optional `article_ids`
- `CoolantPairDataset.__getitem__()` transposes captions from `[seq_len, 768]` to `[768, seq_len]` for the patched FastCNN.
- `create_coolant_dataloaders()` returns `loaders` and `datasets` dictionaries for `train`, `dev`, and `test`, with `num_workers=0` for HDF5 compatibility.
- Phase 2 stores matched pairs only. Phase 3 creates unmatched/fake pairs dynamically.

### 3. Dynamic negative utilities are canonical and should remain unchanged

- `src/processing/coolant/training_utils.py` provides:
  - `make_coolant_pairs(caption, image, shift=3)` for matched/unmatched similarity pairs.
  - `make_detection_batch(caption, image, shift=3)` for balanced real/fake detection batches.
  - `soft_cross_entropy(logits, soft_target)` for CLIP distillation loss.
- These utilities use image rolling with default `shift=3` and assume batch size is large enough to avoid degenerate negatives.
- The notebook should expose `NEGATIVE_SHIFT=3` and `MIN_BATCH_FOR_NEGATIVES=4` in the config.

### 4. Existing training notebooks provide the implementation baseline

Primary references:

- `notebooks/all_stage_final/workflow_coolant/2a_train_simultaneous.ipynb`
- `notebooks/all_stage_final/workflow_coolant/2b_train_phased.ipynb`
- `notebooks/all_stage_final/train_vietnamese_coolant.ipynb`
- `notebooks/all_stage_final/4_train_model.ipynb`

Reusable patterns:

- Separate optimizers for `similarity_module`, `clip_module`, and `detection_module`.
- Composite loss made from:
  - similarity cosine embedding loss
  - CLIP contrastive cross-entropy
  - CLIP soft cross-entropy/distillation
  - detection cross-entropy
  - KL ambiguity loss
- Validation loop collecting accuracy, macro-F1, per-class precision/recall, and confusion matrix.
- Checkpoint save on best validation metric.
- Test evaluation after training.
- `tqdm` batch progress plus printed epoch summaries.

Planning caveat:

- Some legacy notebooks select best by macro-F1, but Phase 3 requirement TRAIN-05 and context D-12 require validation accuracy as the default Phase 4 handoff metric.
- Macro-F1 should still be logged, plotted, and optionally have a secondary `best_macro_f1` checkpoint.

### 5. MLflow should be local and non-blocking

- Existing notebooks import `mlflow` and `mlflow.pytorch`, set experiments, start timestamped runs, log params/metrics, and log artifacts.
- The Phase 3 context requires local tracking under `notebooks/mlruns`.
- Because MLflow setup can fail on some local environments, the notebook should:
  - treat MLflow as optional after dependency check
  - continue saving local artifacts if MLflow logging fails
  - record `mlflow_enabled=False` in `checkpoint_manifest.json` when disabled
- Do not log `.pth` checkpoint files into MLflow by default; keep them under `training/checkpoints_coolant/`.

### 6. Checkpoints must be richer than legacy notebooks

Legacy code often saves only submodule `state_dict`s. Phase 3 needs reproducibility and Phase 4 handoff metadata.

Required checkpoint bundle:

- `model_state_dict`
- `config`
- `epoch`
- validation/test metrics
- feature dimensions
- selection metric
- MLflow run id when available
- training history
- `freeze_for_stage2=True`
- submodule state dicts when straightforward:
  - `similarity_module_state_dict`
  - `clip_module_state_dict`
  - `detection_module_state_dict`

Required companion artifact:

- `checkpoint_manifest.json` with best checkpoint path, metric values, epoch, config hash/run id, expected feature shapes, MLflow status, and Stage 2 output contract.

### 7. COOLANT architecture choice is already locked for this milestone

- `docs/COOLANT_WORKFLOW_ANALYSIS.md` documents discrepancies between the paper, official repository, and current implementation.
- The project decision in `.planning/PROJECT.md` is to use `ResNetCOOLANT` as-is and avoid architecture changes in this milestone.
- Phase 3 should not refactor model internals. It should patch dimensions at notebook setup and train the existing modules.

---

## Recommended Notebook Structure

1. Overview and provenance
2. Single config cell
3. Dependency, path, seed, and device checks
4. HDF5 schema validation
5. DataLoader creation
6. Model construction and dimension patching
7. MLflow and run directory setup
8. Training helper functions
9. Stage 1 training loop
10. Best checkpoint reload and validation
11. Final test evaluation
12. Curves, reports, and manifest writing
13. Stage 2 handoff summary

---

## Implementation Constraints

- Use `notebooks/pipeline/03_coolant_training.ipynb` as the only source notebook deliverable.
- Use relative/config-driven paths only.
- Default HDF5 inputs:
  - `processed_data/hdf5/coolant_train.h5`
  - `processed_data/hdf5/coolant_dev.h5`
  - `processed_data/hdf5/coolant_test.h5`
- Default checkpoint root: `training/checkpoints_coolant/`.
- Default MLflow tracking directory: `notebooks/mlruns`.
- Default run name: `coolant_stage1_{YYYYMMDD_HHMMSS}`.
- Default full-run settings: `BATCH_SIZE=32`, `MAX_EPOCHS=30`, `PATIENCE=7`.
- Include `SMOKE_TEST=False` and config-driven smoke-test limits.
- Use device priority `cuda > mps > cpu`.
- Keep HDF5 dataloading at `num_workers=0`.
- Stop clearly on missing dependencies, missing HDF5 files, schema mismatches, NaN/Inf loss, and CUDA OOM.

---

## Validation Architecture

### Pre-execution checks

- Verify the notebook exists and contains one top-level `CONFIG` dict.
- Verify no absolute local paths such as `/Users/haila/` are present.
- Verify config contains Stage 1 defaults and all required paths.
- Verify notebook imports existing helpers from `src.models.resnet_coolant` and `src.processing.coolant`.

### Runtime smoke validation

With `SMOKE_TEST=True`, the notebook should prove:

- HDF5 files are located or a clear Phase 2 prerequisite error is raised.
- HDF5 schemas contain `caption_features` and `image_features`.
- One batch can be loaded from each split.
- Model patching accepts caption shape `[B, 768, seq_len]` and image shape `[B, 2048]`.
- One forward/loss/checkpoint cycle can run without full training.
- A local checkpoint and `checkpoint_manifest.json` are written.

### Full-run validation

- MLflow logs train/val loss, accuracy, and macro-F1 per epoch.
- Periodic checkpoints are saved every `CHECKPOINT_EVERY` epochs.
- Best checkpoint is selected by validation accuracy.
- Training curves are generated as PNG artifacts and displayed inline.
- Best checkpoint is reloaded before final test evaluation.
- Manifest includes Stage 2 frozen-load contract.

---

## Plan Implications

The plan should be a single executable notebook plan with tasks ordered by dependency:

1. Build notebook skeleton/config/preflight.
2. Add data validation and dataloaders.
3. Add model setup, patching, losses, and optimizers.
4. Add MLflow, checkpoint, and manifest helpers.
5. Add training/evaluation loops with metrics and failure handling.
6. Add plotting, artifact export, and Stage 2 handoff checks.

## RESEARCH COMPLETE
