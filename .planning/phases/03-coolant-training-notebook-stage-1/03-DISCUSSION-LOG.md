# Phase 3: COOLANT Training Notebook (Stage 1) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-11
**Phase:** 3-COOLANT Training Notebook (Stage 1)
**Areas discussed:** Training recipe baseline, Checkpoint handoff to Stage 2, MLflow and plots scope, Runtime defaults and safety controls, Notebook structure and cells, Config naming contract, Training failure handling, Stage 1 feature API for Phase 4

---

## Training Recipe Baseline

**User choices and notes:**
- Delegated baseline choice to Claude; resolved as legacy simultaneous loop because it matches Phase 2 HDF5 and dynamic negatives.
- Selected clean migration over exact preservation or fresh minimal rewrite.
- Initially selected macro-F1 primary, then delegated the roadmap conflict; resolved as preserve validation-accuracy handoff unless requirement is intentionally updated, while logging/saving macro-F1 where practical.
- Selected existing dynamic negative utilities.
- Selected separate optimizers.
- Selected warmup cosine per step.
- Selected include test evaluation.
- Delegated model patching location; resolved as notebook calls existing patch functions.
- Selected full legacy composite loss.
- Selected keep gradient clipping.
- Delegated small-batch and resume behavior; resolved as configurable minimum batch and fresh run by default with explicit resume only.

---

## Checkpoint Handoff to Stage 2

**User choices and notes:**
- Free-text checkpoint naming note: best checkpoint should log winning epoch and validation accuracy.
- Delegated checkpoint contents; resolved as full reproducibility bundle with model/config/metrics/history/feature dims and optional submodule states.
- Delegated periodic checkpointing; resolved as every `CHECKPOINT_EVERY` epochs plus best checkpoint to satisfy TRAIN-03.
- Selected `training/checkpoints_coolant` as checkpoint root.
- Delegated freeze contract, manifest, Stage 2 loading mode, and reload validation; resolved as metadata plus explicit Phase 4 freeze, JSON manifest, full `ResNetCOOLANT` load, and reload-before-test validation.
- Selected optimizer/scheduler states for periodic/latest checkpoints only.

---

## MLflow and Plots Scope

**User choices and notes:**
- Delegated per-epoch MLflow metrics; resolved as core plus class metrics, learning rates, and available loss components.
- Selected config/history/curves/report artifacts, excluding checkpoint `.pth` files from MLflow by default.
- Delegated inline plots; resolved as loss, accuracy, macro-F1, plus final confusion matrix heatmap when test eval runs.
- Delegated MLflow tracking path; resolved as `notebooks/mlruns`.

---

## Runtime Defaults and Safety Controls

**User choices and notes:**
- Selected thesis full run defaults: `BATCH_SIZE=32`, `MAX_EPOCHS=30`, `PATIENCE=7`.
- Delegated smoke mode, dependency checks, output handling, reproducibility, memory cleanup, and non-CUDA behavior; resolved as config-gated smoke path, strict dependency preflight with `AUTO_INSTALL_DEPS=False`, timestamped run dirs, practical RNG seeding, epoch/eval-boundary cleanup, and warn-but-allow CPU/MPS.
- Selected auto device priority `cuda > mps > cpu`.
- Selected fail clearly when HDF5 files are missing or schema/shapes are wrong.

---

## Notebook Structure and Cells

**User choices and notes:**
- Selected linear thesis workflow structure.
- Selected thesis-friendly explanations.
- Delegated config shape; resolved as a single nested `CONFIG` dict.
- Delegated final handoff cell; resolved as explicit executable handoff summary.
- Selected hybrid helper placement.
- Selected `tqdm` plus epoch summary progress display.
- Delegated committed outputs and provenance references; resolved as output-clean source notebook and concise provenance references.

---

## Config Naming Contract

**User choices and notes:**
- Delegated path keys; resolved as nested `CONFIG['paths']` with stable HDF5/checkpoint/MLflow/artifact paths.
- Delegated model dimension keys; resolved as explicit config values validated against HDF5 shapes.
- Delegated hyperparameter naming; resolved as nested descriptive `CONFIG` names with legacy conversion only where needed.
- Selected timestamp-plus-model run naming: `coolant_stage1_{YYYYMMDD_HHMMSS}`.

---

## Training Failure Handling

**User choices and notes:**
- Delegated OOM, NaN/Inf, MLflow failure, and interrupt behavior.
- Resolved as stop with recovery instructions on OOM; stop and diagnose on NaN/Inf; warn and continue local artifacts if MLflow fails; save interrupted checkpoint and cleanly end MLflow on keyboard interrupt.

---

## Stage 1 Feature API for Phase 4

**User choices and notes:**
- Delegated output contract; resolved as MMVF-required outputs: `text_aligned_clip`, `image_aligned_clip`, `attention_weights`, and `fake_prob`.
- Delegated helper ownership; resolved as Phase 3 documents the contract and Phase 4 implements extractor.
- Delegated sanity check; resolved as one-batch output key/shape check after reloading best checkpoint.
- Delegated probability API; resolved as include both `detection_logits` and `fake_prob = softmax(detection_logits)[:, 1]`.

---

## Claude's Discretion

The user frequently selected `You decide`. Resolutions are recorded in `03-CONTEXT.md` under the relevant decisions and in the Claude's Discretion subsection.

## Deferred Ideas

None.
