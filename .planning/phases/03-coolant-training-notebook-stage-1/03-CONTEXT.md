# Phase 3: COOLANT Training Notebook (Stage 1) - Context

**Gathered:** 2026-05-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Build `notebooks/pipeline/03_coolant_training.ipynb`, a reproducible Stage 1 COOLANT training notebook that loads Phase 2 HDF5 feature files, trains the existing `ResNetCOOLANT` / `PatchedCOOLANT` model with dynamic matched/unmatched negatives, logs metrics to MLflow, saves reproducible checkpoints, plots training curves, evaluates the best checkpoint, and produces a clear handoff contract for Phase 4 MM-ViFactCheck integration.

Out of scope: changing the COOLANT architecture, materializing fake/unmatched pairs into Phase 2 HDF5, implementing Stage 2 fusion/training, building an inference API/UI, or pre-extracting all Stage 1 features for Phase 4.

</domain>

<decisions>
## Implementation Decisions

### Training Recipe Baseline
- **D-01:** Use the legacy simultaneous training notebook flow as the Phase 3 baseline, especially `notebooks/all_stage_final/workflow_coolant/2a_train_simultaneous.ipynb`, because it already matches Phase 2 HDF5 and dynamic-negative training.
- **D-02:** Implement a clean migration, not an exact copy: preserve the same model flow and losses, but rewrite cells with a clean config, relative paths, MLflow integration, checkpointing, plots, and thesis-friendly markdown.
- **D-03:** Preserve the existing dynamic negative utilities: call `make_coolant_pairs()` and `make_detection_batch()` as-is, with configurable `NEGATIVE_SHIFT=3`.
- **D-04:** Keep separate optimizers for `similarity_module`, `clip_module`, and `detection_module`.
- **D-05:** Use warmup cosine scheduling per step.
- **D-06:** Preserve the full legacy composite loss: similarity cosine loss, CLIP contrastive CE plus soft CE, and detection CE plus KL ambiguity loss.
- **D-07:** Keep configurable gradient clipping with default `grad_clip=1.0`.
- **D-08:** Use configurable `MIN_BATCH_FOR_NEGATIVES=4`; log skipped-batch counts when tiny batches are skipped.
- **D-09:** Fresh run is the default. Optional resume is allowed only through explicit `RESUME_FROM_CHECKPOINT`; no automatic resume from latest checkpoint.
- **D-10:** Model patching should happen in a clear notebook cell by calling existing functions: `patch_encoding`, `patch_clip_projection`, and `patch_cnn_with_dropout`. Do not add a new abstraction unless planning discovers it is already available.
- **D-11:** Include final test-set evaluation in Phase 3 after reloading the best checkpoint.

### Checkpoint Selection and Handoff
- **D-12:** Preserve roadmap compliance for TRAIN-05 unless research/planning explicitly updates the requirement: validation accuracy remains the default Stage 2 handoff selection metric.
- **D-13:** Macro-F1 must still be logged and plotted. If practical, save both accuracy-best and F1-best checkpoints, but the validation-accuracy checkpoint is the default Phase 4 handoff.
- **D-14:** Use a stable `best_*` checkpoint naming scheme that logs winning epoch and validation accuracy. Prefer `best_model.pth` with metadata unless the planner selects a clearer compatible name.
- **D-15:** Save checkpoints under `training/checkpoints_coolant/` using timestamped run directories to avoid overwriting thesis runs.
- **D-16:** Save periodic `checkpoint_epoch_{epoch}.pth` files every `CHECKPOINT_EVERY` epochs plus the best checkpoint, satisfying TRAIN-03.
- **D-17:** Periodic/latest checkpoints should include optimizer and scheduler states for recovery. The primary best handoff may omit training states or include them only if cheap; Phase 4 ignores optimizer/scheduler states.
- **D-18:** The primary checkpoint must include a reproducibility bundle: `model_state_dict`, config, epoch, validation/test metrics, feature dimensions, selection metric, MLflow run id when available, training history, and separate submodule state dicts if straightforward.
- **D-19:** Add checkpoint metadata `freeze_for_stage2=True`. Phase 4 must load the model and explicitly freeze parameters.
- **D-20:** Write `checkpoint_manifest.json` with best checkpoint path, metric values, epoch, config hash/run id, expected feature shapes, and MLflow status.
- **D-21:** After training, reload the saved best checkpoint, verify keys/shapes, and use the reloaded model for final test evaluation.

### MLflow, Metrics, and Plots
- **D-22:** MLflow should log train/val loss, accuracy, macro-F1, real/fake precision/recall, confusion-matrix values where useful, learning rates, and separate loss components when already available without extra complexity.
- **D-23:** End-of-training artifacts should include config JSON, training history JSON/CSV, curve PNGs, final test report, confusion matrix, and `checkpoint_manifest.json`.
- **D-24:** Do not log checkpoint `.pth` files into MLflow by default; keep them on disk under `training/checkpoints_coolant/`.
- **D-25:** Inline plots should show train/val loss, accuracy, and macro-F1 curves. Add a final confusion matrix heatmap if test evaluation runs.
- **D-26:** Use `notebooks/mlruns` as the local MLflow tracking directory because it already exists and matches prior notebook workflows.

### Runtime Defaults and Safety Controls
- **D-27:** Default training config is thesis full run: `BATCH_SIZE=32`, `MAX_EPOCHS=30`, `PATIENCE=7`.
- **D-28:** Include `SMOKE_TEST=False` plus a lightweight preflight/smoke path that can quickly validate data/model/loss/checkpoint logic when enabled.
- **D-29:** Auto-select device with priority `cuda > mps > cpu`, using existing device behavior where possible. Print selected device and memory notes.
- **D-30:** Warn but allow CPU/MPS fallback for full training, and suggest `SMOKE_TEST=True` for local validation.
- **D-31:** Mirror Phase 2 dependency behavior: check required dependencies, keep `AUTO_INSTALL_DEPS=False` by default, print exact install commands, and stop clearly when dependencies are missing.
- **D-32:** Create a new timestamped run directory under `training/checkpoints_coolant/` for each run. Do not overwrite prior runs by default.
- **D-33:** Seed Python, NumPy, PyTorch, and CUDA. Log seed/device settings, but do not force slow deterministic algorithms unless exposed by config.
- **D-34:** Run memory cleanup at epoch/eval boundaries by default. Provide optional `AGGRESSIVE_MEMORY_CLEANUP=True` for more frequent cleanup if needed.
- **D-35:** Validate Phase 2 HDF5 files before training. If files are missing or schema/shapes are wrong, fail with clear instructions that Phase 2 must run first.

### Notebook Structure and Config Contract
- **D-36:** Notebook section flow should be: Overview → Config → Dependency/device checks → Data validation/loaders → Model setup → Training → Best reload/test eval → Plots/artifacts → Stage 2 handoff.
- **D-37:** Use thesis-friendly markdown: short explanations before each major section covering purpose, inputs, outputs, and why choices match COOLANT/Phase 2.
- **D-38:** Use one top-level nested `CONFIG` dict grouped by paths, model dimensions, training, MLflow, checkpointing, and safety flags. This config must be loggable to MLflow and embeddable in checkpoints.
- **D-39:** Use existing `src/` helpers where available; keep notebook-specific helpers such as plotting/checkpoint glue in clearly labeled notebook cells.
- **D-40:** Display `tqdm` batch progress plus concise printed epoch summaries with train/val metrics and best-checkpoint notices.
- **D-41:** The committed/source notebook should be output-clean by default; generated run artifacts carry execution outputs.
- **D-42:** Include concise provenance references early in the notebook to the legacy training notebook, Phase 2 context, and COOLANT workflow analysis.
- **D-43:** Add a final executable handoff summary cell printing best checkpoint path, manifest path, selected metric/epoch, expected Stage 1 feature outputs, and next notebook path.
- **D-44:** Use nested `CONFIG['paths']` with stable train/dev/test HDF5 paths, checkpoint root/run dir, MLflow tracking dir, and artifacts dir keys.
- **D-45:** Keep model/dimension config explicit and validate it against HDF5 shapes before training.
- **D-46:** Use nested descriptive training hyperparameter names in `CONFIG`; convert to legacy names only where `ResNetCOOLANT` expects them.
- **D-47:** Use run naming convention `coolant_stage1_{YYYYMMDD_HHMMSS}` with model variant and selection metric stored in metadata.

### Failure Handling
- **D-48:** On CUDA OOM, stop with recovery steps: clear cache, save latest recoverable checkpoint if possible, and print instructions to lower `batch_size` or enable smoke mode. Do not silently auto-reduce batch size.
- **D-49:** On NaN/Inf loss, stop and diagnose with recent metrics/loss components and shape checks. Do not silently skip or auto-lower LR.
- **D-50:** If MLflow setup/logging fails, warn clearly and continue saving local artifacts. Mark `mlflow_enabled=False` in `checkpoint_manifest.json`.
- **D-51:** On keyboard interrupt, save `interrupted_epoch_{epoch}.pth` with config/history when possible, end MLflow cleanly if active, and print explicit resume instructions.

### Stage 1 Feature API for Phase 4
- **D-52:** Phase 3 documents the Stage 1 output contract; Phase 4 implements the actual extractor.
- **D-53:** Phase 4 must be able to extract `text_aligned_clip`, `image_aligned_clip`, `attention_weights`, and `fake_prob` from the frozen Stage 1 checkpoint.
- **D-54:** The contract should also expose `detection_logits`, with `fake_prob = softmax(detection_logits, dim=-1)[:, 1]`.
- **D-55:** Phase 3 should include a one-batch sanity check after reloading the best checkpoint to verify the Stage 2 output keys/shapes.
- **D-56:** Do not pre-extract all Stage 1 outputs in Phase 3; that belongs to Phase 4.

### Claude's Discretion
- Several user choices delegated to Claude. Locked resolutions: legacy simultaneous loop baseline; validation-accuracy handoff despite macro-F1 interest unless the requirement is intentionally updated; full reproducibility checkpoint bundle; checkpoint manifest; full `ResNetCOOLANT` load in Phase 4; reload best checkpoint before test eval; `notebooks/mlruns`; smoke-test support; dependency checks; timestamped run dirs; seed practical RNGs; epoch-boundary memory cleanup; output-clean source notebook; concise provenance references; stop-and-diagnose failure behavior; Phase 4 owns the feature extractor.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Roadmap and project scope
- `.planning/ROADMAP.md` — Phase 3 objective, deliverables, requirements TRAIN-01 through TRAIN-05 and NB-01 through NB-03.
- `.planning/PROJECT.md` — project constraints: use `ResNetCOOLANT` as-is, HDF5 workflow, MLflow tracking, config-driven notebooks, no hardcoded absolute paths, notebooks import shared `src/` utilities.
- `.planning/REQUIREMENTS.md` — v1 training requirements and out-of-scope boundaries.
- `.planning/phases/02-preprocessing-notebook/02-CONTEXT.md` — Phase 2 HDF5 contract, dynamic-negative decision, text/image feature shapes, dataset path decisions, and pair generation boundaries.

### COOLANT architecture and training references
- `docs/COOLANT_WORKFLOW_ANALYSIS.md` — documents paper vs official repo vs current implementation discrepancies; confirms milestone decision to use `ResNetCOOLANT` as-is.
- `notebooks/all_stage_final/workflow_coolant/2a_train_simultaneous.ipynb` — primary legacy training baseline for dynamic negatives, full composite loss, separate optimizers, model patching, metrics, and checkpoint behavior.
- `notebooks/all_stage_final/train_vietnamese_coolant.ipynb` — additional legacy notebook with MLflow logging, checkpoint saving, warmup scheduler, and final evaluation patterns.
- `examples/train_coolant_official.py` — official-style trainer reference for separate tasks/optimizers and COOLANT loss flow; use as context, not the primary notebook baseline.

### Existing implementation assets
- `src/models/resnet_coolant.py` — `PatchedCOOLANT` / `ResNetCOOLANT` and patch helpers for ResNet/PhoBERT feature dimensions.
- `src/processing/coolant/pair_dataset.py` — `CoolantPairDataset` and dataloader creation for HDF5 `caption_features`, `image_features`, and `article_ids`.
- `src/processing/coolant/training_utils.py` — `make_coolant_pairs()`, `make_detection_batch()`, and `soft_cross_entropy()` used for dynamic matched/unmatched pair training.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ResNetCOOLANT` / `PatchedCOOLANT` in `src/models/resnet_coolant.py`: primary model for Stage 1 training; supports arbitrary feature dimensions through patch helpers.
- `patch_encoding`, `patch_clip_projection`, `patch_cnn_with_dropout`: reuse directly in notebook model setup to adapt 2048-dim image features and 768-dim PhoBERT token embeddings.
- `CoolantPairDataset` in `src/processing/coolant/pair_dataset.py`: reads Phase 2 HDF5 datasets and transposes caption features to `[768, seq_len]` for Conv1d.
- `create_coolant_dataloaders()` in `src/processing/coolant/pair_dataset.py`: baseline loader helper, using `num_workers=0` for HDF5 compatibility.
- `make_coolant_pairs()` and `make_detection_batch()` in `src/processing/coolant/training_utils.py`: canonical dynamic negative creation for similarity and detection training.
- `soft_cross_entropy()` in `src/processing/coolant/training_utils.py`: reusable soft-label loss for CLIP distillation in legacy training flow.

### Established Patterns
- Pipeline notebooks should use a single top config cell, relative/config-driven paths, and clear markdown sections.
- Phase 2 writes per-split HDF5 files under repo-level `processed_data/hdf5/`: `coolant_train.h5`, `coolant_dev.h5`, `coolant_test.h5`.
- HDF5 rows represent matched article/image-caption pairs only; fake/unmatched training labels are created dynamically in Phase 3.
- Existing COOLANT training uses separate module optimizers and a composite training loop rather than a single monolithic optimizer.
- HDF5 dataloading should keep `num_workers=0` to avoid file-handle issues.
- Project target execution includes GPU instances such as vast.ai, with local MPS/CPU fallback for smoke validation.

### Integration Points
- Input: `processed_data/hdf5/coolant_train.h5`, `processed_data/hdf5/coolant_dev.h5`, `processed_data/hdf5/coolant_test.h5` from Phase 2.
- Output notebook: `notebooks/pipeline/03_coolant_training.ipynb`.
- Checkpoint outputs: timestamped directories under `training/checkpoints_coolant/`.
- MLflow tracking: `notebooks/mlruns`.
- Phase 4 handoff: full `ResNetCOOLANT` checkpoint plus `checkpoint_manifest.json`, with frozen-load expectation and Stage 1 output contract.

</code_context>

<specifics>
## Specific Ideas

- The user selected all initial and additional gray areas, so the context is intentionally comprehensive.
- The user explicitly preferred a full thesis default run (`BATCH_SIZE=32`, `MAX_EPOCHS=30`, `PATIENCE=7`) while still allowing smoke-test support.
- The user selected `training/checkpoints_coolant` as the checkpoint root.
- The user clarified the primary best checkpoint should log the winning epoch and validation accuracy.
- The user selected `Warmup cosine per step`, `Full legacy composite`, `Separate optimizers`, `Include test evaluation`, `Thesis-friendly explanations`, `Hybrid` helper placement, `tqdm plus epoch summary`, and timestamp-based run naming.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within Phase 3 scope.

</deferred>

---

*Phase: 3-COOLANT Training Notebook (Stage 1)*
*Context gathered: 2026-05-11*
