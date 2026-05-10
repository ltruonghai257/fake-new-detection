# Phase 3 Summary: COOLANT Training Notebook (Stage 1)

**Status:** Complete  
**Completed:** 2026-05-11  
**Plan:** 03-PLAN.md (1 plan, 6 tasks)

---

## What Was Built

`notebooks/pipeline/03_coolant_training.ipynb` — a thesis-ready COOLANT Stage 1 training notebook.

### Deliverables

- **Config cell** — single nested `CONFIG` dict (paths / model / training / loss / mlflow / checkpointing / safety). All tunable values in one place; no hardcoded absolute paths.
- **Dependency preflight** — checks torch, h5py, numpy, pandas, matplotlib, seaborn, tqdm, sklearn, mlflow. Prints exact pip/conda install commands on failure; respects `AUTO_INSTALL_DEPS`.
- **Phase 2 HDF5 validation** — `validate_coolant_hdf5()` verifies `caption_features` and `image_features` datasets with shape constraints before training begins.
- **Device/seed setup** — `select_device()` (cuda > mps > cpu), `seed_everything()`, smoke-test dataloader wrapper.
- **Model construction** — `build_model()` creates `ResNetCOOLANT(model_config)` and applies all patch helpers: `patch_encoding` (3×), `patch_clip_projection` (2×), `patch_cnn_with_dropout` (3×).
- **Losses** — `nn.CosineEmbeddingLoss`, `nn.CrossEntropyLoss`, `nn.KLDivLoss(reduction="batchmean")` plus `soft_cross_entropy` for CLIP distillation.
- **Separate optimizers** — `optim_similarity`, `optim_clip`, `optim_detection` (Adam) with per-step warmup cosine schedulers via `make_warmup_cosine_scheduler`.
- **MLflow** — `coolant-stage1-training` experiment, per-epoch metric logging, graceful fallback (`MLflow disabled; continuing with local artifacts only`).
- **Checkpoint helpers** — `save_checkpoint()` (full reproducibility bundle: model/submodule state dicts, config, config_hash, epoch, metrics, history, feature_dims, freeze_for_stage2=True, mlflow_run_id), `write_checkpoint_manifest()`, `load_training_checkpoint()`.
- **Training loop** — `train_one_epoch()` with dynamic negatives (`make_coolant_pairs`, `make_detection_batch`), composite losses, per-step scheduler stepping, gradient clipping, NaN/Inf guard (`assert_finite_loss` → `FloatingPointError`), skipped-batch counter.
- **Evaluation** — `evaluate()` with accuracy, macro-F1, per-class precision/recall, confusion matrix.
- **Epoch loop** — early stopping on `val_accuracy`, periodic checkpoint (`checkpoint_epoch_{epoch}.pth`), `best_model.pth` (val_accuracy), `best_macro_f1.pth`, per-epoch `training_history.json` / `training_history.csv`.
- **Failure handling** — CUDA OOM: saves `interrupted_epoch_{epoch}.pth`, raises with `CUDA OOM: lower CONFIG['training']['batch_size'] or enable CONFIG['safety']['smoke_test']`; KeyboardInterrupt: saves checkpoint, prints `Set CONFIG['safety']['resume_from_checkpoint'] to this path to resume explicitly`.
- **Best checkpoint reload** — `load_best_model_for_eval()` builds fresh model, loads `checkpoint["model_state_dict"]`, runs final test evaluation.
- **Artifacts** — `plot_training_curves()` (loss_curves.png, accuracy_curves.png, macro_f1_curves.png), `plot_confusion_matrix()` (test_confusion_matrix.png), `test_report.json`, all logged to MLflow when available.
- **Stage 2 handoff** — `stage2_handoff_sanity_check()` verifies output keys (text_aligned_clip, image_aligned_clip, attention_weights, detection_logits, fake_prob) and prints their shapes; `checkpoint_manifest.json` documents the full Phase 4 contract.

---

## Requirements Coverage

| Requirement | Status |
|-------------|--------|
| TRAIN-01: Config cell for model variant, hyperparameters, data paths | ✓ |
| TRAIN-02: MLflow logging (loss, accuracy, F1 per epoch) | ✓ |
| TRAIN-03: Checkpoints every N epochs with embedded config | ✓ |
| TRAIN-04: Inline training curves | ✓ |
| TRAIN-05: Best checkpoint by val accuracy; freeze_for_stage2=True | ✓ |
| NB-01: Single config cell | ✓ |
| NB-02: Relative/config-driven paths only | ✓ |
| NB-03: Clear markdown section headers | ✓ |

---

## Commit

`feat(03): create COOLANT Stage 1 training notebook` — 1 file, 1242 insertions

---

## Phase 4 Handoff

- **Best checkpoint:** `training/checkpoints_coolant/<run_name>/best_model.pth`
- **Manifest:** `training/checkpoints_coolant/<run_name>/checkpoint_manifest.json`
- **Selection metric:** `val_accuracy`
- **freeze_for_stage2:** `True`
- **Stage 1 output keys:** `text_aligned_clip`, `image_aligned_clip`, `attention_weights`, `detection_logits`, `fake_prob`
- **Expected inputs:** `caption_features: [batch, 768, 128]`, `image_features: [batch, 2048]`
