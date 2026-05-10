---
phase: 3
plan: 03-PLAN
type: execute
wave: 1
depends_on:
  - Phase 2: Preprocessing Notebook
files_modified:
  - notebooks/pipeline/03_coolant_training.ipynb
autonomous: true
requirements:
  - TRAIN-01
  - TRAIN-02
  - TRAIN-03
  - TRAIN-04
  - TRAIN-05
  - NB-01
  - NB-02
  - NB-03
---

# Phase 3 Plan: COOLANT Training Notebook (Stage 1)

**Phase:** 3 — COOLANT Training Notebook (Stage 1)  
**Goal:** Build `notebooks/pipeline/03_coolant_training.ipynb`, a reproducible Stage 1 COOLANT training notebook that loads Phase 2 HDF5 feature files, trains existing `ResNetCOOLANT` / `PatchedCOOLANT`, logs MLflow metrics, saves reproducible checkpoints, plots training curves, evaluates the best checkpoint, and prints a Phase 4 handoff contract.  
**Requirements:** TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05, NB-01, NB-02, NB-03  
**Planned:** 2026-05-11  
**Status:** Ready to execute

---

<objective>
Create a clean thesis-ready COOLANT Stage 1 training notebook that reuses existing `src/` model and COOLANT dataset utilities, keeps all tunable values in one config cell, validates Phase 2 HDF5 inputs, trains with dynamic negatives and composite losses, logs experiment metrics, saves validation-accuracy-selected checkpoints, and produces a verified frozen-checkpoint handoff for Phase 4.
</objective>

---

## Source Truths and Must-Haves

<must_haves>

### Roadmap and Requirement Truths

- **TRAIN-01:** Notebook has a single config cell for PatchedCOOLANT variant, hyperparameters, and data paths.
- **TRAIN-02:** Training logs loss, accuracy, and F1 to MLflow per epoch.
- **TRAIN-03:** Checkpoints are saved every N epochs with embedded config dict.
- **TRAIN-04:** Notebook displays inline training curves for loss and accuracy per epoch.
- **TRAIN-05:** Best checkpoint is selected and saved by validation accuracy; metadata marks weights as frozen for Stage 2.
- **NB-01:** Notebook has one top config cell for all tunable parameters and paths.
- **NB-02:** Notebook uses relative/config-driven paths only.
- **NB-03:** Notebook has clear markdown section headers.

### Context Decision Coverage

- Use `ResNetCOOLANT` / `PatchedCOOLANT` as-is; do not change COOLANT architecture.
- Use Phase 2 HDF5 files from `processed_data/hdf5/coolant_train.h5`, `coolant_dev.h5`, and `coolant_test.h5`.
- Use existing `create_coolant_dataloaders()`, `make_coolant_pairs()`, `make_detection_batch()`, and `soft_cross_entropy()` utilities.
- Expose `NEGATIVE_SHIFT=3` and `MIN_BATCH_FOR_NEGATIVES=4` in config.
- Patch model dimensions in the notebook with `patch_encoding`, `patch_clip_projection`, and `patch_cnn_with_dropout`.
- Keep separate optimizers for `similarity_module`, `clip_module`, and `detection_module`.
- Use composite losses: similarity cosine loss, CLIP contrastive CE, soft CE, detection CE, and KL ambiguity loss.
- Default full-run config is `BATCH_SIZE=32`, `MAX_EPOCHS=30`, and `PATIENCE=7`.
- Include `SMOKE_TEST=False` and smoke-test limits for local validation.
- Save checkpoints under timestamped run directories in `training/checkpoints_coolant/`.
- Use `notebooks/mlruns` as MLflow tracking directory.
- Select primary best checkpoint by validation accuracy; also log macro-F1.
- Reload the saved best checkpoint before final test evaluation.
- Write `checkpoint_manifest.json` and print a Stage 2 handoff summary.

</must_haves>

---

<threat_model>

## Security, Reproducibility, and Data-Safety Considerations

- **Environment mutation:** The notebook must not install packages unless `AUTO_INSTALL_DEPS=True`; default behavior prints exact install commands and stops clearly.
- **Path safety:** The notebook must not contain hardcoded absolute local paths. All paths must derive from `PROJECT_ROOT` or config keys.
- **Expensive accidental overwrite:** Each run must create a timestamped directory under `training/checkpoints_coolant/`; it must not overwrite previous thesis runs by default.
- **MLflow failure:** If MLflow setup or logging fails, the notebook must continue local artifact/checkpoint saving and mark `mlflow_enabled=False` in `checkpoint_manifest.json`.
- **CUDA OOM:** The notebook must stop with recovery instructions instead of silently changing batch size.
- **NaN/Inf loss:** The notebook must stop and print recent loss components and shape checks instead of silently skipping bad batches.
- **Keyboard interrupt:** The notebook should save `interrupted_epoch_{epoch}.pth` with config/history when possible and print explicit resume instructions using `RESUME_FROM_CHECKPOINT`.

</threat_model>

---

<tasks>

## Task 1 — Create notebook skeleton, config contract, and preflight gates

**Type:** execute  
**Files:** `notebooks/pipeline/03_coolant_training.ipynb`  
**Requirements:** TRAIN-01, NB-01, NB-02, NB-03

<read_first>

- `.planning/phases/03-coolant-training-notebook-stage-1/03-CONTEXT.md`
- `.planning/phases/03-coolant-training-notebook-stage-1/03-RESEARCH.md`
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`
- `environment.yml`
- `requirements.txt`

</read_first>

<action>

Create `notebooks/pipeline/03_coolant_training.ipynb` with output-clean cells and these sections:

1. Markdown title: `# COOLANT Training — Stage 1 Multimodal Alignment`.
2. Markdown overview containing these exact strings:
   - `Input: processed_data/hdf5/coolant_train.h5, coolant_dev.h5, coolant_test.h5`
   - `Output: training/checkpoints_coolant/<run_name>/best_model.pth`
   - `Phase 4 handoff: frozen ResNetCOOLANT checkpoint plus checkpoint_manifest.json`
3. One top config cell containing exactly one top-level `CONFIG` dict with nested groups:
   - `paths`
   - `model`
   - `training`
   - `loss`
   - `mlflow`
   - `checkpointing`
   - `safety`
4. Include these concrete default config values:
   - `PROJECT_ROOT = Path.cwd().parent.parent if Path.cwd().name == "pipeline" else Path.cwd()`
   - `CONFIG["paths"]["hdf5_dir"] = PROJECT_ROOT / "processed_data" / "hdf5"`
   - `CONFIG["paths"]["train_hdf5"] = CONFIG["paths"]["hdf5_dir"] / "coolant_train.h5"`
   - `CONFIG["paths"]["dev_hdf5"] = CONFIG["paths"]["hdf5_dir"] / "coolant_dev.h5"`
   - `CONFIG["paths"]["test_hdf5"] = CONFIG["paths"]["hdf5_dir"] / "coolant_test.h5"`
   - `CONFIG["paths"]["checkpoint_root"] = PROJECT_ROOT / "training" / "checkpoints_coolant"`
   - `CONFIG["paths"]["mlflow_dir"] = PROJECT_ROOT / "notebooks" / "mlruns"`
   - `CONFIG["model"]["variant"] = "ResNetCOOLANT"`
   - `CONFIG["model"]["image_dim"] = 2048`
   - `CONFIG["model"]["text_embed_dim"] = 768`
   - `CONFIG["model"]["text_seq_len"] = 128`
   - `CONFIG["training"]["batch_size"] = 32`
   - `CONFIG["training"]["max_epochs"] = 30`
   - `CONFIG["training"]["patience"] = 7`
   - `CONFIG["training"]["negative_shift"] = 3`
   - `CONFIG["training"]["min_batch_for_negatives"] = 4`
   - `CONFIG["training"]["grad_clip"] = 1.0`
   - `CONFIG["checkpointing"]["selection_metric"] = "val_accuracy"`
   - `CONFIG["checkpointing"]["checkpoint_every"] = 5`
   - `CONFIG["safety"]["smoke_test"] = False`
   - `CONFIG["safety"]["auto_install_deps"] = False`
   - `CONFIG["safety"]["resume_from_checkpoint"] = None`
5. Add a dependency preflight cell that checks imports for `torch`, `h5py`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`, `sklearn`, and `mlflow`.
6. If dependencies are missing and `auto_install_deps` is false, print these exact commands and raise `RuntimeError`:
   - `pip install mlflow seaborn scikit-learn tqdm h5py`
   - `conda install -n fake_news mlflow seaborn scikit-learn tqdm h5py -c conda-forge`
7. Add setup cell that inserts `PROJECT_ROOT` into `sys.path`, creates `checkpoint_root`, imports `ResNetCOOLANT`, `patch_encoding`, `patch_clip_projection`, `patch_cnn_with_dropout`, `create_coolant_dataloaders`, `make_coolant_pairs`, `make_detection_batch`, and `soft_cross_entropy`.

</action>

<acceptance_criteria>

- `notebooks/pipeline/03_coolant_training.ipynb` exists.
- Notebook text contains `# COOLANT Training — Stage 1 Multimodal Alignment`.
- Notebook text contains exactly one top-level assignment string `CONFIG = {`.
- Notebook text contains `"variant": "ResNetCOOLANT"`.
- Notebook text contains `"batch_size": 32`.
- Notebook text contains `"max_epochs": 30`.
- Notebook text contains `"patience": 7`.
- Notebook text contains `"negative_shift": 3`.
- Notebook text contains `"selection_metric": "val_accuracy"`.
- Notebook text contains `"smoke_test": False`.
- Notebook text contains `pip install mlflow seaborn scikit-learn tqdm h5py`.
- Notebook text contains `conda install -n fake_news mlflow seaborn scikit-learn tqdm h5py -c conda-forge`.
- Notebook text contains `from src.models.resnet_coolant import ResNetCOOLANT, patch_encoding, patch_clip_projection, patch_cnn_with_dropout`.
- Notebook text does not contain hardcoded absolute user paths.

</acceptance_criteria>

---

## Task 2 — Implement HDF5 validation, device setup, and dataloaders

**Type:** execute  
**Files:** `notebooks/pipeline/03_coolant_training.ipynb`  
**Requirements:** TRAIN-01, NB-02, NB-03

<read_first>

- `notebooks/pipeline/03_coolant_training.ipynb`
- `.planning/phases/02-preprocessing-notebook/02-PLAN.md`
- `.planning/phases/03-coolant-training-notebook-stage-1/03-CONTEXT.md`
- `src/processing/coolant/pair_dataset.py`
- `src/processing/hdf5_dataset.py`
- `src/utils/device.py`

</read_first>

<action>

Add notebook sections and code for input validation and dataloading:

1. Add markdown section `## Step 1: Validate Phase 2 HDF5 Inputs`.
2. Implement `validate_coolant_hdf5(path, split_name)` that opens each file with `h5py.File(path, "r")` and verifies required datasets:
   - `caption_features`
   - `image_features`
3. Validate shape constraints:
   - `caption_features.shape[1] == CONFIG["model"]["text_seq_len"]`
   - `caption_features.shape[2] == CONFIG["model"]["text_embed_dim"]`
   - `image_features.shape[1] == CONFIG["model"]["image_dim"]`
4. If any HDF5 file is missing, raise `FileNotFoundError` with this exact message fragment: `Run Phase 2 preprocessing first: notebooks/pipeline/02_preprocessing.ipynb`.
5. Print a dataframe summary with split, rows, caption shape, image shape, and file size in MB.
6. Add markdown section `## Step 2: Device, Seed, and DataLoaders`.
7. Implement `select_device()` with priority `cuda`, then `mps`, then `cpu`, and print selected device.
8. Implement `seed_everything(seed)` to seed Python `random`, NumPy, PyTorch, and CUDA when available.
9. Call `create_coolant_dataloaders(str(train_hdf5), str(dev_hdf5), str(test_hdf5), batch_size=CONFIG["training"]["batch_size"])`.
10. Add a smoke-test branch: if `CONFIG["safety"]["smoke_test"]` is true, wrap loaders so only `CONFIG["safety"].get("smoke_batches", 2)` batches are used.
11. Load and print one train batch shape, expecting caption `[B, 768, 128]` and image `[B, 2048]`.

</action>

<acceptance_criteria>

- Notebook text contains `def validate_coolant_hdf5`.
- Notebook text contains `caption_features`.
- Notebook text contains `image_features`.
- Notebook text contains `Run Phase 2 preprocessing first: notebooks/pipeline/02_preprocessing.ipynb`.
- Notebook text contains `def select_device`.
- Notebook text contains `torch.backends.mps.is_available()`.
- Notebook text contains `def seed_everything`.
- Notebook text contains `create_coolant_dataloaders(`.
- Notebook text contains `smoke_batches`.
- Notebook text contains `[B, 768, 128]`.
- Notebook text contains `[B, 2048]`.

</acceptance_criteria>

---

## Task 3 — Implement model setup, patching, losses, optimizers, and schedulers

**Type:** execute  
**Files:** `notebooks/pipeline/03_coolant_training.ipynb`  
**Requirements:** TRAIN-01, NB-03

<read_first>

- `notebooks/pipeline/03_coolant_training.ipynb`
- `src/models/resnet_coolant.py`
- `src/models/coolant_official.py`
- `src/processing/coolant/training_utils.py`
- `notebooks/all_stage_final/workflow_coolant/2b_train_phased.ipynb`
- `notebooks/all_stage_final/workflow_coolant/2a_train_simultaneous.ipynb`

</read_first>

<action>

Add notebook sections and code for model construction:

1. Add markdown section `## Step 3: Build and Patch ResNetCOOLANT`.
2. Implement `build_model(config, device)` that creates `model = ResNetCOOLANT(model_config)` where `model_config` contains these keys copied from `CONFIG["model"]` or `CONFIG["training"]`:
   - `shared_dim`
   - `sim_dim`
   - `clip_embed_dim`
   - `feature_dim`
   - `h_dim`
   - `lr`
   - `weight_decay`
   - `dropout`
3. In `build_model`, call these patch helpers exactly:
   - `patch_encoding(model.similarity_module.encoding, image_dim=CONFIG["model"]["image_dim"])`
   - `patch_encoding(model.detection_module.encoding, image_dim=CONFIG["model"]["image_dim"])`
   - `patch_encoding(model.detection_module.ambiguity_module.encoding, image_dim=CONFIG["model"]["image_dim"])`
   - `patch_clip_projection(model.clip_module, target_dim=CONFIG["model"]["image_dim"], is_image=True)`
   - `patch_clip_projection(model.clip_module, target_dim=CONFIG["model"]["text_embed_dim"], is_image=False)`
   - `patch_cnn_with_dropout(model.similarity_module.encoding.shared_text_encoding, CONFIG["model"]["text_embed_dim"], CONFIG["model"]["dropout"])`
   - `patch_cnn_with_dropout(model.detection_module.encoding.shared_text_encoding, CONFIG["model"]["text_embed_dim"], CONFIG["model"]["dropout"])`
   - `patch_cnn_with_dropout(model.detection_module.ambiguity_module.encoding.shared_text_encoding, CONFIG["model"]["text_embed_dim"], CONFIG["model"]["dropout"])`
4. Print parameter counts for total, trainable, `similarity_module`, `clip_module`, and `detection_module`.
5. Add markdown section `## Step 4: Losses, Optimizers, and Schedulers`.
6. Create losses:
   - `loss_cos = nn.CosineEmbeddingLoss(margin=CONFIG["loss"]["cosine_margin"])`
   - `loss_ce = nn.CrossEntropyLoss(label_smoothing=CONFIG["loss"]["label_smoothing"])`
   - `loss_kl = nn.KLDivLoss(reduction="batchmean")`
7. Create separate optimizers named exactly:
   - `optim_similarity`
   - `optim_clip`
   - `optim_detection`
8. Use warmup cosine scheduling per step. If no existing project helper exists, implement `make_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)` in the notebook using `torch.optim.lr_scheduler.LambdaLR`.
9. Add a one-batch forward/loss sanity cell that computes similarity, CLIP, and detection losses from a train batch and prints finite loss values.

</action>

<acceptance_criteria>

- Notebook text contains `def build_model`.
- Notebook text contains `model = ResNetCOOLANT(model_config)`.
- Notebook text contains `patch_encoding(model.similarity_module.encoding`.
- Notebook text contains `patch_encoding(model.detection_module.encoding`.
- Notebook text contains `patch_encoding(model.detection_module.ambiguity_module.encoding`.
- Notebook text contains `patch_clip_projection(model.clip_module, target_dim=CONFIG["model"]["image_dim"], is_image=True)`.
- Notebook text contains `patch_clip_projection(model.clip_module, target_dim=CONFIG["model"]["text_embed_dim"], is_image=False)`.
- Notebook text contains `patch_cnn_with_dropout(model.similarity_module.encoding.shared_text_encoding`.
- Notebook text contains `optim_similarity`.
- Notebook text contains `optim_clip`.
- Notebook text contains `optim_detection`.
- Notebook text contains `def make_warmup_cosine_scheduler`.
- Notebook text contains `nn.CosineEmbeddingLoss`.
- Notebook text contains `nn.KLDivLoss(reduction="batchmean")`.

</acceptance_criteria>

---

## Task 4 — Implement MLflow, checkpoint, manifest, and resume helpers

**Type:** execute  
**Files:** `notebooks/pipeline/03_coolant_training.ipynb`  
**Requirements:** TRAIN-02, TRAIN-03, TRAIN-05, NB-02, NB-03

<read_first>

- `notebooks/pipeline/03_coolant_training.ipynb`
- `.planning/phases/03-coolant-training-notebook-stage-1/03-CONTEXT.md`
- `.planning/phases/03-coolant-training-notebook-stage-1/03-RESEARCH.md`
- `notebooks/all_stage_final/train_vietnamese_coolant.ipynb`
- `notebooks/all_stage_final/4_train_model.ipynb`

</read_first>

<action>

Add notebook sections and helpers for experiment tracking and reproducible checkpoints:

1. Add markdown section `## Step 5: MLflow and Run Directory Setup`.
2. Compute `run_name = f"coolant_stage1_{timestamp}"` using `datetime.now().strftime("%Y%m%d_%H%M%S")`.
3. Create `run_dir = CONFIG["paths"]["checkpoint_root"] / run_name` and `artifact_dir = run_dir / "artifacts"`.
4. Configure MLflow tracking with `mlflow.set_tracking_uri((CONFIG["paths"]["mlflow_dir"]).as_uri())`.
5. Start MLflow experiment named exactly `coolant-stage1-training`.
6. Wrap MLflow setup/logging in try/except. On failure, set `mlflow_enabled = False` and print `MLflow disabled; continuing with local artifacts only`.
7. Add markdown section `## Step 6: Checkpoint and Manifest Helpers`.
8. Implement `config_to_jsonable(CONFIG)` for Path-safe config serialization.
9. Implement `compute_config_hash(config)` using `hashlib.sha256` over sorted JSON.
10. Implement `save_checkpoint(path, model, epoch, config, history, metrics, selection_metric, mlflow_run_id=None, optimizer_states=None, scheduler_states=None)`.
11. Ensure every saved checkpoint includes:
   - `model_state_dict`
   - `similarity_module_state_dict`
   - `clip_module_state_dict`
   - `detection_module_state_dict`
   - `config`
   - `config_hash`
   - `epoch`
   - `metrics`
   - `selection_metric`
   - `training_history`
   - `feature_dims`
   - `freeze_for_stage2=True`
   - `mlflow_run_id`
12. Implement periodic checkpoint saving as `checkpoint_epoch_{epoch}.pth` when `epoch % CONFIG["checkpointing"]["checkpoint_every"] == 0`.
13. Implement best checkpoint path `best_model.pth` selected by `val_accuracy`.
14. Implement optional secondary best checkpoint path `best_macro_f1.pth` selected by `val_macro_f1`.
15. Implement `write_checkpoint_manifest(manifest_path, best_checkpoint_path, best_epoch, best_metrics, config_hash, mlflow_enabled, mlflow_run_id)` that writes `checkpoint_manifest.json` with Stage 2 expected output keys:
   - `text_aligned_clip`
   - `image_aligned_clip`
   - `attention_weights`
   - `detection_logits`
   - `fake_prob`
16. Implement `load_training_checkpoint(path, model, optimizers=None, schedulers=None)` for explicit `RESUME_FROM_CHECKPOINT` only.

</action>

<acceptance_criteria>

- Notebook text contains `run_name = f"coolant_stage1_{timestamp}"`.
- Notebook text contains `coolant-stage1-training`.
- Notebook text contains `mlflow.set_tracking_uri`.
- Notebook text contains `MLflow disabled; continuing with local artifacts only`.
- Notebook text contains `def config_to_jsonable`.
- Notebook text contains `def compute_config_hash`.
- Notebook text contains `def save_checkpoint`.
- Notebook text contains `model_state_dict`.
- Notebook text contains `similarity_module_state_dict`.
- Notebook text contains `clip_module_state_dict`.
- Notebook text contains `detection_module_state_dict`.
- Notebook text contains `freeze_for_stage2`.
- Notebook text contains `checkpoint_epoch_{epoch}.pth`.
- Notebook text contains `best_model.pth`.
- Notebook text contains `best_macro_f1.pth`.
- Notebook text contains `def write_checkpoint_manifest`.
- Notebook text contains `checkpoint_manifest.json`.
- Notebook text contains `def load_training_checkpoint`.

</acceptance_criteria>

---

## Task 5 — Implement training, validation, failure handling, and per-epoch logging

**Type:** execute  
**Files:** `notebooks/pipeline/03_coolant_training.ipynb`  
**Requirements:** TRAIN-02, TRAIN-03, TRAIN-05, NB-03

<read_first>

- `notebooks/pipeline/03_coolant_training.ipynb`
- `src/processing/coolant/training_utils.py`
- `notebooks/all_stage_final/workflow_coolant/2a_train_simultaneous.ipynb`
- `notebooks/all_stage_final/workflow_coolant/2b_train_phased.ipynb`
- `notebooks/all_stage_final/workflow_coolant_adabelief/2b_train_adabelief_phased.ipynb`

</read_first>

<action>

Add notebook sections and code for training and validation:

1. Add markdown section `## Step 7: Training and Validation Functions`.
2. Implement `cleanup_memory(device, aggressive=False)` that calls `gc.collect()` and `torch.cuda.empty_cache()` when CUDA is active.
3. Implement `assert_finite_loss(loss, loss_components)` that raises `FloatingPointError` and prints the loss component dict if NaN/Inf appears.
4. Implement `compute_classification_metrics(y_true, y_pred, prefix)` returning keys:
   - `{prefix}_accuracy`
   - `{prefix}_macro_f1`
   - `{prefix}_real_precision`
   - `{prefix}_real_recall`
   - `{prefix}_fake_precision`
   - `{prefix}_fake_recall`
5. Implement `train_one_epoch(epoch, model, loaders, optimizers, schedulers, losses, device, config)` using:
   - `make_coolant_pairs(caption, image, shift=CONFIG["training"]["negative_shift"])`
   - `make_detection_batch(caption, image, shift=CONFIG["training"]["negative_shift"])`
   - skip and count batches where `caption.size(0) < CONFIG["training"]["min_batch_for_negatives"]`
   - gradient clipping with `CONFIG["training"]["grad_clip"]`
   - separate optimizer steps for similarity, CLIP, and detection modules
6. Implement `evaluate(model, loader, device, config, split_name)` that returns loss, accuracy, macro-F1, per-class precision/recall, and confusion matrix values.
7. Add markdown section `## Step 8: Run Training`.
8. Implement the epoch loop with early stopping on validation accuracy using `CONFIG["training"]["patience"]`.
9. Log per-epoch MLflow metrics when `mlflow_enabled` is true, including at minimum:
   - `train_loss`
   - `train_accuracy`
   - `train_macro_f1`
   - `val_loss`
   - `val_accuracy`
   - `val_macro_f1`
10. Append every epoch to a `history` list containing epoch, train/val loss, accuracy, macro-F1, learning rates, skipped batch count, and checkpoint path when saved.
11. Save `training_history.json` and `training_history.csv` under `artifact_dir` after every epoch.
12. Save periodic checkpoints every `CHECKPOINT_EVERY` epochs with optimizer and scheduler states.
13. Save `best_model.pth` whenever `val_accuracy` improves.
14. Save `best_macro_f1.pth` whenever `val_macro_f1` improves.
15. On CUDA OOM, catch `RuntimeError` containing `out of memory`, attempt to save `interrupted_epoch_{epoch}.pth`, and raise with the message fragment `CUDA OOM: lower CONFIG['training']['batch_size'] or enable CONFIG['safety']['smoke_test']`.
16. On `KeyboardInterrupt`, save `interrupted_epoch_{epoch}.pth`, end MLflow if active, and print `Set CONFIG['safety']['resume_from_checkpoint'] to this path to resume explicitly`.

</action>

<acceptance_criteria>

- Notebook text contains `def cleanup_memory`.
- Notebook text contains `def assert_finite_loss`.
- Notebook text contains `FloatingPointError`.
- Notebook text contains `def compute_classification_metrics`.
- Notebook text contains `def train_one_epoch`.
- Notebook text contains `make_coolant_pairs(caption, image, shift=CONFIG["training"]["negative_shift"])`.
- Notebook text contains `make_detection_batch(caption, image, shift=CONFIG["training"]["negative_shift"])`.
- Notebook text contains `skipped_batches`.
- Notebook text contains `def evaluate`.
- Notebook text contains `val_accuracy`.
- Notebook text contains `mlflow.log_metrics`.
- Notebook text contains `training_history.json`.
- Notebook text contains `training_history.csv`.
- Notebook text contains `CUDA OOM: lower CONFIG['training']['batch_size'] or enable CONFIG['safety']['smoke_test']`.
- Notebook text contains `Set CONFIG['safety']['resume_from_checkpoint'] to this path to resume explicitly`.

</acceptance_criteria>

---

## Task 6 — Implement best-checkpoint reload, test evaluation, plots, artifacts, and Stage 2 handoff

**Type:** execute  
**Files:** `notebooks/pipeline/03_coolant_training.ipynb`  
**Requirements:** TRAIN-02, TRAIN-04, TRAIN-05, NB-03

<read_first>

- `notebooks/pipeline/03_coolant_training.ipynb`
- `.planning/phases/03-coolant-training-notebook-stage-1/03-CONTEXT.md`
- `.planning/phases/03-coolant-training-notebook-stage-1/03-RESEARCH.md`
- `src/models/resnet_coolant.py`

</read_first>

<action>

Add notebook sections and code for post-training verification and handoff:

1. Add markdown section `## Step 9: Reload Best Checkpoint and Evaluate Test Split`.
2. Implement `load_best_model_for_eval(best_checkpoint_path, config, device)` that builds a fresh model with `build_model(config, device)`, loads `checkpoint["model_state_dict"]`, sets `model.eval()`, and returns the model plus checkpoint metadata.
3. Run final test evaluation using the reloaded best validation-accuracy checkpoint, not the in-memory training model.
4. Save `test_report.json` with test loss, accuracy, macro-F1, per-class precision/recall, and confusion matrix.
5. Add markdown section `## Step 10: Training Curves and Artifacts`.
6. Implement `plot_training_curves(history, artifact_dir)` that saves and displays:
   - `loss_curves.png`
   - `accuracy_curves.png`
   - `macro_f1_curves.png`
7. Implement `plot_confusion_matrix(confusion_matrix, artifact_dir)` that saves and displays `test_confusion_matrix.png`.
8. Log non-checkpoint artifacts to MLflow when `mlflow_enabled` is true:
   - `training_history.json`
   - `training_history.csv`
   - `test_report.json`
   - curve PNG files
   - `checkpoint_manifest.json`
9. Call `write_checkpoint_manifest(...)` after test evaluation. Ensure manifest includes:
   - `best_checkpoint_path`
   - `best_epoch`
   - `selection_metric: val_accuracy`
   - `freeze_for_stage2: true`
   - `expected_input_shapes.caption_features: [batch, 768, 128]`
   - `expected_input_shapes.image_features: [batch, 2048]`
   - `stage2_output_keys`
10. Add markdown section `## Step 11: Stage 2 Handoff Sanity Check`.
11. Implement `stage2_handoff_sanity_check(model, sample_batch, device)` that returns a dict containing keys:
   - `text_aligned_clip`
   - `image_aligned_clip`
   - `attention_weights`
   - `detection_logits`
   - `fake_prob`
12. Print each handoff output key and tensor shape from one test batch.
13. Add final summary cell printing exact labels:
   - `Best checkpoint:`
   - `Manifest:`
   - `Selection metric: val_accuracy`
   - `Phase 4 notebook: notebooks/pipeline/04_mm_vifactcheck_integration.ipynb`
14. End the MLflow run cleanly if active.

</action>

<acceptance_criteria>

- Notebook text contains `def load_best_model_for_eval`.
- Notebook text contains `checkpoint["model_state_dict"]`.
- Notebook text contains `test_report.json`.
- Notebook text contains `def plot_training_curves`.
- Notebook text contains `loss_curves.png`.
- Notebook text contains `accuracy_curves.png`.
- Notebook text contains `macro_f1_curves.png`.
- Notebook text contains `def plot_confusion_matrix`.
- Notebook text contains `test_confusion_matrix.png`.
- Notebook text contains `stage2_output_keys`.
- Notebook text contains `def stage2_handoff_sanity_check`.
- Notebook text contains `text_aligned_clip`.
- Notebook text contains `image_aligned_clip`.
- Notebook text contains `attention_weights`.
- Notebook text contains `detection_logits`.
- Notebook text contains `fake_prob`.
- Notebook text contains `Best checkpoint:`.
- Notebook text contains `Phase 4 notebook: notebooks/pipeline/04_mm_vifactcheck_integration.ipynb`.
- Notebook text contains `mlflow.end_run()`.

</acceptance_criteria>

</tasks>

---

<verification>

## Verification Steps

Run these checks after executing the plan:

1. `rtk test -f notebooks/pipeline/03_coolant_training.ipynb && echo notebook_exists`
2. `rtk grep -n "CONFIG = {" notebooks/pipeline/03_coolant_training.ipynb`
3. `rtk grep -n "val_accuracy" notebooks/pipeline/03_coolant_training.ipynb`
4. `rtk grep -n "checkpoint_manifest.json" notebooks/pipeline/03_coolant_training.ipynb`
5. `rtk grep -n "stage2_handoff_sanity_check" notebooks/pipeline/03_coolant_training.ipynb`
6. `rtk grep -n "/Users/" notebooks/pipeline/03_coolant_training.ipynb` must return no matches.
7. Run the notebook with `CONFIG["safety"]["smoke_test"] = True` after Phase 2 HDF5 outputs exist; verify it writes `best_model.pth`, `checkpoint_manifest.json`, `training_history.json`, and curve PNG artifacts under a timestamped `training/checkpoints_coolant/coolant_stage1_*` directory.

</verification>

---

<success_criteria>

- `notebooks/pipeline/03_coolant_training.ipynb` exists and is output-clean by default.
- The notebook has one top config cell and no hardcoded absolute local paths.
- The notebook validates Phase 2 HDF5 inputs before training.
- The notebook trains existing `ResNetCOOLANT` with dynamic negatives and existing project helpers.
- MLflow logs per-epoch loss, accuracy, and macro-F1 when available.
- Periodic checkpoints include embedded config and recovery state.
- `best_model.pth` is selected by validation accuracy and includes `freeze_for_stage2=True`.
- Training curves and test confusion matrix are displayed inline and saved as artifacts.
- Best checkpoint is reloaded before test evaluation.
- `checkpoint_manifest.json` documents the Phase 4 handoff contract.
- All requirements TRAIN-01 through TRAIN-05 and NB-01 through NB-03 are covered.

</success_criteria>

## PLANNING COMPLETE
