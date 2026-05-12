# Phase 4: MM-ViFactCheck Integration Notebook - Context

**Gathered:** 2026-05-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Build `notebooks/pipeline/04_mm_vifactcheck_integration.ipynb`, a thesis-ready Stage 2 integration notebook that: (1) loads the frozen COOLANT Stage 1 checkpoint from Phase 3, (2) pre-extracts per-article COOLANT features (text_aligned_clip + image_aligned_clip) into a cached HDF5 with ViFactCheck ground-truth labels, (3) trains a new GatedFusionHead classifier on those cached features across 4 ablation configs (A: text-only, B: image-only, C: concat, D: gated fusion), (4) evaluates each config on the test split, (5) renders and saves an ablation table, (6) exports a comprehensive JSON report for thesis documentation.

Out of scope: changing the COOLANT architecture, re-running Phase 2 preprocessing, building an inference API or UI, multi-GPU training, any new crawling or preprocessing steps.

</domain>

<decisions>
## Implementation Decisions

### Training Architecture
- **D-01:** Use a new `GatedFusionHead` module (not the existing COOLANT detection head). Takes `text_aligned_clip` (128-dim) and `image_aligned_clip` (128-dim) from frozen COOLANT as inputs. Projects each with `h_text_proj` and `h_mm_proj` linear layers, computes a learned gate, fuses, and classifies.
- **D-02:** COOLANT Stage 1 model stays fully frozen throughout Phase 4 — no fine-tuning of COOLANT weights. `freeze_for_stage2=True` from the Phase 3 checkpoint manifest is enforced.
- **D-03:** Stage 1 feature contract used: `text_aligned_clip` (128-dim) + `image_aligned_clip` (128-dim) only. `attention_weights` and `fake_prob` are not passed to the fusion head.
- **D-04:** Pre-extract all COOLANT features in a dedicated notebook cell before training. Saves `training/stage2_features/stage2_{split}.h5` with datasets `text_features` (128-dim), `image_features` (128-dim), and `labels`. Training loop reads only the small cached HDF5 — no COOLANT forward pass per epoch.

### Label Space & HDF5 Input
- **D-05:** Phase 4 creates its own enriched HDF5 files (`training/stage2_features/stage2_{split}.h5`) during feature pre-extraction. Does NOT modify Phase 2's `processed_data/hdf5/coolant_*.h5` outputs.
- **D-06:** Labels joined from root-labeled JSON files (`notebooks/data/json/news_data_vifactcheck_{split}_labeled.json`) using `article_ids` from Phase 2 HDF5. Label variant: `root` (raw ViFactCheck: 0=Supported, 1=Refuted, 2=NEI).
- **D-07:** Label space is config-switchable with `NUM_CLASSES` in the top config cell. Default = 2 (binary). When `NUM_CLASSES=2`, NEI (label 2) is remapped to 0 (Supported/Real) at dataset load time. When `NUM_CLASSES=3`, labels used as-is.
- **D-08:** Class-weighted CrossEntropyLoss used for all training to handle ViFactCheck class imbalance (Refuted is rare).

### Ablation Configs (A–D)
- **D-09:** Four ablation configs, all fully trained to convergence on the same cached stage2 features:
  - **Config A — text-only:** GatedFusionHead receives only `text_features` (128-dim). Image input zeroed or dropped.
  - **Config B — image-only:** GatedFusionHead receives only `image_features` (128-dim). Text input zeroed or dropped.
  - **Config C — concat (no gating):** GatedFusionHead receives concat([text, image], dim=-1) (256-dim), passed directly through a linear classifier — no learned gate.
  - **Config D — full gated fusion:** Full `GatedFusionHead` with learned gate over `text_aligned_clip` + `image_aligned_clip`.
- **D-10:** All 4 configs get their own MLflow run, training loop, best checkpoint, and test evaluation. Each config run name follows the pattern `stage2_{config_label}_{YYYYMMDD_HHMMSS}`.

### Training Hyperparameters
- **D-11:** Default full-run recipe: `MAX_EPOCHS=30`, `PATIENCE=7`, `BATCH_SIZE=32`, `AdamW lr=3e-4`, `weight_decay=1e-4`, `label_smoothing=0.1`, `grad_clip=1.0`.
- **D-12:** Scheduler: OneCycleLR with 5% warmup steps, mirroring Phase 3 and legacy notebook patterns.
- **D-13:** Include `SMOKE_TEST=False` config flag. When True: 5 epochs, 2 batches/split, skips full ablation loop — for local validation on MPS/CPU.
- **D-14:** `AUTO_INSTALL_DEPS=False` by default. Check required deps (torch, h5py, numpy, pandas, matplotlib, seaborn, tqdm, sklearn, mlflow); print exact install commands and stop clearly if missing.

### Checkpoint Selection
- **D-15:** Best checkpoint selected by **val macro-F1** (MMVF-04 requirement). Macro-F1 is appropriate for imbalanced ViFactCheck classes.
- **D-16:** Checkpoint saved as `best_model.pth` per config, under `training/checkpoints_stage2/{config}_{timestamp}/`.
- **D-17:** Checkpoint bundle includes: `model_state_dict`, config dict, epoch, val metrics (acc + macro-F1 + per-class), num_classes, config_label, mlflow_run_id when available.

### Results & Export
- **D-18:** Ablation table displayed as a Pandas DataFrame inline in the notebook AND saved to `training/stage2_results/ablation_table.csv`. Rows = configs A–D, cols = accuracy / macro-F1 / per-class F1 (Supported, Refuted, NEI if 3-class).
- **D-19:** Confusion matrix plotted as seaborn heatmap for config D (primary result), logged to MLflow when available.
- **D-20:** Final JSON export saved to `training/stage2_results/mm_vifactcheck_results.json`. Contains: `ablation_summary` (all 4 configs — accuracy, macro-F1, per-class F1, best epoch), `best_config` (config D — full sklearn classification report, confusion matrix as nested list, hyperparameters, best epoch, val macro-F1, stage1 checkpoint path, stage1 epoch).
- **D-21:** MLflow experiment name: `mm-vifactcheck-stage2`. Tracking dir: `notebooks/mlruns` (existing, consistent with Phase 3 D-26). Each config run logged separately.

### Notebook Structure & Config Contract
- **D-22:** Section flow: Overview → Config → Dependency/device checks → Stage 1 checkpoint loading → Feature pre-extraction → Stage 2 dataset/loaders → GatedFusionHead definition → Ablation training loop (A–D) → Results & ablation table → Confusion matrix plot → JSON export → Summary.
- **D-23:** Single top-level nested `CONFIG` dict: `paths` (stage1_checkpoint, stage1_manifest, hdf5_dir, stage2_features_dir, checkpoints_dir, results_dir, mlflow_dir), `model` (text_dim=128, image_dim=128, fusion_hidden_dim, num_classes), `training` (batch_size, max_epochs, patience, lr, weight_decay, label_smoothing, grad_clip), `mlflow` (experiment_name, tracking_uri), `safety` (smoke_test, auto_install_deps, seed).
- **D-24:** Stage 1 checkpoint loaded via `checkpoint_manifest.json` path (from Phase 3 D-20). If manifest is missing, fall back to direct path config.
- **D-25:** Concise thesis-friendly markdown before each major section. Provenance references to Phase 3 notebook and COOLANT workflow analysis.
- **D-26:** Output-clean source notebook. Committed without cell outputs; run artifacts carry execution outputs.
- **D-27:** device selection: `cuda > mps > cpu` using `src/utils/device.py` select_device() or equivalent. `num_workers=0` for HDF5 compatibility.

### Failure Handling
- **D-28:** If Phase 2 HDF5 is missing or `article_ids` don't match labeled JSON: fail clearly with instructions to run Phase 2 first and check `notebooks/data/json/`.
- **D-29:** If Phase 3 checkpoint is missing: fail with instructions to run Phase 3 first. Read from `checkpoint_manifest.json` path.
- **D-30:** If MLflow fails: warn and continue with local artifacts. Mark `mlflow_enabled=False` in results JSON.
- **D-31:** If feature pre-extraction HDF5 already exists: skip by default (consistent with Phase 2 D-33). Config flag `FORCE_REBUILD_FEATURES=False` to override.

### Claude's Discretion
- Training architecture: gated fusion (over legacy fine-tuning) — cleaner thesis story, better ablation clarity.
- Feature extraction: pre-extract once, cache to HDF5, train fusion head on cached features — faster iteration.
- Label variant: `root`, NEI remapping at load time via `NUM_CLASSES`.
- Hyperparameters: 30 epochs, patience 7, AdamW 3e-4, class-weighted CE, OneCycleLR 5% warmup.
- Ablation depth: all 4 configs trained fully — real numbers for thesis committee.
- Ablation table: DataFrame inline + CSV file.
- JSON export: full report (all configs + config D detailed metrics).
- Checkpoint directory: `training/checkpoints_stage2/{config}_{timestamp}/`, results at `training/stage2_results/`.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Roadmap and project scope
- `.planning/ROADMAP.md` — Phase 4 objective, deliverables, requirements MMVF-01 through MMVF-07 and NB-01 through NB-03.
- `.planning/PROJECT.md` — project constraints: config-driven notebooks, no hardcoded absolute paths, notebooks import shared `src/` utilities, ResNetCOOLANT as-is, HDF5 workflow.
- `.planning/phases/03-coolant-training-notebook-stage-1/03-CONTEXT.md` — Stage 1 output contract (D-52 through D-56): `text_aligned_clip`, `image_aligned_clip`, `attention_weights`, `detection_logits`, `fake_prob` shapes; `freeze_for_stage2=True`; checkpoint manifest schema.
- `.planning/phases/03-coolant-training-notebook-stage-1/03-SUMMARY.md` — Phase 3 deliverables and Stage 2 handoff details including checkpoint_manifest.json structure.
- `.planning/phases/02-preprocessing-notebook/02-CONTEXT.md` — Phase 2 HDF5 schema (D-15 through D-19): `caption_features` [N,max_len,768], `image_features` [N,2048], `article_ids`; split files at `processed_data/hdf5/coolant_{split}.h5`.

### Stage 2 legacy reference
- `notebooks/all_stage_final/research/9_stage2_vifactcheck_supervised.ipynb` — existing Stage 2 research notebook implementing two-phase fine-tuning (2a frozen backbone + 2b full fine-tune) on ViFactCheck labels. Reference for: `HDF5DatasetLabeled` pattern, class-weighted loss, OneCycleLR with 5% warmup, `apply_phase()` freeze/unfreeze helpers, MLflow logging pattern, confusion matrix display. Phase 4 diverges architecturally (gated fusion vs detection head) but inherits training loop patterns.

### COOLANT implementation assets
- `src/models/resnet_coolant.py` — `PatchedCOOLANT` / `ResNetCOOLANT` and patch helpers. Phase 4 loads this model from the Phase 3 checkpoint, applies the same patches, then freezes it for feature extraction.
- `src/processing/coolant/pair_dataset.py` — `CoolantPairDataset` for reading Phase 2 HDF5 during feature pre-extraction step.
- `src/utils/device.py` — `select_device()` for cuda/mps/cpu selection.

### Label data
- `notebooks/data/json/news_data_vifactcheck_train_labeled.json` — root-labeled train split with `article_id` and `label` fields for joining to Phase 2 HDF5.
- `notebooks/data/json/news_data_vifactcheck_dev_labeled.json` — root-labeled dev split.
- `notebooks/data/json/news_data_vifactcheck_test_labeled.json` — root-labeled test split.

### Notebook conventions (carry forward from Phase 3)
- `notebooks/pipeline/03_coolant_training.ipynb` — Phase 3 notebook; reference for config cell structure, dependency check pattern, MLflow setup, checkpoint saving conventions, and output-clean commit style.
- `docs/COOLANT_WORKFLOW_ANALYSIS.md` — documents paper vs implementation discrepancies; confirms milestone decision to use ResNetCOOLANT as-is.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `PatchedCOOLANT` / `ResNetCOOLANT` in `src/models/resnet_coolant.py`: load from Phase 3 checkpoint, apply same 6 patches (patch_encoding ×3, patch_clip_projection ×2, patch_cnn_with_dropout ×3), freeze all params, use for feature extraction only.
- `CoolantPairDataset` in `src/processing/coolant/pair_dataset.py`: reads Phase 2 `coolant_{split}.h5` with `caption_features` [N,768,128] and `image_features` [N,2048]. Use in pre-extraction step to iterate all articles through frozen COOLANT.
- `select_device()` in `src/utils/device.py`: existing cuda/mps/cpu detection, respects `FORCE_DEVICE` env var.
- Phase 3 pattern for `run_epoch_supervised` from legacy `9_stage2_vifactcheck_supervised.ipynb`: batch loop structure, `backbone_frozen` no_grad optimization, tqdm progress, metric accumulation — adapt for GatedFusionHead training.
- `HDF5DatasetLabeled` class from legacy notebook: pattern for loading HDF5 + label remapping by `NUM_CLASSES`. Adapt for Phase 4's `stage2_{split}.h5` schema.

### Established Patterns
- All pipeline notebooks: single top config cell, relative/config-driven paths, `notebooks/pipeline/` location, `AUTO_INSTALL_DEPS=False`, dependency checks with conda install instructions.
- Checkpointing: timestamped run dirs under `training/checkpoints_*/`, `best_model.pth` with full reproducibility bundle, `checkpoint_manifest.json`.
- MLflow: `notebooks/mlruns` tracking dir, per-epoch metric logging, graceful fallback on failure.
- HDF5: `num_workers=0` for all DataLoaders, skip existing outputs by default (`FORCE_REBUILD=False`).
- Failure handling: validate upstream dependencies (Phase 2 HDF5, Phase 3 checkpoint) before any computation; fail with clear instructions.

### Integration Points
- Input from Phase 3: `training/checkpoints_coolant/{run_name}/best_model.pth` + `checkpoint_manifest.json` (Stage 1 contract).
- Input from Phase 2: `processed_data/hdf5/coolant_{split}.h5` — `caption_features`, `image_features`, `article_ids` used during pre-extraction.
- Input labels: `notebooks/data/json/news_data_vifactcheck_{split}_labeled.json` — joined via `article_ids`.
- Phase 4 intermediate output: `training/stage2_features/stage2_{split}.h5` — `text_features` [N,128], `image_features` [N,128], `labels` [N].
- Phase 4 checkpoints: `training/checkpoints_stage2/{config}_{timestamp}/best_model.pth`.
- Phase 4 results: `training/stage2_results/ablation_table.csv` + `training/stage2_results/mm_vifactcheck_results.json`.
- Notebook output: `notebooks/pipeline/04_mm_vifactcheck_integration.ipynb`.
- MLflow: `notebooks/mlruns`, experiment `mm-vifactcheck-stage2`.

</code_context>

<specifics>
## Specific Ideas

- User explicitly selected "You decide" for training architecture, feature extraction strategy, training hyperparameters, label variant, ablation training depth, ablation table format, JSON export scope, and checkpoint directory — Claude has significant discretion in these areas.
- User confirmed: gated fusion (not legacy 2-phase fine-tune), val macro-F1 as selection metric, config-switchable label space (default binary), all 4 ablation configs train fully.
- The legacy `9_stage2_vifactcheck_supervised.ipynb` is the primary reference for training loop patterns even though Phase 4 diverges architecturally.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within Phase 4 scope.

</deferred>

---

*Phase: 4-MM-ViFactCheck Integration Notebook*
*Context gathered: 2026-05-12*
