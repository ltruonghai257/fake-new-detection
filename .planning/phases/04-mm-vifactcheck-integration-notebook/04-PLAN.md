---
phase: 4
plan: 04-PLAN
type: execute
wave: 1
depends_on:
  - Phase 3: COOLANT Training Notebook (Stage 1)
files_modified:
  - notebooks/pipeline/04_mm_vifactcheck_integration.ipynb
autonomous: true
requirements:
  - MMVF-01
  - MMVF-02
  - MMVF-03
  - MMVF-04
  - MMVF-05
  - MMVF-06
  - MMVF-07
  - NB-01
  - NB-02
  - NB-03
---

# Phase 4 Plan: MM-ViFactCheck Integration Notebook (Stage 2)

**Phase:** 4 — MM-ViFactCheck Integration Notebook (Stage 2)
**Goal:** Build `notebooks/pipeline/04_mm_vifactcheck_integration.ipynb`, a thesis-ready Stage 2 notebook that loads the frozen COOLANT Stage 1 checkpoint from Phase 3, pre-extracts per-pair COOLANT features (text_aligned_clip + image_aligned_clip) into a cached HDF5 joined with ViFactCheck root labels, trains four ablation configs (A text-only / B image-only / C concat / D full gated fusion) of a new `GatedFusionHead` with class-weighted CE + AdamW + OneCycleLR + macro-F1 early stop, evaluates each on the test split, renders an ablation table + confusion matrix, and exports a JSON results report for thesis documentation.
**Requirements:** MMVF-01, MMVF-02, MMVF-03, MMVF-04, MMVF-05, MMVF-06, MMVF-07, NB-01, NB-02, NB-03
**Planned:** 2026-05-12
**Status:** Ready to execute

---

<objective>
Create a clean thesis-ready MM-ViFactCheck Stage 2 notebook that (1) loads and freezes the Phase 3 ResNetCOOLANT checkpoint via the manifest path, (2) pre-extracts per-pair `text_aligned_clip` + `image_aligned_clip` features into a cached `stage2_{split}.h5` joined with ViFactCheck root labels by `article_id`, (3) defines a single `GatedFusionHead` with four selectable modes (text-only / image-only / concat / gated), (4) trains each ablation config with class-weighted CE + AdamW + OneCycleLR and saves best-by-val-macro-F1 checkpoints, (5) reloads each best checkpoint and evaluates on the test split, and (6) emits `ablation_table.csv`, `test_confusion_matrix.png`, and `mm_vifactcheck_results.json` for thesis documentation. The notebook keeps all tunable values in one `CONFIG` cell, uses relative/config-driven paths only, runs idempotently (skip cached features unless `FORCE_REBUILD_FEATURES`), and never installs packages unless `AUTO_INSTALL_DEPS=True`.
</objective>

---

## Must Haves

<must_haves>

**Roadmap and Requirement Truths**

- **MMVF-01:** Notebook loads the frozen PatchedCOOLANT checkpoint and extracts `text_aligned_clip` + `image_aligned_clip` per article-pair (Stage 1 D-52..D-56 contract; `attention_weights`/`fake_prob` are not used per D-03).
- **MMVF-02:** Resolved per D-03 — no separate Stage 2 PhoBERT encoder; PhoBERT-base-v2 token features from Phase 2 flow through COOLANT's text projector into `text_aligned_clip`. Notebook must include a markdown cell that documents this resolution against the original requirement text.
- **MMVF-03:** Notebook defines a `GatedFusionHead` with `h_text_proj` and `h_mm_proj` projection layers and a learned gate; selectable mode covers all four ablation configs.
- **MMVF-04:** Each ablation config is trained on ViFactCheck train split; best checkpoint per config is selected by `val_macro_f1`; early stop at `patience=7`.
- **MMVF-05:** Test-split evaluation produces accuracy, macro-F1, per-class precision / recall / F1, and a confusion matrix.
- **MMVF-06:** Ablation table compares configs A–D on accuracy, macro-F1, and per-class F1.
- **MMVF-07:** Final results exported to `training/stage2_results/mm_vifactcheck_results.json`.
- **NB-01:** One top `CONFIG` cell for all tunable parameters and paths.
- **NB-02:** All paths relative or `CONFIG`-driven; no hardcoded absolute local paths.
- **NB-03:** Clear markdown section headers explaining each step.

**Context Decision Coverage**

- **D-01:** New `GatedFusionHead` (not the COOLANT detection head) with `h_text_proj` + `h_mm_proj` projections, learned gate, fuse, classify — defined in Task 4.
- **D-02:** COOLANT Stage 1 fully frozen; `freeze_for_stage2=True` asserted from Phase 3 manifest — enforced in Task 2.
- **D-03:** Only `text_aligned_clip` + `image_aligned_clip` flow into the fusion head (no `attention_weights` / `fake_prob` consumed) — pre-extraction in Task 3.
- **D-04:** Pre-extract once into `training/stage2_features/stage2_{split}.h5` with `text_features` (128), `image_features` (128), `labels`; training reads only cached HDF5 — Task 3 + Task 4.
- **D-05:** Phase 4 writes its own HDF5; does NOT modify Phase 2 `processed_data/hdf5/coolant_*.h5` — Task 3.
- **D-06:** Labels joined from `notebooks/data/json/news_data_vifactcheck_{split}_labeled.json` via `article_ids`; root label variant — Task 3.
- **D-07:** `NUM_CLASSES` config-switchable; default 2 (NEI remapped to 0 at dataset load), `3` uses labels as-is — Task 1 (CONFIG) + Task 3 (remap) + Task 4 (dataset).
- **D-08:** Class-weighted `CrossEntropyLoss` (inverse-frequency) — Task 5.
- **D-09:** Four ablation configs (A `text_only` / B `image_only` / C `concat` / D `gated`), all fully trained on cached features — Task 4 (mode) + Task 5 (loop).
- **D-10:** Per-config MLflow run named `stage2_{config_label}_{YYYYMMDD_HHMMSS}` — Task 5.
- **D-11:** Defaults `BATCH_SIZE=32`, `MAX_EPOCHS=30`, `PATIENCE=7`, AdamW `lr=3e-4`, `weight_decay=1e-4`, `label_smoothing=0.1`, `grad_clip=1.0` — Task 1 (CONFIG) + Task 5 (apply).
- **D-12:** OneCycleLR with 5% warmup (`pct_start=0.05`) — Task 5.
- **D-13:** `SMOKE_TEST=False` default; True → 5 epochs / 2 batches per split / skip full ablation — Task 1 (CONFIG) + Task 3 / Task 5 (apply).
- **D-14:** `AUTO_INSTALL_DEPS=False` default; print exact `pip install` and `conda install` commands and raise `RuntimeError` — Task 1.
- **D-15:** Best checkpoint per config selected by `val_macro_f1` — Task 5.
- **D-16:** `best_model.pth` saved under `training/checkpoints_stage2/{config_label}_{timestamp}/` — Task 5.
- **D-17:** Checkpoint bundle: `model_state_dict`, `config_dict`, `config_label`, `mode`, `num_classes`, `epoch`, `metrics` (val acc + macro-F1 + per-class), `mlflow_run_id` — Task 5.
- **D-18:** Ablation table as inline Pandas DataFrame + saved to `training/stage2_results/ablation_table.csv`; rows A–D, columns accuracy / macro-F1 / per-class F1 — Task 6.
- **D-19:** Confusion matrix seaborn heatmap for config `gated`; logged to MLflow when enabled — Task 6.
- **D-20:** `training/stage2_results/mm_vifactcheck_results.json` with `ablation_summary` (all 4 configs) + `best_config` block (full sklearn `classification_report`, nested-list `confusion_matrix`, hyperparameters, best epoch, val macro-F1, stage1 checkpoint path, stage1 epoch) — Task 6.
- **D-21:** MLflow experiment name `mm-vifactcheck-stage2`; tracking dir `notebooks/mlruns`; per-config runs — Task 1 (CONFIG) + Task 5 (`set_tracking_uri` + `set_experiment`).
- **D-22:** Section flow Overview → Config → Dependency/device → Stage 1 checkpoint loading → Feature pre-extraction → Stage 2 dataset/loaders → GatedFusionHead → Ablation training loop (A–D) → Results & table → Confusion matrix → JSON export → Summary — Tasks 1 through 6.
- **D-23:** Single top-level nested `CONFIG` dict with groups `paths`, `model`, `training`, `mlflow`, `safety` — Task 1.
- **D-24:** Stage 1 checkpoint loaded via `checkpoint_manifest.json` path; fallback to direct path — Task 2 (`resolve_stage1_checkpoint`).
- **D-25:** Concise thesis-friendly markdown before each major section with provenance refs to Phase 3 + COOLANT_WORKFLOW_ANALYSIS — Tasks 1–6 (per-step markdown headers).
- **D-26:** Output-clean source notebook; run artifacts carry execution outputs — Task 1 (initial scaffold) + execution discipline.
- **D-27:** Device priority `cuda > mps > cpu` via `src/utils/device.get_device()`; `num_workers=0` — Task 1 (device setup) + Task 4 (loaders).
- **D-28:** Missing Phase 2 HDF5 / mismatched `article_ids` → fail clearly with instructions to run Phase 2 first — Task 3 (`validate_phase2_hdf5` + `article_id out of range` assert).
- **D-29:** Missing Phase 3 checkpoint → fail with instruction `Run Phase 3 first: notebooks/pipeline/03_coolant_training.ipynb` — Task 2.
- **D-30:** MLflow init/log failure → warn `MLflow disabled; continuing with local artifacts only`, continue with local artifacts, mark `mlflow_enabled: false` in results JSON — Task 5 (setup wrap) + Task 6 (results metadata).
- **D-31:** Feature pre-extraction skipped by default if cache exists; `FORCE_REBUILD_FEATURES=False` override — Task 1 (CONFIG) + Task 3 (skip-if-exists).

</must_haves>

---

<threat_model>

## Security, Reproducibility, and Data-Safety Considerations

| ID | Threat | Mitigation |
|----|--------|------------|
| **T-04-01** | Hardcoded absolute paths leak local environment / break on other machines | `PROJECT_ROOT = Path.cwd().parent.parent if Path.cwd().name == "pipeline" else Path.cwd()`; all paths derived from `CONFIG["paths"]` keys; acceptance grep verifies no `/Users/` literal. |
| **T-04-02** | Missing Phase 3 checkpoint silently produces a randomly-initialized COOLANT that "trains" and yields garbage thesis numbers | Explicit `manifest`/`checkpoint` existence check at top of loading cell; `assert ckpt.get("freeze_for_stage2") is True`; fail-fast `FileNotFoundError` with instruction to run Phase 3 first. |
| **T-04-03** | `article_id` join mismatch silently mis-pairs features to labels (Pitfall 1 in RESEARCH.md) | Pre-extraction cell asserts `max(article_ids) < len(articles_labeled)`; if the Phase 2 HDF5 stored `source_urls` / `titles`, cross-check `articles_labeled[article_id]["source_url"]` against HDF5 metadata for the first 5 pairs; fail with clear instructions on mismatch. |
| **T-04-04** | Ablation mode bug — e.g. config A "text-only" accidentally uses image input through unused-but-active layer | `GatedFusionHead.forward()` branches by `self.mode`; per-config test in plan acceptance asserts `mode` is set correctly per config and that disabled inputs are not used in `fused`. |
| **T-04-05** | Test-time data leak — evaluating with last-epoch in-memory model instead of saved best checkpoint inflates thesis numbers | Per-config evaluation reloads `best_model.pth` from disk before computing test metrics; acceptance grep checks the reload call appears. |
| **T-04-06** | MLflow run leak across ablation configs — all 4 configs log into one parent run, ablation comparison breaks | Per-config `mlflow.start_run(run_name=...)` wrapped in try/finally with `mlflow.end_run()`; acceptance grep verifies `mlflow.end_run` call. |
| **T-04-07** | Environment mutation — auto-installing deps changes the user's env without consent | `AUTO_INSTALL_DEPS=False` default; check imports; print exact `pip install` and `conda install -n fake_news ... -c conda-forge` commands and raise `RuntimeError`. |
| **T-04-08** | CUDA OOM during pre-extraction or training silently corrupts the run | Catch `RuntimeError` with `"out of memory"` substring; print recovery instructions (`lower CONFIG["training"]["batch_size"]` or enable `SMOKE_TEST=True`) and re-raise. |
| **T-04-09** | MLflow disabled mid-run breaks ablation aggregation | All MLflow calls wrapped in try/except; on failure, set `mlflow_enabled=False`, continue local artifact save, record `mlflow_enabled: false` in `mm_vifactcheck_results.json` metadata. |
| **T-04-10** | Accidental overwrite of previous thesis run | Each config writes to a fresh `training/checkpoints_stage2/{config_label}_{timestamp}/` directory; feature HDF5 skipped by default per D-31 unless `FORCE_REBUILD_FEATURES=True`. |

</threat_model>

---

<tasks>

## Task 1 — Notebook skeleton, CONFIG cell, dependency preflight, device + seed setup

**Type:** execute
**Files:** `notebooks/pipeline/04_mm_vifactcheck_integration.ipynb`
**Requirements:** NB-01, NB-02, NB-03

<read_first>

- `.planning/phases/04-mm-vifactcheck-integration-notebook/04-CONTEXT.md`
- `.planning/phases/04-mm-vifactcheck-integration-notebook/04-RESEARCH.md`
- `.planning/phases/03-coolant-training-notebook-stage-1/03-PLAN.md`
- `notebooks/pipeline/03_coolant_training.ipynb` (if produced by Phase 3 execution; otherwise consult Phase 3 PLAN tasks 1–2)
- `src/utils/device.py`
- `environment.yml`
- `requirements.txt`

</read_first>

<action>

Create `notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` with output-clean cells and these sections:

1. Markdown title: `# MM-ViFactCheck Integration — Stage 2 (Frozen COOLANT + Gated Fusion)`.
2. Overview markdown cell containing these exact strings:
   - `Input: training/checkpoints_coolant/<run_name>/best_model.pth (Phase 3 Stage 1 frozen checkpoint)`
   - `Input: processed_data/hdf5/coolant_{train,dev,test}.h5 (Phase 2 features)`
   - `Input: notebooks/data/json/news_data_vifactcheck_{train,dev,test}_labeled.json (ViFactCheck labels)`
   - `Output: training/stage2_features/stage2_{train,dev,test}.h5 (cached per-pair COOLANT outputs + labels)`
   - `Output: training/checkpoints_stage2/{config}_<timestamp>/best_model.pth`
   - `Output: training/stage2_results/{ablation_table.csv, test_confusion_matrix.png, mm_vifactcheck_results.json}`
3. Markdown sub-section explaining the MMVF-02 resolution (per Plan Must-Have): the original requirement text described a separate Stage 2 PhoBERT [Statement; SEP; Evidence] encoder; the locked architecture (D-03) uses the PhoBERT-base-v2 token features from Phase 2 that flow through COOLANT's text projector into `text_aligned_clip` (128-dim), eliminating a redundant second PhoBERT pass.
4. One top config cell containing exactly one top-level `CONFIG` dict with nested groups:
   - `paths`
   - `model`
   - `training`
   - `mlflow`
   - `safety`
5. Include these concrete default config values (exact string literals required):
   - `PROJECT_ROOT = Path.cwd().parent.parent if Path.cwd().name == "pipeline" else Path.cwd()`
   - `CONFIG["paths"]["hdf5_dir"] = PROJECT_ROOT / "processed_data" / "hdf5"`
   - `CONFIG["paths"]["train_hdf5"] = CONFIG["paths"]["hdf5_dir"] / "coolant_train.h5"`
   - `CONFIG["paths"]["dev_hdf5"] = CONFIG["paths"]["hdf5_dir"] / "coolant_dev.h5"`
   - `CONFIG["paths"]["test_hdf5"] = CONFIG["paths"]["hdf5_dir"] / "coolant_test.h5"`
   - `CONFIG["paths"]["labeled_json_dir"] = PROJECT_ROOT / "notebooks" / "data" / "json"`
   - `CONFIG["paths"]["stage1_checkpoint_root"] = PROJECT_ROOT / "training" / "checkpoints_coolant"`
   - `CONFIG["paths"]["stage1_manifest"] = None  # auto-detect newest checkpoint_manifest.json under stage1_checkpoint_root`
   - `CONFIG["paths"]["stage2_features_dir"] = PROJECT_ROOT / "training" / "stage2_features"`
   - `CONFIG["paths"]["stage2_checkpoint_root"] = PROJECT_ROOT / "training" / "checkpoints_stage2"`
   - `CONFIG["paths"]["stage2_results_dir"] = PROJECT_ROOT / "training" / "stage2_results"`
   - `CONFIG["paths"]["mlflow_dir"] = PROJECT_ROOT / "notebooks" / "mlruns"`
   - `CONFIG["model"]["text_dim"] = 128`
   - `CONFIG["model"]["image_dim"] = 128`
   - `CONFIG["model"]["fusion_hidden_dim"] = 256`
   - `CONFIG["model"]["num_classes"] = 2`
   - `CONFIG["model"]["dropout"] = 0.3`
   - `CONFIG["training"]["batch_size"] = 32`
   - `CONFIG["training"]["max_epochs"] = 30`
   - `CONFIG["training"]["patience"] = 7`
   - `CONFIG["training"]["lr"] = 3e-4`
   - `CONFIG["training"]["weight_decay"] = 1e-4`
   - `CONFIG["training"]["label_smoothing"] = 0.1`
   - `CONFIG["training"]["grad_clip"] = 1.0`
   - `CONFIG["training"]["onecycle_pct_start"] = 0.05`
   - `CONFIG["training"]["ablation_configs"] = ["text_only", "image_only", "concat", "gated"]`
   - `CONFIG["mlflow"]["experiment_name"] = "mm-vifactcheck-stage2"`
   - `CONFIG["safety"]["smoke_test"] = False`
   - `CONFIG["safety"]["smoke_batches"] = 2`
   - `CONFIG["safety"]["smoke_epochs"] = 5`
   - `CONFIG["safety"]["auto_install_deps"] = False`
   - `CONFIG["safety"]["force_rebuild_features"] = False`
   - `CONFIG["safety"]["seed"] = 42`
6. Add markdown section `## Step 1: Dependency Preflight`.
7. Add a dependency preflight cell that imports `torch`, `h5py`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`, `sklearn`, and `mlflow`. If any are missing AND `auto_install_deps` is false, print these exact commands and raise `RuntimeError`:
   - `pip install mlflow seaborn scikit-learn tqdm h5py`
   - `conda install -n fake_news mlflow seaborn scikit-learn tqdm h5py -c conda-forge`
8. Add markdown section `## Step 2: Device, Seed, and Project Setup`.
9. Insert `str(PROJECT_ROOT)` into `sys.path`; create `stage2_features_dir`, `stage2_checkpoint_root`, `stage2_results_dir`, `mlflow_dir` with `mkdir(parents=True, exist_ok=True)`.
10. Implement `select_device()` as a thin wrapper around `src.utils.device.get_device()` (priority cuda > mps > cpu) and print the selected device.
11. Implement `seed_everything(seed)` seeding `random`, NumPy, PyTorch, and CUDA when available.
12. Import `ResNetCOOLANT`, `patch_encoding`, `patch_clip_projection`, `patch_cnn_with_dropout` from `src.models.resnet_coolant`; import `CoolantPairDataset` from `src.processing.coolant.pair_dataset`.

</action>

<acceptance_criteria>

- `notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` exists.
- Notebook text contains `# MM-ViFactCheck Integration — Stage 2 (Frozen COOLANT + Gated Fusion)`.
- Notebook text contains exactly one top-level assignment string `CONFIG = {`.
- Notebook text contains `"fusion_hidden_dim": 256`.
- Notebook text contains `"num_classes": 2`.
- Notebook text contains `"batch_size": 32`.
- Notebook text contains `"max_epochs": 30`.
- Notebook text contains `"patience": 7`.
- Notebook text contains `"onecycle_pct_start": 0.05`.
- Notebook text contains `"experiment_name": "mm-vifactcheck-stage2"`.
- Notebook text contains `"smoke_test": False`.
- Notebook text contains `"force_rebuild_features": False`.
- Notebook text contains `"ablation_configs": ["text_only", "image_only", "concat", "gated"]`.
- Notebook text contains `pip install mlflow seaborn scikit-learn tqdm h5py`.
- Notebook text contains `conda install -n fake_news mlflow seaborn scikit-learn tqdm h5py -c conda-forge`.
- Notebook text contains `from src.models.resnet_coolant import ResNetCOOLANT, patch_encoding, patch_clip_projection, patch_cnn_with_dropout`.
- Notebook text contains `from src.processing.coolant.pair_dataset import CoolantPairDataset`.
- Notebook text contains `from src.utils.device import get_device`.
- Notebook text contains `def seed_everything`.
- Notebook text contains `seed_everything(CONFIG["safety"]["seed"])`.
- Notebook text contains `## Step 1: Dependency Preflight`.
- Notebook text contains `## Step 2: Device, Seed, and Project Setup`.
- `rtk grep -c "/Users/" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` returns `0`.

</acceptance_criteria>

---

## Task 2 — Load and freeze Phase 3 Stage 1 checkpoint (manifest-first, 6-patch chain)

**Type:** execute
**Files:** `notebooks/pipeline/04_mm_vifactcheck_integration.ipynb`
**Requirements:** MMVF-01, NB-02, NB-03

<read_first>

- `notebooks/pipeline/04_mm_vifactcheck_integration.ipynb`
- `.planning/phases/04-mm-vifactcheck-integration-notebook/04-CONTEXT.md`
- `.planning/phases/04-mm-vifactcheck-integration-notebook/04-RESEARCH.md`
- `.planning/phases/03-coolant-training-notebook-stage-1/03-PLAN.md`
- `.planning/phases/03-coolant-training-notebook-stage-1/03-CONTEXT.md`
- `src/models/resnet_coolant.py`

</read_first>

<action>

Add notebook sections and code for Stage 1 loading:

1. Add markdown section `## Step 3: Load Frozen COOLANT Stage 1 Checkpoint`.
2. Implement `resolve_stage1_checkpoint(config)` that:
   - If `config["paths"]["stage1_manifest"]` is not None and the file exists, read `json.loads(...)` and return `Path(manifest["best_checkpoint_path"])`.
   - Else search `config["paths"]["stage1_checkpoint_root"]` recursively for the newest file named `checkpoint_manifest.json` and return the `best_checkpoint_path` recorded in it.
   - If no manifest or checkpoint is found, raise `FileNotFoundError` containing this exact message fragment: `Run Phase 3 first: notebooks/pipeline/03_coolant_training.ipynb`.
3. Implement `load_frozen_coolant(checkpoint_path, device)` that:
   - Calls `ckpt = torch.load(checkpoint_path, map_location=device)`.
   - Asserts `ckpt.get("freeze_for_stage2") is True` with error message `Phase 3 checkpoint missing freeze_for_stage2 flag — re-train Stage 1`.
   - Builds `model = ResNetCOOLANT(ckpt["config"])`.
   - Applies the 6 patches in this exact order (image_dim, text_embed_dim, dropout values pulled from `ckpt["config"]` or sensible defaults `2048`, `768`, `0.3`):
     - `patch_encoding(model.similarity_module.encoding, image_dim=...)`
     - `patch_encoding(model.detection_module.encoding, image_dim=...)`
     - `patch_encoding(model.detection_module.ambiguity_module.encoding, image_dim=...)`
     - `patch_clip_projection(model.clip_module, target_dim=..., is_image=True)`
     - `patch_clip_projection(model.clip_module, target_dim=..., is_image=False)`
     - `patch_cnn_with_dropout(model.similarity_module.encoding.shared_text_encoding, text_embed_dim, dropout)`
     - `patch_cnn_with_dropout(model.detection_module.encoding.shared_text_encoding, text_embed_dim, dropout)`
     - `patch_cnn_with_dropout(model.detection_module.ambiguity_module.encoding.shared_text_encoding, text_embed_dim, dropout)`
   - Calls `model.load_state_dict(ckpt["model_state_dict"])` (use `strict=True`; raise on mismatch).
   - Calls `model.eval()` and sets `param.requires_grad = False` for every parameter.
   - Moves the model to `device` and returns `(model, ckpt)`.
4. Call the loader; print a one-line summary: stage1 checkpoint path, stage1 epoch (`ckpt.get("epoch", "unknown")`), stage1 val_accuracy (`ckpt.get("metrics", {}).get("val_accuracy", "?")`), and total/trainable parameter counts (trainable must be 0).
5. Discover Stage 1 forward output structure: run one minibatch from `CoolantPairDataset(train_hdf5)` (batch_size=2 via `torch.utils.data.DataLoader`) through `model(caption, image)`. Detect whether the return is a `dict` / `tuple` / object with attributes by branching on `isinstance(out, dict)` / `hasattr(out, "text_aligned_clip")`; pick the access path that yields a `[B, 128]` tensor for both `text_aligned_clip` and `image_aligned_clip`. Cache the chosen access mode in a local variable `STAGE1_OUTPUT_MODE` (one of `"dict"` / `"attrs"` / `"tuple"`) for use in Task 3.
6. Print the resolved shapes of both 128-dim outputs to confirm the contract.

</action>

<acceptance_criteria>

- Notebook text contains `## Step 3: Load Frozen COOLANT Stage 1 Checkpoint`.
- Notebook text contains `def resolve_stage1_checkpoint`.
- Notebook text contains `checkpoint_manifest.json`.
- Notebook text contains `Run Phase 3 first: notebooks/pipeline/03_coolant_training.ipynb`.
- Notebook text contains `def load_frozen_coolant`.
- Notebook text contains `freeze_for_stage2`.
- Notebook text contains `Phase 3 checkpoint missing freeze_for_stage2 flag`.
- Notebook text contains `ResNetCOOLANT(ckpt["config"])`.
- Notebook text contains `patch_encoding(model.similarity_module.encoding`.
- Notebook text contains `patch_encoding(model.detection_module.encoding`.
- Notebook text contains `patch_encoding(model.detection_module.ambiguity_module.encoding`.
- Notebook text contains `patch_clip_projection(model.clip_module, target_dim=`.
- Notebook text contains `is_image=True`.
- Notebook text contains `is_image=False`.
- Notebook text contains `patch_cnn_with_dropout(model.similarity_module.encoding.shared_text_encoding`.
- Notebook text contains `model.load_state_dict(ckpt["model_state_dict"])`.
- Notebook text contains `param.requires_grad = False`.
- Notebook text contains `model.eval()`.
- Notebook text contains `STAGE1_OUTPUT_MODE`.
- Notebook text contains `text_aligned_clip`.
- Notebook text contains `image_aligned_clip`.

</acceptance_criteria>

---

## Task 3 — Pre-extract per-pair COOLANT features into stage2_{split}.h5 (with label join + validation guards)

**Type:** execute
**Files:** `notebooks/pipeline/04_mm_vifactcheck_integration.ipynb`
**Requirements:** MMVF-01, MMVF-02 (resolved per D-03), NB-02, NB-03

<read_first>

- `notebooks/pipeline/04_mm_vifactcheck_integration.ipynb`
- `.planning/phases/04-mm-vifactcheck-integration-notebook/04-CONTEXT.md`
- `.planning/phases/04-mm-vifactcheck-integration-notebook/04-RESEARCH.md`
- `src/processing/coolant/pair_dataset.py`
- `src/processing/coolant/pair_extractor.py`
- `notebooks/data/json/news_data_vifactcheck_dev_labeled.json` (sample only, do not load full file unless needed)

</read_first>

<action>

Add notebook sections and code for feature pre-extraction and label join:

1. Add markdown section `## Step 4: Pre-extract COOLANT Features and Join ViFactCheck Labels`.
2. Implement `load_labeled_articles(labeled_json_path)` that returns the list of article dicts via `json.load`, and asserts every element has a `label` field in `{0, 1, 2}` (raise `ValueError` with message containing `Invalid label space in {labeled_json_path}; expected 0/1/2` on failure).
3. Implement `validate_phase2_hdf5(hdf5_path, split_name)` that opens with `h5py.File(path, "r")` and verifies datasets `caption_features`, `image_features`, `article_ids`. Raise `FileNotFoundError` with this exact message fragment if missing: `Run Phase 2 preprocessing first: notebooks/pipeline/02_preprocessing.ipynb`.
4. Implement `extract_and_cache_stage2_features(split_name, hdf5_path, labeled_json_path, model, config, device, output_path)` that:
   - If `output_path` exists AND `config["safety"]["force_rebuild_features"]` is False: print `[stage2 cache] {split_name}: using existing {output_path}` and return early.
   - Loads articles via `load_labeled_articles(labeled_json_path)`.
   - Creates `ds = CoolantPairDataset(str(hdf5_path))` and a DataLoader with `batch_size=config["training"]["batch_size"]`, `shuffle=False`, `num_workers=0`.
   - Asserts every `article_id` returned by the dataset satisfies `0 <= article_id < len(articles_labeled)`; on failure raise `ValueError` containing the exact substring `article_id out of range — Phase 2 HDF5 and labeled JSON are not aligned`.
   - Iterates the loader under `torch.no_grad()` with tqdm; for each batch, runs `out = model(caption, image)` and uses `STAGE1_OUTPUT_MODE` from Task 2 to extract `text_aligned_clip` and `image_aligned_clip` (both `[B, 128]`).
   - For each item in the batch, looks up `raw_label = articles_labeled[article_id]["label"]`. When `config["model"]["num_classes"] == 2`: remap `2 -> 0` (NEI → Supported); when `num_classes == 3`: keep as-is.
   - Accumulates lists and at the end writes a fresh HDF5 with `text_features` (`float32`, `[N, 128]`), `image_features` (`float32`, `[N, 128]`), `labels` (`int64`, `[N]`), `article_ids` (`int64`, `[N]`).
   - Records HDF5 attributes: `n_samples`, `num_classes`, `stage1_checkpoint_path`, `stage1_epoch`.
   - If `config["safety"]["smoke_test"]` is True, stop after `config["safety"]["smoke_batches"]` batches and tag the file via attribute `smoke_test=True`.
5. Add a smoke-or-full driver cell that calls the extractor for each split in order `train`, `dev`, `test` using paths derived from `CONFIG["paths"]` (no hardcoded literals) and prints a summary table with split, n_samples, label histogram (`numpy.bincount`), and output path.
6. After extraction, run a sanity-check cell: for each split, open `stage2_{split}.h5` with h5py, assert `text_features.shape[1] == 128` and `image_features.shape[1] == 128`, and print the first 3 `article_ids` along with their resolved labels.

</action>

<acceptance_criteria>

- Notebook text contains `## Step 4: Pre-extract COOLANT Features and Join ViFactCheck Labels`.
- Notebook text contains `def load_labeled_articles`.
- Notebook text contains `Invalid label space in`.
- Notebook text contains `def validate_phase2_hdf5`.
- Notebook text contains `Run Phase 2 preprocessing first: notebooks/pipeline/02_preprocessing.ipynb`.
- Notebook text contains `def extract_and_cache_stage2_features`.
- Notebook text contains `force_rebuild_features`.
- Notebook text contains `article_id out of range — Phase 2 HDF5 and labeled JSON are not aligned`.
- Notebook text contains `CoolantPairDataset(str(hdf5_path))`.
- Notebook text contains `torch.no_grad()`.
- Notebook text contains `text_features`.
- Notebook text contains `image_features`.
- Notebook text contains `stage2_{split}.h5` (literal pattern as an f-string template or via `format`).
- Notebook text contains `articles_labeled[article_id]["label"]` (literal token sequence).
- Notebook text contains `num_classes == 2`.
- Notebook text contains `stage1_checkpoint_path`.
- Notebook text contains `numpy.bincount` or `np.bincount`.

</acceptance_criteria>

---

## Task 4 — Stage 2 Dataset, DataLoaders, GatedFusionHead module with 4 ablation modes

**Type:** execute
**Files:** `notebooks/pipeline/04_mm_vifactcheck_integration.ipynb`
**Requirements:** MMVF-03, NB-03

<read_first>

- `notebooks/pipeline/04_mm_vifactcheck_integration.ipynb`
- `.planning/phases/04-mm-vifactcheck-integration-notebook/04-CONTEXT.md`
- `.planning/phases/04-mm-vifactcheck-integration-notebook/04-RESEARCH.md`
- `notebooks/all_stage_final/research/9_stage2_vifactcheck_supervised.ipynb` (reference only — adapt `HDF5DatasetLabeled` to cached 128-dim features)

</read_first>

<action>

Add notebook sections and code for the cached-feature dataset and the fusion head:

1. Add markdown section `## Step 5: Stage 2 Cached-Feature Dataset and DataLoaders`.
2. Implement `class HDF5DatasetStage2(torch.utils.data.Dataset)` that:
   - Takes `(hdf5_path, num_classes)`.
   - Opens the file once in `__init__`, loads `text_features` / `image_features` / `labels` into in-memory numpy arrays (small after pre-extraction).
   - When `num_classes == 2`, remaps any `label == 2` to `0`.
   - Stores `self.labels` as `np.int64`.
   - `__getitem__(idx)` returns `(text_tensor, image_tensor, label_int)` with `text_tensor` and `image_tensor` as `torch.float32` and label as Python `int`.
3. Implement `create_stage2_dataloaders(config)` that returns `loaders, datasets` dicts keyed by `train`, `dev`, `test`. Each loader uses `batch_size=config["training"]["batch_size"]`, `num_workers=0`, `shuffle=True` for train, `False` for dev/test. If `config["safety"]["smoke_test"]` is True, wrap each loader to yield at most `config["safety"]["smoke_batches"]` batches.
4. Print one batch shape from train (`text [B, 128]`, `image [B, 128]`, `label [B]`) and a label histogram for the train set.
5. Add markdown section `## Step 6: GatedFusionHead Module`.
6. Implement:
   ```python
   class GatedFusionHead(nn.Module):
       MODES = ("text_only", "image_only", "concat", "gated")
       def __init__(self, text_dim, image_dim, fusion_hidden_dim, num_classes, mode, dropout):
           super().__init__()
           assert mode in self.MODES, f"Unknown mode {mode}"
           self.mode = mode
           self.h_text_proj = nn.Linear(text_dim, fusion_hidden_dim)
           self.h_mm_proj   = nn.Linear(image_dim, fusion_hidden_dim)
           if mode == "concat":
               self.classifier = nn.Sequential(
                   nn.Dropout(dropout),
                   nn.Linear(2 * fusion_hidden_dim, num_classes),
               )
           else:
               if mode == "gated":
                   self.gate = nn.Linear(2 * fusion_hidden_dim, fusion_hidden_dim)
               self.classifier = nn.Sequential(
                   nn.Dropout(dropout),
                   nn.Linear(fusion_hidden_dim, num_classes),
               )

       def forward(self, text_feat, image_feat):
           h_t = torch.relu(self.h_text_proj(text_feat))
           h_m = torch.relu(self.h_mm_proj(image_feat))
           if self.mode == "text_only":
               fused = h_t
           elif self.mode == "image_only":
               fused = h_m
           elif self.mode == "concat":
               fused = torch.cat([h_t, h_m], dim=-1)
           else:
               g = torch.sigmoid(self.gate(torch.cat([h_t, h_m], dim=-1)))
               fused = g * h_t + (1 - g) * h_m
           return self.classifier(fused)
   ```
7. Add a one-batch sanity-forward cell that builds `GatedFusionHead(mode="gated")` and runs a forward on one batch; assert the output shape equals `[B, num_classes]`.

</action>

<acceptance_criteria>

- Notebook text contains `## Step 5: Stage 2 Cached-Feature Dataset and DataLoaders`.
- Notebook text contains `class HDF5DatasetStage2`.
- Notebook text contains `def create_stage2_dataloaders`.
- Notebook text contains `## Step 6: GatedFusionHead Module`.
- Notebook text contains `class GatedFusionHead`.
- Notebook text contains `h_text_proj`.
- Notebook text contains `h_mm_proj`.
- Notebook text contains `text_only`.
- Notebook text contains `image_only`.
- Notebook text contains `concat`.
- Notebook text contains `gated`.
- Notebook text contains `torch.sigmoid(self.gate(`.
- Notebook text contains `Unknown mode`.
- Notebook text contains `num_workers=0`.
- Notebook text contains `smoke_batches`.

</acceptance_criteria>

---

## Task 5 — Training loop, class weights, AdamW + OneCycleLR, best-by-macro-F1 checkpointing

**Type:** execute
**Files:** `notebooks/pipeline/04_mm_vifactcheck_integration.ipynb`
**Requirements:** MMVF-04, MMVF-05, NB-03

<read_first>

- `notebooks/pipeline/04_mm_vifactcheck_integration.ipynb`
- `.planning/phases/04-mm-vifactcheck-integration-notebook/04-CONTEXT.md`
- `.planning/phases/04-mm-vifactcheck-integration-notebook/04-RESEARCH.md`
- `notebooks/all_stage_final/research/9_stage2_vifactcheck_supervised.ipynb`

</read_first>

<action>

Add notebook sections and code for the training loop:

1. Add markdown section `## Step 7: Training Utilities (Class Weights, Train/Eval Functions)`.
2. Implement `compute_class_weights(train_labels, num_classes, device)` using inverse-frequency:
   `class_weights = len(train_labels) / (num_classes * np.bincount(train_labels, minlength=num_classes).astype(np.float64))`
   and return as a `torch.float32` tensor on `device`.
3. Implement `compute_classification_metrics(y_true, y_pred, num_classes, prefix)` using `sklearn.metrics` with `zero_division=0`. Return a dict containing keys `{prefix}_accuracy`, `{prefix}_macro_f1`, and per-class F1 keys `{prefix}_f1_class{i}` for `i in range(num_classes)`.
4. Implement `run_epoch(head, loader, criterion, optimizer, scheduler, device, train, grad_clip)` that:
   - Sets `head.train()` or `head.eval()`.
   - Iterates `(text, image, label)`, moves to `device`, casts label to `torch.long`.
   - Runs forward, computes loss, accumulates predictions.
   - Under train: `loss.backward()`, gradient clip via `torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)`, `optimizer.step()`, `scheduler.step()`, `optimizer.zero_grad()`.
   - Returns mean loss, `y_true_array`, `y_pred_array`.
5. Implement `assert_finite_loss(loss)` that raises `FloatingPointError` when `not torch.isfinite(loss)`.
6. Add markdown section `## Step 8: Ablation Training Loop (Configs A–D)`.
7. Before the ablation loop, configure MLflow once: call `mlflow.set_tracking_uri(CONFIG["paths"]["mlflow_dir"].as_uri())` and `mlflow.set_experiment(CONFIG["mlflow"]["experiment_name"])`. Wrap both calls in try/except; on failure set `mlflow_enabled = False`, print `MLflow disabled; continuing with local artifacts only`, and continue.
8. Implement `train_one_config(config_label, mode, datasets, loaders, class_weights_t, config, device, mlflow_enabled, run_dir_root)` that:
   - Creates a fresh `head = GatedFusionHead(text_dim=128, image_dim=128, fusion_hidden_dim=CONFIG["model"]["fusion_hidden_dim"], num_classes=CONFIG["model"]["num_classes"], mode=mode, dropout=CONFIG["model"]["dropout"]).to(device)`.
   - Creates `optimizer = torch.optim.AdamW(head.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])`.
   - Creates `scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG["training"]["lr"], total_steps=len(loaders["train"]) * effective_epochs, pct_start=CONFIG["training"]["onecycle_pct_start"])`.
   - `criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=CONFIG["training"]["label_smoothing"])`.
   - Creates `run_dir = run_dir_root / f"{config_label}_{timestamp}"`; `run_dir.mkdir(parents=True, exist_ok=True)`.
   - If `mlflow_enabled`, calls `mlflow.start_run(run_name=f"stage2_{config_label}_{timestamp}")` wrapped in try/finally with `mlflow.end_run()` in the finally block.
   - Trains for up to `effective_epochs` (full `CONFIG["training"]["max_epochs"]` or `CONFIG["safety"]["smoke_epochs"]` when smoke) with early stopping when `val_macro_f1` has not improved for `CONFIG["training"]["patience"]` epochs.
   - After every epoch logs `train_loss`, `train_accuracy`, `train_macro_f1`, `val_loss`, `val_accuracy`, `val_macro_f1` to MLflow when enabled.
   - Saves `best_model.pth` whenever `val_macro_f1` strictly improves; the saved bundle contains:
     - `model_state_dict`
     - `config_dict` (full `CONFIG` snapshot via `json.loads(json.dumps(CONFIG, default=str))`)
     - `config_label`
     - `mode`
     - `num_classes`
     - `epoch`
     - `metrics` (latest val metrics dict)
     - `mlflow_run_id` (or `None` when MLflow disabled)
   - On `RuntimeError` with `out of memory`: prints `CUDA OOM: lower CONFIG['training']['batch_size'] or enable CONFIG['safety']['smoke_test']` and re-raises.
   - On NaN/Inf loss: `assert_finite_loss(loss)` raises and the run aborts.
   - Returns `(best_epoch, best_val_metrics, run_dir, best_checkpoint_path)`.
9. Add a one-batch sanity-train cell that runs `train_one_config(... mode="gated" ...)` for **2 batches** when `SMOKE_TEST` is True before kicking off the full ablation loop.
10. Build `train_labels_all = np.asarray(datasets["train"].labels, dtype=np.int64)` from the dataset created in Task 4, then `class_weights_t = compute_class_weights(train_labels_all, CONFIG["model"]["num_classes"], device)`. Print the resulting weight vector.
11. Iterate `for config_label, mode in zip(CONFIG["training"]["ablation_configs"], CONFIG["training"]["ablation_configs"]): train_one_config(...)` (each entry in `ablation_configs` is both the config label and the `GatedFusionHead.mode`); aggregate per-config best results in a Python dict `ablation_results[config_label]` containing `best_epoch`, `best_checkpoint_path`, `val_metrics`, and `run_dir`.

</action>

<acceptance_criteria>

- Notebook text contains `## Step 7: Training Utilities (Class Weights, Train/Eval Functions)`.
- Notebook text contains `def compute_class_weights`.
- Notebook text contains `class_weights = len(train_labels) / (num_classes * np.bincount(train_labels, minlength=num_classes).astype(np.float64))`.
- Notebook text contains `def compute_classification_metrics`.
- Notebook text contains `zero_division=0`.
- Notebook text contains `def run_epoch`.
- Notebook text contains `clip_grad_norm_`.
- Notebook text contains `def assert_finite_loss`.
- Notebook text contains `FloatingPointError`.
- Notebook text contains `## Step 8: Ablation Training Loop (Configs A–D)`.
- Notebook text contains `def train_one_config`.
- Notebook text contains `torch.optim.AdamW`.
- Notebook text contains `torch.optim.lr_scheduler.OneCycleLR`.
- Notebook text contains `pct_start=CONFIG["training"]["onecycle_pct_start"]`.
- Notebook text contains `nn.CrossEntropyLoss(weight=class_weights_t`.
- Notebook text contains `label_smoothing=CONFIG["training"]["label_smoothing"]`.
- Notebook text contains `val_macro_f1`.
- Notebook text contains `best_model.pth`.
- Notebook text contains `mlflow.set_tracking_uri(CONFIG["paths"]["mlflow_dir"].as_uri())`.
- Notebook text contains `mlflow.set_experiment(CONFIG["mlflow"]["experiment_name"])`.
- Notebook text contains `MLflow disabled; continuing with local artifacts only`.
- Notebook text contains `mlflow.start_run(run_name=f"stage2_{config_label}_{timestamp}")`.
- Notebook text contains `mlflow.end_run()`.
- Notebook text contains `CUDA OOM: lower CONFIG['training']['batch_size'] or enable CONFIG['safety']['smoke_test']`.
- Notebook text contains `ablation_results[config_label]`.

</acceptance_criteria>

---

## Task 6 — Reload best checkpoints, run test eval, build ablation table, confusion matrix, JSON export, summary

**Type:** execute
**Files:** `notebooks/pipeline/04_mm_vifactcheck_integration.ipynb`
**Requirements:** MMVF-05, MMVF-06, MMVF-07, NB-02, NB-03

<read_first>

- `notebooks/pipeline/04_mm_vifactcheck_integration.ipynb`
- `.planning/phases/04-mm-vifactcheck-integration-notebook/04-CONTEXT.md`
- `.planning/phases/04-mm-vifactcheck-integration-notebook/04-RESEARCH.md`

</read_first>

<action>

Add notebook sections and code for evaluation, ablation aggregation, and export:

1. Add markdown section `## Step 9: Reload Best Checkpoints and Evaluate on Test Split`.
2. Implement `load_best_head_for_eval(best_checkpoint_path, device)` that:
   - Calls `ckpt = torch.load(best_checkpoint_path, map_location=device)`.
   - Builds `head = GatedFusionHead(text_dim=128, image_dim=128, fusion_hidden_dim=ckpt["config_dict"]["model"]["fusion_hidden_dim"], num_classes=ckpt["num_classes"], mode=ckpt["mode"], dropout=ckpt["config_dict"]["model"]["dropout"]).to(device)`.
   - Calls `head.load_state_dict(ckpt["model_state_dict"], strict=True)`.
   - Calls `head.eval()` and returns `(head, ckpt)`.
3. For each `config_label` in `ablation_results`:
   - Reload via `load_best_head_for_eval(ablation_results[config_label]["best_checkpoint_path"], device)`.
   - Run a no-grad pass over `loaders["test"]` collecting `y_true`, `y_pred`, and predicted probabilities.
   - Compute `accuracy`, `macro_f1`, and `sklearn.metrics.classification_report(y_true, y_pred, output_dict=True, zero_division=0)`; compute `sklearn.metrics.confusion_matrix(y_true, y_pred).tolist()`.
   - Store into `ablation_results[config_label]["test_metrics"]`.
4. Add markdown section `## Step 10: Ablation Table`.
5. Build a `pandas.DataFrame` with rows = `[text_only, image_only, concat, gated]`, columns = `accuracy`, `macro_f1`, per-class F1 (`f1_class0`, `f1_class1`, and `f1_class2` when `num_classes == 3`), and `best_epoch`. Display inline and write to `training/stage2_results/ablation_table.csv` (via `CONFIG["paths"]["stage2_results_dir"]`).
6. Add markdown section `## Step 11: Confusion Matrix (Config D — Full Gated Fusion)`.
7. Implement `plot_confusion_matrix(cm, class_names, output_path)` using `seaborn.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)`, save to `output_path`, and `plt.show()`.
8. Call the plotter for config `gated` with class names `["Supported", "Refuted"]` (or `["Supported", "Refuted", "NEI"]` when 3-class); save as `training/stage2_results/test_confusion_matrix.png`; log to MLflow when `mlflow_enabled`.
9. Add markdown section `## Step 12: JSON Results Export`.
10. Build the export dict with this exact top-level schema:
    - `metadata`: `{run_completed_at, stage1_checkpoint_path, stage1_epoch, num_classes, mlflow_enabled, mlflow_experiment_name}`
    - `ablation_summary`: list of dicts, one per config, each with `config_label`, `mode`, `accuracy`, `macro_f1`, `f1_per_class`, `best_epoch`
    - `best_config`: dict for the `gated` config with `mode`, `config_label`, `classification_report` (full sklearn dict), `confusion_matrix` (nested list), `hyperparameters` (lr, weight_decay, batch_size, label_smoothing, fusion_hidden_dim, dropout), `best_epoch`, `val_macro_f1`, `stage1_checkpoint_path`, `stage1_epoch`.
11. Write the dict to `CONFIG["paths"]["stage2_results_dir"] / "mm_vifactcheck_results.json"` via `json.dump(..., indent=2, ensure_ascii=False)`.
12. Add markdown section `## Step 13: Summary`.
13. Print a final summary cell containing these exact labels:
    - `Best config: gated`
    - `Ablation CSV:`
    - `Confusion matrix PNG:`
    - `Results JSON:`
    - `Phase 5 (if applicable): /gsd-plan-phase 5`

</action>

<acceptance_criteria>

- Notebook text contains `## Step 9: Reload Best Checkpoints and Evaluate on Test Split`.
- Notebook text contains `def load_best_head_for_eval`.
- Notebook text contains `ckpt["model_state_dict"]`.
- Notebook text contains `strict=True`.
- Notebook text contains `sklearn.metrics.classification_report` or `from sklearn.metrics import classification_report`.
- Notebook text contains `sklearn.metrics.confusion_matrix` or `from sklearn.metrics import confusion_matrix`.
- Notebook text contains `## Step 10: Ablation Table`.
- Notebook text contains `ablation_table.csv`.
- Notebook text contains `pd.DataFrame` or `pandas.DataFrame`.
- Notebook text contains `## Step 11: Confusion Matrix (Config D — Full Gated Fusion)`.
- Notebook text contains `def plot_confusion_matrix`.
- Notebook text contains `sns.heatmap` or `seaborn.heatmap`.
- Notebook text contains `test_confusion_matrix.png`.
- Notebook text contains `## Step 12: JSON Results Export`.
- Notebook text contains `mm_vifactcheck_results.json`.
- Notebook text contains `"ablation_summary"`.
- Notebook text contains `"best_config"`.
- Notebook text contains `"classification_report"`.
- Notebook text contains `"confusion_matrix"`.
- Notebook text contains `json.dump`.
- Notebook text contains `ensure_ascii=False`.
- Notebook text contains `## Step 13: Summary`.
- Notebook text contains `Best config: gated`.
- Notebook text contains `Phase 5 (if applicable): /gsd-plan-phase 5`.

</acceptance_criteria>

</tasks>

---

<verification>

## Verification Steps

Run these checks after executing the plan:

1. `rtk test -f notebooks/pipeline/04_mm_vifactcheck_integration.ipynb && echo notebook_exists`
2. `rtk grep -c "^CONFIG = {" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` must return `1`.
3. `rtk grep -c "/Users/" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` must return `0`.
4. `rtk grep -c "class GatedFusionHead" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` must return at least `1`.
5. `rtk grep -n "text_aligned_clip\|image_aligned_clip\|freeze_for_stage2\|val_macro_f1\|OneCycleLR\|ablation_table.csv\|mm_vifactcheck_results.json\|test_confusion_matrix.png\|stage2_features" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` should match every listed string.
6. `rtk jupyter nbconvert --to script notebooks/pipeline/04_mm_vifactcheck_integration.ipynb --stdout > /tmp/nb4.py && python -c 'import py_compile; py_compile.compile("/tmp/nb4.py")'` exits 0.
7. Smoke run (gated on Phase 2 + Phase 3 completion): set `CONFIG["safety"]["smoke_test"] = True`, run the notebook end-to-end, and verify these outputs exist:
   - `training/stage2_features/stage2_train.h5`, `stage2_dev.h5`, `stage2_test.h5`
   - `training/checkpoints_stage2/text_only_*/best_model.pth`
   - `training/checkpoints_stage2/image_only_*/best_model.pth`
   - `training/checkpoints_stage2/concat_*/best_model.pth`
   - `training/checkpoints_stage2/gated_*/best_model.pth`
   - `training/stage2_results/ablation_table.csv`
   - `training/stage2_results/test_confusion_matrix.png`
   - `training/stage2_results/mm_vifactcheck_results.json`

</verification>

---

<success_criteria>

- `notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` exists and is output-clean by default (D-26).
- One top `CONFIG` cell with nested groups; no hardcoded absolute local paths (NB-01, NB-02).
- Stage 1 checkpoint loads via manifest with the 6-patch chain replayed BEFORE `load_state_dict`; `freeze_for_stage2=True` is asserted; all model parameters have `requires_grad=False` (MMVF-01, D-02, D-24).
- Per-pair COOLANT features pre-extracted into `training/stage2_features/stage2_{split}.h5` with `text_features [N,128]` + `image_features [N,128]` + `labels` + `article_ids`, labels joined from `news_data_vifactcheck_{split}_labeled.json` by `article_id` with NEI remap when `num_classes == 2`, idempotent by default (D-04, D-05, D-06, D-07, D-31).
- `GatedFusionHead` defined with selectable mode covering all four ablation configs (`text_only`, `image_only`, `concat`, `gated`) and projection layers `h_text_proj`, `h_mm_proj` (MMVF-03, D-01, D-03, D-09).
- Training uses class-weighted `CrossEntropyLoss`, AdamW + OneCycleLR with 5% warmup, early stop on val macro-F1, per-config MLflow run, best-by-macro-F1 checkpoint at `training/checkpoints_stage2/{config}_{timestamp}/best_model.pth` with full reproducibility bundle (MMVF-04, D-08, D-10–D-17, D-21).
- Each config's `best_model.pth` is reloaded from disk before test evaluation; test metrics include accuracy, macro-F1, per-class P/R/F1, confusion matrix (MMVF-05, D-18, D-19).
- Ablation DataFrame inline and `ablation_table.csv` written (MMVF-06, D-18).
- `test_confusion_matrix.png` rendered via seaborn for config `gated` (D-19).
- `mm_vifactcheck_results.json` with `ablation_summary` (all 4 configs) and `best_config` (full classification report, confusion matrix, hyperparameters, best epoch, val macro-F1, stage1 checkpoint path + epoch) and metadata block (MMVF-07, D-20).
- All threat-model mitigations (T-04-01..T-04-10) appear as concrete code or asserts in the notebook.
- All requirements MMVF-01 through MMVF-07 and NB-01 through NB-03 are covered by at least one task.

</success_criteria>

## PLANNING COMPLETE
