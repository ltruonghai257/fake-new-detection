# Phase 4: MM-ViFactCheck Integration Notebook (Stage 2) — Research

**Researched:** 2026-05-12
**Domain:** PyTorch multimodal fine-tuning on frozen feature backbone; HDF5 feature caching; MLflow experiment tracking; ablation evaluation for Vietnamese fact-checking.
**Confidence:** HIGH on stack + patterns (verified against project sources); MEDIUM on one open question (HDF5↔label join key).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Training Architecture:**
- **D-01:** New `GatedFusionHead` module (not existing COOLANT detection head). Inputs: `text_aligned_clip` (128-dim) + `image_aligned_clip` (128-dim). Projects each via `h_text_proj` / `h_mm_proj`, learned gate, fuse, classify.
- **D-02:** COOLANT Stage 1 stays fully frozen; `freeze_for_stage2=True` enforced.
- **D-03:** Stage 1 contract: `text_aligned_clip` + `image_aligned_clip` only. `attention_weights` / `fake_prob` not passed to fusion head.
- **D-04:** Pre-extract all COOLANT features in a dedicated cell into `training/stage2_features/stage2_{split}.h5` with `text_features` (128), `image_features` (128), `labels`. Training reads cached HDF5 only.

**Label Space & HDF5 Input:**
- **D-05:** Phase 4 creates its own enriched HDF5 files; does NOT modify Phase 2 outputs.
- **D-06:** Labels joined from `notebooks/data/json/news_data_vifactcheck_{split}_labeled.json` using `article_ids` from Phase 2 HDF5. Label variant: `root` (0=Supported, 1=Refuted, 2=NEI).
- **D-07:** `NUM_CLASSES` config-switchable. Default = 2 (binary). When 2: NEI (2) remapped to 0. When 3: as-is.
- **D-08:** Class-weighted CrossEntropyLoss.

**Ablation Configs (A–D):**
- **D-09:** Four configs — A: text-only / B: image-only / C: concat (no gating, 256-dim → linear) / D: full gated fusion.
- **D-10:** All 4 fully trained; per-config MLflow run; run name `stage2_{config_label}_{YYYYMMDD_HHMMSS}`.

**Training Hyperparameters:**
- **D-11:** `MAX_EPOCHS=30`, `PATIENCE=7`, `BATCH_SIZE=32`, AdamW `lr=3e-4`, `weight_decay=1e-4`, `label_smoothing=0.1`, `grad_clip=1.0`.
- **D-12:** OneCycleLR with 5% warmup.
- **D-13:** `SMOKE_TEST=False` flag; True → 5 epochs, 2 batches/split, skip ablation loop.
- **D-14:** `AUTO_INSTALL_DEPS=False`; print exact commands, stop clearly.

**Checkpoint Selection:**
- **D-15:** Best by **val macro-F1**.
- **D-16:** `best_model.pth` per config under `training/checkpoints_stage2/{config}_{timestamp}/`.
- **D-17:** Checkpoint bundle: `model_state_dict`, config, epoch, val metrics (acc + macro-F1 + per-class), `num_classes`, `config_label`, `mlflow_run_id` if available.

**Results & Export:**
- **D-18:** Ablation table = Pandas DataFrame inline + `training/stage2_results/ablation_table.csv`. Rows = configs A–D, cols = accuracy / macro-F1 / per-class F1.
- **D-19:** Confusion matrix seaborn heatmap for config D; logged to MLflow.
- **D-20:** `training/stage2_results/mm_vifactcheck_results.json` with `ablation_summary` (all 4: acc, macro-F1, per-class F1, best epoch) + `best_config` (config D — sklearn classification_report, confusion matrix nested list, hyperparameters, best epoch, val macro-F1, stage1 checkpoint path, stage1 epoch).
- **D-21:** MLflow experiment name `mm-vifactcheck-stage2`. Tracking dir `notebooks/mlruns`. Per-config runs.

**Notebook Structure:**
- **D-22:** Section flow: Overview → Config → Dependency/device → Stage 1 checkpoint loading → Feature pre-extraction → Stage 2 dataset/loaders → GatedFusionHead → Ablation training loop (A–D) → Results & table → Confusion matrix → JSON export → Summary.
- **D-23:** Single nested `CONFIG` dict (`paths` / `model` / `training` / `mlflow` / `safety`).
- **D-24:** Stage 1 checkpoint via `checkpoint_manifest.json`; fallback to direct path.
- **D-25:** Concise thesis-friendly markdown; provenance refs to Phase 3 + COOLANT_WORKFLOW_ANALYSIS.
- **D-26:** Output-clean source notebook.
- **D-27:** Device `cuda > mps > cpu` via `src/utils/device.get_device()`. `num_workers=0`.

**Failure Handling:**
- **D-28:** Missing Phase 2 HDF5 / mismatched `article_ids` → fail with clear instructions.
- **D-29:** Missing Phase 3 checkpoint → fail with instructions; read manifest.
- **D-30:** MLflow failure → warn + continue local; mark `mlflow_enabled=False` in results JSON.
- **D-31:** Existing feature HDF5 → skip by default; `FORCE_REBUILD_FEATURES=False` override.

### Claude's Discretion
- Training architecture: gated fusion (over legacy fine-tuning).
- Feature extraction: pre-extract once → HDF5 cache → train fusion head on cached.
- Label variant: `root`, NEI remap at load time via `NUM_CLASSES`.
- Hyperparameters: 30 epochs, patience 7, AdamW 3e-4, class-weighted CE, OneCycleLR 5% warmup.
- Ablation depth: all 4 configs trained fully.
- Ablation table format: DataFrame inline + CSV.
- JSON export scope: full report (all configs + config D detailed).
- Checkpoint dir layout: `training/checkpoints_stage2/{config}_{timestamp}/`; results at `training/stage2_results/`.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within Phase 4 scope.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MMVF-01 | Load frozen COOLANT checkpoint, extract features per article | `src/models/resnet_coolant.py` `ResNetCOOLANT = PatchedCOOLANT`; load via `torch.load(manifest["best_checkpoint_path"], map_location=device)`; apply same 6 patches as Phase 3 (`patch_encoding ×3`, `patch_clip_projection ×2`, `patch_cnn_with_dropout ×3`) BEFORE `load_state_dict`; then `model.eval()` + freeze all params; run `model.forward(caption, image)` under `torch.no_grad()` and capture `text_aligned_clip` + `image_aligned_clip` outputs (per Stage 1 D-52..D-56 in Phase 3 CONTEXT). |
| MMVF-02 | PhoBERT-base-v2 encodes [Statement; SEP; Evidence] → [CLS] (768-dim) | **Note:** D-03 supersedes this — the locked Phase 4 architecture (gated fusion over `text_aligned_clip` + `image_aligned_clip` only) does NOT use a Stage 2 PhoBERT encoder. The requirement is satisfied through Phase 2's PhoBERT-base-v2 token features that flow into COOLANT's `text_aligned_clip`. Plans should document this resolution in markdown so the thesis committee sees the lineage. |
| MMVF-03 | Gated fusion module (h_text_proj + h_mm_proj) | Implement `GatedFusionHead(nn.Module)` with `h_text_proj=nn.Linear(128, fusion_hidden_dim)`, `h_mm_proj=nn.Linear(128, fusion_hidden_dim)`, `gate=nn.Linear(2*fusion_hidden_dim, fusion_hidden_dim)` with `sigmoid`, fused = `gate * h_text + (1-gate) * h_image`, then `classifier=nn.Linear(fusion_hidden_dim, num_classes)`. Naming: D-03 says `h_nli_proj + h_mm_proj` but Phase 4's discretion section + D-01 normalize this to `h_text_proj + h_mm_proj` (no Stage 2 PhoBERT NLI head, so `h_text` is more accurate). Plans should use `h_text_proj` and `h_mm_proj`. |
| MMVF-04 | Train on ViFactCheck train; best checkpoint by val macro-F1 | Adapt `run_epoch_supervised` pattern from `notebooks/all_stage_final/research/9_stage2_vifactcheck_supervised.ipynb` (lines 624–676). Use AdamW + OneCycleLR per D-11/D-12. Track `val_macro_f1` per epoch; save `best_model.pth` when macro-F1 improves; early stop at `patience=7`. |
| MMVF-05 | Evaluation report on test split | Reload best checkpoint per config → `model.eval()` → run test loop → compute `sklearn.metrics.classification_report(output_dict=True)`, `accuracy_score`, `f1_score(average="macro")`, `confusion_matrix`. Save into per-config metrics dict. |
| MMVF-06 | Ablation table (text-only → full MM-ViFactCheck) | Aggregate per-config metrics into Pandas DataFrame; columns = accuracy, macro-F1, F1_Supported, F1_Refuted (+ F1_NEI when 3-class); rows = A/B/C/D; display + save CSV. |
| MMVF-07 | Export results to JSON | `json.dump({"ablation_summary": [...], "best_config": {...}, "metadata": {...}}, ...)` with `indent=2, ensure_ascii=False`. |
| NB-01 | Single config cell | One `CONFIG = {...}` at top, nested per D-23. |
| NB-02 | Relative/config-driven paths | `PROJECT_ROOT = Path.cwd().parent.parent if Path.cwd().name == "pipeline" else Path.cwd()` pattern from Phase 3; all paths via `CONFIG["paths"][...]`. |
| NB-03 | Clear markdown section headers | Section flow per D-22. |

</phase_requirements>

## Summary

Phase 4 builds a single thesis-ready notebook `notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` that operates downstream of Phase 3's frozen COOLANT checkpoint. Architecture is fully locked by CONTEXT.md (D-01..D-31). The notebook executes in 4 stages: (1) load frozen Stage 1, (2) pre-extract per-pair `text_aligned_clip`+`image_aligned_clip` features into a cached HDF5 with ViFactCheck labels joined by `article_id`, (3) train 4 ablation configs (A text-only / B image-only / C concat / D full gated fusion) on the cached features with class-weighted CE + AdamW + OneCycleLR + macro-F1 early stop, (4) emit ablation CSV + JSON + confusion matrix.

**Primary recommendation:** Mirror Phase 3's plan structure — single horizontal plan with 6–7 tasks ordered by section flow (D-22). Reuse Phase 3's notebook scaffolding patterns (config cell, dependency preflight, MLflow-with-fallback, timestamped run dirs, sanity-check before training). The legacy `9_stage2_vifactcheck_supervised.ipynb` is the primary reference for training loop patterns even though Phase 4 diverges architecturally (cached features + new fusion head vs. fine-tune-the-backbone).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Stage 1 checkpoint loading | Notebook (top, single load) | `src/models/resnet_coolant.py` | Reuse `PatchedCOOLANT` + patch helpers. Loaded once, frozen for entire phase. |
| Feature pre-extraction | Notebook cell (one-shot) | `CoolantPairDataset` for input iteration | Idempotent; writes `training/stage2_features/stage2_{split}.h5` once; subsequent runs skip per D-31. |
| Stage 2 dataset (cached features + labels) | Notebook helper class | h5py + torch.utils.data.Dataset | Tiny custom class — reads `text_features`, `image_features`, `labels`; remaps per `NUM_CLASSES`. |
| GatedFusionHead | Notebook `nn.Module` | — | Small (≈4 layers); thesis-clearest to keep inline rather than promote to `src/` until validated. |
| Ablation orchestration | Notebook training loop | MLflow per-config run | Per-config: build model variant → train → eval → save → append to results dict. |
| Results aggregation/export | Notebook final cells | pandas + json + sklearn | DataFrame + CSV + JSON + confusion matrix PNG. |

## Standard Stack

### Core (all already in project)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.x [VERIFIED: existing in environment.yml + Phase 3 plan] | Tensors, training | Project standard |
| h5py | 3.x [VERIFIED: used in Phases 2–3] | Cached feature HDF5 read/write | Project standard for feature caches |
| numpy | 1.x [VERIFIED] | Array operations, label remapping | Project standard |
| pandas | 1.x/2.x [VERIFIED] | Ablation table DataFrame + CSV | Project standard (used in Phase 3) |
| scikit-learn | 1.x [VERIFIED] | `classification_report`, `confusion_matrix`, `f1_score(average="macro")`, `accuracy_score` | Project standard (used in Phase 3 metrics) |
| matplotlib | 3.x [VERIFIED] | Loss/F1 curves | Project standard |
| seaborn | 0.x [VERIFIED] | Confusion matrix heatmap (D-19) | Project standard |
| tqdm | 4.x [VERIFIED] | Per-epoch / per-batch progress | Project standard |
| mlflow | 2.x [VERIFIED: Phase 3] | Per-config experiment tracking | Project standard (D-21) |

### Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Macro-F1 / classification report | Manual per-class counters | `sklearn.metrics.f1_score(..., average="macro")` + `classification_report(..., output_dict=True)` | Edge cases (zero-division for absent classes) handled by sklearn |
| OneCycleLR | Manual cosine schedule | `torch.optim.lr_scheduler.OneCycleLR` per D-12 | Stdlib in torch, exactly the pattern legacy notebook uses |
| Class weight computation | Manual | `weights = len(train_labels) / (NUM_CLASSES * np.bincount(train_labels, minlength=NUM_CLASSES))` (legacy line 580) | Inverse-frequency formula already chosen; copy verbatim |
| Confusion matrix heatmap | Custom matplotlib | `seaborn.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)` | Idiomatic, one-liner |
| HDF5 write of mixed types | Manual | `h5py.File(path, "w").create_dataset(name, data=array)` per dataset | Phase 2 already established the pattern |

## Architecture Patterns

### Recommended Notebook Section Flow (per D-22)

```
00. Title + thesis-friendly overview markdown
01. CONFIG cell (single nested dict — paths, model, training, mlflow, safety)
02. Dependency preflight (check imports, print install commands if missing)
03. Device + seed setup
04. Load frozen Stage 1 COOLANT
    - Read checkpoint_manifest.json (or fallback to CONFIG["paths"]["stage1_checkpoint"])
    - Build model + apply 6 patches
    - load_state_dict
    - Freeze all params (param.requires_grad = False)
    - model.eval()
05. Feature pre-extraction (skipped if exists & not FORCE_REBUILD_FEATURES)
    - For each split: iterate CoolantPairDataset, run frozen COOLANT under no_grad
    - Stack text_aligned_clip + image_aligned_clip per pair
    - Join labels via article_id → JSON articles[article_id]["label"]
    - Write stage2_{split}.h5 with text_features/image_features/labels/article_ids
06. Stage 2 dataset (HDF5DatasetStage2 + DataLoaders for train/dev/test)
07. GatedFusionHead definition (one class, variants A/B/C/D via constructor arg)
08. Ablation training loop (for each config in [A, B, C, D]):
    - build_head(config_label, num_classes)
    - per-config MLflow run "stage2_{label}_{timestamp}"
    - train with AdamW + OneCycleLR + class-weighted CE
    - early stop on val_macro_f1, patience=7
    - save best_model.pth + per-config metadata
    - reload best, evaluate on test, store metrics
09. Ablation table (pandas DataFrame inline + CSV save)
10. Confusion matrix heatmap (config D)
11. JSON export (training/stage2_results/mm_vifactcheck_results.json)
12. Summary cell (best config, paths, next steps)
```

### Pattern 1: Pre-extract + cache features (D-04)

**What:** Iterate every (caption, image) pair through frozen COOLANT once, save (text_aligned_clip, image_aligned_clip, label) to small HDF5. Training loop reads only this cache.
**When to use:** Backbone is frozen and small downstream head is cheap to train repeatedly across ablation configs.
**Why:** 4 ablation configs × 30 epochs would otherwise run COOLANT forward 120× over the dataset. Cache once → train 4 heads from RAM-budget HDF5 cache.

```python
# Pre-extraction (run once per split, idempotent)
def extract_stage2_features(model, loader, articles_by_id, num_classes, device):
    text_feats, img_feats, labels, article_ids = [], [], [], []
    model.eval()
    with torch.no_grad():
        for caption, image, article_id in tqdm(loader, desc="Extracting"):
            caption = caption.to(device); image = image.to(device)
            out = model(caption, image)  # dict with text_aligned_clip / image_aligned_clip
            text_feats.append(out["text_aligned_clip"].cpu().numpy())
            img_feats.append(out["image_aligned_clip"].cpu().numpy())
            for aid in article_id.tolist():
                raw_label = articles_by_id[aid]["label"]
                if num_classes == 2 and raw_label == 2:
                    raw_label = 0  # NEI -> Supported
                labels.append(raw_label)
                article_ids.append(aid)
    return (np.vstack(text_feats), np.vstack(img_feats),
            np.array(labels, dtype=np.int64), np.array(article_ids, dtype=np.int64))
```

### Pattern 2: GatedFusionHead with mode switch

```python
class GatedFusionHead(nn.Module):
    def __init__(self, text_dim=128, image_dim=128, fusion_hidden_dim=256,
                 num_classes=2, mode="gated", dropout=0.3):
        super().__init__()
        self.mode = mode  # "text_only" | "image_only" | "concat" | "gated"
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
        h_t = torch.relu(self.h_text_proj(text_feat))   # [B, H]
        h_m = torch.relu(self.h_mm_proj(image_feat))    # [B, H]
        if self.mode == "text_only":
            fused = h_t
        elif self.mode == "image_only":
            fused = h_m
        elif self.mode == "concat":
            fused = torch.cat([h_t, h_m], dim=-1)
        else:  # gated
            g = torch.sigmoid(self.gate(torch.cat([h_t, h_m], dim=-1)))
            fused = g * h_t + (1 - g) * h_m
        return self.classifier(fused)
```

### Pattern 3: Class-weighted CE + OneCycleLR (verbatim from legacy)

```python
# From legacy 9_stage2_vifactcheck_supervised.ipynb lines 578–610
train_labels_all = np.array(train_ds.labels, dtype=np.int64)
class_counts = np.bincount(train_labels_all, minlength=NUM_CLASSES).astype(np.float64)
class_weights = len(train_labels_all) / (NUM_CLASSES * class_counts)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights_t,
                                label_smoothing=CONFIG["training"]["label_smoothing"])
optimizer = torch.optim.AdamW(head.parameters(),
                              lr=CONFIG["training"]["lr"],
                              weight_decay=CONFIG["training"]["weight_decay"])
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=CONFIG["training"]["lr"],
    total_steps=len(train_loader) * CONFIG["training"]["max_epochs"],
    pct_start=0.05,  # 5% warmup per D-12
)
```

### Anti-Patterns to Avoid

- **Re-running COOLANT every epoch.** Phase 4 explicitly caches features (D-04). Running frozen COOLANT inside the training loop wastes >90% of GPU budget.
- **Using `model.detection_module.classifier`.** That is COOLANT's binary head from Phase 3. Phase 4 uses a brand-new `GatedFusionHead` (D-01).
- **Per-pair PhoBERT re-encoding in Stage 2.** MMVF-02 in REQUIREMENTS.md describes the original two-encoder design; D-03 supersedes — Phase 4 reuses Phase 2 PhoBERT features that flow through COOLANT's text projector. Adding a Stage 2 PhoBERT call would contradict the locked architecture.
- **Hardcoded absolute paths.** Use `PROJECT_ROOT` derived from `Path.cwd()` per Phase 3 pattern; this notebook may be run from `notebooks/pipeline/` or repo root.
- **Sharing one MLflow run across 4 configs.** D-10 + D-21 require per-config runs so the thesis writer can compare ablation cleanly.

## Common Pitfalls

### Pitfall 1: `article_id` join mismatch (HIGH risk, ties to D-06 + D-28)
**What goes wrong:** HDF5 `article_ids` are dataset-relative integer indices (from `pair_extractor.py`'s `article_idx`, the position of the parent article in the source JSON). The labeled JSON has no `article_id` field — labels are looked up by **list index**.
**Why it happens:** Two artifacts (Phase 2 HDF5 + ViFactCheck labeled JSON) reference articles by position in the same JSON ordering, but neither stores an explicit common key.
**How to avoid:** When joining, treat `article_ids[i]` as **the index into the source JSON list**: `articles = json.load(f); label = articles[article_id]["label"]`. The feature-extraction cell MUST assert `max(article_ids) < len(articles)` and fail per D-28 if not.
**Warning signs:** `KeyError` / `IndexError` during pre-extraction; label distribution doesn't match ViFactCheck published splits.

### Pitfall 2: Forgetting to apply 6 patches before `load_state_dict`
**What goes wrong:** `ResNetCOOLANT(model_config)` instantiates COOLANT with the official-paper dims (e.g., text_dim=300 instead of 768). Loading Phase 3 weights → shape mismatch error.
**Why:** Phase 3 explicitly patches dims via `patch_encoding`, `patch_clip_projection`, `patch_cnn_with_dropout` BEFORE training. Phase 4 must replay the same 6 patches before `load_state_dict`.
**How to avoid:** Copy Phase 3 Plan Task 3 `build_model()` block verbatim into Phase 4 feature-extraction cell. Confirm via `assert model.state_dict()["..."].shape == checkpoint["model_state_dict"]["..."].shape` for one key.
**Warning signs:** PyTorch raises `RuntimeError: Error(s) in loading state_dict ... size mismatch ...`.

### Pitfall 3: Test-time data leak via in-memory model
**What goes wrong:** Evaluating with the last-epoch model instead of the saved best checkpoint inflates test metrics with overfit weights.
**How to avoid:** Per Phase 3 pattern, **reload `best_model.pth` from disk before test evaluation** for every config. Plan Task 5 must include this explicit reload step.
**Warning signs:** Test macro-F1 > val macro-F1 by a large margin.

### Pitfall 4: MLflow run leak across ablation configs
**What goes wrong:** Forgetting `mlflow.end_run()` before starting the next config → all 4 configs log into one parent run; ablation comparison breaks.
**How to avoid:** Use `with mlflow.start_run(run_name=f"stage2_{cfg}_{ts}"):` per config OR explicit `start_run()` / `end_run()` per config + try/finally.

### Pitfall 5: Macro-F1 zero-division on degenerate configs
**What goes wrong:** Config B (image-only) may collapse to a single class on a small split → sklearn raises `UndefinedMetricWarning`, macro-F1 = 0.0.
**How to avoid:** Use `f1_score(..., average="macro", zero_division=0)` and `classification_report(..., zero_division=0)`. Log a warning if any per-class F1 is 0.0 so the thesis writer notices.

### Pitfall 6: MPS device subtleties (Apple Silicon)
**What goes wrong:** `mps` doesn't support `torch.float64` and may silently fall back for some ops. `num_workers > 0` with HDF5 can deadlock.
**How to avoid:** D-27 already locks `num_workers=0` and `cuda > mps > cpu` priority. Cast labels to `torch.long` explicitly; cast features to `torch.float32` before moving to device.

## Code Examples

### Loading frozen Stage 1 via manifest (D-24)
```python
import json, torch
from pathlib import Path
from src.models.resnet_coolant import (
    ResNetCOOLANT, patch_encoding, patch_clip_projection, patch_cnn_with_dropout,
)

manifest_path = Path(CONFIG["paths"]["stage1_manifest"])
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    best_ckpt_path = Path(manifest["best_checkpoint_path"])
else:
    best_ckpt_path = Path(CONFIG["paths"]["stage1_checkpoint"])
if not best_ckpt_path.exists():
    raise FileNotFoundError(
        f"Stage 1 checkpoint not found at {best_ckpt_path}. "
        "Run Phase 3 first: notebooks/pipeline/03_coolant_training.ipynb"
    )

ckpt = torch.load(best_ckpt_path, map_location=device)
model = ResNetCOOLANT(ckpt["config"])  # use embedded Stage 1 config
patch_encoding(model.similarity_module.encoding, image_dim=2048)
patch_encoding(model.detection_module.encoding, image_dim=2048)
patch_encoding(model.detection_module.ambiguity_module.encoding, image_dim=2048)
patch_clip_projection(model.clip_module, target_dim=2048, is_image=True)
patch_clip_projection(model.clip_module, target_dim=768,  is_image=False)
patch_cnn_with_dropout(model.similarity_module.encoding.shared_text_encoding, 768, 0.3)
patch_cnn_with_dropout(model.detection_module.encoding.shared_text_encoding, 768, 0.3)
patch_cnn_with_dropout(model.detection_module.ambiguity_module.encoding.shared_text_encoding, 768, 0.3)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
for p in model.parameters():
    p.requires_grad = False
model.to(device)
assert ckpt.get("freeze_for_stage2") is True, "Phase 3 checkpoint missing freeze_for_stage2 flag"
```

### Cached stage2 HDF5 dataset
```python
class HDF5DatasetStage2(torch.utils.data.Dataset):
    """Reads (text_features [128], image_features [128], label) from stage2_{split}.h5.
       Remaps NEI (2) -> 0 when num_classes == 2.
    """
    def __init__(self, hdf5_path, num_classes=2):
        self.path = Path(hdf5_path)
        with h5py.File(self.path, "r") as f:
            self.text  = f["text_features"][:].astype(np.float32)
            self.image = f["image_features"][:].astype(np.float32)
            self.labels = f["labels"][:].astype(np.int64)
        if num_classes == 2:
            self.labels = np.where(self.labels == 2, 0, self.labels)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.text[idx]),
                torch.from_numpy(self.image[idx]),
                int(self.labels[idx]))
```

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 7.x + grep-based notebook acceptance checks (Phase 3 precedent) |
| Config file | `pytest.ini` (project root) |
| Quick run command | `rtk grep -n "<acceptance_string>" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` |
| Full suite command | `rtk grep -c "<all_required_strings>" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` + smoke-test notebook run when Phase 3 checkpoint exists |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MMVF-01 | Frozen COOLANT loads + extracts features | smoke / grep | `rtk grep -n "freeze_for_stage2" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` | ❌ Wave 0 |
| MMVF-02 | Text features flow through COOLANT (per D-03 resolution) | grep | `rtk grep -n "text_aligned_clip" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` | ❌ Wave 0 |
| MMVF-03 | GatedFusionHead with h_text_proj + h_mm_proj | grep | `rtk grep -n "class GatedFusionHead" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` | ❌ Wave 0 |
| MMVF-04 | Best by val macro-F1 + early stop patience=7 | grep | `rtk grep -n "best_macro_f1\|val_macro_f1" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` | ❌ Wave 0 |
| MMVF-05 | Eval report (acc / macro-F1 / per-class / cm) | grep + smoke | `rtk grep -n "classification_report\|confusion_matrix" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` | ❌ Wave 0 |
| MMVF-06 | Ablation table A–D | grep | `rtk grep -n "ablation_table.csv" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` | ❌ Wave 0 |
| MMVF-07 | JSON export to thesis path | grep + file-exists post-smoke | `rtk grep -n "mm_vifactcheck_results.json" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` | ❌ Wave 0 |
| NB-01 | Single CONFIG cell | grep | `rtk grep -c "^CONFIG = {" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` must == 1 | ❌ Wave 0 |
| NB-02 | No hardcoded absolute paths | grep | `rtk grep -n "/Users/" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` must == 0 | ❌ Wave 0 |
| NB-03 | Clear section headers | grep | `rtk grep -c "^## Step" notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` must >= 10 | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** acceptance-criteria grep checks (per task in PLAN.md).
- **Per wave merge:** Full grep sweep across all acceptance strings + `nbconvert --to script` to confirm cells are parseable Python.
- **Phase gate:** Smoke run (`CONFIG["safety"]["smoke_test"] = True`) when Phase 3 checkpoint exists; verify per-config `best_model.pth` + ablation CSV + JSON + confusion matrix PNG are produced under timestamped run dirs.

### Wave 0 Gaps
- [ ] `notebooks/pipeline/04_mm_vifactcheck_integration.ipynb` does not exist yet — all tests are grep-on-the-built-notebook.
- [ ] Phase 3 checkpoint does not exist yet (Phase 3 hasn't executed). Smoke run is a phase-gate verification, not a wave-merge gate.
- [ ] No pytest test file is required — Phase 3 used grep-on-notebook acceptance and the pattern carries forward.

## Security Domain

This is a notebook-only phase (no network endpoints, no auth, no user input parsing, no secrets). ASVS does not meaningfully apply.

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | partial | Validate `article_id` < `len(articles)` per Pitfall 1; raise clear error per D-28. |
| V6 Cryptography | no | — |

### Reproducibility / Data-Safety Considerations (non-security)
- **Environment mutation:** Notebook must not install packages unless `AUTO_INSTALL_DEPS=True` (D-14). Default behaviour: print exact `pip install ...` and `conda install ...` commands then stop with `RuntimeError`.
- **Path safety:** No hardcoded absolute local paths (NB-02). All paths derived from `PROJECT_ROOT` or `CONFIG` keys.
- **Expensive accidental overwrite:** Each config run creates a fresh timestamped directory under `training/checkpoints_stage2/{label}_{YYYYMMDD_HHMMSS}/`. Skip feature-extraction by default if cache exists (D-31); flag `FORCE_REBUILD_FEATURES=False` to override.
- **MLflow failure:** Wrap MLflow init + per-call logging in try/except. On failure, set `mlflow_enabled=False`, continue local artifact save, mark `mlflow_enabled=False` in `mm_vifactcheck_results.json` (D-30).
- **CUDA OOM:** Catch `RuntimeError` containing `out of memory` during pre-extraction or training; print recovery instructions (`lower CONFIG["training"]["batch_size"]` or enable `SMOKE_TEST`).
- **KeyboardInterrupt during ablation loop:** Save partial `ablation_table.csv` with whatever configs completed; print which config was interrupted so user can resume by tweaking the loop.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Phase 2 HDF5 (`processed_data/hdf5/coolant_{split}.h5`) | Feature pre-extraction | ✗ (not yet executed) | — | Fail per D-28 with instruction to run Phase 2 first |
| Phase 3 checkpoint (`training/checkpoints_coolant/*/best_model.pth` + manifest) | Frozen backbone | ✗ (not yet executed) | — | Fail per D-29 with instruction to run Phase 3 first |
| Labeled JSON (`notebooks/data/json/news_data_vifactcheck_{split}_labeled.json`) | Label join during pre-extraction | ✓ | 634 / 1268 / 2535 articles dev/test/train | — |
| `src/models/resnet_coolant.py` | `ResNetCOOLANT` + patches | ✓ | — | — |
| `src/processing/coolant/pair_dataset.py` | `CoolantPairDataset` for input iteration | ✓ | — | — |
| `src/utils/device.py` | `get_device()` | ✓ | — | — |
| torch / h5py / numpy / pandas / sklearn / matplotlib / seaborn / tqdm / mlflow | Training + analysis | ✓ (Phase 3 + Phase 2 used them) | per environment.yml | — |

**Missing dependencies with no fallback:** None — the missing artifacts (Phase 2 HDF5, Phase 3 checkpoint) are not "missing" but "not yet produced upstream". They are the explicit phase prerequisite (`Depends on: Phase 3`).

**Missing dependencies with fallback:** None.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Legacy two-phase fine-tune (2a frozen, 2b full unfreeze) on COOLANT detection head | Frozen COOLANT + new `GatedFusionHead` on cached features | This phase (D-01..D-04 locked) | Cleaner thesis story; clearer ablation; ~10× faster per-config training (no backbone forward in loop) |
| `model.detection_module.classifier = nn.Linear(...)` head swap | Brand-new `GatedFusionHead` module | D-01 | Decouples evaluation from Stage 1 detection head; lets ablation isolate fusion contribution |
| Cosine warmup schedule from Phase 3 | OneCycleLR 5% warmup (matches legacy stage2 reference) | D-12 | Aligns with legacy reference notebook so thesis writer can cite "same training recipe as `9_stage2_vifactcheck_supervised.ipynb` adapted for cached features" |

**Deprecated/outdated:**
- Per-epoch COOLANT forward (kept in legacy notebook only). Phase 4 explicitly pre-extracts (D-04).
- 2-class classifier head swap on `model.detection_module.classifier` (D-01 replaces with `GatedFusionHead`).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `article_ids` in Phase 2 HDF5 are integer indices into the **same JSON file ordering** as `news_data_vifactcheck_{split}_labeled.json` | Pitfall 1 + MMVF-01 join logic | [ASSUMED, but verified by reading `src/processing/coolant/pair_extractor.py` line 117 + 179: `article_idx = enumerate(articles)`]. Phase 2 reads from `news_data_vifactcheck_{split}_cleaned.json`; the `_labeled.json` files have the **same article ordering + indexing** if labeling preserved order. **Plan must add an explicit assert in the pre-extraction cell** to fail fast if `len(articles_labeled) != len(articles_used_by_phase_2)` (compare via title or source_url at index 0). |
| A2 | Phase 3 checkpoint stores the Stage 1 config under `ckpt["config"]` and is loadable via `ResNetCOOLANT(ckpt["config"])` after replaying the 6 patches | Code example "Loading frozen Stage 1 via manifest" | [VERIFIED: matches Phase 3 03-PLAN.md Task 4 acceptance criteria — checkpoint bundle includes `config` + module-level state dicts] |
| A3 | The COOLANT model returns a dict containing `text_aligned_clip` and `image_aligned_clip` as 128-dim tensors per pair | Pattern 1 + D-03 | [ASSUMED, but locked by D-03 + Phase 3 D-52..D-56 contract.] If the model returns a tuple/single tensor instead, the pre-extraction cell needs `text_aligned_clip = out[0]` or `out.text_aligned_clip` adjustment. **Plan task should grep `src/models/resnet_coolant.py` for return type during execution** and update accordingly. Note: `src/models/coolant_official.py` is larger and may have the canonical forward; plan should read both. |

## Open Questions

1. **`article_id` join correctness (carry into PLAN.md as an explicit pre-extraction assertion)**
   - What we know: HDF5 stores `article_ids` = index in source JSON (verified in `pair_extractor.py`); labeled JSON has same ordering by convention.
   - What's unclear: Whether `news_data_vifactcheck_{split}_labeled.json` was ever resorted relative to `news_data_vifactcheck_{split}_cleaned.json`.
   - Recommendation: Plan task adds an assertion in the pre-extraction cell — when reading the labeled JSON, compare `articles[article_id]["title"]` against the title recorded in HDF5 metadata (if `source_urls`/`titles` were saved per Phase 2 D-27); if mismatch, fail with clear instructions to align JSON files.

2. **COOLANT forward output shape**
   - What we know: Phase 3 manifest declares output keys `text_aligned_clip`, `image_aligned_clip`, `attention_weights`, `detection_logits`, `fake_prob`.
   - What's unclear: Exact return type (dict vs. tuple vs. NamedTuple) of `ResNetCOOLANT.forward()`.
   - Recommendation: Plan task includes a "verify forward signature" sub-step at the top of the pre-extraction cell — print `type(out)` and adjust extraction code accordingly. Acceptance criterion: `notebook text contains "text_aligned_clip"` + `notebook text contains "image_aligned_clip"`.

3. **`fusion_hidden_dim` choice**
   - What we know: D-23 lists `fusion_hidden_dim` under `model` in CONFIG but doesn't specify a value.
   - What's unclear: Whether 256 (Claude's recommendation, matches legacy NLI head dim) or 128 (matches COOLANT output dim) is preferred for thesis-clean ablation.
   - Recommendation: Default to `256` in plan (Claude's discretion area per CONTEXT.md), document the choice in markdown.

## Sources

### Primary (HIGH confidence — codebase-verified)
- `src/models/resnet_coolant.py` — `ResNetCOOLANT = PatchedCOOLANT`; patch helpers referenced.
- `src/processing/coolant/pair_extractor.py` — `article_idx = enumerate(articles)` join semantics.
- `src/processing/coolant/pair_dataset.py` — `CoolantPairDataset` iteration shape `[768, seq_len]` for caption / `[2048]` for image / `int(article_id)` per sample.
- `src/utils/device.py` — `get_device()` cuda > mps > cpu fallback; respects `FORCE_DEVICE` env var.
- `.planning/phases/03-coolant-training-notebook-stage-1/03-PLAN.md` — Stage 1 build pattern + patch chain + checkpoint bundle contract.
- `.planning/phases/02-preprocessing-notebook/02-PLAN.md` — HDF5 schema `caption_features [N,128,768]` / `image_features [N,2048]` / `article_ids`.
- `notebooks/all_stage_final/research/9_stage2_vifactcheck_supervised.ipynb` — class-weight formula (line 580), `OneCycleLR` block (line 601), `run_epoch_supervised` (line 624), `HDF5DatasetLabeled` (line 208).
- `notebooks/data/json/news_data_vifactcheck_dev_labeled.json` — verified 634 articles, root-level `label ∈ {0,1,2}`, no `article_id` field (join by list index).

### Secondary (MEDIUM confidence)
- `.planning/phases/02-preprocessing-notebook/02-CONTEXT.md` + `02-SUMMARY.md` — HDF5 contract description (no source-of-truth Phase 2 notebook execution yet because Phase 2 not run).

### Tertiary (LOW confidence)
- None — research grounded entirely in existing project artifacts.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every library already in environment.yml + used in Phase 3.
- Architecture: HIGH — all key decisions locked in CONTEXT.md; integration points verified against codebase.
- Pitfalls: MEDIUM — Pitfall 1 (article_id join) and Open Question 2 (COOLANT forward output shape) are real and need explicit guards in plan tasks.
- Validation: HIGH — grep-on-notebook precedent established by Phase 3.

**Research date:** 2026-05-12
**Valid until:** 2026-06-12 (30 days) or until Phase 3 checkpoint contract changes (whichever first).

## RESEARCH COMPLETE
