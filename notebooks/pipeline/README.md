# Pipeline Notebooks

Training/preprocessing stages for the fake-news detection thesis. Numbering:
`NN_` = main stage, `NN.M_` = substage (decimal, runs alongside `NN_`), `NNx_` = variant
(letter suffix, alternative approach to `NN_`).

## Stages

| Notebook | Stage | Purpose |
|---|---|---|
| `01_data_crawling.ipynb` | 1 | Crawl raw news articles + images |
| `02_preprocessing.ipynb` | 2 | Extract PhoBERT text + ResNet50 image features → `coolant_{train,dev,test}.h5` |
| `03_coolant_training.ipynb` | 3 | Train `ResNetCOOLANT` (contrastive image-text fake-news model) |
| `03-1_coolant_visualize.ipynb` | 3.1 | Visualize Stage 3 results |
| `03a_anchored_coolant_training.ipynb` | 3a | Variant: anchored-negative COOLANT training |
| `03.5_ai_art_detection_training.ipynb` | 3.5 | Train CLIP ViT-L/14 + head binary classifier: real artwork vs AI/GAN-generated image |
| `03.6_ai_art_signal_fusion.ipynb` | 3.6 | Apply Stage 3.5 classifier to COOLANT images → per-image `ai_generated_score` signal |
| `03.9_vifactcheck_training.ipynb` | 3.9 | Train PhoBERT classifier on ViFactCheck text |
| `04_mm_vifactcheck_integration.ipynb` | 4 | Multimodal integration of ViFactCheck |
| `05a`-`05d` | 5 | Fusion variants (MIL attention, asymmetric gated, misinformation-aware MIL) |
| `06a`-`06d` | 6 | Fusion variants (cross-attention, late decision, finetune) |
| `07a`-`07d` | 7 | Vietnamese COOLANT (sanity, pretrain, finetune, ablations) |

## Stage 3.5 — AI-Art Detection Training

**Input:** HF dataset `hmnshudhmn24/real-fake-ai-generated-art-images` (21,642 images,
balanced real WikiArt / GAN-generated, single `train` split — no HF token needed).

**What it does:**
1. Loads dataset, stratified 80/10/10 split (fixed seed, since only one HF split exists).
2. Extracts frozen CLIP ViT-L/14 features (`ImagePreprocessor` from
   `src/preprocessing/image_preprocessing.py`), caches to
   `$DATA_ROOT/training/ai_art_features/*.npz`.
3. Trains small MLP head (1024→256→1, BCE) on top — backbone stays frozen.
4. Early stops on `val_f1`, logs to MLflow (experiment `ai-art-detection-stage35`).
5. Writes best checkpoint + `checkpoint_manifest.json` under
   `$DATA_ROOT/training/checkpoints_ai_art/<run>/`.

**Output:** `checkpoint_manifest.json` — handoff contract for Stage 3.6.

**Why:** AI-generated images are a common fake-news artifact. This gives a reusable
"is this image AI-generated" signal.

## Stage 3.6 — AI-Art Signal Fusion

**Input:** Stage 3.5's `checkpoint_manifest.json` (auto-picks latest under
`checkpoints_ai_art/` if not set explicitly) + existing COOLANT HDF5 files
(`coolant_{train,dev,test}.h5`, from Stage 2 — reads their `image_paths`/`article_ids`).

**What it does:**
1. Rebuilds classifier head from checkpoint, loads matching CLIP backbone.
2. Runs every COOLANT image through it → `ai_generated_score = sigmoid(logit)` in `[0, 1]`.
3. Exports `ai_art_signal_{train,dev,test}.npz` under `$DATA_ROOT/training/ai_art_signal/`,
   keyed by `article_id`.

**Output:** scalar signal file per split, joined by `article_id`.

**Design decision:** kept as a separate scalar signal, not fused into COOLANT's 2048-dim
ResNet50 image features or `ResNetCOOLANT`'s forward pass. Dimension mismatch (CLIP
1024-dim vs ResNet50 2048-dim) plus untested correlation with the fake-news label make
premature fusion wasted plumbing. Before wiring this into training, check
`groupby(label).ai_generated_score.mean()` — COOLANT images are news photos, not
artwork/GAN images, so the signal may not transfer. Full fusion is a follow-up if it does.

**Both notebooks use `safety.smoke_test=True` (small subset, few images) for local
validation before a full run — same convention as Stage 3.9.**
