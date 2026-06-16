# Training Stages Documentation

This document summarizes the two main training stages in the fake news detection pipeline.

---

## Stage 3: COOLANT Training (Multimodal Alignment)

**Notebook:** `notebooks/pipeline/03_coolant_training.ipynb`

### Purpose
Trains `ResNetCOOLANT` / `PatchedCOOLANT` on Phase 2 HDF5 feature files using dynamic matched/unmatched negatives. This stage performs multimodal alignment between text (captions) and image features.

### Key Configuration
- **Model Variant:** ResNetCOOLANT
- **Image Dimension:** 2048 (ResNet50 features)
- **Text Embed Dimension:** 768 (PhoBERT token embeddings)
- **Text Sequence Length:** 128
- **Shared Dimension:** 128
- **Similarity Dimension:** 64
- **CLIP Embed Dimension:** 64
- **Feature Dimension:** 96 (64 + 16 + 16)
- **Hidden Dimension:** 64
- **Learning Rate:** 1e-4
- **Weight Decay:** 1e-5
- **Dropout:** 0.1
- **Batch Size:** 32
- **Max Epochs:** 30
- **Patience:** 7 (early stopping)
- **Negative Shift:** 3
- **Warmup Steps:** 200
- **Seed:** 42

### Loss Weights
- **Similarity Weight:** 0.5
- **CLIP Weight:** 0.2
- **Detection Weight:** 1.0
- **Cosine Margin:** 0.2
- **Label Smoothing:** 0.0

### Input Data
- **Train:** `processed_data/hdf5/coolant_train.h5` (6,724 pairs)
- **Dev:** `processed_data/hdf5/coolant_dev.h5` (862 pairs)
- **Test:** `processed_data/hdf5/coolant_test.h5` (2,053 pairs)

### Model Architecture
- **Total Parameters:** 5,610,288
- **Similarity Module:** 1,003,266 parameters
- **CLIP Module:** 754,305 parameters
- **Detection Module:** 3,852,717 parameters

### Training Strategy
- Separate optimizers for each module (similarity, clip, detection)
- Composite losses: cosine similarity, CLIP contrastive CE + soft CE, detection CE + KL ambiguity
- Per-step warmup cosine LR scheduling
- Early stopping on validation accuracy

### Output
- **Best Model Path:** `training/checkpoints_coolant/coolant_stage1_20260610_020928/best_model.pth`
- **Checkpoint Manifest:** `training/checkpoints_coolant/coolant_stage1_20260610_020928/checkpoint_manifest.json`
- **MLflow Experiment:** `coolant-stage1-training`

### Stage 4 Handoff
- Frozen ResNetCOOLANT checkpoint
- Checkpoint manifest documenting expected input shapes and output keys
- Output keys: `text_aligned_clip`, `image_aligned_clip`, `attention_weights`, `detection_logits`, `fake_prob`

---

## Stage 3.9: ViFactCheck Text Classifier Training

**Notebook:** `notebooks/pipeline/03.9_vifactcheck_training.ipynb`

### Purpose
Fine-tunes `vinai/phobert-base-v2` as a Vietnamese fake-news **text-only** classifier on the ViFactCheck dataset. Serves as a strong standalone text baseline and exports a frozen checkpoint that Stage 4 can optionally load as its text encoder.

### Key Configuration
- **Backbone:** vinai/phobert-base-v2
- **Dropout:** 0.3 (paper §4.2)
- **Number of Classes:** 3 (three-class classification)
- **Batch Size:** 16 (paper §4.2)
- **Max Epochs:** 15 (extended from paper's 10)
- **Patience:** 5 (early stopping on val_macro_f1)
- **Learning Rate:** 5e-6 (paper §4.2 for PLMs on ViFactCheck)
- **Weight Decay:** 0.01
- **Warmup Ratio:** 0.1
- **Grad Clip:** 1.0
- **Label Smoothing:** 0.0
- **Max Length:** 256
- **Seed:** 42

### Data Sources
1. **Primary:** HuggingFace `tranthaihoa/vifactcheck` (native claim + evidence pairs)
2. **Fallback:** `data/json/labeled_nei_as_real/news_data_vifactcheck_{split}_labeled.json`

### Label Strategy (Three-Class)
- **Supported (0)** → Class 0
- **Refuted (1)** → Class 1
- **NEI (2)** → Class 2

### Text Fields
- **Statement:** Claim to verify (main text field)
- **Evidence:** Gold evidence (context field, best mode per paper §4.3)
- **Word Segmentation:** Optional (underthesea for VnCoreNLP-style segmentation)

### Dataset Statistics
- **Train:** 5,062 examples (label dist: [1751, 1658, 1653])
- **Dev:** 723 examples (label dist: [256, 244, 223])
- **Test:** 1,447 examples (label dist: [508, 468, 471])

### Model Architecture
- **Total Parameters:** 135,000,579
- **Hidden Size:** 768 (PhoBERT base)
- **Classifier:** Dropout + Linear(768 → 3)

### Training Strategy
- AdamW optimizer with linear warmup + linear decay
- Class weights derived from training label frequencies
- Early stopping on validation macro-F1
- Tokenizer saved alongside checkpoint for Stage 4 loading

### Output
- **Best Model Path:** `training/checkpoints_vifactcheck/vifactcheck_stage39_20260616_082028/best_model.pth`
- **Tokenizer Path:** `training/checkpoints_vifactcheck/vifactcheck_stage39_20260616_082028/tokenizer/`
- **Checkpoint Manifest:** `training/checkpoints_vifactcheck/vifactcheck_stage39_20260616_082028/checkpoint_manifest.json`
- **Stage 39 Features:** `training/stage39_features/stage39_{train,dev,test}.h5` (pre-extracted CLS features)
- **MLflow Experiment:** `vifactcheck-stage39-text`

### Stage 4 Integration (Ablation E - phobert_gated)
- **Feature Dimension:** 768-dim CLS vector from PhoBERT
- **Method:** `get_cls_features(input_ids, attention_mask)` → [B, 768]
- **Fusion:** GatedFusionHead(text_dim=768, image_dim=64)
- **Ablation Name:** `phobert_gated`
- **Description:** Replace COOLANT text_aligned_clip (64-dim) with Stage3.9 PhoBERT CLS (768-dim). Pair with COOLANT image_aligned_clip (64-dim).

### Usage Example
```python
from src.models.phobert_classifier import PhoBERTClassifier
ckpt = torch.load(best_checkpoint_path)
model = PhoBERTClassifier(backbone, num_classes, dropout)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
feats = model.get_cls_features(input_ids, attention_mask)  # [B, 768]
```

---

## Summary

| Stage | Model | Best Model Path | Primary Output |
|-------|-------|-----------------|----------------|
| 3 | ResNetCOOLANT | `training/checkpoints_coolant/coolant_stage1_20260610_020928/best_model.pth` | Multimodal alignment features |
| 3.9 | PhoBERT Classifier | `training/checkpoints_vifactcheck/vifactcheck_stage39_20260616_082028/best_model.pth` | Text-only CLS features (768-dim) |

Both stages produce frozen checkpoints that can be loaded in Stage 4 for multimodal fusion experiments.
