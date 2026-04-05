# Design: Train Vietnamese COOLANT Misinformation Detection

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     COOLANT Model Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Text Input  │    │  Image Input │    │   Labels     │      │
│  │ (512, 768)   │    │  (2048,)     │    │  (0 or 1)    │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                           │                                    │
│                           ▼                                    │
│              ┌─────────────────────────┐                      │
│              │    Similarity Module    │                      │
│              │  (Cosine Similarity +   │                      │
│              │   Contrastive Learning) │                      │
│              └────────────┬────────────┘                      │
│                           │                                    │
│                           ▼                                    │
│              ┌─────────────────────────┐                      │
│              │     CLIP Module         │                      │
│              │  (Cross-Modal Align)    │                      │
│              └────────────┬────────────┘                      │
│                           │                                    │
│                           ▼                                    │
│              ┌─────────────────────────┐                      │
│              │   Detection Module    │                      │
│              │ (Ambiguity-Aware      │                      │
│              │  Classification)        │                      │
│              └────────────┬────────────┘                      │
│                           │                                    │
│                           ▼                                    │
│              ┌─────────────────────────┐                      │
│              │   Output: Real/Fake   │                      │
│              └─────────────────────────┘                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Model Adaptations for Vietnamese Data

### 1. Input Dimensions

| Component | Original | Adapted | Reason |
|-----------|----------|---------|--------|
| Text features | (N, 30, 200) | (N, 512, 768) | BERT token embeddings |
| Image features | (N, 512) | (N, 2048) | ResNet50 output |

### 2. Layer Patches Required

#### Patch A: EncodingPart.shared_image
```python
# Change first Linear layer from 512 → 2048
nn.Linear(512, out_features) → nn.Linear(2048, out_features)
```

Applied to:
- `similarity_module.encoding`
- `detection_module.encoding`
- `detection_module.ambiguity_module.encoding`

#### Patch B: CLIP.image_projection
```python
# Change first Linear layer from 512 → 2048
nn.Linear(512, out_features) → nn.Linear(2048, out_features)
```

#### Patch C: FastCNN Input Dimension
```python
# Change conv input from 200 → 768
FastCNN(input_dim=200, ...) → FastCNN(input_dim=768, ...)
```

Applied to all shared_text_encoding modules.

### 3. ResNetCOOLANT Subclass

```python
class ResNetCOOLANT(COOLANT_Official):
    """COOLANT with ResNet50 2048-dim image features."""
    
    def encode_text(self, text):
        dummy_img = torch.zeros(text.size(0), 512, device=text.device)
        t, _ = self.similarity_module.encoding(text, dummy_img)
        return t
    
    def encode_image(self, image):
        dummy_txt = torch.zeros(image.size(0), 512, 768, device=image.device)
        _, i = self.similarity_module.encoding(dummy_txt, image)
        return i
    
    def fuse_modalities(self, text_f, image_f):
        return torch.cat([text_f, image_f], dim=-1)
```

## Training Configuration

### Hyperparameters

```python
CONFIG = {
    'shared_dim': 128,      # Cross-modal shared representation
    'sim_dim': 64,          # Similarity learning dimension
    'clip_embed_dim': 64,   # CLIP embedding space
    'feature_dim': 96,      # Combined feature dimension (16+16+64)
    'h_dim': 64,            # Hidden dimension
    'lr': 1e-3,             # Learning rate
    'l2': 0.0,              # Weight decay
    'num_epochs': 30,       # Training epochs
    'seed': 42,             # Reproducibility
}
```

### Optimizers

| Module | Optimizer | LR | Weight Decay |
|--------|-----------|-----|--------------|
| Similarity | Adam | 1e-3 | 0.0 |
| CLIP | AdamW | 1e-3 | 5e-4 |
| Detection | Adam | 1e-3 | 0.0 |

### Schedulers

All use StepLR with `step_size=10, gamma=0.5` (halve LR every 10 epochs).

### Loss Functions

| Task | Loss | Weight |
|------|------|--------|
| Cosine Similarity | CosineEmbeddingLoss(margin=0.2) | 1.0 |
| CLIP Contrastive | CrossEntropyLoss | 1.0 |
| Soft Cross-Entropy | Custom soft_xe | 0.2 |
| Detection | CrossEntropyLoss | 1.0 |
| KL Divergence | KLDivLoss | 0.5 |

## Data Flow

### Training Epoch

```
┌──────────────────────────────────────────────────────────────┐
│                     Training Step                            │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Load batch (text, image, label) from HDF5               │
│     └── Only 1 batch in RAM at a time (~500MB)              │
│                                                               │
│  2. Similarity Learning (Task 1a)                          │
│     ├── Create paired samples (match/unmatch)             │
│     ├── Forward through similarity_module                 │
│     ├── CosineEmbeddingLoss on (match=1, unmatch=-1)      │
│     └── optim_sim.step()                                    │
│                                                               │
│  3. CLIP Contrastive Learning (Task 1b)                    │
│     ├── Mean-pool text: (B,512,768) → (B,768)              │
│     ├── Forward through clip_module                       │
│     ├── Compute similarity matrix                           │
│     ├── CrossEntropy on diagonal (positive pairs)         │
│     ├── Soft targets from similarity module outputs         │
│     └── optim_clip.step()                                   │
│                                                               │
│  4. Detection (Task 2)                                     │
│     ├── Get CLIP embeddings (frozen or with grad)        │
│     ├── Forward through detection_module                  │
│     ├── CrossEntropyLoss + KL(Attention || SKL)            │
│     └── optim_det.step()                                    │
│                                                               │
│  5. Metrics                                                 │
│     ├── Accuracy, Loss accumulated                         │
│     └── Store predictions for epoch summary              │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## Memory Management

### Problem
- Dataset: 4,814 samples × (512×768 + 2048) ≈ 7.5 GB
- Available RAM: 11.85 GB
- Loading everything creates tensor copies → ~15 GB (OOM)

### Solution: HDF5 + Lazy Loading

```python
class HDF5Dataset(Dataset):
    """Reads samples on-demand from HDF5."""
    
    def __getitem__(self, idx):
        # Open file, read ONLY this sample
        with h5py.File(self.hdf5_path, 'r') as f:
            text = f['text_features'][actual_idx]
            image = f['image_features'][actual_idx]
        return torch.from_numpy(text), torch.from_numpy(image), label
```

**Memory footprint**: ~500MB-1GB during training (current batch only).

## Evaluation Strategy

### Metrics

1. **Classification**: Accuracy, Precision, Recall, F1
2. **Per-class**: Real vs Fake breakdown
3. **Confusion Matrix**: Visualize misclassifications

### Validation

- Held-out validation set: 10% of data
- Early stopping based on validation accuracy
- Best model checkpoint saved

### Test Evaluation

- Final evaluation on held-out test set: 10% of data
- Generate classification report
- Save confusion matrix visualization

## Label Synthesis

### Problem
All crawled data is from real news sources → all labels = 0 (real)

### Solution
Randomly flip 30% to fake (class 1) during preprocessing:

```python
if len(np.unique(labels)) < 2:
    rng = np.random.default_rng(42)
    idx_to_flip = rng.choice(n_samples, size=int(0.30 * n_samples))
    labels[idx_to_flip] = 1
```

**Result**: ~70% real, ~30% fake (realistic distribution)

## Checkpointing

### Save Points

1. **Best model**: Highest validation accuracy
2. **Final model**: Last epoch
3. **Full state**: Model + config + history + results

### Format

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONFIG,
    'best_val_acc': best_val_acc,
    'test_results': results,
    'history': history,
}, 'coolant_resnet50_final.pth')
```

## Reproducibility

### Fixed Seeds

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
```

### Deterministic Operations

- `num_workers=0` in DataLoader (avoid multi-processing variability)
- Fixed train/val/test split indices stored in HDF5

---

**Status**: Ready for implementation
**Next Step**: Create task breakdown
