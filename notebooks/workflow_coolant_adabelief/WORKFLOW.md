# COOLANT + AdaBelief: Caption-Pair Workflow

Paper-faithful COOLANT (arXiv:2302.14057) trained on image-caption pairs
with **AdaBelief optimizer** instead of Adam/AdamW.

## Data Labeling (same as workflow_coolant)

- **Real (label 0)**: Matched pair (caption_i, image_i) from same article
- **Fake (label 1)**: Unmatched pair (caption_i, image_j) from different articles
- Labels created **dynamically per batch** — no external labels needed
- **Balanced by construction**: 50% Real, 50% Fake in every batch

## Pipeline

```
../workflow_coolant/pairs/pairs_{train,dev,test}.json
    |
    v  [1_preprocess_from_pairs.ipynb]
    |  PhoBERT [128, 768] + ResNet50 [2048]
    v
./processed/coolant_{train,dev,test}.h5
    |
    +---> [2a] Simultaneous training (all 3 tasks per epoch)
    |
    +---> [2b] Phased training (4-phase progressive)
```

## Optimizer: AdaBelief

Key differences from Adam/AdamW:
- `eps=1e-16` (not 1e-8) — adapts step size based on "belief" in gradient direction
- `weight_decouple=True` — decoupled weight decay (like AdamW)
- `rectify=True` — variance rectification (like RAdam)

## Steps

| Step | Notebook | Input | Output |
|------|----------|-------|--------|
| 1 | `1_preprocess_from_pairs.ipynb` | `../workflow_coolant/pairs/*.json` | `./processed/*.h5` |
| 2a | `2a_train_adabelief_simultaneous.ipynb` | `./processed/*.h5` | `./checkpoints/` |
| 2b | `2b_train_adabelief_phased.ipynb` | `./processed/*.h5` | `./checkpoints_phased/` |

## Configuration

```
lr=1e-3, eps=1e-16, weight_decouple=True, rectify=True
batch_size=32, max_length=128, dropout=0.3
label_smoothing=0.1, grad_clip=1.0
patience=5, seed=42
```
