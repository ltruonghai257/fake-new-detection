#!/usr/bin/env python3
"""Generate 4_train_model.ipynb programmatically."""
import json, pathlib

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [src]})

def code(src):
    cells.append({
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [], "source": [src]
    })

# ──────────────────────────────────────────────────────────────────────────────
md("""# 4. Train COOLANT Model on Preprocessed ResNet50 Features

**Data layout (`combined_dataset.npz`):**
- `text_features`: `(N, 512, 768)` — BERT token embeddings
- `image_features`: `(N, 2048)` — ResNet50 features
- `labels`: `(N,)` — class labels (0 = real, 1 = fake)

**Adaptations applied at runtime:**
1. `ResNetCOOLANT` subclass satisfies the abstract stubs from `MultimodalModel`
2. `EncodingPart.shared_image` first linear: `512 → 2048`
3. `CLIP.image_projection` first linear: `512 → 2048`
4. `FastCNN.fast_cnn` conv input dim: `200 → 768` (BERT embed size)
5. Single-class label guard: if all labels are identical, 30 % are flipped to class 1
""")

# ── 1. Setup ──────────────────────────────────────────────────────────────────
md("## 1. Setup & Dependencies")
code("""\
import sys, os, math, random, json, warnings
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix, classification_report,
)
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

warnings.filterwarnings('ignore')

# Device
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

print(f'Device : {DEVICE}')
print(f'PyTorch: {torch.__version__}')
""")

# ── 2. Load data ──────────────────────────────────────────────────────────────
md("## 2. Load Preprocessed Data")
code("""\
DATA_PATH = './processed_data/crawled/combined_dataset.npz'
npz = np.load(DATA_PATH)

text_features  = npz['text_features']   # (N, 512, 768)
image_features = npz['image_features']  # (N, 2048)
labels         = npz['labels']          # (N,)

print(f'text_features  : {text_features.shape}  {text_features.dtype}')
print(f'image_features : {image_features.shape}  {image_features.dtype}')
print(f'labels         : {labels.shape}  {labels.dtype}')
print(f'Unique labels  : {dict(zip(*np.unique(labels, return_counts=True)))}')

N = len(labels)
""")

code("""\
# ── Guard: synthesize binary labels if only one class exists ──────────────────
unique = np.unique(labels)
if len(unique) < 2:
    print('[WARNING] Single class detected. Randomly marking 30% as class 1.')
    rng    = np.random.default_rng(42)
    labels = labels.copy()
    labels[rng.choice(N, size=int(0.30 * N), replace=False)] = 1
    print(f'After synthesis: {dict(zip(*np.unique(labels, return_counts=True)))}')
else:
    print(f'Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}')
""")

# ── 3. Split & DataLoader ─────────────────────────────────────────────────────
md("## 3. Train / Val / Test Split & DataLoaders")
code("""\
SEED       = 42
BATCH_SIZE = 32
TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

T_text  = torch.tensor(text_features,  dtype=torch.float32)
T_image = torch.tensor(image_features, dtype=torch.float32)
T_label = torch.tensor(labels,         dtype=torch.long)

dataset = TensorDataset(T_text, T_image, T_label)
n_train = int(TRAIN_FRAC * N)
n_val   = int(VAL_FRAC   * N)
n_test  = N - n_train - n_val

train_ds, val_ds, test_ds = random_split(
    dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED),
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f'Split  → train:{n_train}  val:{n_val}  test:{n_test}')
print(f'Batches→ train:{len(train_loader)}  val:{len(val_loader)}  test:{len(test_loader)}')
""")

# ── 4. Model ──────────────────────────────────────────────────────────────────
md("## 4. Model Initialization (ResNet50-compatible COOLANT)")
code("""\
import src.models.coolant_official as _co_mod
from src.models.base import FastCNN

IMAGE_DIM  = 2048   # ResNet50 output dim
TEXT_EMBED = 768    # BERT hidden size

CONFIG = {
    'shared_dim'     : 128,
    'sim_dim'        : 64,
    'clip_embed_dim' : 64,
    'feature_dim'    : 96,   # 16+16+64
    'h_dim'          : 64,
    'lr'             : 1e-3,
    'l2'             : 0.0,
    'num_epochs'     : 30,
    'seed'           : SEED,
    'device'         : str(DEVICE),
    'save_dir'       : './training/checkpoints',
}

# ── Concrete subclass: satisfies the three abstract methods ───────────────────
class ResNetCOOLANT(_co_mod.COOLANT_Official):
    \"\"\"COOLANT_Official concrete wrapper for ResNet50 2048-dim image features.\"\"\"
    def encode_text(self, text):
        dummy_img = torch.zeros(text.size(0), 512, device=text.device)
        t, _ = self.similarity_module.encoding(text, dummy_img)
        return t

    def encode_image(self, image):
        dummy_txt = torch.zeros(image.size(0), 512, TEXT_EMBED, device=image.device)
        _, i = self.similarity_module.encoding(dummy_txt, image)
        return i

    def fuse_modalities(self, text_f, image_f):
        return torch.cat([text_f, image_f], dim=-1)

model = ResNetCOOLANT(CONFIG)

# ── Patch 1: EncodingPart.shared_image  512 → IMAGE_DIM ──────────────────────
def _patch_enc(enc):
    layers, done = [], False
    for l in enc.shared_image:
        if isinstance(l, nn.Linear) and not done:
            layers.append(nn.Linear(IMAGE_DIM, l.out_features)); done = True
        else:
            layers.append(l)
    enc.shared_image = nn.Sequential(*layers)

_patch_enc(model.similarity_module.encoding)
_patch_enc(model.detection_module.encoding)
_patch_enc(model.detection_module.ambiguity_module.encoding)

# ── Patch 2: CLIP.image_projection  512 → IMAGE_DIM ─────────────────────────
layers, done = [], False
for l in model.clip_module.image_projection:
    if isinstance(l, nn.Linear) and not done:
        layers.append(nn.Linear(IMAGE_DIM, l.out_features)); done = True
    else:
        layers.append(l)
model.clip_module.image_projection = nn.Sequential(*layers)

# ── Patch 3: FastCNN conv input  200 → TEXT_EMBED ────────────────────────────
def _patch_cnn(m):
    m.fast_cnn = FastCNN(input_dim=TEXT_EMBED, channel=32,
                         kernel_size=(1, 2, 4, 8)).fast_cnn

_patch_cnn(model.similarity_module.encoding.shared_text_encoding)
_patch_cnn(model.detection_module.encoding.shared_text_encoding)
_patch_cnn(model.detection_module.ambiguity_module.encoding.shared_text_encoding)

model = model.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable parameters: {n_params:,}')
print('Model ready ✓')
""")

# ── 5. Optimizers ─────────────────────────────────────────────────────────────
md("## 5. Training Configuration")
code("""\
optim_sim = torch.optim.Adam(
    model.similarity_module.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['l2'])
optim_clip = torch.optim.AdamW(
    model.clip_module.parameters(), lr=1e-3, weight_decay=5e-4)
optim_det = torch.optim.Adam(
    model.detection_module.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['l2'])

sch_sim  = torch.optim.lr_scheduler.StepLR(optim_sim,  step_size=10, gamma=0.5)
sch_clip = torch.optim.lr_scheduler.StepLR(optim_clip, step_size=10, gamma=0.5)
sch_det  = torch.optim.lr_scheduler.StepLR(optim_det,  step_size=10, gamma=0.5)

loss_cos = nn.CosineEmbeddingLoss(margin=0.2)
loss_ce  = nn.CrossEntropyLoss()
loss_kl  = nn.KLDivLoss(reduction='batchmean')

NUM_EPOCHS = CONFIG['num_epochs']
SAVE_DIR   = Path(CONFIG['save_dir'])
SAVE_DIR.mkdir(parents=True, exist_ok=True)
print('Optimizers ready ✓')
""")

# ── 6. Helpers ────────────────────────────────────────────────────────────────
md("## 6. Helper Functions")
code("""\
def make_sim_pairs(text, image, label):
    \"\"\"Paired/unpaired samples from real-news rows for similarity learning.\"\"\"
    idx = (label == 0).nonzero(as_tuple=True)[0].tolist() or list(range(2))
    t = text[idx].clone()
    i_match   = image[idx].clone()
    i_unmatch = image[idx].clone().roll(3, 0)
    return t, i_match, i_unmatch


def soft_xe(logits, soft_target):
    return -(soft_target * F.log_softmax(logits, 1)).sum() / logits.size(0)


def run_epoch(loader, train=True):
    if train:
        model.similarity_module.train()
        model.clip_module.train()
        model.detection_module.train()
    else:
        model.eval()

    tot_loss, tot_ok, tot_n = 0.0, 0, 0
    all_y, all_p = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for text, image, label in loader:
            text  = text.to(DEVICE)
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            bs    = label.size(0)

            # ── Task 1a: Cosine Similarity ────────────────────────────────
            ft, fi_m, fi_u = make_sim_pairs(text, image, label)
            ta_m, ia_m, _  = model.similarity_module(ft, fi_m)
            ta_u, ia_u, _  = model.similarity_module(ft, fi_u)
            t_cat = torch.cat([ta_m, ta_u])
            i_cat = torch.cat([ia_m, ia_u])
            y_cos = torch.cat([
                 torch.ones(ta_m.size(0), device=DEVICE),
                -torch.ones(ta_u.size(0), device=DEVICE),
            ])
            ls = loss_cos(t_cat, i_cat, y_cos)
            if train:
                optim_sim.zero_grad(); ls.backward(); optim_sim.step()

            # ── Task 1b: CLIP Contrastive ─────────────────────────────────
            text_1d = text.mean(1)          # (B, 768) mean-pool over tokens
            ie, te  = model.clip_module(image, text_1d)
            logits  = ie @ te.T * math.exp(0.07)
            ids     = torch.arange(bs, device=DEVICE)

            ts, is_, _ = model.similarity_module(text, image)
            soft_m = is_ @ ts.T * math.exp(0.07)

            lc = (loss_ce(logits, ids) + loss_ce(logits.T, ids)) / 2
            ls2 = (soft_xe(logits, F.softmax(soft_m, 1)) +
                   soft_xe(logits.T, F.softmax(soft_m.T, 1))) / 2
            l_clip = lc + 0.2 * ls2
            if train:
                optim_clip.zero_grad(); l_clip.backward(); optim_clip.step()

            # ── Task 2: Detection ─────────────────────────────────────────
            with (torch.no_grad() if not train else torch.enable_grad()):
                ie2, te2 = model.clip_module(image, text_1d)
            det, attn, skl = model.detection_module(text, image, te2, ie2)
            ld = (loss_ce(det, label) +
                  0.5 * loss_kl(F.log_softmax(attn, 1), F.softmax(skl, 1)))
            if train:
                optim_det.zero_grad(); ld.backward(); optim_det.step()

            # ── Metrics ───────────────────────────────────────────────────
            pred     = det.argmax(1)
            tot_ok  += pred.eq(label).sum().item()
            tot_n   += bs
            tot_loss += ld.item() * bs
            all_y.extend(label.cpu().numpy())
            all_p.extend(pred.cpu().numpy())

    return tot_loss / tot_n, tot_ok / tot_n, all_y, all_p
""")

# ── 7. Training loop ──────────────────────────────────────────────────────────
md("## 7. Training Loop")
code("""\
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    tr_loss, tr_acc, _,  _        = run_epoch(train_loader, train=True)
    vl_loss, vl_acc, vl_y, vl_p  = run_epoch(val_loader,   train=False)
    sch_sim.step(); sch_clip.step(); sch_det.step()

    history['train_loss'].append(tr_loss)
    history['train_acc'].append(tr_acc)
    history['val_loss'].append(vl_loss)
    history['val_acc'].append(vl_acc)

    print(f'Epoch [{epoch:02d}/{NUM_EPOCHS}]  '
          f'train loss={tr_loss:.4f} acc={tr_acc:.4f}  |  '
          f'val loss={vl_loss:.4f} acc={vl_acc:.4f}')

    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), SAVE_DIR / 'best_model.pth')
        print(f'  ✓ Best val_acc={best_val_acc:.4f} saved.')

print(f'\\nTraining done. Best val acc: {best_val_acc:.4f}')
""")

# ── 8. Save history ───────────────────────────────────────────────────────────
md("## 8. Save Training History")
code("""\
with open(SAVE_DIR / 'training_history.json', 'w') as f:
    json.dump(history, f, indent=2)
print('History saved.')
""")

# ── 9. Test evaluation ────────────────────────────────────────────────────────
md("## 9. Final Evaluation on Test Set")
code("""\
model.load_state_dict(torch.load(SAVE_DIR / 'best_model.pth', map_location=DEVICE))
te_loss, te_acc, te_y, te_p = run_epoch(test_loader, train=False)

pre, rec, f1, _ = precision_recall_fscore_support(
    te_y, te_p, average='weighted', zero_division=0)

print(f'\\n=== Test Results ===')
print(f'Loss      : {te_loss:.4f}')
print(f'Accuracy  : {te_acc:.4f}')
print(f'Precision : {pre:.4f}')
print(f'Recall    : {rec:.4f}')
print(f'F1-Score  : {f1:.4f}')
print('\\n' + classification_report(te_y, te_p,
      target_names=['Real', 'Fake'], zero_division=0))

results = {'test_loss': te_loss, 'test_accuracy': te_acc,
           'precision': float(pre), 'recall': float(rec), 'f1': float(f1)}
with open(SAVE_DIR / 'test_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Test results saved.')
""")

# ── 10. Visualizations ────────────────────────────────────────────────────────
md("## 10. Visualizations")
code("""\
epochs = range(1, NUM_EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('COOLANT Training Curves', fontsize=14, fontweight='bold')

axes[0].plot(epochs, history['train_loss'], label='Train')
axes[0].plot(epochs, history['val_loss'],   label='Val')
axes[0].set(title='Detection Loss', xlabel='Epoch', ylabel='Loss')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(epochs, history['train_acc'], label='Train')
axes[1].plot(epochs, history['val_acc'],   label='Val')
axes[1].set(title='Detection Accuracy', xlabel='Epoch', ylabel='Accuracy')
axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(SAVE_DIR / 'training_curves.png', dpi=150)
plt.show()
""")

code("""\
cm = confusion_matrix(te_y, te_p)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real','Fake'], yticklabels=['Real','Fake'], ax=ax)
ax.set(title='Confusion Matrix — Test Set', xlabel='Predicted', ylabel='Actual')
plt.tight_layout()
plt.savefig(SAVE_DIR / 'confusion_matrix.png', dpi=150)
plt.show()
""")

# ── 11. Save final model ──────────────────────────────────────────────────────
md("## 11. Save Final Model")
code("""\
torch.save({
    'model_state_dict': model.state_dict(),
    'config'          : CONFIG,
    'best_val_acc'    : best_val_acc,
    'test_results'    : results,
    'history'         : history,
}, SAVE_DIR / 'coolant_resnet50_final.pth')
print(f'Saved → {SAVE_DIR}/coolant_resnet50_final.pth')
print(f'  Best val acc : {best_val_acc:.4f}')
print(f'  Test acc     : {te_acc:.4f}')
print(f'  Test F1      : {f1:.4f}')
""")

# ──────────────────────────────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "fake_news",
            "language": "python",
            "name": "fake_news",
        },
        "language_info": {"name": "python", "version": "3.13.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = pathlib.Path(__file__).parent / "4_train_model.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Written: {out}")
