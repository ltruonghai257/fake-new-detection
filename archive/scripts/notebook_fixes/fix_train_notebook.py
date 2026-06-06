#!/usr/bin/env python3
"""Regenerate src/4_train_model.ipynb with all ResNet50/2048-dim fixes applied."""
import json, pathlib

with open('src/4_train_model.ipynb') as f:
    nb = json.load(f)

cells = nb['cells']

# ── Helper to set a code cell's source ────────────────────────────────────────
def set_src(idx, src):
    cells[idx]['source'] = [src]

# ── Cell 6: remove (duplicate import attempt) → make it a no-op comment ───────
set_src(6, "# Model import handled in cell 7 below\n")

# ── Cell 7: replace with patched ResNetCOOLANT ────────────────────────────────
set_src(7, """\
import sys
sys.path.insert(0, '..')          # ensure src/ parent is on path
import math
import torch.nn as nn
import torch.nn.functional as F
from src.models.coolant import COOLANT
from src.models.base import FastCNN

IMAGE_DIM  = 2048   # ResNet50 output dim
TEXT_EMBED = 768    # BERT/PhoBERT hidden dim

model_config = {
    'shared_dim'      : 128,
    'sim_dim'         : 64,
    'feature_dim'     : 96,   # 64 + 16 + 16
    'h_dim'           : 64,
    'cnn_channel'     : 32,
    'cnn_kernel_size' : (1, 2, 4, 8),
    'contrastive_weight'  : 1.0,
    'classification_weight': 1.0,
    'similarity_weight'   : 0.5,
    'temperature'         : 0.07,
}

# ── Concrete subclass: satisfies the three abstract stubs ──────────────────────
class ResNetCOOLANT(COOLANT):
    \"\"\"COOLANT adapted for ResNet50 2048-dim image + BERT 768-dim text.\"\"\"
    def encode_text(self, text):
        t, _ = self.similarity_module.encoding(
            text, torch.zeros(text.size(0), 512, device=text.device))
        return t
    def encode_image(self, image):
        _, i = self.similarity_module.encoding(
            torch.zeros(image.size(0), 30, 200, device=image.device), image)
        return i
    def fuse_modalities(self, tf, imf):
        return torch.cat([tf, imf], dim=-1)

model = ResNetCOOLANT(model_config)

# ── Patch 1: EncodingPart.shared_image  Linear(512,256) → Linear(2048,256) ────
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

# ── Patch 2: FastCNN conv input_dim  200 → 768 ────────────────────────────────
def _patch_cnn(m):
    m.fast_cnn = FastCNN(input_dim=TEXT_EMBED, channel=32,
                         kernel_size=(1, 2, 4, 8)).fast_cnn

_patch_cnn(model.similarity_module.encoding.shared_text_encoding)
_patch_cnn(model.detection_module.encoding.shared_text_encoding)

model = model.to(device)
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ ResNetCOOLANT ready — {n:,} trainable parameters")
print(f"   Image input : {IMAGE_DIM}-dim  (ResNet50)")
print(f"   Text input  : ({TEXT_EMBED}-dim BERT embeddings)")
model_config_saved = model_config.copy()
""")

# ── Cell 8: fix test — call forward via compute_total_loss interface (COOLANT
#            has no CLIP module, so its forward IS usable after patching) ──────
set_src(8, """\
# 🧪 Test model with sample batch
print("🧪 Testing model with sample batch...")
with torch.no_grad():
    sample_batch = next(iter(train_loader))
    text_sample  = sample_batch["text"].to(device)      # (B, 512, 768)
    image_sample = sample_batch["image"].to(device)     # (B, 2048)
    label_sample = sample_batch["label"].to(device)

    print(f"Text input shape:  {text_sample.shape}")
    print(f"Image input shape: {image_sample.shape}")

    try:
        outputs = model(text_sample, image_sample)
        print("✅ Model forward pass successful!")
        print(f"   Logits shape:           {outputs['logits'].shape}")
        print(f"   Attention weights shape: {outputs['attention_weights'].shape}")
        print(f"   Ambiguity weights shape: {outputs['ambiguity_weights'].shape}")
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        raise
""")

# ── Cell 9: insert training config (was empty) ─────────────────────────────────
set_src(9, """\
# Training configuration
NUM_EPOCHS   = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4

optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True)

print(f"✅ Training config ready")
print(f"   Epochs: {NUM_EPOCHS}   LR: {LEARNING_RATE}   WD: {WEIGHT_DECAY}")
""")

# ── Cell 10: keep structure but fix forward call to use compute_total_loss ─────
# compute_total_loss calls model.forward internally (which is now patched)
set_src(10, """\
# 🚀 Training loop
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_loss = float("inf")
best_model_path = "./best_coolant_resnet50.pth"

print("🚀 Starting training...")
print("=" * 50)

for epoch in range(NUM_EPOCHS):
    # ── Train ─────────────────────────────────────────────────────────────────
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    for batch in train_pbar:
        text   = batch["text"].to(device)
        image  = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        loss_dict = model.compute_total_loss(text, image, labels)
        loss = loss_dict["total_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            logits = model(text, image)["logits"]
            _, predicted = torch.max(logits, 1)
        train_loss    += loss.item()
        train_total   += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        train_pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc":  f"{100.*train_correct/train_total:.2f}%"
        })

    # ── Validate ──────────────────────────────────────────────────────────────
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
        for batch in val_pbar:
            text   = batch["text"].to(device)
            image  = batch["image"].to(device)
            labels = batch["label"].to(device)

            loss_dict = model.compute_total_loss(text, image, labels)
            loss = loss_dict["total_loss"]
            logits = model(text, image)["logits"]
            _, predicted = torch.max(logits, 1)

            val_loss    += loss.item()
            val_total   += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc":  f"{100.*val_correct/val_total:.2f}%"
            })

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss   = val_loss   / len(val_loader)
    train_acc      = 100. * train_correct / train_total
    val_acc        = 100. * val_correct   / val_total

    train_losses.append(avg_train_loss);  val_losses.append(avg_val_loss)
    train_accuracies.append(train_acc);   val_accuracies.append(val_acc)
    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "config": model_config,
        }, best_model_path)
        print(f"💾 Best model saved (val_loss: {avg_val_loss:.4f})")

    print(f"\\nEpoch {epoch+1}/{NUM_EPOCHS}:")
    print(f"  Train — loss: {avg_train_loss:.4f}  acc: {train_acc:.2f}%")
    print(f"  Val   — loss: {avg_val_loss:.4f}  acc: {val_acc:.2f}%")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
    print("-" * 50)

print("\\n🎉 Training completed!")
""")

# ── Cell 12: evaluation also needs patched model — it already loads state_dict
#             which is fine since model was patched before training.
#             Just fix forward calls to use logits key from COOLANT.forward ─────
set_src(12, """\
# 🔄 Load best model and evaluate on test set
print("🔄 Loading best model...")
checkpoint = torch.load(best_model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
print(f"✅ Loaded epoch {checkpoint['epoch']+1}  val_loss={checkpoint['val_loss']:.4f}")

model.eval()
all_predictions, all_labels_list = [], []
test_loss = 0.0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        text   = batch["text"].to(device)
        image  = batch["image"].to(device)
        labels = batch["label"].to(device)

        loss_dict = model.compute_total_loss(text, image, labels)
        test_loss += loss_dict["total_loss"].item()

        logits = model(text, image)["logits"]
        _, predicted = torch.max(logits, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels_list.extend(labels.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
test_accuracy = accuracy_score(all_labels_list, all_predictions)
# re-bind name used by subsequent cells
all_labels = all_labels_list

print(f"\\n📊 Final Test Results:")
print(f"  Test Loss:     {avg_test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f}  ({test_accuracy*100:.2f}%)")
print(f"\\n📋 Classification Report:")
print(classification_report(all_labels, all_predictions, target_names=["Real","Fake"], zero_division=0))
""")

nb['cells'] = cells
out = pathlib.Path('src/4_train_model.ipynb')
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Written: {out}")
