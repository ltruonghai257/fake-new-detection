#!/usr/bin/env python3
"""Quick verification script to test the text dimension understanding."""

import sys
import os
from pathlib import Path

# Setup path
cwd = Path.cwd()
if cwd.name == "notebooks":
    BASE_PATH = str(cwd.parent)
else:
    BASE_PATH = str(cwd)

if BASE_PATH not in sys.path:
    sys.path.insert(0, BASE_PATH)

from src.processing.hdf5_dataset import HDF5Dataset, create_hdf5_dataloaders
import torch

HDF5_PATH = os.path.join(BASE_PATH, "notebooks/processed_data/crawled/dataset.h5")

if not os.path.exists(HDF5_PATH):
    print(f"❌ HDF5 file not found at {HDF5_PATH}")
    sys.exit(1)

print(f"Loading dataset from: {HDF5_PATH}")

# Test individual dataset
print("\n1. Testing HDF5Dataset (individual samples):")
dataset = HDF5Dataset(HDF5_PATH, split="train")
text, image, label = dataset[0]

print(f"   Text shape:   {text.shape}  (expected: [768, 512])")
print(f"   Image shape:  {image.shape}  (expected: [2048])")
print(f"   Label:        {label}")

# Verify shapes
assert text.shape == torch.Size([768, 512]), f"Text shape mismatch! Got {text.shape}"
assert image.shape == torch.Size([2048]), f"Image shape mismatch! Got {image.shape}"
print("   ✓ Individual sample shapes are correct!")

# Test DataLoader batching
print("\n2. Testing DataLoader (batched samples):")
train_loader, val_loader, test_loader = create_hdf5_dataloaders(
    HDF5_PATH, batch_size=4, num_workers=0
)

batch_text, batch_image, batch_label = next(iter(train_loader))
print(f"   Batch text shape:   {batch_text.shape}  (expected: [4, 768, 512])")
print(f"   Batch image shape:  {batch_image.shape}  (expected: [4, 2048])")
print(f"   Batch label shape:  {batch_label.shape}  (expected: [4])")

assert batch_text.shape == torch.Size(
    [4, 768, 512]
), f"Batch text shape mismatch! Got {batch_text.shape}"
assert batch_image.shape == torch.Size(
    [4, 2048]
), f"Batch image shape mismatch! Got {batch_image.shape}"
print("   ✓ Batch shapes are correct!")

# Test that FastCNN can process the text (using same patching as notebook)
print("\n3. Testing FastCNN processing (notebook-style patching):")
import torch.nn as nn


class PatchedFastCNN(nn.Module):
    """FastCNN without internal permute - matches notebook's _patch_cnn approach."""

    def __init__(
        self, input_dim: int = 768, channel: int = 32, kernel_size: tuple = (1, 2, 4, 8)
    ):
        super().__init__()
        self.fast_cnn = nn.ModuleList()
        for kernel in kernel_size:
            self.fast_cnn.append(
                nn.Sequential(
                    nn.Conv1d(input_dim, channel, kernel_size=kernel),
                    nn.BatchNorm1d(channel),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.AdaptiveMaxPool1d(1),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No permute! Input should already be (B, C, L) = (B, 768, 512)
        x_out = []
        for module in self.fast_cnn:
            x_out.append(module(x).squeeze(-1))
        x_out = torch.cat(x_out, 1)
        return x_out


# Test with patched FastCNN (no permute)
fast_cnn = PatchedFastCNN(input_dim=768, channel=32, kernel_size=(1, 2, 4, 8))
text_encoded = fast_cnn(batch_text)
print(f"   FastCNN input shape:  {batch_text.shape}")
print(f"   FastCNN output shape: {text_encoded.shape}")
print("   ✓ FastCNN can process transposed text features!")

print("\n✅ All verifications passed! The dimension fix is working correctly.")
print("\nSummary of the fix:")
print("  - HDF5 stores text as (512, 768) [seq_len, embed_dim]")
print("  - HDF5Dataset transposes to (768, 512) [channels, seq_len] for Conv1d")
print("  - Batched shape: (B, 768, 512) [batch, channels, seq_len]")
print("  - Notebook's FastCNN uses raw Conv1d (no internal permute)")
print("  - CLIP module pools to (B, 768) by averaging over sequence dimension")
