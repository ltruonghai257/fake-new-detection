#!/usr/bin/env python3
"""
Diagnostic script to check ViFactCheck label availability.
Run this to find where your ViFactCheck labels are stored.
"""
import json
from pathlib import Path

data_root = Path("/Users/haila/Library/CloudStorage/GoogleDrive-latruonghai@gmail.com/My Drive/Thesis_Final/fake-news-data-for-thesis")

# Check various possible locations
possible_locations = [
    data_root / "data" / "json" / "vifactcheck_claims.json",
    data_root / "data" / "json" / "claims.json",
    data_root / "data" / "vifactcheck" / "train.json",
    data_root / "data" / "vifactcheck" / "dev.json",
    data_root / "data" / "vifactcheck" / "test.json",
    data_root / "raw" / "vifactcheck_train.json",
    data_root / "raw" / "vifactcheck_dev.json",
    data_root / "raw" / "vifactcheck_test.json",
]

print("Checking for ViFactCheck claim files...")
for loc in possible_locations:
    if loc.exists():
        print(f"  ✓ Found: {loc}")
        # Peek at structure
        with open(loc) as f:
            data = json.load(f)
            if isinstance(data, list) and data:
                print(f"     Records: {len(data)}")
                print(f"     First record keys: {list(data[0].keys())[:5]}")
            elif isinstance(data, dict):
                print(f"     Top-level keys: {list(data.keys())[:5]}")
    else:
        print(f"  ✗ Not found: {loc}")

# Check if labels might be in the HDF5 files
print("\nChecking HDF5 files for labels...")
import h5py
hdf5_dir = data_root / "processed_data" / "hdf5"
for split in ["train", "dev", "test"]:
    hdf5_path = hdf5_dir / f"coolant_{split}.h5"
    if hdf5_path.exists():
        with h5py.File(hdf5_path, "r") as f:
            datasets = list(f.keys())
            print(f"  {hdf5_path.name}: datasets = {datasets}")
            if "source_labels" in f:
                labels = f["source_labels"][:10]
                print(f"    First 10 source_labels: {labels}")

print("\n" + "="*60)
print("RECOMMENDATION:")
print("If you have ViFactCheck labels in a separate file, you need to")
print("merge them with the article JSONs before running notebook 04.")
print("\nIf you don't have labels yet, you need to:")
print("1. Run ViFactCheck claim verification on your articles, or")
print("2. Use a dataset that already has ground-truth labels")
