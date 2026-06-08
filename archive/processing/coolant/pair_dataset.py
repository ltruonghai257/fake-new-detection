"""
COOLANT Pair Dataset for training with matched/unmatched image-caption pairs.

Labels are created dynamically per batch:
- matched pair (caption_i, image_i) -> label 0 (Real)
- unmatched pair (caption_i, image_j) -> label 1 (Fake)
"""

import torch
import h5py
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Union

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.device import get_device


class CoolantPairDataset(Dataset):
    """
    Dataset for COOLANT training from pre-extracted features.

    Each sample is an (image, caption) pair. Labels are NOT stored —
    they are constructed dynamically during training by the collate/training loop.
    """

    def __init__(
        self,
        hdf5_path: str,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.device = get_device(str(device) if device else None)

        with h5py.File(self.hdf5_path, "r") as f:
            self.n_samples = f["caption_features"].shape[0]
            self.caption_shape = f["caption_features"].shape[1:]
            self.image_shape = f["image_features"].shape[1:]

            # Load article IDs for avoiding same-article negatives
            if "article_ids" in f:
                self.article_ids = f["article_ids"][:]
            else:
                self.article_ids = np.arange(self.n_samples)

        print(
            f"CoolantPairDataset: {self.n_samples} pairs from {self.hdf5_path.name}"
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        with h5py.File(self.hdf5_path, "r") as f:
            caption = f["caption_features"][idx]
            image = f["image_features"][idx]
            article_id = self.article_ids[idx]

        # Caption: transpose [seq_len, 768] -> [768, seq_len] for Conv1d
        caption_tensor = torch.from_numpy(caption).float().transpose(0, 1)
        image_tensor = torch.from_numpy(image).float()

        caption_tensor = caption_tensor.to(self.device)
        image_tensor = image_tensor.to(self.device)

        return caption_tensor, image_tensor, int(article_id)


def create_coolant_dataloaders(
    train_path: str,
    dev_path: str,
    test_path: str,
    batch_size: int = 32,
) -> Tuple[dict, dict]:
    """
    Create DataLoaders for COOLANT training.

    All loaders use shuffle=True for training (needed for diverse negatives)
    and shuffle=False for dev/test.

    Returns:
        (loaders_dict, datasets_dict)
    """
    train_ds = CoolantPairDataset(train_path)
    dev_ds = CoolantPairDataset(dev_path)
    test_ds = CoolantPairDataset(test_path)

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0
        ),
        "dev": DataLoader(
            dev_ds, batch_size=batch_size, shuffle=False, num_workers=0
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=0
        ),
    }
    datasets = {"train": train_ds, "dev": dev_ds, "test": test_ds}

    print(f"\nCOOLANT DataLoaders created:")
    print(f"  Train: {len(loaders['train'])} batches ({len(train_ds)} pairs)")
    print(f"  Dev:   {len(loaders['dev'])} batches ({len(dev_ds)} pairs)")
    print(f"  Test:  {len(loaders['test'])} batches ({len(test_ds)} pairs)")

    return loaders, datasets
