"""
HDF5 Dataset for memory-efficient loading of preprocessed features.
"""

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Callable


class HDF5Dataset(Dataset):
    """
    PyTorch Dataset that reads samples lazily from an HDF5 file.

    This avoids loading the entire dataset into RAM, making it suitable
    for large datasets that exceed available memory.
    """

    def __init__(
        self,
        hdf5_path: str,
        split: str = "train",
        transform_text: Optional[Callable] = None,
        transform_image: Optional[Callable] = None,
    ):
        """
        Initialize HDF5 dataset.

        Args:
            hdf5_path: Path to HDF5 file created by convert_npz_to_hdf5.py
            split: One of 'train', 'val', 'test'
            transform_text: Optional transform to apply to text features
            transform_image: Optional transform to apply to image features
        """
        self.hdf5_path = Path(hdf5_path)
        self.split = split
        self.transform_text = transform_text
        self.transform_image = transform_image

        # Open file to read metadata (file stays open for __getitem__ calls)
        with h5py.File(self.hdf5_path, "r") as f:
            # Get indices for this split
            split_idx = f[f"{split}_idx"][:]
            self.indices = split_idx

            # Read metadata
            self.n_samples = f.attrs["n_samples"]
            self.text_shape = f.attrs["text_shape"]
            self.image_shape = f.attrs["image_shape"]

            # Load labels into memory (small, always needed)
            all_labels = f["labels"][:]
            self.labels = all_labels[split_idx]

        self.size = len(self.indices)
        print(f"HDF5Dataset [{split}]: {self.size} samples")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the HDF5 file.

        Returns:
            (text_features, image_features, label) as tensors
        """
        # Map dataset index to actual sample index in HDF5
        actual_idx = self.indices[idx]

        # Open file on each access (HDF5 handles caching efficiently)
        with h5py.File(self.hdf5_path, "r") as f:
            # Read only the required sample
            text = f["text_features"][actual_idx]
            image = f["image_features"][actual_idx]
            label = self.labels[idx]  # Already in memory

        # Convert to tensors
        text_tensor = torch.from_numpy(text).float()
        image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Apply transforms if provided
        if self.transform_text:
            text_tensor = self.transform_text(text_tensor)
        if self.transform_image:
            image_tensor = self.transform_image(image_tensor)

        return text_tensor, image_tensor, label_tensor

    def get_sample_by_index(self, idx: int) -> dict:
        """
        Get a sample with additional metadata.

        Returns:
            Dictionary with features and metadata
        """
        actual_idx = self.indices[idx]

        with h5py.File(self.hdf5_path, "r") as f:
            text = f["text_features"][actual_idx]
            image = f["image_features"][actual_idx]
            label = self.labels[idx]

        return {
            "text_features": torch.from_numpy(text).float(),
            "image_features": torch.from_numpy(image).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "index": int(actual_idx),
            "split": self.split,
        }


class HDF5DatasetFull(Dataset):
    """
    Alternative implementation that keeps HDF5 file open (faster but less safe).
    Use this if opening/closing the file on each access causes slowdown.
    """

    def __init__(self, hdf5_path: str, split: str = "train"):
        """
        Initialize with file kept open.

        Note: This requires careful handling in DataLoader with num_workers=0
        """
        self.hdf5_path = Path(hdf5_path)
        self.split = split

        # Keep file open
        self.file = h5py.File(self.hdf5_path, "r")

        # Get indices
        split_idx = self.file[f"{split}_idx"][:]
        self.indices = split_idx

        # Read metadata
        self.n_samples = self.file.attrs["n_samples"]
        self.labels = self.file["labels"][split_idx]
        self.size = len(self.indices)

        print(f"HDF5DatasetFull [{split}]: {self.size} samples (file open)")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actual_idx = self.indices[idx]

        text = self.file["text_features"][actual_idx]
        image = self.file["image_features"][actual_idx]
        label = self.labels[idx]

        return (
            torch.from_numpy(text).float(),
            torch.from_numpy(image).float(),
            torch.tensor(label, dtype=torch.long),
        )

    def close(self):
        """Close the HDF5 file."""
        if hasattr(self, "file") and self.file:
            self.file.close()

    def __del__(self):
        """Destructor to ensure file is closed."""
        try:
            self.close()
        except Exception:
            pass


def create_hdf5_dataloaders(
    hdf5_path: str, batch_size: int = 32, num_workers: int = 0
) -> Tuple:
    """
    Create train/val/test DataLoaders from an HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file
        batch_size: Batch size for all loaders
        num_workers: Number of workers (must be 0 for HDF5)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_ds = HDF5Dataset(hdf5_path, split="train")
    val_ds = HDF5Dataset(hdf5_path, split="val")
    test_ds = HDF5Dataset(hdf5_path, split="test")

    # Create dataloaders (num_workers must be 0 for HDF5)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader)} batches")

    return train_loader, val_loader, test_loader
