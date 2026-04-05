#!/usr/bin/env python3
"""
Convert NPZ dataset to HDF5 format with compression for memory-efficient loading.
"""

import numpy as np
import h5py
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse


def convert_npz_to_hdf5(npz_path: str, hdf5_path: str, train_frac=0.8, val_frac=0.1, seed=42):
    """
    Convert NPZ file to HDF5 with train/val/test splits.
    
    Args:
        npz_path: Path to input NPZ file
        hdf5_path: Path to output HDF5 file
        train_frac: Fraction for training set
        val_frac: Fraction for validation set
        seed: Random seed for reproducibility
    """
    print(f"Loading NPZ file: {npz_path}")
    
    # Load with memory mapping to avoid loading everything into RAM
    npz = np.load(npz_path, mmap_mode='r')
    
    text_features = npz['text_features']   # (N, 512, 768)
    image_features = npz['image_features']  # (N, 2048)
    labels = npz['labels']                  # (N,)
    
    n_samples = len(labels)
    print(f"Dataset size: {n_samples} samples")
    print(f"Text features: {text_features.shape}")
    print(f"Image features: {image_features.shape}")
    
    # Create train/val/test splits
    indices = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(
        indices, test_size=(1 - train_frac), random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - val_frac / (1 - train_frac)), random_state=seed
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_idx)} ({len(train_idx)/n_samples*100:.1f}%)")
    print(f"  Val:   {len(val_idx)} ({len(val_idx)/n_samples*100:.1f}%)")
    print(f"  Test:  {len(test_idx)} ({len(test_idx)/n_samples*100:.1f}%)")
    
    # Create HDF5 file with chunked, compressed datasets
    print(f"\nCreating HDF5 file: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'w') as f:
        # Store split indices
        f.create_dataset('train_idx', data=train_idx)
        f.create_dataset('val_idx', data=val_idx)
        f.create_dataset('test_idx', data=test_idx)
        
        # Create chunked datasets with compression
        # Chunks of 64 samples balance I/O efficiency and memory
        chunk_size = min(64, n_samples)
        
        # Text features: large array, use compression
        f.create_dataset(
            'text_features',
            shape=text_features.shape,
            dtype=text_features.dtype,
            chunks=(chunk_size, text_features.shape[1], text_features.shape[2]),
            compression='gzip',
            compression_opts=4,  # Moderate compression (speed vs size tradeoff)
            data=text_features
        )
        
        # Image features: smaller, still compress
        f.create_dataset(
            'image_features',
            shape=image_features.shape,
            dtype=image_features.dtype,
            chunks=(chunk_size, image_features.shape[1]),
            compression='gzip',
            compression_opts=4,
            data=image_features
        )
        
        # Labels: small, no compression needed
        f.create_dataset(
            'labels',
            shape=labels.shape,
            dtype=labels.dtype,
            data=labels
        )
        
        # Store metadata
        f.attrs['n_samples'] = n_samples
        f.attrs['train_size'] = len(train_idx)
        f.attrs['val_size'] = len(val_idx)
        f.attrs['test_size'] = len(test_idx)
        f.attrs['text_shape'] = text_features.shape
        f.attrs['image_shape'] = image_features.shape
    
    print(f"\nHDF5 file created successfully!")
    
    # Print file sizes for comparison
    npz_size = Path(npz_path).stat().st_size / 1e9
    hdf5_size = Path(hdf5_path).stat().st_size / 1e9
    print(f"\nFile sizes:")
    print(f"  NPZ:  {npz_size:.2f} GB")
    print(f"  HDF5: {hdf5_size:.2f} GB")
    print(f"  Ratio: {hdf5_size/npz_size*100:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert NPZ to HDF5')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input NPZ file path')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output HDF5 file path')
    parser.add_argument('--train-frac', type=float, default=0.8,
                        help='Training fraction (default: 0.8)')
    parser.add_argument('--val-frac', type=float, default=0.1,
                        help='Validation fraction (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    convert_npz_to_hdf5(
        args.input,
        args.output,
        args.train_frac,
        args.val_frac,
        args.seed
    )
