"""
Data Utilities for COOLANT Fake News Detection
Utility functions for loading, saving, and managing preprocessed data
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
import shutil
from pathlib import Path


class DataManager:
    """
    Utility class for managing preprocessed data files
    """
    
    def __init__(self, base_dir: str = "./processed_data"):
        """
        Initialize data manager
        
        Args:
            base_dir: Base directory for processed data
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def save_pickle(self, data: Any, filename: str, subfolder: str = "") -> str:
        """
        Save data to pickle file
        
        Args:
            data: Data to save
            filename: Filename
            subfolder: Subfolder within base directory
            
        Returns:
            Full path to saved file
        """
        save_dir = self.base_dir / subfolder if subfolder else self.base_dir
        save_dir.mkdir(exist_ok=True)
        
        filepath = save_dir / filename
        if not filename.endswith('.pkl'):
            filepath = filepath.with_suffix('.pkl')
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved data to {filepath}")
        return str(filepath)
    
    def load_pickle(self, filename: str, subfolder: str = "") -> Any:
        """
        Load data from pickle file
        
        Args:
            filename: Filename
            subfolder: Subfolder within base directory
            
        Returns:
            Loaded data
        """
        load_dir = self.base_dir / subfolder if subfolder else self.base_dir
        filepath = load_dir / filename
        
        if not filename.endswith('.pkl'):
            filepath = filepath.with_suffix('.pkl')
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded data from {filepath}")
        return data
    
    def save_npz(self, data_dict: Dict[str, np.ndarray], filename: str, subfolder: str = "") -> str:
        """
        Save numpy arrays to npz file
        
        Args:
            data_dict: Dictionary of numpy arrays to save
            filename: Filename
            subfolder: Subfolder within base directory
            
        Returns:
            Full path to saved file
        """
        save_dir = self.base_dir / subfolder if subfolder else self.base_dir
        save_dir.mkdir(exist_ok=True)
        
        filepath = save_dir / filename
        if not filename.endswith('.npz'):
            filepath = filepath.with_suffix('.npz')
        
        np.savez(filepath, **data_dict)
        print(f"Saved numpy data to {filepath}")
        return str(filepath)
    
    def load_npz(self, filename: str, subfolder: str = "") -> Dict[str, np.ndarray]:
        """
        Load numpy arrays from npz file
        
        Args:
            filename: Filename
            subfolder: Subfolder within base directory
            
        Returns:
            Dictionary of loaded numpy arrays
        """
        load_dir = self.base_dir / subfolder if subfolder else self.base_dir
        filepath = load_dir / filename
        
        if not filename.endswith('.npz'):
            filepath = filepath.with_suffix('.npz')
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        loaded = np.load(filepath)
        data_dict = {key: loaded[key] for key in loaded.files}
        print(f"Loaded numpy data from {filepath}")
        return data_dict
    
    def save_json(self, data: Dict, filename: str, subfolder: str = "") -> str:
        """
        Save data to JSON file
        
        Args:
            data: Dictionary to save
            filename: Filename
            subfolder: Subfolder within base directory
            
        Returns:
            Full path to saved file
        """
        save_dir = self.base_dir / subfolder if subfolder else self.base_dir
        save_dir.mkdir(exist_ok=True)
        
        filepath = save_dir / filename
        if not filename.endswith('.json'):
            filepath = filepath.with_suffix('.json')
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved JSON data to {filepath}")
        return str(filepath)
    
    def load_json(self, filename: str, subfolder: str = "") -> Dict:
        """
        Load data from JSON file
        
        Args:
            filename: Filename
            subfolder: Subfolder within base directory
            
        Returns:
            Loaded dictionary
        """
        load_dir = self.base_dir / subfolder if subfolder else self.base_dir
        filepath = load_dir / filename
        
        if not filename.endswith('.json'):
            filepath = filepath.with_suffix('.json')
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded JSON data from {filepath}")
        return data
    
    def list_files(self, subfolder: str = "", extension: str = "") -> List[str]:
        """
        List files in directory
        
        Args:
            subfolder: Subfolder within base directory
            extension: File extension to filter by
            
        Returns:
            List of file paths
        """
        search_dir = self.base_dir / subfolder if subfolder else self.base_dir
        
        if extension:
            if not extension.startswith('.'):
                extension = '.' + extension
            files = list(search_dir.glob(f"*{extension}"))
        else:
            files = list(search_dir.glob("*"))
        
        return [str(f) for f in files if f.is_file()]
    
    def copy_file(self, src: str, dest_subfolder: str = "", dest_filename: str = "") -> str:
        """
        Copy file to data directory
        
        Args:
            src: Source file path
            dest_subfolder: Destination subfolder
            dest_filename: Destination filename (optional)
            
        Returns:
            Full path to copied file
        """
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src}")
        
        dest_dir = self.base_dir / dest_subfolder if dest_subfolder else self.base_dir
        dest_dir.mkdir(exist_ok=True)
        
        if dest_filename:
            dest_path = dest_dir / dest_filename
        else:
            dest_path = dest_dir / src_path.name
        
        shutil.copy2(src_path, dest_path)
        print(f"Copied file to {dest_path}")
        return str(dest_path)
    
    def delete_file(self, filename: str, subfolder: str = ""):
        """
        Delete file from data directory
        
        Args:
            filename: Filename
            subfolder: Subfolder within base directory
        """
        delete_dir = self.base_dir / subfolder if subfolder else self.base_dir
        filepath = delete_dir / filename
        
        if filepath.exists():
            filepath.unlink()
            print(f"Deleted file: {filepath}")
        else:
            print(f"File not found: {filepath}")
    
    def get_file_info(self, filename: str, subfolder: str = "") -> Dict[str, Any]:
        """
        Get file information
        
        Args:
            filename: Filename
            subfolder: Subfolder within base directory
            
        Returns:
            Dictionary with file information
        """
        file_dir = self.base_dir / subfolder if subfolder else self.base_dir
        filepath = file_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        stat = filepath.stat()
        return {
            'name': filepath.name,
            'path': str(filepath),
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'created': stat.st_ctime,
            'modified': stat.st_mtime
        }


class DatasetSplitter:
    """
    Utility class for splitting datasets
    """
    
    @staticmethod
    def split_data(data: List[Any], 
                  labels: List[int],
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  random_seed: int = 42,
                  stratify: bool = True) -> Tuple[List[Any], List[Any], List[Any], List[int], List[int], List[int]]:
        """
        Split data into train, validation, and test sets
        
        Args:
            data: List of data samples
            labels: List of corresponding labels
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_seed: Random seed for reproducibility
            stratify: Whether to stratify by labels
            
        Returns:
            Tuple of (train_data, val_data, test_data, train_labels, val_labels, test_labels)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        np.random.seed(random_seed)
        
        if stratify:
            # Group by label
            label_groups = {}
            for i, label in enumerate(labels):
                if label not in label_groups:
                    label_groups[label] = []
                label_groups[label].append(i)
            
            train_indices, val_indices, test_indices = [], [], []
            
            for label, indices in label_groups.items():
                np.random.shuffle(indices)
                n = len(indices)
                
                train_end = int(n * train_ratio)
                val_end = train_end + int(n * val_ratio)
                
                train_indices.extend(indices[:train_end])
                val_indices.extend(indices[train_end:val_end])
                test_indices.extend(indices[val_end:])
        else:
            # Random split
            indices = np.random.permutation(len(data))
            n = len(data)
            
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]
        
        # Create splits
        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        test_data = [data[i] for i in test_indices]
        
        train_labels = [labels[i] for i in train_indices]
        val_labels = [labels[i] for i in val_indices]
        test_labels = [labels[i] for i in test_indices]
        
        return train_data, val_data, test_data, train_labels, val_labels, test_labels


class DataValidator:
    """
    Utility class for validating preprocessed data
    """
    
    @staticmethod
    def validate_text_features(features: np.ndarray, expected_shape: Optional[Tuple] = None) -> Dict[str, Any]:
        """
        Validate text feature array
        
        Args:
            features: Text feature array
            expected_shape: Expected shape (optional)
            
        Returns:
            Validation report
        """
        report = {
            'valid': True,
            'shape': features.shape,
            'dtype': features.dtype,
            'min': float(np.min(features)),
            'max': float(np.max(features)),
            'mean': float(np.mean(features)),
            'std': float(np.std(features)),
            'has_nan': bool(np.isnan(features).any()),
            'has_inf': bool(np.isinf(features).any()),
            'warnings': [],
            'errors': []
        }
        
        if expected_shape:
            if features.shape != expected_shape:
                report['errors'].append(f"Shape mismatch: expected {expected_shape}, got {features.shape}")
                report['valid'] = False
        
        if report['has_nan']:
            report['warnings'].append("Contains NaN values")
        
        if report['has_inf']:
            report['warnings'].append("Contains infinite values")
        
        if features.dtype not in [np.float32, np.float64]:
            report['warnings'].append(f"Unexpected dtype: {features.dtype}")
        
        return report
    
    @staticmethod
    def validate_image_features(features: np.ndarray, expected_dim: int = 512) -> Dict[str, Any]:
        """
        Validate image feature array
        
        Args:
            features: Image feature array
            expected_dim: Expected feature dimension
            
        Returns:
            Validation report
        """
        report = {
            'valid': True,
            'shape': features.shape,
            'dtype': features.dtype,
            'min': float(np.min(features)),
            'max': float(np.max(features)),
            'mean': float(np.mean(features)),
            'std': float(np.std(features)),
            'has_nan': bool(np.isnan(features).any()),
            'has_inf': bool(np.isinf(features).any()),
            'warnings': [],
            'errors': []
        }
        
        if len(features.shape) != 2:
            report['errors'].append(f"Expected 2D array, got {len(features.shape)}D")
            report['valid'] = False
        
        if features.shape[1] != expected_dim:
            report['warnings'].append(f"Feature dimension {features.shape[1]} differs from expected {expected_dim}")
        
        if report['has_nan']:
            report['warnings'].append("Contains NaN values")
        
        if report['has_inf']:
            report['warnings'].append("Contains infinite values")
        
        return report
    
    @staticmethod
    def validate_labels(labels: np.ndarray, num_classes: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate label array
        
        Args:
            labels: Label array
            num_classes: Expected number of classes (optional)
            
        Returns:
            Validation report
        """
        unique_labels = np.unique(labels)
        
        report = {
            'valid': True,
            'shape': labels.shape,
            'dtype': labels.dtype,
            'num_samples': len(labels),
            'unique_labels': unique_labels.tolist(),
            'num_classes': len(unique_labels),
            'class_distribution': {int(label): int(np.sum(labels == label)) for label in unique_labels},
            'warnings': [],
            'errors': []
        }
        
        if num_classes and len(unique_labels) != num_classes:
            report['warnings'].append(f"Number of classes {len(unique_labels)} differs from expected {num_classes}")
        
        if labels.dtype not in [np.int32, np.int64]:
            report['warnings'].append(f"Unexpected dtype: {labels.dtype}")
        
        # Check for negative labels
        if np.any(labels < 0):
            report['errors'].append("Contains negative labels")
            report['valid'] = False
        
        return report


def create_sample_data(num_samples: int = 1000, 
                      save_dir: str = "./sample_data") -> Dict[str, str]:
    """
    Create sample data for testing
    
    Args:
        num_samples: Number of samples to create
        save_dir: Directory to save sample data
        
    Returns:
        Dictionary with paths to created files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate sample texts
    texts = [
        f"This is sample news article number {i} about various topics."
        for i in range(num_samples)
    ]
    
    # Generate sample labels (0: real, 1: fake)
    labels = np.random.randint(0, 2, num_samples).tolist()
    
    # Generate sample image paths
    image_paths = [f"image_{i}.jpg" for i in range(num_samples)]
    
    # Create metadata
    metadata = {
        'num_samples': num_samples,
        'texts': texts,
        'image_paths': image_paths,
        'labels': labels,
        'created_for': 'testing'
    }
    
    # Save data
    manager = DataManager(save_dir)
    metadata_path = manager.save_json(metadata, "sample_metadata.json")
    
    # Create sample data in COOLANT format
    sample_data = []
    for i in range(num_samples):
        sample_data.append({
            'id': i,
            'text': texts[i],
            'image_name': image_paths[i],
            'label': labels[i]
        })
    
    data_path = manager.save_json(sample_data, "sample_data.json")
    
    return {
        'metadata': metadata_path,
        'data': data_path,
        'directory': save_dir
    }


def convert_format(input_path: str, 
                  output_path: str, 
                  input_format: str = "pkl",
                  output_format: str = "npz"):
    """
    Convert data between pickle and numpy formats
    
    Args:
        input_path: Input file path
        output_path: Output file path
        input_format: Input format ('pkl' or 'npz')
        output_format: Output format ('pkl' or 'npz')
    """
    if input_format == "pkl":
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
    elif input_format == "npz":
        data = dict(np.load(input_path))
    else:
        raise ValueError("input_format must be 'pkl' or 'npz'")
    
    if output_format == "pkl":
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
    elif output_format == "npz":
        np.savez(output_path, **data)
    else:
        raise ValueError("output_format must be 'pkl' or 'npz'")
    
    print(f"Converted {input_path} from {input_format} to {output_path}: {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Data Utilities for COOLANT")
    
    # Initialize data manager
    manager = DataManager("./test_data")
    
    # Save and load test data
    test_data = {
        'features': np.random.randn(100, 512),
        'labels': np.random.randint(0, 2, 100)
    }
    
    # Save as pickle
    pickle_path = manager.save_pickle(test_data, "test_data.pkl")
    
    # Load pickle
    loaded_pickle = manager.load_pickle("test_data.pkl")
    print(f"Loaded pickle data shape: {loaded_pickle['features'].shape}")
    
    # Save as numpy
    npz_path = manager.save_npz(test_data, "test_data.npz")
    
    # Load numpy
    loaded_npz = manager.load_npz("test_data.npz")
    print(f"Loaded numpy data shape: {loaded_npz['features'].shape}")
    
    # Validate data
    validator = DataValidator()
    text_report = validator.validate_text_features(test_data['features'])
    image_report = validator.validate_image_features(test_data['features'])
    label_report = validator.validate_labels(test_data['labels'])
    
    print("Text validation report:", text_report['valid'])
    print("Image validation report:", image_report['valid'])
    print("Label validation report:", label_report['valid'])
    
    # Create sample data
    sample_info = create_sample_data(100)
    print("Created sample data:", sample_info)
    
    print("Data utilities test completed successfully!")
