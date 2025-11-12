import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import logging
import hashlib
import re

logger = logging.getLogger(__name__)


class FakeNewsDataset(Dataset):
    """
    PyTorch Dataset for multimodal fake news detection.
    
    This dataset handles loading and preprocessing of text and image data
    on-the-fly, making it memory efficient for large datasets.
    """
    
    def __init__(self,
                 json_path: str,
                 image_base_dir: str,
                 transform: Optional[Any] = None,
                 text_processor: Optional[Any] = None,
                 image_processor: Optional[Any] = None,
                 max_length: int = 30,
                 embed_dim: int = 200,
                 feature_dim: int = 512,
                 labels: Optional[List[int]] = None,
                 cache_images: bool = True,
                 cache_text: bool = False):
        """
        Initialize the dataset.
        
        Args:
            json_path: Path to JSON data file
            image_base_dir: Base directory containing image folders
            transform: Optional custom image transforms
            text_processor: Optional custom text processor
            image_processor: Optional custom image processor
            max_length: Maximum text sequence length
            embed_dim: Text embedding dimension
            feature_dim: Image feature dimension
            labels: Optional list of labels
            cache_images: Whether to cache processed images
            cache_text: Whether to cache processed text
        """
        self.json_path = json_path
        self.image_base_dir = image_base_dir
        self.labels = labels
        self.cache_images = cache_images
        self.cache_text = cache_text
        
        # Initialize processors
        self.text_processor = text_processor or self._default_text_processor(max_length, embed_dim)
        self.image_processor = image_processor or self._default_image_processor(feature_dim)
        
        # Load and validate data
        self.data = self._load_data()
        
        # Initialize caches
        self.image_cache = {} if cache_images else None
        self.text_cache = {} if cache_text else None
        
        logger.info(f"Dataset initialized with {len(self.data)} samples")
    
    def _default_text_processor(self, max_length: int, embed_dim: int):
        """Create default text processor."""
        from .multimodal_processor import TextProcessor
        return TextProcessor(max_length=max_length, embed_dim=embed_dim)
    
    def _default_image_processor(self, feature_dim: int):
        """Create default image processor."""
        from .multimodal_processor import ImageProcessor
        return ImageProcessor(feature_dim=feature_dim)
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load and validate data from JSON file."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            if not isinstance(raw_data, list):
                raw_data = [raw_data]
            
            # Filter valid entries
            valid_data = []
            for i, item in enumerate(raw_data):
                if (item.get('title') and 
                    item.get('text') and 
                    item.get('title') != 'Error' and
                    'Could not crawl' not in item.get('text', '')):
                    
                    # Add label
                    if self.labels and i < len(self.labels):
                        item['label'] = self.labels[i]
                    else:
                        item['label'] = 0  # Default to real news
                    
                    valid_data.append(item)
            
            logger.info(f"Loaded {len(valid_data)} valid entries from {len(raw_data)} total")
            return valid_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return []
    
    def find_image_path(self, article: Dict[str, Any]) -> Optional[str]:
        """Find corresponding image for an article."""
        # Strategy 1: Use images field if available
        if article.get('images'):
            images = article['images']
            if isinstance(images, str):
                image_name = os.path.basename(images)
            elif isinstance(images, list) and images:
                image_name = os.path.basename(images[0]) if isinstance(images[0], str) else None
            elif isinstance(images, dict):
                image_name = os.path.basename(images.get('path', images.get('url', '')))
            else:
                image_name = None
            
            if image_name:
                # Search in all source directories
                for source_dir in os.listdir(self.image_base_dir):
                    source_path = os.path.join(self.image_base_dir, source_dir)
                    if os.path.isdir(source_path):
                        # Try exact match
                        image_path = os.path.join(source_path, image_name)
                        if os.path.exists(image_path):
                            return image_path
                        
                        # Try with different extensions
                        base_name = os.path.splitext(image_name)[0]
                        for ext in ['.jpg', '.jpeg', '.png']:
                            test_path = os.path.join(source_path, base_name + ext)
                            if os.path.exists(test_path):
                                return test_path
        
        # Strategy 2: Match by title hash
        title = article.get('title', '')
        if title:
            title_hash = hashlib.md5(title.encode()).hexdigest()[:12]
            
            for source_dir in os.listdir(self.image_base_dir):
                source_path = os.path.join(self.image_base_dir, source_dir)
                if os.path.isdir(source_path):
                    for file in os.listdir(source_path):
                        if (title_hash in file.lower() and 
                            file.lower().endswith(('.jpg', '.jpeg', '.png'))):
                            return os.path.join(source_path, file)
        
        # Strategy 3: Use any available image (fallback for testing)
        for source_dir in os.listdir(self.image_base_dir):
            source_path = os.path.join(self.image_base_dir, source_dir)
            if os.path.isdir(source_path):
                files = [f for f in os.listdir(source_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    return os.path.join(source_path, files[0])
        
        return None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary containing:
            - text_features: (max_length, embed_dim)
            - image_features: (feature_dim,)
            - label: (1,)
            - title: str
            - image_path: str
        """
        article = self.data[idx]
        
        # Process text (with caching)
        text = article.get('text', '')
        if self.cache_text and idx in self.text_cache:
            text_features = self.text_cache[idx]
        else:
            text_features = self.text_processor.preprocess_text(text)
            if self.cache_text:
                self.text_cache[idx] = text_features
        
        # Process image (with caching)
        image_path = self.find_image_path(article)
        if image_path:
            if self.cache_images and image_path in self.image_cache:
                image_features = self.image_cache[image_path]
            else:
                image_features = self.image_processor.preprocess_image(image_path)
                if self.cache_images:
                    self.image_cache[image_path] = image_features
        else:
            # Use random features if no image found
            image_features = torch.randn(self.image_processor.feature_dim)
            image_path = 'None'
        
        label = torch.tensor(article['label'], dtype=torch.long)
        
        return {
            'text_features': text_features,
            'image_features': image_features,
            'label': label,
            'title': article.get('title', ''),
            'image_path': image_path
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if len(self.data) == 0:
            return {}
        
        labels = [item['label'] for item in self.data]
        text_lengths = [len(item.get('text', '').split()) for item in self.data]
        
        return {
            'total_samples': len(self.data),
            'label_distribution': {
                'real': labels.count(0),
                'fake': labels.count(1)
            },
            'avg_text_length': np.mean(text_lengths),
            'max_text_length': max(text_lengths),
            'min_text_length': min(text_lengths)
        }


class FakeNewsDataLoader:
    """
    Enhanced DataLoader wrapper for fake news dataset with additional utilities.
    """
    
    def __init__(self,
                 dataset: FakeNewsDataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 2,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 collate_fn: Optional[Any] = None):
        """
        Initialize the data loader.
        
        Args:
            dataset: FakeNewsDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop the last incomplete batch
            collate_fn: Custom collate function
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        
        # Use default collate function if none provided
        self.collate_fn = collate_fn or self._default_collate_fn
        
        # Create PyTorch DataLoader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self.collate_fn
        )
    
    def _default_collate_fn(self, batch):
        """
        Default collate function for batching samples.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Dictionary with batched tensors
        """
        text_features = torch.stack([item['text_features'] for item in batch])
        image_features = torch.stack([item['image_features'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        
        return {
            'text_features': text_features,
            'image_features': image_features,
            'labels': labels,
            'titles': [item['title'] for item in batch],
            'image_paths': [item['image_path'] for item in batch]
        }
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    def get_batch_iterator(self):
        """Get iterator over batches."""
        return iter(self.dataloader)
    
    def get_sample_batch(self) -> Dict[str, torch.Tensor]:
        """Get a single sample batch for testing."""
        return next(iter(self.dataloader))


class PreprocessedDataset(Dataset):
    """
    Dataset for pre-computed features (memory efficient for training).
    """
    
    def __init__(self,
                 text_features: torch.Tensor,
                 image_features: torch.Tensor,
                 labels: torch.Tensor,
                 titles: Optional[List[str]] = None,
                 image_paths: Optional[List[str]] = None):
        """
        Initialize with pre-computed features.
        
        Args:
            text_features: Text features tensor (N, seq_len, embed_dim)
            image_features: Image features tensor (N, feature_dim)
            labels: Labels tensor (N,)
            titles: Optional list of titles
            image_paths: Optional list of image paths
        """
        assert len(text_features) == len(image_features) == len(labels), \
            "All tensors must have the same length"
        
        self.text_features = text_features
        self.image_features = image_features
        self.labels = labels
        self.titles = titles or [''] * len(labels)
        self.image_paths = image_paths or [''] * len(labels)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'text_features': self.text_features[idx],
            'image_features': self.image_features[idx],
            'label': self.labels[idx],
            'title': self.titles[idx],
            'image_path': self.image_paths[idx]
        }


def create_dataloaders(json_path: str,
                      image_base_dir: str,
                      batch_size: int = 32,
                      test_size: float = 0.2,
                      val_size: float = 0.1,
                      num_workers: int = 2,
                      labels: Optional[List[int]] = None,
                      random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to create train/val/test dataloaders.
    
    Args:
        json_path: Path to JSON data file
        image_base_dir: Base directory containing image folders
        batch_size: Batch size
        test_size: Fraction of data for test set
        val_size: Fraction of data for validation set
        num_workers: Number of worker processes
        labels: Optional list of labels
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Create full dataset
    full_dataset = FakeNewsDataset(
        json_path=json_path,
        image_base_dir=image_base_dir,
        labels=labels
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    test_size_abs = int(total_size * test_size)
    val_size_abs = int(total_size * val_size)
    train_size_abs = total_size - test_size_abs - val_size_abs
    
    # Create indices for splits
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size_abs]
    val_indices = indices[train_size_abs:train_size_abs + val_size_abs]
    test_indices = indices[train_size_abs + val_size_abs:]
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created dataloaders: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def create_sample_dataloader(batch_size: int = 32, 
                           num_samples: int = 1000) -> DataLoader:
    """
    Create a sample dataloader for testing models.
    
    Args:
        batch_size: Batch size
        num_samples: Number of samples
        
    Returns:
        DataLoader with sample data
    """
    # Generate sample data
    text_features = torch.randn(num_samples, 30, 200)
    image_features = torch.randn(num_samples, 512)
    labels = torch.randint(0, 2, (num_samples,))
    titles = [f"Sample article {i}" for i in range(num_samples)]
    
    # Create dataset
    dataset = PreprocessedDataset(
        text_features=text_features,
        image_features=image_features,
        labels=labels,
        titles=titles
    )
    
    def sample_collate_fn(batch):
        """Collate function for sample data."""
        text_features = torch.stack([item['text_features'] for item in batch])
        image_features = torch.stack([item['image_features'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        
        return {
            'text_features': text_features,
            'image_features': image_features,
            'labels': labels,
            'titles': [item['title'] for item in batch],
            'image_paths': [item['image_path'] for item in batch]
        }
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single-threaded for sample data
        pin_memory=False,
        collate_fn=sample_collate_fn
    )
    
    logger.info(f"Created sample dataloader with {num_samples} samples")
    
    return dataloader


# Utility functions for working with dataloaders
def get_dataloader_info(dataloader: DataLoader) -> Dict[str, Any]:
    """Get information about a dataloader."""
    dataset = dataloader.dataset
    
    info = {
        'batch_size': dataloader.batch_size,
        'num_batches': len(dataloader),
        'dataset_size': len(dataset),
        'num_workers': dataloader.num_workers,
        'pin_memory': dataloader.pin_memory
    }
    
    try:
        # Get a sample batch to check tensor shapes
        sample_batch = next(iter(dataloader))
        
        # Handle different batch formats (dict or tuple)
        if isinstance(sample_batch, dict):
            info.update({
                'text_features_shape': sample_batch['text_features'].shape,
                'image_features_shape': sample_batch['image_features'].shape,
                'labels_shape': sample_batch['labels'].shape
            })
        elif isinstance(sample_batch, (list, tuple)):
            # Standard PyTorch DataLoader format
            info.update({
                'text_features_shape': sample_batch[0].shape,
                'image_features_shape': sample_batch[1].shape,
                'labels_shape': sample_batch[2].shape
            })
        else:
            logger.warning("Unknown batch format in dataloader")
            
    except Exception as e:
        logger.error(f"Error getting sample batch: {e}")
        info.update({
            'text_features_shape': 'Unknown',
            'image_features_shape': 'Unknown',
            'labels_shape': 'Unknown'
        })
    
    return info


def preview_dataloader(dataloader: DataLoader, num_batches: int = 2) -> None:
    """Preview a few batches from the dataloader."""
    logger.info(f"Previewing {num_batches} batches from dataloader...")
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        logger.info(f"Batch {batch_idx + 1}:")
        
        # Handle different batch formats
        if isinstance(batch, dict):
            logger.info(f"  Text features: {batch['text_features'].shape}")
            logger.info(f"  Image features: {batch['image_features'].shape}")
            logger.info(f"  Labels: {batch['labels'].shape}")
            if 'titles' in batch:
                logger.info(f"  Sample titles: {batch['titles'][:2]}")
            logger.info(f"  Label distribution: {torch.bincount(batch['labels'])}")
        elif isinstance(batch, (list, tuple)):
            logger.info(f"  Text features: {batch[0].shape}")
            logger.info(f"  Image features: {batch[1].shape}")
            logger.info(f"  Labels: {batch[2].shape}")
            logger.info(f"  Label distribution: {torch.bincount(batch[2])}")
        else:
            logger.warning(f"  Unknown batch format: {type(batch)}")
