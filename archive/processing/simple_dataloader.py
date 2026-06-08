#!/usr/bin/env python3
"""
Simplified DataLoader for ViFactCheck Dataset

Clean, focused implementation using separate text and image processors.
"""

import os
import json
import torch
from typing import List, Dict, Tuple, Optional, Any, Union
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging

from .text_processor import TextProcessor
from .image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class ViFactCheckDataset(Dataset):
    """
    Simplified ViFactCheck dataset using separate text and image processors.
    """
    
    def __init__(self,
                 data_path: str,
                 image_base_dir: str,
                 text_processor: Optional[TextProcessor] = None,
                 image_processor: Optional[ImageProcessor] = None,
                 label_mapping: Optional[Dict[str, int]] = None):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSON data file
            image_base_dir: Base directory containing images
            text_processor: Text processor instance
            image_processor: Image processor instance
            label_mapping: Mapping from source to label
        """
        self.data_path = Path(data_path)
        self.image_base_dir = Path(image_base_dir)
        self.label_mapping = label_mapping or {}
        
        # Initialize processors
        self.text_processor = text_processor or TextProcessor()
        self.image_processor = image_processor or ImageProcessor()
        
        # Load data
        self.samples = self._load_data()
        
        logger.info(f"Dataset initialized with {len(self.samples)} samples")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            samples = []
            
            # Handle different data formats
            if isinstance(data, list):
                if data and 'text' in data[0] and 'image_path' in data[0]:
                    # Preprocessed format
                    samples = data
                else:
                    # Raw ViFactCheck format
                    samples = self._parse_raw_data(data)
            elif isinstance(data, dict) and 'images' in data:
                samples = self._parse_raw_data([data])
            
            # Add labels
            for sample in samples:
                if 'label' not in sample:
                    source = sample.get('source', 'unknown')
                    sample['label'] = self.label_mapping.get(source, 0)
            
            return samples
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return []
    
    def _parse_raw_data(self, raw_data: List[Dict]) -> List[Dict[str, Any]]:
        """Parse raw ViFactCheck format."""
        samples = []
        
        for entry_idx, entry in enumerate(raw_data):
            if not isinstance(entry, dict) or 'images' not in entry:
                continue
            
            for img_idx, img_data in enumerate(entry['images']):
                if not isinstance(img_data, dict):
                    continue
                
                caption = img_data.get('caption', '')
                folder_path = img_data.get('folder_path', '')
                
                if caption and folder_path:
                    # Extract source from path
                    source = 'unknown'
                    path_parts = Path(folder_path).parts
                    if len(path_parts) >= 2:
                        source = path_parts[1]
                    
                    sample = {
                        'text': caption,
                        'image_path': folder_path,
                        'source': source,
                        'sample_id': f"{entry_idx}_{img_idx}"
                    }
                    samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Process text
        text_result = self.text_processor.process(sample['text'])
        
        # Process image
        image_path = self.image_base_dir / sample['image_path']
        image_result = self.image_processor.process(image_path)
        
        return {
            'text_features': text_result['features'],
            'text_tokens': text_result['input_ids'],
            'text_mask': text_result['attention_mask'],
            'image_features': image_result['features'],
            'image_tensor': image_result['tensor'],
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'text': text_result['text'],
            'image_path': image_result['image_path'],
            'source': sample.get('source', 'unknown'),
            'sample_id': sample.get('sample_id', str(idx)),
            'text_valid': text_result['valid'],
            'image_valid': image_result['valid']
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching samples.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched data
    """
    return {
        'text_features': torch.stack([item['text_features'] for item in batch]),
        'text_tokens': torch.stack([item['text_tokens'] for item in batch]),
        'text_mask': torch.stack([item['text_mask'] for item in batch]),
        'image_features': torch.stack([item['image_features'] for item in batch]),
        'image_tensors': torch.stack([item['image_tensor'] for item in batch]),
        'labels': torch.stack([item['label'] for item in batch]),
        'texts': [item['text'] for item in batch],
        'image_paths': [item['image_path'] for item in batch],
        'sources': [item['source'] for item in batch],
        'sample_ids': [item['sample_id'] for item in batch],
        'text_valid': [item['text_valid'] for item in batch],
        'image_valid': [item['image_valid'] for item in batch]
    }


def create_dataloader(data_path: str,
                     image_base_dir: str,
                     batch_size: int = 32,
                     shuffle: bool = False,
                     num_workers: int = 2,
                     text_max_length: int = 128,
                     image_size: Tuple[int, int] = (224, 224),
                     image_feature_dim: int = 512,
                     label_mapping: Optional[Dict[str, int]] = None) -> DataLoader:
    """
    Create a simple dataloader for ViFactCheck data.
    
    Args:
        data_path: Path to data JSON file
        image_base_dir: Base directory containing images
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        text_max_length: Maximum text sequence length
        image_size: Target image size
        image_feature_dim: Image feature dimension
        label_mapping: Source to label mapping
        
    Returns:
        DataLoader instance
    """
    # Create processors
    text_processor = TextProcessor(max_length=text_max_length)
    image_processor = ImageProcessor(
        image_size=image_size,
        feature_dim=image_feature_dim
    )
    
    # Create dataset
    dataset = ViFactCheckDataset(
        data_path=data_path,
        image_base_dir=image_base_dir,
        text_processor=text_processor,
        image_processor=image_processor,
        label_mapping=label_mapping
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    logger.info(f"Created dataloader with {len(dataset)} samples")
    return dataloader


def create_train_val_test_loaders(train_path: str,
                                 val_path: str,
                                 test_path: str,
                                 image_base_dir: str,
                                 batch_size: int = 32,
                                 num_workers: int = 2,
                                 label_mapping: Optional[Dict[str, int]] = None,
                                 **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data
        image_base_dir: Base directory containing images
        batch_size: Batch size
        num_workers: Number of workers
        label_mapping: Source to label mapping
        **kwargs: Additional arguments for dataloader creation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = create_dataloader(
        data_path=train_path,
        image_base_dir=image_base_dir,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        label_mapping=label_mapping,
        **kwargs
    )
    
    val_loader = create_dataloader(
        data_path=val_path,
        image_base_dir=image_base_dir,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        label_mapping=label_mapping,
        **kwargs
    )
    
    test_loader = create_dataloader(
        data_path=test_path,
        image_base_dir=image_base_dir,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        label_mapping=label_mapping,
        **kwargs
    )
    
    logger.info("Created train/val/test dataloaders")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    label_mapping = {
        'thanh_nien': 0,
        'vn_express': 0,
        'dan_tri': 1,
        'bao_tin_tuc': 1
    }
    
    dataloader = create_dataloader(
        data_path="path/to/data.json",
        image_base_dir="path/to/images",
        batch_size=16,
        label_mapping=label_mapping
    )
    
    # Test the dataloader
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("Text features shape:", batch['text_features'].shape)
        print("Image features shape:", batch['image_features'].shape)
        print("Labels shape:", batch['labels'].shape)
        break
