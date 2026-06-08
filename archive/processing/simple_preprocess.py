#!/usr/bin/env python3
"""
Simplified Preprocessing for ViFactCheck Dataset

Clean, focused preprocessing using separate text and image processors.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

from .text_processor import TextProcessor
from .image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class SimplePreprocessor:
    """
    Simplified preprocessor for ViFactCheck data.
    """
    
    def __init__(self,
                 json_path: str,
                 image_base_dir: str,
                 output_dir: str = "preprocessed"):
        """
        Initialize preprocessor.
        
        Args:
            json_path: Path to ViFactCheck JSON file
            image_base_dir: Base directory containing images
            output_dir: Output directory for processed data
        """
        self.json_path = Path(json_path)
        self.image_base_dir = Path(image_base_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processors
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        
        # Statistics
        self.stats = {
            'total_entries': 0,
            'valid_samples': 0,
            'invalid_text': 0,
            'invalid_images': 0
        }
    
    def load_raw_data(self) -> List[Dict[str, Any]]:
        """Load raw ViFactCheck data."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                data = [data]
            
            self.stats['total_entries'] = len(data)
            logger.info(f"Loaded {len(data)} raw entries")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return []
    
    def extract_samples(self, raw_data: List[Dict]) -> List[Dict[str, Any]]:
        """Extract samples from raw ViFactCheck data."""
        samples = []
        
        for entry_idx, entry in enumerate(raw_data):
            if not isinstance(entry, dict) or 'images' not in entry:
                continue
            
            for img_idx, img_data in enumerate(entry['images']):
                if not isinstance(img_data, dict):
                    continue
                
                caption = img_data.get('caption', '')
                folder_path = img_data.get('folder_path', '')
                
                if not caption or not folder_path:
                    continue
                
                # Extract source from path
                source = 'unknown'
                path_parts = Path(folder_path).parts
                if len(path_parts) >= 2:
                    source = path_parts[1]
                
                sample = {
                    'text': caption,
                    'image_path': folder_path,
                    'source': source,
                    'sample_id': f"{entry_idx}_{img_idx}",
                    'entry_idx': entry_idx,
                    'img_idx': img_idx
                }
                samples.append(sample)
        
        logger.info(f"Extracted {len(samples)} samples")
        return samples
    
    def validate_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate text and images."""
        valid_samples = []
        
        for sample in samples:
            # Validate text
            cleaned_text = self.text_processor.clean_text(sample['text'])
            if not cleaned_text:
                self.stats['invalid_text'] += 1
                continue
            
            # Validate image
            image_path = self.image_base_dir / sample['image_path']
            if not self.image_processor.validate_image(image_path):
                self.stats['invalid_images'] += 1
                continue
            
            # Update sample with cleaned text
            sample['text'] = cleaned_text
            sample['text_length'] = len(cleaned_text.split())
            valid_samples.append(sample)
            self.stats['valid_samples'] += 1
        
        logger.info(f"Validated {len(valid_samples)} samples")
        return valid_samples
    
    def create_splits(self, 
                     samples: List[Dict[str, Any]],
                     test_size: float = 0.2,
                     val_size: float = 0.1,
                     random_state: int = 42) -> Tuple[List, List, List]:
        """Create train/val/test splits."""
        if not samples:
            return [], [], []
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            samples, 
            test_size=test_size, 
            random_state=random_state,
            stratify=[s['source'] for s in samples]
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=[s['source'] for s in train_val]
        )
        
        logger.info(f"Created splits: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test
    
    def save_splits(self, train: List, val: List, test: List):
        """Save data splits."""
        splits = {'train': train, 'val': val, 'test': test}
        
        for split_name, split_data in splits.items():
            # Save JSON
            json_path = self.output_dir / f"{split_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            
            # Save CSV
            df = pd.DataFrame(split_data)
            csv_path = self.output_dir / f"{split_name}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Save statistics
        stats_path = self.output_dir / "stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Saved splits to {self.output_dir}")
    
    def process(self, 
                test_size: float = 0.2,
                val_size: float = 0.1,
                random_state: int = 42) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline.
        
        Args:
            test_size: Test set size
            val_size: Validation set size  
            random_state: Random seed
            
        Returns:
            Processing results
        """
        logger.info("Starting preprocessing...")
        
        # Load raw data
        raw_data = self.load_raw_data()
        if not raw_data:
            return {}
        
        # Extract samples
        samples = self.extract_samples(raw_data)
        if not samples:
            return {}
        
        # Validate samples
        valid_samples = self.validate_samples(samples)
        if not valid_samples:
            return {}
        
        # Create splits
        train, val, test = self.create_splits(
            valid_samples, test_size, val_size, random_state
        )
        
        # Save splits
        self.save_splits(train, val, test)
        
        # Log statistics
        self._log_stats()
        
        return {
            'total_samples': len(valid_samples),
            'train_samples': len(train),
            'val_samples': len(val),
            'test_samples': len(test),
            'output_dir': str(self.output_dir),
            'files': {
                'train': str(self.output_dir / "train.json"),
                'val': str(self.output_dir / "val.json"),
                'test': str(self.output_dir / "test.json"),
                'stats': str(self.output_dir / "stats.json")
            }
        }
    
    def _log_stats(self):
        """Log preprocessing statistics."""
        logger.info("=== Preprocessing Statistics ===")
        logger.info(f"Total entries: {self.stats['total_entries']}")
        logger.info(f"Valid samples: {self.stats['valid_samples']}")
        logger.info(f"Invalid text: {self.stats['invalid_text']}")
        logger.info(f"Invalid images: {self.stats['invalid_images']}")


def preprocess_vifactcheck(json_path: str,
                          image_base_dir: str,
                          output_dir: str = "preprocessed",
                          test_size: float = 0.2,
                          val_size: float = 0.1,
                          random_state: int = 42) -> Dict[str, Any]:
    """
    Convenience function for preprocessing ViFactCheck data.
    
    Args:
        json_path: Path to ViFactCheck JSON file
        image_base_dir: Base directory containing images
        output_dir: Output directory
        test_size: Test set size
        val_size: Validation set size
        random_state: Random seed
        
    Returns:
        Processing results
    """
    preprocessor = SimplePreprocessor(json_path, image_base_dir, output_dir)
    return preprocessor.process(test_size, val_size, random_state)


if __name__ == "__main__":
    # Example usage
    results = preprocess_vifactcheck(
        json_path="src/data/json/news_data_vifactcheck_train.json",
        image_base_dir="src/data/jpg",
        output_dir="preprocessed_simple"
    )
    
    print("Preprocessing results:")
    for key, value in results.items():
        if key != 'files':
            print(f"  {key}: {value}")
    
    print("\nGenerated files:")
    for file_type, file_path in results.get('files', {}).items():
        print(f"  {file_type}: {file_path}")
