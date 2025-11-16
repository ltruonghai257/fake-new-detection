"""
COOLANT Preprocessing Module

This module provides comprehensive preprocessing pipelines for multimodal fake news detection,
based on the COOLANT repository: https://github.com/wishever/COOLANT

Components:
- TextPreprocessor: BERT-based text preprocessing for English and Chinese
- ImagePreprocessor: ResNet-based image preprocessing
- CombinedPreprocessor: Integrated multimodal preprocessing
- DataManager: Utilities for data management and validation
"""

from .text_preprocessing import TextPreprocessor, TextDataset
from .image_preprocessing import ImagePreprocessor, ImageDataset, FeatureDataset
from .combined_preprocessing import CombinedPreprocessor, MultimodalDataset, create_dataloaders
from .data_utils import DataManager, DatasetSplitter, DataValidator, create_sample_data, convert_format

__version__ = "1.0.0"
__author__ = "COOLANT Preprocessing Team"

# Convenience functions for quick access
def preprocess_text_dataset(texts, labels, save_path=None, language="en", **kwargs):
    """
    Quick function to preprocess text dataset
    
    Args:
        texts: List of text strings
        labels: List of corresponding labels
        save_path: Path to save processed data (optional)
        language: Language code ('en' or 'zh')
        **kwargs: Additional arguments for TextPreprocessor
        
    Returns:
        Tuple of (features, labels)
    """
    preprocessor = TextPreprocessor(language=language, **kwargs)
    return preprocessor.preprocess_dataset(texts, labels, save_path)

def preprocess_image_dataset(image_paths, labels, save_path=None, **kwargs):
    """
    Quick function to preprocess image dataset
    
    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        save_path: Path to save processed data (optional)
        **kwargs: Additional arguments for ImagePreprocessor
        
    Returns:
        Tuple of (features, labels)
    """
    preprocessor = ImagePreprocessor(**kwargs)
    return preprocessor.preprocess_dataset(image_paths, labels, save_path)

def preprocess_multimodal_dataset(texts, image_paths, labels, save_dir, language="en", **kwargs):
    """
    Quick function to preprocess multimodal dataset
    
    Args:
        texts: List of text strings
        image_paths: List of image file paths
        labels: List of corresponding labels
        save_dir: Directory to save processed data
        language: Language code ('en' or 'zh')
        **kwargs: Additional arguments for CombinedPreprocessor
        
    Returns:
        Dictionary with train, val, test datasets
    """
    preprocessor = CombinedPreprocessor(language=language, **kwargs)
    return preprocessor.create_data_splits(texts, image_paths, labels, save_dir=save_dir)

__all__ = [
    # Main classes
    'TextPreprocessor',
    'ImagePreprocessor', 
    'CombinedPreprocessor',
    'DataManager',
    'DatasetSplitter',
    'DataValidator',
    
    # Dataset classes
    'TextDataset',
    'ImageDataset',
    'FeatureDataset',
    'MultimodalDataset',
    
    # Utility functions
    'create_dataloaders',
    'create_sample_data',
    'convert_format',
    
    # Convenience functions
    'preprocess_text_dataset',
    'preprocess_image_dataset', 
    'preprocess_multimodal_dataset',
]
