#!/usr/bin/env python3
"""
Image Processing Module for ViFactCheck Dataset

Handles image loading, preprocessing, and feature extraction.
"""

import os
import torch
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Union
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Image processor for fake news detection.
    
    Handles image loading, preprocessing, and feature extraction.
    """
    
    def __init__(self,
                 image_size: Tuple[int, int] = (224, 224),
                 feature_dim: int = 512,
                 model_name: str = "resnet50",
                 normalize: bool = True):
        """
        Initialize image processor.
        
        Args:
            image_size: Target image size (height, width)
            feature_dim: Output feature dimension
            model_name: Pretrained model name
            normalize: Whether to apply ImageNet normalization
        """
        self.image_size = image_size
        self.feature_dim = feature_dim
        self.model_name = model_name
        self.normalize = normalize
        
        # Initialize transforms and model
        self.transform = self._create_transforms()
        self.model = None
        self.projection = None
        self._init_model()
    
    def _create_transforms(self) -> transforms.Compose:
        """Create image preprocessing transforms."""
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor()
        ]
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        return transforms.Compose(transform_list)
    
    def _init_model(self):
        """Initialize pretrained model for feature extraction."""
        try:
            if self.model_name == "resnet50":
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                feature_size = 2048
            elif self.model_name == "resnet18":
                self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                feature_size = 512
            elif self.model_name == "efficientnet_b0":
                self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
                feature_size = 1280
            else:
                logger.warning(f"Unknown model {self.model_name}, using ResNet50")
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                feature_size = 2048
            
            # Remove classification layer
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            
            # Create projection layer if needed
            if feature_size != self.feature_dim:
                self.projection = torch.nn.Linear(feature_size, self.feature_dim)
            
            logger.info(f"Loaded image model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Could not load image model: {e}")
            self.model = None
    
    def load_image(self, image_path: Union[str, Path]) -> Optional[Image.Image]:
        """
        Load image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image or None if loading failed
        """
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                return None
            
            image = Image.open(image_path).convert('RGB')
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def validate_image(self, image_path: Union[str, Path]) -> bool:
        """
        Validate if image exists and is readable.
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if image is valid
        """
        try:
            image = self.load_image(image_path)
            if image is None:
                return False
            
            # Try to get image size
            _ = image.size
            return True
            
        except Exception:
            return False
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess PIL image to tensor.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed image tensor
        """
        try:
            return self.transform(image)
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            # Return dummy tensor
            return torch.zeros(3, *self.image_size)
    
    def extract_features(self, image: Image.Image) -> torch.Tensor:
        """
        Extract features from image.
        
        Args:
            image: PIL Image
            
        Returns:
            Feature tensor
        """
        if self.model is None:
            # Return random features if model not loaded
            return torch.randn(self.feature_dim)
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image).unsqueeze(0)
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.squeeze().flatten()
                
                # Apply projection if needed
                if self.projection is not None:
                    features = self.projection(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting image features: {e}")
            return torch.randn(self.feature_dim)
    
    def process(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Complete image processing pipeline.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with processed image data
        """
        # Load image
        image = self.load_image(image_path)
        
        if image is None:
            return {
                'tensor': torch.zeros(3, *self.image_size),
                'features': torch.randn(self.feature_dim),
                'image_path': str(image_path),
                'valid': False,
                'size': (0, 0)
            }
        
        # Process image
        image_tensor = self.preprocess_image(image)
        features = self.extract_features(image)
        
        return {
            'tensor': image_tensor,
            'features': features,
            'image_path': str(image_path),
            'valid': True,
            'size': image.size
        }
    
    def batch_process(self, image_paths: List[Union[str, Path]]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Batched processed data
        """
        processed = [self.process(path) for path in image_paths]
        
        return {
            'tensors': torch.stack([p['tensor'] for p in processed]),
            'features': torch.stack([p['features'] for p in processed]),
            'image_paths': [p['image_path'] for p in processed],
            'valid': [p['valid'] for p in processed],
            'sizes': [p['size'] for p in processed]
        }
    
    def create_dummy_image(self, color: str = 'gray') -> Image.Image:
        """
        Create a dummy image for missing files.
        
        Args:
            color: Background color
            
        Returns:
            PIL Image
        """
        return Image.new('RGB', self.image_size, color=color)
    
    def get_image_stats(self, image_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        Get statistics about a list of images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Dictionary with image statistics
        """
        valid_count = 0
        sizes = []
        
        for path in image_paths:
            if self.validate_image(path):
                valid_count += 1
                image = self.load_image(path)
                if image:
                    sizes.append(image.size)
        
        if sizes:
            widths, heights = zip(*sizes)
            return {
                'total_images': len(image_paths),
                'valid_images': valid_count,
                'invalid_images': len(image_paths) - valid_count,
                'avg_width': np.mean(widths),
                'avg_height': np.mean(heights),
                'min_width': min(widths),
                'max_width': max(widths),
                'min_height': min(heights),
                'max_height': max(heights)
            }
        else:
            return {
                'total_images': len(image_paths),
                'valid_images': 0,
                'invalid_images': len(image_paths)
            }


class ImageAugmentor:
    """
    Image augmentation for training data.
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (224, 224),
                 augment_prob: float = 0.5):
        """
        Initialize image augmentor.
        
        Args:
            image_size: Target image size
            augment_prob: Probability of applying augmentation
        """
        self.image_size = image_size
        self.augment_prob = augment_prob
        
        self.augment_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.normal_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image: Image.Image, training: bool = True) -> torch.Tensor:
        """
        Apply augmentation to image.
        
        Args:
            image: PIL Image
            training: Whether in training mode
            
        Returns:
            Augmented image tensor
        """
        if training and np.random.random() < self.augment_prob:
            return self.augment_transform(image)
        else:
            return self.normal_transform(image)


def create_image_processor(image_size: Tuple[int, int] = (224, 224),
                          feature_dim: int = 512,
                          model_name: str = "resnet50") -> ImageProcessor:
    """
    Factory function to create an image processor.
    
    Args:
        image_size: Target image size
        feature_dim: Output feature dimension
        model_name: Pretrained model name
        
    Returns:
        ImageProcessor instance
    """
    return ImageProcessor(
        image_size=image_size,
        feature_dim=feature_dim,
        model_name=model_name
    )


if __name__ == "__main__":
    # Example usage
    processor = ImageProcessor()
    
    # Create a dummy image for testing
    dummy_image = processor.create_dummy_image()
    
    # Process the image
    result = processor.extract_features(dummy_image)
    
    print("Image processing result:")
    print(f"Features shape: {result.shape}")
    print(f"Feature dimension: {result.size(0)}")
    
    # Test image augmentation
    augmentor = ImageAugmentor()
    augmented = augmentor(dummy_image, training=True)
    print(f"Augmented image shape: {augmented.shape}")
