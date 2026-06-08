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


class ViFactCheckTextProcessor:
    """Text processing for ViFactCheck dataset."""
    
    def __init__(self, 
                 max_length: int = 30,
                 embed_dim: int = 200,
                 tokenizer_name: str = "vinai/phobert-base"):
        
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.tokenizer_name = tokenizer_name
        
        # Initialize tokenizer and model
        self._init_text_models()
    
    def _init_text_models(self):
        """Initialize text tokenizer and embedding model."""
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.text_model = AutoModel.from_pretrained(self.tokenizer_name)
            self.text_model.eval()
            logger.info(f"Loaded text model: {self.tokenizer_name}")
        except Exception as e:
            logger.warning(f"Could not load {self.tokenizer_name}: {e}")
            self.tokenizer = None
            self.text_model = None
    
    def preprocess_text(self, text: str) -> torch.Tensor:
        """
        Preprocess raw text to model-compatible format.
        
        Args:
            text: Raw Vietnamese text from caption
            
        Returns:
            Tensor of shape (max_length, embed_dim)
        """
        if not text or not isinstance(text, str):
            return torch.zeros(self.max_length, self.embed_dim)
        
        # Clean text
        text = self._clean_text(text)
        
        if self.text_model is None:
            # Fallback: simple word-based processing
            return self._fallback_text_processing(text)
        
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.text_model(**inputs)
                embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)
                
                # Project to target dimension if needed
                if embeddings.size(-1) != self.embed_dim:
                    if not hasattr(self, 'text_projection'):
                        self.text_projection = torch.nn.Linear(
                            embeddings.size(-1), self.embed_dim
                        ).to(embeddings.device)
                    embeddings = self.text_projection(embeddings)
            
            return embeddings
        except Exception as e:
            logger.error(f"Error in text processing: {e}")
            return self._fallback_text_processing(text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove HTML tags if present
        text = re.sub(r'<[^>]+>', '', text)
        
        # Keep only reasonable length
        if len(text) > 2000:
            text = text[:2000] + "..."
        
        return text
    
    def _fallback_text_processing(self, text: str) -> torch.Tensor:
        """Fallback text processing when model is unavailable."""
        # Simple word tokenization
        words = text.split()[:self.max_length]
        
        # Pad or truncate
        if len(words) < self.max_length:
            words.extend(['<pad>'] * (self.max_length - len(words)))
        
        # Create random embeddings (in real usage, you'd use word embeddings)
        embeddings = torch.randn(len(words), self.embed_dim)
        
        return embeddings


class ViFactCheckImageProcessor:
    """Image processing for ViFactCheck dataset."""
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (224, 224),
                 feature_dim: int = 512,
                 model_name: str = "resnet50"):
        
        self.image_size = image_size
        self.feature_dim = feature_dim
        self.model_name = model_name
        
        # Initialize transforms and model
        self._init_image_models()
    
    def _init_image_models(self):
        """Initialize image transforms and feature extractor."""
        # Standard ImageNet transforms
        self.transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load pretrained model
        try:
            if self.model_name == "resnet50":
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            elif self.model_name == "resnet18":
                self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            elif self.model_name == "efficientnet_b0":
                self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            
            self.model.eval()
            
            # Remove final classification layer for feature extraction
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            logger.info(f"Loaded image model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Could not load image model: {e}")
            self.model = None
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image to model-compatible format.
        
        Args:
            image_path: Path to image file from folder_path
            
        Returns:
            Tensor of shape (feature_dim,)
        """
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            return torch.randn(self.feature_dim)
        
        if self.model is None:
            return torch.randn(self.feature_dim)
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transforms(image).unsqueeze(0)  # Add batch dimension
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.squeeze().flatten()  # Remove batch and spatial dims
                
                # Project to target dimension if needed
                if features.size(0) != self.feature_dim:
                    if not hasattr(self, 'image_projection'):
                        self.image_projection = torch.nn.Linear(
                            features.size(0), self.feature_dim
                        ).to(features.device)
                    features = self.image_projection(features)
            
            return features
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return torch.randn(self.feature_dim)


class ViFactCheckDataset(Dataset):
    """
    PyTorch Dataset for ViFactCheck multimodal fake news detection.
    
    This dataset handles the ViFactCheck format where:
    - images[i].caption contains the text
    - images[i].folder_path contains the image path
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
        Initialize the ViFactCheck dataset.
        
        Args:
            json_path: Path to ViFactCheck JSON data file
            image_base_dir: Base directory containing images (folder_path is relative to this)
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
        self.text_processor = text_processor or ViFactCheckTextProcessor(max_length, embed_dim)
        self.image_processor = image_processor or ViFactCheckImageProcessor(feature_dim=feature_dim)
        
        # Load and validate data
        self.data = self._load_vifactcheck_data()
        
        # Initialize caches
        self.image_cache = {} if cache_images else None
        self.text_cache = {} if cache_text else None
        
        logger.info(f"ViFactCheck Dataset initialized with {len(self.data)} samples")
    
    def _load_vifactcheck_data(self) -> List[Dict[str, Any]]:
        """Load and validate ViFactCheck format data from JSON file."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # ViFactCheck format: list of image objects with caption and folder_path
            valid_samples = []
            
            if isinstance(raw_data, list):
                # Direct list of image objects
                for i, item in enumerate(raw_data):
                    if (isinstance(item, dict) and 
                        item.get('caption') and 
                        item.get('folder_path')):
                        
                        sample = {
                            'text': item['caption'],
                            'image_path': item['folder_path'],
                            'original_index': i
                        }
                        
                        # Add label
                        if self.labels and i < len(self.labels):
                            sample['label'] = self.labels[i]
                        else:
                            sample['label'] = 0  # Default to real news
                        
                        valid_samples.append(sample)
                        
            elif isinstance(raw_data, dict) and 'images' in raw_data:
                # Nested format with 'images' key
                for i, item in enumerate(raw_data['images']):
                    if (isinstance(item, dict) and 
                        item.get('caption') and 
                        item.get('folder_path')):
                        
                        sample = {
                            'text': item['caption'],
                            'image_path': item['folder_path'],
                            'original_index': i
                        }
                        
                        # Add label
                        if self.labels and i < len(self.labels):
                            sample['label'] = self.labels[i]
                        else:
                            sample['label'] = 0  # Default to real news
                        
                        valid_samples.append(sample)
            
            logger.info(f"Loaded {len(valid_samples)} valid ViFactCheck entries")
            return valid_samples
            
        except Exception as e:
            logger.error(f"Error loading ViFactCheck data: {e}")
            return []
    
    def _get_full_image_path(self, folder_path: str) -> str:
        """Get full image path from folder_path."""
        # folder_path might be relative or absolute
        if os.path.isabs(folder_path):
            return folder_path
        else:
            return os.path.join(self.image_base_dir, folder_path)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the ViFactCheck dataset.
        
        Returns:
            Dictionary containing:
            - text_features: (max_length, embed_dim)
            - image_features: (feature_dim,)
            - label: (1,)
            - caption: str
            - image_path: str
        """
        sample = self.data[idx]
        
        # Process text (with caching)
        text = sample['text']
        if self.cache_text and idx in self.text_cache:
            text_features = self.text_cache[idx]
        else:
            text_features = self.text_processor.preprocess_text(text)
            if self.cache_text:
                self.text_cache[idx] = text_features
        
        # Process image (with caching)
        image_path = self._get_full_image_path(sample['image_path'])
        if self.cache_images and image_path in self.image_cache:
            image_features = self.image_cache[image_path]
        else:
            image_features = self.image_processor.preprocess_image(image_path)
            if self.cache_images:
                self.image_cache[image_path] = image_features
        
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return {
            'text_features': text_features,
            'image_features': image_features,
            'label': label,
            'caption': sample['text'],
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


def create_vifactcheck_dataloaders(json_path: str,
                                 image_base_dir: str,
                                 batch_size: int = 32,
                                 test_size: float = 0.2,
                                 val_size: float = 0.1,
                                 num_workers: int = 2,
                                 labels: Optional[List[int]] = None,
                                 random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to create train/val/test dataloaders for ViFactCheck data.
    
    Args:
        json_path: Path to ViFactCheck JSON data file
        image_base_dir: Base directory containing images
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
    full_dataset = ViFactCheckDataset(
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
    
    def vifactcheck_collate_fn(batch):
        """Collate function for ViFactCheck data."""
        text_features = torch.stack([item['text_features'] for item in batch])
        image_features = torch.stack([item['image_features'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        
        return {
            'text_features': text_features,
            'image_features': image_features,
            'labels': labels,
            'captions': [item['caption'] for item in batch],
            'image_paths': [item['image_path'] for item in batch]
        }
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=vifactcheck_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=vifactcheck_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=vifactcheck_collate_fn
    )
    
    logger.info(f"Created ViFactCheck dataloaders: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def create_sample_vifactcheck_dataloader(batch_size: int = 32, 
                                       num_samples: int = 1000) -> DataLoader:
    """
    Create a sample ViFactCheck dataloader for testing models.
    
    Args:
        batch_size: Batch size
        num_samples: Number of samples
        
    Returns:
        DataLoader with sample ViFactCheck data
    """
    # Generate sample data
    text_features = torch.randn(num_samples, 30, 200)
    image_features = torch.randn(num_samples, 512)
    labels = torch.randint(0, 2, (num_samples,))
    captions = [f"Sample caption {i}" for i in range(num_samples)]
    image_paths = [f"sample_image_{i}.jpg" for i in range(num_samples)]
    
    # Create dataset
    dataset = ViFactCheckDataset.__new__(ViFactCheckDataset)  # Create without calling __init__
    dataset.data = [
        {
            'text_features': text_features[i],
            'image_features': image_features[i],
            'label': labels[i],
            'caption': captions[i],
            'image_path': image_paths[i]
        }
        for i in range(num_samples)
    ]
    
    def sample_collate_fn(batch):
        """Collate function for sample data."""
        text_features = torch.stack([item['text_features'] for item in batch])
        image_features = torch.stack([item['image_features'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        
        return {
            'text_features': text_features,
            'image_features': image_features,
            'labels': labels,
            'captions': [item['caption'] for item in batch],
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
    
    logger.info(f"Created sample ViFactCheck dataloader with {num_samples} samples")
    
    return dataloader
