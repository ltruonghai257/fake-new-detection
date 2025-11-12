import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import logging
import hashlib
import re

logger = logging.getLogger(__name__)


class TextProcessor:
    """Text processing for Vietnamese news articles."""
    
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
            text: Raw Vietnamese text
            
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


class ImageProcessor:
    """Image processing for news article images."""
    
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
                self.model = models.resnet50(pretrained=True)
            elif self.model_name == "resnet18":
                self.model = models.resnet18(pretrained=True)
            elif self.model_name == "efficientnet_b0":
                self.model = models.efficientnet_b0(pretrained=True)
            else:
                self.model = models.resnet50(pretrained=True)
            
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
            image_path: Path to image file
            
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
    
    def preprocess_image_from_pil(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image directly."""
        if self.model is None:
            return torch.randn(self.feature_dim)
        
        try:
            image_tensor = self.transforms(image).unsqueeze(0)
            
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.squeeze().flatten()
                
                if features.size(0) != self.feature_dim:
                    if not hasattr(self, 'image_projection'):
                        self.image_projection = torch.nn.Linear(
                            features.size(0), self.feature_dim
                        ).to(features.device)
                    features = self.image_projection(features)
            
            return features
        except Exception as e:
            logger.error(f"Error processing PIL image: {e}")
            return torch.randn(self.feature_dim)


class MultimodalDataset:
    """Dataset class for loading and processing multimodal fake news data."""
    
    def __init__(self,
                 json_path: str,
                 image_base_dir: str,
                 max_length: int = 30,
                 embed_dim: int = 200,
                 feature_dim: int = 512,
                 labels: Optional[List[int]] = None):
        
        self.json_path = json_path
        self.image_base_dir = image_base_dir
        self.labels = labels
        
        # Initialize processors
        self.text_processor = TextProcessor(max_length, embed_dim)
        self.image_processor = ImageProcessor(feature_dim=feature_dim)
        
        # Load and process data
        self.data = self._load_data()
        
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
        # Try different strategies to find image
        
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
        
        # Strategy 3: Look for any image in the same source directory
        # (This is a fallback for testing)
        for source_dir in os.listdir(self.image_base_dir):
            source_path = os.path.join(self.image_base_dir, source_dir)
            if os.path.isdir(source_path):
                files = [f for f in os.listdir(source_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    # Return first available image
                    return os.path.join(source_path, files[0])
        
        return None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get processed item at index."""
        article = self.data[idx]
        
        # Process text
        text = article.get('text', '')
        text_features = self.text_processor.preprocess_text(text)
        
        # Process image
        image_path = self.find_image_path(article)
        if image_path:
            image_features = self.image_processor.preprocess_image(image_path)
        else:
            logger.warning(f"No image found for article {idx}, using random features")
            image_features = torch.randn(self.image_processor.feature_dim)
        
        label = torch.tensor(article['label'], dtype=torch.long)
        
        return {
            'text_features': text_features,
            'image_features': image_features,
            'label': label,
            'title': article.get('title', ''),
            'image_path': image_path or 'None'
        }
    
    def get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """Get a batch of processed items."""
        batch_items = [self[i] for i in indices]
        
        text_features = torch.stack([item['text_features'] for item in batch_items])
        image_features = torch.stack([item['image_features'] for item in batch_items])
        labels = torch.stack([item['label'] for item in batch_items])
        
        return {
            'text_features': text_features,
            'image_features': image_features,
            'labels': labels,
            'titles': [item['title'] for item in batch_items],
            'image_paths': [item['image_path'] for item in batch_items]
        }


class DataPreprocessor:
    """Main data preprocessing pipeline."""
    
    def __init__(self, 
                 json_path: str,
                 image_base_dir: str,
                 output_dir: str = "./processed_data",
                 max_length: int = 30,
                 embed_dim: int = 200,
                 feature_dim: int = 512):
        
        self.json_path = json_path
        self.image_base_dir = image_base_dir
        self.output_dir = output_dir
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def preprocess_dataset(self, 
                          labels: Optional[List[int]] = None,
                          test_size: float = 0.2,
                          val_size: float = 0.1) -> Dict[str, Any]:
        """
        Preprocess the entire dataset and save splits.
        
        Args:
            labels: Optional list of labels
            test_size: Fraction of data for test set
            val_size: Fraction of data for validation set
            
        Returns:
            Dictionary with dataset information
        """
        logger.info("Starting dataset preprocessing...")
        
        # Create dataset
        dataset = MultimodalDataset(
            self.json_path,
            self.image_base_dir,
            self.max_length,
            self.embed_dim,
            self.feature_dim,
            labels
        )
        
        if len(dataset) == 0:
            raise ValueError("No valid data found")
        
        # Generate all features
        all_text_features = []
        all_image_features = []
        all_labels = []
        
        logger.info(f"Processing {len(dataset)} samples...")
        for i in range(len(dataset)):
            item = dataset[i]
            all_text_features.append(item['text_features'])
            all_image_features.append(item['image_features'])
            all_labels.append(item['label'])
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} samples")
        
        # Convert to tensors
        text_features = torch.stack(all_text_features)
        image_features = torch.stack(all_image_features)
        labels = torch.stack(all_labels)
        
        # Split dataset
        total_size = len(text_features)
        test_size = int(total_size * test_size)
        val_size = int(total_size * val_size)
        train_size = total_size - test_size - val_size
        
        # Random split
        indices = torch.randperm(total_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create splits
        splits = {
            'train': {
                'text_features': text_features[train_indices],
                'image_features': image_features[train_indices],
                'labels': labels[train_indices]
            },
            'val': {
                'text_features': text_features[val_indices],
                'image_features': image_features[val_indices],
                'labels': labels[val_indices]
            },
            'test': {
                'text_features': text_features[test_indices],
                'image_features': image_features[test_indices],
                'labels': labels[test_indices]
            }
        }
        
        # Save splits
        for split_name, split_data in splits.items():
            split_dir = os.path.join(self.output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            torch.save(split_data['text_features'], 
                      os.path.join(split_dir, 'text_features.pt'))
            torch.save(split_data['image_features'], 
                      os.path.join(split_dir, 'image_features.pt'))
            torch.save(split_data['labels'], 
                      os.path.join(split_dir, 'labels.pt'))
        
        # Save metadata
        metadata = {
            'total_samples': total_size,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'text_shape': text_features.shape,
            'image_shape': image_features.shape,
            'max_length': self.max_length,
            'embed_dim': self.embed_dim,
            'feature_dim': self.feature_dim
        }
        
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Preprocessing completed. Data saved to {self.output_dir}")
        logger.info(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        return metadata
    
    def create_sample_data(self, num_samples: int = 1000) -> Dict[str, torch.Tensor]:
        """Create sample data for testing models."""
        logger.info(f"Creating sample dataset with {num_samples} samples...")
        
        text_features = torch.randn(num_samples, self.max_length, self.embed_dim)
        image_features = torch.randn(num_samples, self.feature_dim)
        labels = torch.randint(0, 2, (num_samples,))
        
        # Save sample data
        sample_dir = os.path.join(self.output_dir, 'sample')
        os.makedirs(sample_dir, exist_ok=True)
        
        torch.save(text_features, os.path.join(sample_dir, 'text_features.pt'))
        torch.save(image_features, os.path.join(sample_dir, 'image_features.pt'))
        torch.save(labels, os.path.join(sample_dir, 'labels.pt'))
        
        logger.info(f"Sample data saved to {sample_dir}")
        
        return {
            'text_features': text_features,
            'image_features': image_features,
            'labels': labels
        }


# Utility functions
def load_processed_data(data_dir: str, split: str = 'train') -> Dict[str, torch.Tensor]:
    """Load preprocessed data."""
    split_dir = os.path.join(data_dir, split)
    
    text_features = torch.load(os.path.join(split_dir, 'text_features.pt'))
    image_features = torch.load(os.path.join(split_dir, 'image_features.pt'))
    labels = torch.load(os.path.join(split_dir, 'labels.pt'))
    
    return {
        'text_features': text_features,
        'image_features': image_features,
        'labels': labels
    }


def create_dataloader(data_dict: Dict[str, torch.Tensor], 
                     batch_size: int = 32,
                     shuffle: bool = True) -> torch.utils.data.DataLoader:
    """Create PyTorch DataLoader from processed data."""
    from torch.utils.data import TensorDataset
    
    dataset = TensorDataset(
        data_dict['text_features'],
        data_dict['image_features'],
        data_dict['labels']
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True
    )
