"""
Image Preprocessing Pipeline for COOLANT Fake News Detection
Based on the COOLANT repository: https://github.com/wishever/COOLANT
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import resnet18, resnet50
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm

# Optional OpenCV import
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class ImagePreprocessor:
    """
    Image preprocessing pipeline for multimodal fake news detection
    Compatible with COOLANT's ResNet-based feature extraction
    """
    
    def __init__(self, 
                 model_name: str = "resnet18",
                 pretrained: bool = True,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize image preprocessor
        
        Args:
            model_name: Backbone model name ('resnet18' or 'resnet50')
            pretrained: Whether to use pretrained weights
            device: Device to run preprocessing on
            image_size: Target image size for resizing
        """
        self.device = device
        self.image_size = image_size
        self.model_name = model_name
        
        # Initialize ResNet model
        if model_name == "resnet18":
            self.model = resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == "resnet50":
            self.model = resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError("model_name must be 'resnet18' or 'resnet50'")
        
        # Remove the final classification layer to get features
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(device)
        self.model.eval()
        
        # Define image transforms (matching COOLANT preprocessing)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0)
            return image_tensor.to(self.device)
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a zero tensor as fallback
            return torch.zeros(1, 3, *self.image_size).to(self.device)
    
    def extract_features(self, image_paths: List[str]) -> np.ndarray:
        """
        Extract ResNet features for a list of images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            numpy array of image features (batch_size, feature_dim)
        """
        features = []
        
        with torch.no_grad():
            for image_path in tqdm(image_paths, desc="Extracting image features"):
                image_tensor = self.load_and_preprocess_image(image_path)
                
                # Extract features
                feature = self.model(image_tensor)
                feature = feature.squeeze().cpu().numpy()
                
                # Ensure feature is 1D
                if feature.ndim == 0:
                    feature = np.array([feature])
                    
                features.append(feature)
        
        return np.vstack(features)
    
    def extract_features_from_images(self, images: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
        """
        Extract features from a list of PIL Images or numpy arrays
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            numpy array of image features (batch_size, feature_dim)
        """
        features = []
        
        with torch.no_grad():
            for img in tqdm(images, desc="Extracting image features"):
                if isinstance(img, np.ndarray):
                    # Convert numpy array to PIL Image
                    img = Image.fromarray(img)
                elif not isinstance(img, Image.Image):
                    raise ValueError("Images must be PIL Image or numpy array")
                
                # Apply transforms
                image_tensor = self.transform(img).unsqueeze(0)
                image_tensor = image_tensor.to(self.device)
                
                # Extract features
                feature = self.model(image_tensor)
                feature = feature.squeeze().cpu().numpy()
                
                # Ensure feature is 1D
                if feature.ndim == 0:
                    feature = np.array([feature])
                    
                features.append(feature)
        
        return np.vstack(features)
    
    def preprocess_dataset(self, 
                          image_paths: List[str], 
                          labels: List[int],
                          save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess a complete dataset of images
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            save_path: Path to save preprocessed data (optional)
            
        Returns:
            Tuple of (features, labels)
        """
        features = self.extract_features(image_paths)
        labels_array = np.array(labels)
        
        if save_path:
            self.save_preprocessed_data(features, labels_array, save_path)
            
        return features, labels_array
    
    def save_preprocessed_data(self, 
                              features: np.ndarray, 
                              labels: np.ndarray, 
                              save_path: str):
        """
        Save preprocessed data to pickle or npz format
        
        Args:
            features: Preprocessed image features
            labels: Corresponding labels
            save_path: Path to save the data
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if save_path.endswith('.pkl'):
            with open(save_path, 'wb') as f:
                pickle.dump({'features': features, 'labels': labels}, f)
        elif save_path.endswith('.npz'):
            np.savez(save_path, data=features, label=labels)
        else:
            raise ValueError("save_path must end with .pkl or .npz")
            
        print(f"Saved preprocessed image data to {save_path}")
    
    @staticmethod
    def load_preprocessed_data(load_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load preprocessed data from file
        
        Args:
            load_path: Path to load data from
            
        Returns:
            Tuple of (features, labels)
        """
        if load_path.endswith('.pkl'):
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
                return data['features'], data['labels']
        elif load_path.endswith('.npz'):
            loaded = np.load(load_path)
            return loaded['data'], loaded['label']
        else:
            raise ValueError("load_path must end with .pkl or .npz")


class ImageDataset(Dataset):
    """
    PyTorch Dataset for preprocessed image data
    Compatible with COOLANT training pipeline
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset
        
        Args:
            features: Preprocessed image features
            labels: Corresponding labels
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class FeatureDataset(Dataset):
    """
    Combined dataset for text and image features (from COOLANT)
    """
    
    def __init__(self, text_file: str, image_file: str):
        """
        Initialize combined dataset
        
        Args:
            text_file: Path to text features file (.npz)
            image_file: Path to image features file (.npz)
        """
        # Load text features
        text_data = np.load(text_file)
        self.text_features = torch.from_numpy(text_data["data"]).float()
        self.labels = torch.from_numpy(text_data["label"]).long()
        
        # Load image features
        image_data = np.load(image_file)
        self.image_features = torch.from_numpy(image_data["data"]).squeeze().float()
        
    def __len__(self):
        return self.text_features.shape[0]
    
    def __getitem__(self, idx):
        return self.text_features[idx], self.image_features[idx], self.labels[idx]


def preprocess_twitter_images(data_path: str, 
                             image_dir: str,
                             save_dir: str = "./processed_data/twitter",
                             save_format: str = "npz") -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess Twitter dataset images
    
    Args:
        data_path: Path to Twitter metadata file
        image_dir: Directory containing Twitter images
        save_dir: Directory to save processed data
        save_format: Format to save ('pkl' or 'npz')
        
    Returns:
        Tuple of (features, labels)
    """
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(model_name="resnet18")
    
    # Load metadata (assuming JSON format)
    import json
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Extract image paths and labels
    image_paths = []
    labels = []
    
    for item in raw_data:
        image_path = os.path.join(image_dir, item['image_name'])
        image_paths.append(image_path)
        labels.append(item['label'])
    
    # Preprocess
    features, labels_array = preprocessor.preprocess_dataset(
        image_paths, labels,
        save_path=f"{save_dir}/image_features.{save_format}"
    )
    
    return features, labels_array


def preprocess_weibo_images(data_path: str,
                           image_dir: str,
                           save_dir: str = "./processed_data/weibo", 
                           save_format: str = "npz") -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess Weibo dataset images
    
    Args:
        data_path: Path to Weibo metadata file
        image_dir: Directory containing Weibo images
        save_dir: Directory to save processed data
        save_format: Format to save ('pkl' or 'npz')
        
    Returns:
        Tuple of (features, labels)
    """
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(model_name="resnet18")
    
    # Load metadata (assuming JSON format)
    import json
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Extract image paths and labels
    image_paths = []
    labels = []
    
    for item in raw_data:
        image_path = os.path.join(image_dir, item['image_name'])
        image_paths.append(image_path)
        labels.append(item['label'])
    
    # Preprocess
    features, labels_array = preprocessor.preprocess_dataset(
        image_paths, labels,
        save_path=f"{save_dir}/image_features.{save_format}"
    )
    
    return features, labels_array


def create_augmented_image_features(image_paths: List[str], 
                                   augmentations: int = 3) -> np.ndarray:
    """
    Create augmented image features for robust training
    
    Args:
        image_paths: List of image file paths
        augmentations: Number of augmentations per image
        
    Returns:
        Augmented image features
    """
    # Define augmentation transforms
    augment_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    preprocessor = ImagePreprocessor()
    preprocessor.transform = augment_transforms
    
    all_features = []
    
    for aug_idx in range(augmentations):
        print(f"Processing augmentation {aug_idx + 1}/{augmentations}")
        features = preprocessor.extract_features(image_paths)
        all_features.append(features)
    
    return np.vstack(all_features)


if __name__ == "__main__":
    # Example usage
    print("Image Preprocessing Pipeline for COOLANT")
    
    # Example for preprocessing sample images
    sample_image_paths = [
        "./sample_images/image1.jpg",
        "./sample_images/image2.jpg"
    ]
    sample_labels = [0, 1]  # 0: real news, 1: fake news
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(model_name="resnet18")
    
    # Create sample images for demonstration
    os.makedirs("./sample_images", exist_ok=True)
    for i, path in enumerate(sample_image_paths):
        if not os.path.exists(path):
            # Create a dummy image
            dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            dummy_img.save(path)
    
    # Preprocess and save
    features, labels = preprocessor.preprocess_dataset(
        sample_image_paths, 
        sample_labels,
        save_path="./sample_image_data.pkl"
    )
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print("Image preprocessing completed successfully!")
