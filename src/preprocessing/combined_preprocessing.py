"""
Combined Multimodal Preprocessing Pipeline for COOLANT Fake News Detection
Based on the COOLANT repository: https://github.com/wishever/COOLANT
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
import json
import pandas as pd
from PIL import Image

from .text_preprocessing import TextPreprocessor
from .image_preprocessing import ImagePreprocessor


class MultimodalDataset(Dataset):
    """
    Combined multimodal dataset for text and image features
    Compatible with COOLANT training pipeline
    """

    def __init__(
        self,
        text_features: np.ndarray,
        image_features: np.ndarray,
        labels: np.ndarray,
        text_dim: Optional[int] = None,
        image_dim: Optional[int] = None,
    ):
        """
        Initialize multimodal dataset

        Args:
            text_features: Preprocessed text features
            image_features: Preprocessed image features
            labels: Corresponding labels
            text_dim: Target text dimension (for padding/truncation)
            image_dim: Target image dimension (for padding/truncation)
        """
        self.text_features = torch.FloatTensor(text_features)
        self.image_features = torch.FloatTensor(image_features)
        self.labels = torch.LongTensor(labels)

        # Handle dimensionality adjustments
        if text_dim and self.text_features.shape[-1] != text_dim:
            if self.text_features.shape[-1] > text_dim:
                # Truncate
                self.text_features = self.text_features[..., :text_dim]
            else:
                # Pad with zeros
                padding = text_dim - self.text_features.shape[-1]
                self.text_features = torch.nn.functional.pad(
                    self.text_features, (0, padding), "constant", 0
                )

        if image_dim and self.image_features.shape[-1] != image_dim:
            if self.image_features.shape[-1] > image_dim:
                # Truncate
                self.image_features = self.image_features[..., :image_dim]
            else:
                # Pad with zeros
                padding = image_dim - self.image_features.shape[-1]
                self.image_features = torch.nn.functional.pad(
                    self.image_features, (0, padding), "constant", 0
                )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text_features[idx], self.image_features[idx], self.labels[idx]


class CombinedPreprocessor:
    """
    Combined preprocessing pipeline for multimodal fake news detection
    Handles both text and image preprocessing with coordinated saving
    """

    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        image_model_name: str = "resnet18",
        language: str = "en",
        max_text_length: int = 512,
        image_size: Tuple[int, int] = (224, 224),
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    ):
        """
        Initialize combined preprocessor

        Args:
            text_model_name: BERT model name for text
            image_model_name: ResNet model name for images
            language: Language code ('en' or 'zh')
            max_text_length: Maximum text sequence length
            image_size: Target image size
            device: Device to run preprocessing on
        """
        self.device = device
        self.language = language

        # Initialize individual preprocessors
        self.text_preprocessor = TextPreprocessor(
            model_name=text_model_name,
            max_length=max_text_length,
            language=language,
            device=device,
        )

        self.image_preprocessor = ImagePreprocessor(
            model_name=image_model_name, device=device, image_size=image_size
        )

    def preprocess_sample(
        self, text: str, image_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess a single text-image pair

        Args:
            text: Text content
            image_path: Path to image file

        Returns:
            Tuple of (text_features, image_features)
        """
        # Preprocess text
        text_encoded = self.text_preprocessor.tokenize_text(text)
        with torch.no_grad():
            text_outputs = self.text_preprocessor.bert_model(**text_encoded)
            text_features = text_outputs.last_hidden_state.cpu().numpy()

        # Preprocess image
        image_tensor = self.image_preprocessor.load_and_preprocess_image(image_path)
        with torch.no_grad():
            image_features = self.image_preprocessor.model(image_tensor)
            image_features = image_features.squeeze().cpu().numpy()

        return text_features, image_features

    def preprocess_dataset(
        self,
        texts: List[str],
        image_paths: List[str],
        labels: List[int],
        save_dir: str,
        save_format: str = "npz",
        batch_size: int = 32,
    ) -> Tuple[MultimodalDataset, Dict]:
        """
        Preprocess a complete multimodal dataset

        Args:
            texts: List of text strings
            image_paths: List of image file paths
            labels: List of corresponding labels
            save_dir: Directory to save processed data
            save_format: Format to save ('pkl' or 'npz')
            batch_size: Batch size for processing

        Returns:
            Tuple of (dataset, metadata)
        """
        os.makedirs(save_dir, exist_ok=True)

        print("Preprocessing text features...")
        text_features, _ = self.text_preprocessor.preprocess_dataset(
            texts,
            labels,
            save_path=os.path.join(save_dir, f"text_features.{save_format}"),
            extract_type="token_embeddings",
        )

        print("Preprocessing image features...")
        image_features, _ = self.image_preprocessor.preprocess_dataset(
            image_paths,
            labels,
            save_path=os.path.join(save_dir, f"image_features.{save_format}"),
        )

        # Create combined dataset
        dataset = MultimodalDataset(text_features, image_features, np.array(labels))

        # Save combined dataset
        combined_save_path = os.path.join(save_dir, f"combined_dataset.{save_format}")
        self.save_combined_dataset(
            text_features, image_features, np.array(labels), combined_save_path
        )

        # Create metadata
        metadata = {
            "num_samples": len(texts),
            "text_feature_shape": text_features.shape,
            "image_feature_shape": image_features.shape,
            "num_classes": len(set(labels)),
            "language": self.language,
            "text_model": self.text_preprocessor.model_name,
            "image_model": self.image_preprocessor.model_name,
            "save_dir": save_dir,
            "files": {
                "text_features": f"text_features.{save_format}",
                "image_features": f"image_features.{save_format}",
                "combined": f"combined_dataset.{save_format}",
            },
        }

        # Save metadata
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        return dataset, metadata

    def save_combined_dataset(
        self,
        text_features: np.ndarray,
        image_features: np.ndarray,
        labels: np.ndarray,
        save_path: str,
    ):
        """
        Save combined multimodal dataset

        Args:
            text_features: Preprocessed text features
            image_features: Preprocessed image features
            labels: Corresponding labels
            save_path: Path to save the combined data
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if save_path.endswith(".pkl"):
            with open(save_path, "wb") as f:
                pickle.dump(
                    {
                        "text_features": text_features,
                        "image_features": image_features,
                        "labels": labels,
                    },
                    f,
                )
        elif save_path.endswith(".npz"):
            np.savez(
                save_path,
                text_features=text_features,
                image_features=image_features,
                labels=labels,
            )
        else:
            raise ValueError("save_path must end with .pkl or .npz")

        print(f"Saved combined dataset to {save_path}")

    @staticmethod
    def load_combined_dataset(
        load_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load combined multimodal dataset

        Args:
            load_path: Path to load data from

        Returns:
            Tuple of (text_features, image_features, labels)
        """
        if load_path.endswith(".pkl"):
            with open(load_path, "rb") as f:
                data = pickle.load(f)
                return data["text_features"], data["image_features"], data["labels"]
        elif load_path.endswith(".npz"):
            loaded = np.load(load_path)
            return loaded["text_features"], loaded["image_features"], loaded["labels"]
        else:
            raise ValueError("load_path must end with .pkl or .npz")

    def create_data_splits(
        self,
        texts: List[str],
        image_paths: List[str],
        labels: List[int],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        save_dir: str = "./processed_data",
        save_format: str = "npz",
        random_seed: int = 42,
    ) -> Dict[str, MultimodalDataset]:
        """
        Create train/val/test splits and preprocess each

        Args:
            texts: List of text strings
            image_paths: List of image file paths
            labels: List of corresponding labels
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            save_dir: Base directory to save processed data
            save_format: Format to save ('pkl' or 'npz')
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary with train, val, test datasets
        """
        np.random.seed(random_seed)

        # Shuffle data
        indices = np.random.permutation(len(texts))
        texts = [texts[i] for i in indices]
        image_paths = [image_paths[i] for i in indices]
        labels = [labels[i] for i in indices]

        # Calculate split indices
        n_samples = len(texts)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)

        # Split data
        splits = {
            "train": (texts[:train_end], image_paths[:train_end], labels[:train_end]),
            "val": (
                texts[train_end:val_end],
                image_paths[train_end:val_end],
                labels[train_end:val_end],
            ),
            "test": (texts[val_end:], image_paths[val_end:], labels[val_end:]),
        }

        datasets = {}

        for split_name, (split_texts, split_images, split_labels) in splits.items():
            print(f"\nProcessing {split_name} split...")
            split_dir = os.path.join(save_dir, split_name)

            dataset, metadata = self.preprocess_dataset(
                split_texts,
                split_images,
                split_labels,
                save_dir=split_dir,
                save_format=save_format,
            )

            datasets[split_name] = dataset

            # Save split metadata
            with open(os.path.join(split_dir, f"{split_name}_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

        return datasets


def preprocess_twitter_dataset(
    data_path: str,
    image_dir: str,
    save_dir: str = "./processed_data/twitter",
    save_format: str = "npz",
) -> Dict[str, MultimodalDataset]:
    """
    Preprocess complete Twitter dataset

    Args:
        data_path: Path to Twitter metadata file (JSON)
        image_dir: Directory containing Twitter images
        save_dir: Directory to save processed data
        save_format: Format to save ('pkl' or 'npz')

    Returns:
        Dictionary with train, val, test datasets
    """
    # Initialize combined preprocessor
    preprocessor = CombinedPreprocessor(
        text_model_name="bert-base-uncased", image_model_name="resnet18", language="en"
    )

    # Load raw data
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Extract data
    texts = [item["text"] for item in raw_data]
    image_paths = [os.path.join(image_dir, item["image_name"]) for item in raw_data]
    labels = [item["label"] for item in raw_data]

    # Create splits and preprocess
    datasets = preprocessor.create_data_splits(
        texts, image_paths, labels, save_dir=save_dir, save_format=save_format
    )

    return datasets


def preprocess_weibo_dataset(
    data_path: str,
    image_dir: str,
    save_dir: str = "./processed_data/weibo",
    save_format: str = "npz",
) -> Dict[str, MultimodalDataset]:
    """
    Preprocess complete Weibo dataset

    Args:
        data_path: Path to Weibo metadata file (JSON)
        image_dir: Directory containing Weibo images
        save_dir: Directory to save processed data
        save_format: Format to save ('pkl' or 'npz')

    Returns:
        Dictionary with train, val, test datasets
    """
    # Initialize combined preprocessor
    preprocessor = CombinedPreprocessor(
        text_model_name="bert-base-chinese", image_model_name="resnet18", language="zh"
    )

    # Load raw data
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Extract data
    texts = [item["text"] for item in raw_data]
    image_paths = [os.path.join(image_dir, item["image_name"]) for item in raw_data]
    labels = [item["label"] for item in raw_data]

    # Create splits and preprocess
    datasets = preprocessor.create_data_splits(
        texts, image_paths, labels, save_dir=save_dir, save_format=save_format
    )

    return datasets


def create_dataloaders(
    datasets: Dict[str, MultimodalDataset], batch_size: int = 64, num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders from datasets

    Args:
        datasets: Dictionary with train, val, test datasets
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes

    Returns:
        Dictionary with train, val, test DataLoaders
    """
    dataloaders = {}

    for split_name, dataset in datasets.items():
        shuffle = split_name == "train"
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataloaders


if __name__ == "__main__":
    # Example usage
    print("Combined Multimodal Preprocessing Pipeline for COOLANT")

    # Sample data for demonstration
    sample_texts = [
        "This is a real news article about climate change.",
        "Breaking: Scientists discover new planet in solar system!",
        "Celebrity gossip: Famous actor spotted at local restaurant.",
        "Fake news claim: Eating chocolate cures all diseases.",
    ]

    sample_image_paths = [
        "./sample_images/news1.jpg",
        "./sample_images/news2.jpg",
        "./sample_images/news3.jpg",
        "./sample_images/news4.jpg",
    ]

    sample_labels = [0, 0, 1, 1]  # 0: real news, 1: fake news

    # Create sample images
    os.makedirs("./sample_images", exist_ok=True)
    for i, path in enumerate(sample_image_paths):
        if not os.path.exists(path):
            dummy_img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            dummy_img.save(path)

    # Initialize combined preprocessor
    preprocessor = CombinedPreprocessor(language="en")

    # Preprocess dataset
    dataset, metadata = preprocessor.preprocess_dataset(
        sample_texts,
        sample_image_paths,
        sample_labels,
        save_dir="./sample_processed_data",
        save_format="pkl",
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Text features shape: {metadata['text_feature_shape']}")
    print(f"Image features shape: {metadata['image_feature_shape']}")
    print("Combined preprocessing completed successfully!")

    # Example of loading the processed data
    loaded_text, loaded_image, loaded_labels = (
        CombinedPreprocessor.load_combined_dataset(
            "./sample_processed_data/combined_dataset.pkl"
        )
    )
    print(f"Loaded text features shape: {loaded_text.shape}")
    print(f"Loaded image features shape: {loaded_image.shape}")
    print(f"Loaded labels shape: {loaded_labels.shape}")
