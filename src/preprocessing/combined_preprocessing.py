"""
Combined Multimodal Preprocessing Pipeline for COOLANT Fake News Detection
Based on the COOLANT repository: https://github.com/wishever/COOLANT
"""

import os
import pickle
from venv import logger
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
import json
from PIL import Image

# Import centralized device detection
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.device import get_device

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
    """Vietnamese multimodal preprocessing pipeline for COOLANT."""

    def __init__(
        self,
        text_model_name: str = "vinai/phobert-base",
        image_model_name: str = "resnet18",
        language: str = "vi",
        max_text_length: int = 512,
        image_size: Tuple[int, int] = (224, 224),
        device: Optional[str] = None,
        use_word_segmentation: bool = True,
    ):
        """
        Initialize combined preprocessor

        Args:
            text_model_name: BERT model name for Vietnamese text
            image_model_name: Image backbone name. Options:
                "resnet18", "resnet50", "clip-vit-L-14", "siglip-base-224"
                See IMAGE_MODEL_REGISTRY in image_preprocessing.py for details.
            language: Language code ('vi')
            max_text_length: Maximum text sequence length
            image_size: Target image size (overridden by model registry for CLIP/SigLIP)
            device: Device to run preprocessing on (None = auto-detect)
        """
        self.device = get_device(device)
        logger.info(f"Using device: {self.device}")
        self.language = language

        # Initialize individual preprocessors with the same device
        self.text_preprocessor = TextPreprocessor(
            model_name=text_model_name,
            max_length=max_text_length,
            language=language,
            device=str(self.device),
            use_word_segmentation=use_word_segmentation,
        )

        self.image_preprocessor = ImagePreprocessor(
            model_name=image_model_name, device=str(self.device), image_size=image_size
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

    def process_existing_splits(
        self,
        data_dir: str = "./src/data/json",
        save_base_dir: str = "./processed_data",
        save_format: str = "pkl",
        batch_size: int = 32,
        splits: List[str] = ["train", "dev", "test"],
        file_prefix: str = "news_data_vifactcheck_",
    ) -> Dict[str, Tuple[MultimodalDataset, Dict]]:
        """
        Process existing dataset split files (train/dev/test.json)

        Args:
            data_dir: Directory containing split JSON files
            save_base_dir: Base directory to save processed data
            save_format: Format to save ('pkl' or 'npz')
            batch_size: Batch size for processing
            splits: List of split names to process
            file_prefix: Prefix for split files (e.g., "news_data_vifactcheck_")

        Returns:
            Dictionary with split names as keys and (dataset, metadata) tuples as values
        """
        import json

        results = {}

        for split_name in splits:
            print(f"\n🔄 Processing {split_name.upper()} split...")

            # Path to split file with customizable prefix
            split_path = os.path.join(data_dir, f"{file_prefix}{split_name}.json")

            if not os.path.exists(split_path):
                print(f"❌ Split file not found: {split_path}")
                continue

            # Load and extract data
            with open(split_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            # Extract texts, labels, and image paths
            texts = []
            labels = []
            image_paths = []
            base_data_dir = "./src/data"

            for item in raw_data:
                text = item.get("text", item.get("content", ""))
                if text:
                    texts.append(text)
                    labels.append(item.get("label", item.get("is_fake", 0)))

                    # Extract image path
                    images = item.get("images", [])
                    if images and len(images) > 0:
                        folder_path = images[0].get("folder_path", "")
                        if folder_path:
                            full_image_path = os.path.join(base_data_dir, folder_path)
                            image_paths.append(full_image_path)
                        else:
                            image_paths.append(None)
                    else:
                        image_paths.append(None)

            print(f"✓ Extracted {len(texts)} texts for {split_name}")
            print(
                f"✓ Found {sum(1 for path in image_paths if path is not None)} images"
            )

            # Create placeholder images for missing paths
            placeholder_dir = f"./placeholder_images/{split_name}"
            os.makedirs(placeholder_dir, exist_ok=True)

            for i, image_path in enumerate(image_paths):
                if image_path is None or not os.path.exists(image_path):
                    placeholder_path = os.path.join(
                        placeholder_dir, f"placeholder_{i}.jpg"
                    )
                    if not os.path.exists(placeholder_path):
                        from PIL import Image

                        placeholder_array = np.random.randint(
                            128, 200, (224, 224, 3), dtype=np.uint8
                        )
                        placeholder_image = Image.fromarray(placeholder_array)
                        placeholder_image.save(placeholder_path)
                    image_paths[i] = placeholder_path

            # Process the split
            split_save_dir = os.path.join(save_base_dir, f"vietnamese_{split_name}")
            dataset, metadata = self.preprocess_dataset(
                texts,
                image_paths,
                labels,
                save_dir=split_save_dir,
                save_format=save_format,
                batch_size=batch_size,
            )

            results[split_name] = (dataset, metadata)

            print(f"✓ {split_name.upper()} split completed: {len(dataset)} samples")

        return results
