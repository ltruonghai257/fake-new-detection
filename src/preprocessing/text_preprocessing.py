"""
Text Preprocessing Pipeline for COOLANT Fake News Detection
Vietnamese Language Support
Based on the COOLANT repository: https://github.com/wishever/COOLANT
"""

import re
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional, Union
import os
from tqdm import tqdm

# Optional Vietnamese text processing
try:
    import underthesea

    UNDERTHESEA_AVAILABLE = True
except ImportError:
    UNDERTHESEA_AVAILABLE = False


class TextPreprocessor:
    """
    Text preprocessing pipeline for multimodal fake news detection
    Supports Vietnamese language processing
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        max_length: int = 512,
        language: str = "vi",
        device: str = "cuda" if torch.cuda.is_available() else "mps",
    ):
        """
        Initialize text preprocessor for Vietnamese

        Args:
            model_name: BERT model name for Vietnamese (vinai/phobert-base or bert-base-multilingual-cased)
            max_length: Maximum sequence length
            language: Language code ('vi' for Vietnamese)
            device: Device to run preprocessing on
        """
        self.max_length = max_length
        self.language = language
        self.device = device

        # Initialize tokenizer for Vietnamese
        if language == "vi":
            if "phobert" in model_name.lower():
                self.model_name = "vinai/phobert-base"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.bert_model = AutoModel.from_pretrained(self.model_name)
            else:
                self.model_name = "bert-base-multilingual-cased"
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                self.bert_model = BertModel.from_pretrained(self.model_name)
        else:
            raise ValueError("This preprocessor only supports Vietnamese language (vi)")

        self.bert_model.to(device)
        self.bert_model.eval()

    def clean_text(self, text: str) -> str:
        """
        Clean Vietnamese text data

        Args:
            text: Raw Vietnamese text string

        Returns:
            Cleaned Vietnamese text string
        """
        # Vietnamese text cleaning
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove Vietnamese punctuation and special characters
        vietnamese_punctuation = r'[.,;:!?""' "(){}\[\]\\/|`~@#$%^&*+=<>—–]"
        text = re.sub(vietnamese_punctuation, "", text)

        # Remove numbers (optional - keep if important for your use case)
        text = re.sub(r"\d+", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace and convert to lowercase
        text = text.strip().lower()

        # Optional: Use underthesea for advanced Vietnamese text processing
        if UNDERTHESEA_AVAILABLE:
            try:
                # Text normalization
                text = underthesea.text_normalize(text)
            except:
                pass  # Fallback to basic cleaning if underthesea fails

        return text

    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize Vietnamese text using BERT tokenizer

        Args:
            text: Input Vietnamese text string

        Returns:
            Dictionary with tokenized inputs
        """
        cleaned_text = self.clean_text(text)

        encoded = self.tokenizer(
            cleaned_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {k: v.to(self.device) for k, v in encoded.items()}

    def extract_bert_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract BERT features for a list of Vietnamese texts

        Args:
            texts: List of Vietnamese text strings

        Returns:
            numpy array of BERT features (batch_size, hidden_size)
        """
        features = []

        with torch.no_grad():
            for text in tqdm(texts, desc="Extracting Vietnamese BERT features"):
                encoded = self.tokenize_text(text)
                outputs = self.bert_model(**encoded)

                # Use [CLS] token representation (pooled output)
                if hasattr(outputs, "pooler_output"):
                    pooled_output = outputs.pooler_output
                else:
                    # For PhoBERT, use the first token
                    pooled_output = outputs.last_hidden_state[:, 0, :]

                features.append(pooled_output.cpu().numpy())

        return np.vstack(features)

    def extract_token_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Extract token-level embeddings for CNN processing (FastCNN compatible)

        Args:
            texts: List of Vietnamese text strings

        Returns:
            numpy array of token embeddings (batch_size, seq_len, hidden_size)
        """
        embeddings = []

        with torch.no_grad():
            for text in tqdm(texts, desc="Extracting Vietnamese token embeddings"):
                encoded = self.tokenize_text(text)
                outputs = self.bert_model(**encoded)

                # Use last hidden state
                last_hidden_state = outputs.last_hidden_state
                embeddings.append(last_hidden_state.cpu().numpy())

        return np.vstack(embeddings)

    def preprocess_dataset(
        self,
        texts: List[str],
        labels: List[int],
        save_path: Optional[str] = None,
        extract_type: str = "bert_features",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess a complete Vietnamese dataset

        Args:
            texts: List of Vietnamese text strings
            labels: List of corresponding labels
            save_path: Path to save preprocessed data (optional)
            extract_type: Type of features to extract ('bert_features' or 'token_embeddings')

        Returns:
            Tuple of (features, labels)
        """
        if extract_type == "bert_features":
            features = self.extract_bert_features(texts)
        elif extract_type == "token_embeddings":
            features = self.extract_token_embeddings(texts)
        else:
            raise ValueError(
                "extract_type must be 'bert_features' or 'token_embeddings'"
            )

        labels_array = np.array(labels)

        if save_path:
            self.save_preprocessed_data(features, labels_array, save_path)

        return features, labels_array

    def save_preprocessed_data(
        self, features: np.ndarray, labels: np.ndarray, save_path: str
    ):
        """
        Save preprocessed Vietnamese data to pickle or npz format

        Args:
            features: Preprocessed Vietnamese text features
            labels: Corresponding labels
            save_path: Path to save the data
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if save_path.endswith(".pkl"):
            with open(save_path, "wb") as f:
                pickle.dump(
                    {
                        "features": features,
                        "labels": labels,
                        "language": "vietnamese",
                        "model_name": self.model_name,
                    },
                    f,
                )
        elif save_path.endswith(".npz"):
            np.savez(
                save_path,
                data=features,
                label=labels,
                language="vietnamese",
                model_name=self.model_name,
            )
        else:
            raise ValueError("save_path must end with .pkl or .npz")

        print(f"Saved preprocessed Vietnamese text data to {save_path}")

    @staticmethod
    def load_preprocessed_data(load_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load preprocessed Vietnamese data from file

        Args:
            load_path: Path to load data from

        Returns:
            Tuple of (features, labels)
        """
        if load_path.endswith(".pkl"):
            with open(load_path, "rb") as f:
                data = pickle.load(f)
                return data["features"], data["labels"]
        elif load_path.endswith(".npz"):
            loaded = np.load(load_path)
            return loaded["data"], loaded["label"]
        else:
            raise ValueError("load_path must end with .pkl or .npz")


class TextDataset(Dataset):
    """
    PyTorch Dataset for preprocessed Vietnamese text data
    Compatible with COOLANT training pipeline
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset

        Args:
            features: Preprocessed Vietnamese text features
            labels: Corresponding labels
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def preprocess_vietnamese_data(
    data_path: str,
    save_dir: str = "./processed_data/vietnamese",
    save_format: str = "npz",
    model_name: str = "vinai/phobert-base",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess Vietnamese dataset

    Args:
        data_path: Path to raw Vietnamese data
        save_dir: Directory to save processed data
        save_format: Format to save ('pkl' or 'npz')
        model_name: Model name for Vietnamese ('vinai/phobert-base' or 'bert-base-multilingual-cased')

    Returns:
        Tuple of (features, labels)
    """
    # Initialize preprocessor for Vietnamese
    preprocessor = TextPreprocessor(model_name=model_name, language="vi")

    # Load raw data (assuming JSON format)
    import json

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    texts = [item["text"] for item in raw_data]
    labels = [item["label"] for item in raw_data]

    # Preprocess
    features, labels_array = preprocessor.preprocess_dataset(
        texts,
        labels,
        save_path=f"{save_dir}/vietnamese_text_features.{save_format}",
        extract_type="token_embeddings",  # FastCNN compatible
    )

    return features, labels_array


if __name__ == "__main__":
    # Example usage for Vietnamese
    print("Vietnamese Text Preprocessing Pipeline for COOLANT")

    # Example Vietnamese texts
    sample_texts = [
        "Tin tức Việt Nam hôm nay: Chính phủ ban hành chính sách mới về kinh tế.",
        "Cảnh báo: Tin giả về dịch bệnh COVID-19 đang lan truyền trên mạng xã hội.",
        "Khoa học công nghệ Việt Nam đạt nhiều thành tựu quan trọng trong năm 2023.",
        "BREAKING: Phát hiện thuốc chữa bách bệnh - các chuyên gia cảnh báo tin giả.",
    ]
    sample_labels = [0, 1, 0, 1]  # 0: real news, 1: fake news

    # Initialize preprocessor for Vietnamese
    preprocessor = TextPreprocessor(model_name="vinai/phobert-base", language="vi")

    # Test text cleaning
    print("\nOriginal Vietnamese texts:")
    for i, text in enumerate(sample_texts):
        print(f"{i+1}. {text}")

    print("\nCleaned Vietnamese texts:")
    for i, text in enumerate(sample_texts):
        cleaned = preprocessor.clean_text(text)
        print(f"{i+1}. {cleaned}")

    # Preprocess and save
    features, labels = preprocessor.preprocess_dataset(
        sample_texts,
        sample_labels,
        save_path="./sample_vietnamese_text_data.pkl",
        extract_type="token_embeddings",
    )

    print(f"\nFeatures shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print("Vietnamese text preprocessing completed successfully!")
