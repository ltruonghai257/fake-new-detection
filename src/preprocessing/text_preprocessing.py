"""
Text Preprocessing Pipeline for COOLANT Fake News Detection
Based on the COOLANT repository: https://github.com/wishever/COOLANT
"""

import re
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
from typing import List, Dict, Tuple, Optional, Union
import os
from tqdm import tqdm


class TextPreprocessor:
    """
    Text preprocessing pipeline for multimodal fake news detection
    Supports both English (Twitter) and Chinese (Weibo) datasets
    """
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 max_length: int = 512,
                 language: str = "en",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize text preprocessor
        
        Args:
            model_name: BERT model name (bert-base-uncased for English, bert-base-chinese for Chinese)
            max_length: Maximum sequence length
            language: Language code ('en' for English, 'zh' for Chinese)
            device: Device to run preprocessing on
        """
        self.max_length = max_length
        self.language = language
        self.device = device
        
        # Initialize tokenizer based on language
        if language == "zh":
            self.model_name = "bert-base-chinese"
        else:
            self.model_name = model_name
            
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = BertModel.from_pretrained(self.model_name)
        self.bert_model.to(device)
        self.bert_model.eval()
        
    def clean_text(self, text: str) -> str:
        """
        Clean text data based on COOLANT preprocessing
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if self.language == "zh":
            # Chinese text cleaning (from weibo.py)
            text = re.sub(u"[，。 :,.；|-""——_/nbsp+&;@、《》～（）())#O！：【】]", "", text)
        else:
            # English text cleaning
            text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
            text = re.sub(r"\s+", " ", text)     # Remove extra whitespace
            
        return text.strip().lower()
    
    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize text using BERT tokenizer
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with tokenized inputs
        """
        cleaned_text = self.clean_text(text)
        
        encoded = self.tokenizer(
            cleaned_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {k: v.to(self.device) for k, v in encoded.items()}
    
    def extract_bert_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract BERT features for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of BERT features (batch_size, hidden_size)
        """
        features = []
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Extracting BERT features"):
                encoded = self.tokenize_text(text)
                outputs = self.bert_model(**encoded)
                
                # Use [CLS] token representation (pooled output)
                pooled_output = outputs.pooler_output
                features.append(pooled_output.cpu().numpy())
                
        return np.vstack(features)
    
    def extract_token_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Extract token-level embeddings for CNN processing (FastCNN compatible)
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of token embeddings (batch_size, seq_len, hidden_size)
        """
        embeddings = []
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Extracting token embeddings"):
                encoded = self.tokenize_text(text)
                outputs = self.bert_model(**encoded)
                
                # Use last hidden state
                last_hidden_state = outputs.last_hidden_state
                embeddings.append(last_hidden_state.cpu().numpy())
                
        return np.vstack(embeddings)
    
    def preprocess_dataset(self, 
                          texts: List[str], 
                          labels: List[int],
                          save_path: Optional[str] = None,
                          extract_type: str = "bert_features") -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess a complete dataset
        
        Args:
            texts: List of text strings
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
            raise ValueError("extract_type must be 'bert_features' or 'token_embeddings'")
        
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
            features: Preprocessed text features
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
            
        print(f"Saved preprocessed text data to {save_path}")
    
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


class TextDataset(Dataset):
    """
    PyTorch Dataset for preprocessed text data
    Compatible with COOLANT training pipeline
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset
        
        Args:
            features: Preprocessed text features
            labels: Corresponding labels
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def preprocess_twitter_data(data_path: str, 
                          save_dir: str = "./processed_data/twitter",
                          save_format: str = "npz") -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess Twitter dataset (English)
    
    Args:
        data_path: Path to raw Twitter data
        save_dir: Directory to save processed data
        save_format: Format to save ('pkl' or 'npz')
        
    Returns:
        Tuple of (features, labels)
    """
    # Initialize preprocessor for English
    preprocessor = TextPreprocessor(
        model_name="bert-base-uncased",
        language="en"
    )
    
    # Load raw data (assuming JSON format)
    import json
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    texts = [item['text'] for item in raw_data]
    labels = [item['label'] for item in raw_data]
    
    # Preprocess
    features, labels_array = preprocessor.preprocess_dataset(
        texts, labels, 
        save_path=f"{save_dir}/text_features.{save_format}",
        extract_type="token_embeddings"  # FastCNN compatible
    )
    
    return features, labels_array


def preprocess_weibo_data(data_path: str,
                         save_dir: str = "./processed_data/weibo", 
                         save_format: str = "npz") -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess Weibo dataset (Chinese)
    
    Args:
        data_path: Path to raw Weibo data
        save_dir: Directory to save processed data
        save_format: Format to save ('pkl' or 'npz')
        
    Returns:
        Tuple of (features, labels)
    """
    # Initialize preprocessor for Chinese
    preprocessor = TextPreprocessor(
        model_name="bert-base-chinese",
        language="zh"
    )
    
    # Load raw data (assuming JSON format)
    import json
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    texts = [item['text'] for item in raw_data]
    labels = [item['label'] for item in raw_data]
    
    # Preprocess
    features, labels_array = preprocessor.preprocess_dataset(
        texts, labels,
        save_path=f"{save_dir}/text_features.{save_format}",
        extract_type="token_embeddings"  # FastCNN compatible
    )
    
    return features, labels_array


if __name__ == "__main__":
    # Example usage
    print("Text Preprocessing Pipeline for COOLANT")
    
    # Example for preprocessing sample data
    sample_texts = [
        "This is a sample news article about fake news detection.",
        "Another example of text that needs preprocessing."
    ]
    sample_labels = [0, 1]  # 0: real news, 1: fake news
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(language="en")
    
    # Preprocess and save
    features, labels = preprocessor.preprocess_dataset(
        sample_texts, 
        sample_labels,
        save_path="./sample_text_data.pkl",
        extract_type="token_embeddings"
    )
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print("Text preprocessing completed successfully!")
