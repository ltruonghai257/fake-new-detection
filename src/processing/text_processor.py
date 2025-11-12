#!/usr/bin/env python3
"""
Text Processing Module for ViFactCheck Dataset

Handles Vietnamese text preprocessing, cleaning, and feature extraction.
"""

import re
import torch
import numpy as np
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Vietnamese text processor for fake news detection.
    
    Handles text cleaning, tokenization, and feature extraction.
    """
    
    def __init__(self, 
                 max_length: int = 128,
                 min_length: int = 10,
                 tokenizer_name: str = "vinai/phobert-base"):
        """
        Initialize text processor.
        
        Args:
            max_length: Maximum sequence length
            min_length: Minimum text length to keep
            tokenizer_name: Pretrained tokenizer name
        """
        self.max_length = max_length
        self.min_length = min_length
        self.tokenizer_name = tokenizer_name
        
        # Initialize tokenizer
        self.tokenizer = None
        self.model = None
        self._init_models()
    
    def _init_models(self):
        """Initialize tokenizer and model."""
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.model = AutoModel.from_pretrained(self.tokenizer_name)
            self.model.eval()
            logger.info(f"Loaded text model: {self.tokenizer_name}")
        except Exception as e:
            logger.warning(f"Could not load {self.tokenizer_name}: {e}")
            logger.info("Using fallback text processing")
    
    def clean_text(self, text: str) -> Optional[str]:
        """
        Clean and normalize Vietnamese text.
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text or None if invalid
        """
        if not text or not isinstance(text, str):
            return None
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep Vietnamese diacritics and basic punctuation
        text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF.,!?;:()-]', '', text)
        
        # Check length constraints
        if len(text) < self.min_length:
            return None
        
        # Truncate if too long
        if len(text) > 2000:
            text = text[:2000] + "..."
        
        return text.strip()
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize text using the loaded tokenizer.
        
        Args:
            text: Cleaned text
            
        Returns:
            Dictionary with tokenized inputs
        """
        if self.tokenizer is None:
            # Fallback tokenization
            words = text.split()[:self.max_length]
            # Pad with zeros
            input_ids = list(range(len(words))) + [0] * (self.max_length - len(words))
            attention_mask = [1] * len(words) + [0] * (self.max_length - len(words))
            
            return {
                'input_ids': torch.tensor(input_ids[:self.max_length]),
                'attention_mask': torch.tensor(attention_mask[:self.max_length])
            }
        
        # Use transformers tokenizer
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0)
        }
    
    def extract_features(self, text: str) -> torch.Tensor:
        """
        Extract features from text.
        
        Args:
            text: Input text
            
        Returns:
            Text features tensor
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return torch.zeros(768)  # Default embedding size
        
        # Tokenize
        inputs = self.tokenize(cleaned_text)
        
        if self.model is None:
            # Fallback: return random features
            return torch.randn(768)
        
        try:
            # Extract features using the model
            with torch.no_grad():
                input_ids = inputs['input_ids'].unsqueeze(0)
                attention_mask = inputs['attention_mask'].unsqueeze(0)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Use [CLS] token representation or mean pooling
                features = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                
                return features
                
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return torch.randn(768)
    
    def process(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Complete text processing pipeline.
        
        Args:
            text: Raw text input
            
        Returns:
            Dictionary with processed text data
        """
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'features': torch.zeros(768),
                'text': "",
                'valid': False
            }
        
        tokens = self.tokenize(cleaned_text)
        features = self.extract_features(cleaned_text)
        
        return {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
            'features': features,
            'text': cleaned_text,
            'valid': True
        }
    
    def batch_process(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of texts.
        
        Args:
            texts: List of text inputs
            
        Returns:
            Batched processed data
        """
        processed = [self.process(text) for text in texts]
        
        return {
            'input_ids': torch.stack([p['input_ids'] for p in processed]),
            'attention_mask': torch.stack([p['attention_mask'] for p in processed]),
            'features': torch.stack([p['features'] for p in processed]),
            'texts': [p['text'] for p in processed],
            'valid': [p['valid'] for p in processed]
        }


def create_text_processor(max_length: int = 128, 
                         tokenizer_name: str = "vinai/phobert-base") -> TextProcessor:
    """
    Factory function to create a text processor.
    
    Args:
        max_length: Maximum sequence length
        tokenizer_name: Pretrained tokenizer name
        
    Returns:
        TextProcessor instance
    """
    return TextProcessor(max_length=max_length, tokenizer_name=tokenizer_name)


if __name__ == "__main__":
    # Example usage
    processor = TextProcessor()
    
    sample_text = "Chính phủ công bố các biện pháp hỗ trợ người dân trong đại dịch COVID-19."
    result = processor.process(sample_text)
    
    print("Text processing result:")
    print(f"Input IDs shape: {result['input_ids'].shape}")
    print(f"Features shape: {result['features'].shape}")
    print(f"Cleaned text: {result['text']}")
    print(f"Valid: {result['valid']}")
