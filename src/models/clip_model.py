"""
CLIP (Contrastive Language-Image Pre-training) Model Implementation

This module implements a CLIP-style contrastive learning model for multimodal
fake news detection, providing cross-modal alignment between text and image features.

Based on the CLIP architecture adapted for the COOLANT framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence data."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]


class CLIPTextEncoder(nn.Module):
    """Text encoder for CLIP model."""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 512):
        super(CLIPTextEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, embed_dim) or (batch_size, embed_dim)
        if x.dim() == 3:
            # Pool over sequence dimension
            x = torch.mean(x, dim=1)  # (batch_size, embed_dim)
        
        return self.encoder(x)


class CLIPImageEncoder(nn.Module):
    """Image encoder for CLIP model."""
    
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 1024, output_dim: int = 512):
        super(CLIPImageEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, feature_dim)
        return self.encoder(x)


class CLIP(nn.Module):
    """
    CLIP model for cross-modal contrastive learning.
    
    This implementation follows the CLIP architecture adapted for multimodal
    fake news detection, providing text-image alignment through contrastive learning.
    """
    
    def __init__(self, 
                 text_input_dim: int = 768,
                 image_input_dim: int = 2048,
                 embed_dim: int = 512,
                 temperature: float = 0.07,
                 max_seq_len: int = 512):
        super(CLIP, self).__init__()
        
        self.embed_dim = embed_dim
        self.temperature = temperature
        
        # Text and image encoders
        self.text_encoder = CLIPTextEncoder(
            input_dim=text_input_dim,
            hidden_dim=embed_dim * 2,
            output_dim=embed_dim
        )
        
        self.image_encoder = CLIPImageEncoder(
            input_dim=image_input_dim,
            hidden_dim=embed_dim * 2,
            output_dim=embed_dim
        )
        
        # Positional encoding for text sequences
        self.positional_encoding = PositionalEncoding(
            d_model=text_input_dim,
            max_len=max_seq_len
        )
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text features."""
        # Add positional encoding if text is sequential
        if text.dim() == 3:
            # Apply positional encoding
            batch_size, seq_len, embed_dim = text.shape
            text = text + self.positional_encoding.pe[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        
        return self.text_encoder(text)
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image features."""
        return self.image_encoder(image)
    
    def forward(self, 
                text: torch.Tensor, 
                image: torch.Tensor,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CLIP model.
        
        Args:
            text: Text features (batch_size, seq_len, embed_dim) or (batch_size, embed_dim)
            image: Image features (batch_size, feature_dim)
            return_features: Whether to return raw encoded features
            
        Returns:
            Dictionary containing:
            - text_embed: Normalized text embeddings
            - image_embed: Normalized image embeddings
            - logits_per_text: Text-to-image similarity logits
            - logits_per_image: Image-to-text similarity logits
            - text_features: Raw text features (if return_features=True)
            - image_features: Raw image features (if return_features=True)
        """
        # Encode text and image
        text_features = self.encode_text(text)
        image_features = self.encode_image(image)
        
        # Normalize features
        text_embed = F.normalize(text_features, dim=-1)
        image_embed = F.normalize(image_features, dim=-1)
        
        # Compute similarity matrix
        logits_per_text = torch.matmul(text_embed, image_embed.t()) * self.logit_scale.exp()
        logits_per_image = logits_per_text.t()
        
        outputs = {
            'text_embed': text_embed,
            'image_embed': image_embed,
            'logits_per_text': logits_per_text,
            'logits_per_image': logits_per_image
        }
        
        if return_features:
            outputs.update({
                'text_features': text_features,
                'image_features': image_features
            })
        
        return outputs
    
    def compute_contrastive_loss(self, 
                                 text_features: torch.Tensor,
                                 image_features: torch.Tensor,
                                 labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute CLIP contrastive loss.
        
        Args:
            text_features: Text features (batch_size, embed_dim)
            image_features: Image features (batch_size, embed_dim)
            labels: Optional labels for supervised contrastive learning
            
        Returns:
            Contrastive loss
        """
        # Normalize features
        text_features = F.normalize(text_features, dim=-1)
        image_features = F.normalize(image_features, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.t()) * self.logit_scale.exp()
        
        if labels is None:
            # Unsupervised contrastive loss (standard CLIP)
            batch_size = logits.shape[0]
            labels = torch.arange(batch_size, device=logits.device)
        
        # Compute cross-entropy loss
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2
    
    def get_similarity_scores(self, 
                            text: torch.Tensor, 
                            image: torch.Tensor) -> torch.Tensor:
        """
        Get similarity scores between text and image pairs.
        
        Args:
            text: Text features (batch_size, seq_len, embed_dim) or (batch_size, embed_dim)
            image: Image features (batch_size, feature_dim)
            
        Returns:
            Similarity scores (batch_size,)
        """
        with torch.no_grad():
            outputs = self.forward(text, image)
            text_embed = outputs['text_embed']
            image_embed = outputs['image_embed']
            
            # Compute cosine similarity
            similarity = torch.sum(text_embed * image_embed, dim=-1)
            
        return similarity


def create_clip_model(config: Dict[str, Any]) -> CLIP:
    """
    Create a CLIP model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        CLIP model instance
    """
    return CLIP(
        text_input_dim=config.get('text_input_dim', 768),
        image_input_dim=config.get('image_input_dim', 2048),
        embed_dim=config.get('embed_dim', 512),
        temperature=config.get('temperature', 0.07),
        max_seq_len=config.get('max_seq_len', 512)
    )


# Default configuration
def get_default_clip_config() -> Dict[str, Any]:
    """Get default CLIP configuration."""
    return {
        'text_input_dim': 768,
        'image_input_dim': 2048,
        'embed_dim': 512,
        'temperature': 0.07,
        'max_seq_len': 512
    }
