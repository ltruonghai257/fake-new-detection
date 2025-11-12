import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any
from .base import BaseModel, FastCNN


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CLIPTextEncoder(nn.Module):
    """CLIP-style text encoder with both CNN and Transformer options."""
    
    def __init__(self, 
                 vocab_size: int = 49408,
                 embed_dim: int = 768,
                 num_heads: int = 8,
                 num_layers: int = 12,
                 hidden_dim: int = 2048,
                 dropout: float = 0.1,
                 use_cnn: bool = True,
                 cnn_channel: int = 32,
                 cnn_kernel_size: Tuple[int, ...] = (1, 2, 4, 8),
                 output_dim: int = 512):
        super(CLIPTextEncoder, self).__init__()
        
        self.use_cnn = use_cnn
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        
        if use_cnn:
            # CNN-based text encoding (FastCNN)
            self.text_encoding = FastCNN(
                channel=cnn_channel,
                kernel_size=cnn_kernel_size
            )
            self.text_projection = nn.Sequential(
                nn.Linear(cnn_channel * len(cnn_kernel_size), 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU()
            )
        else:
            # Transformer-based text encoding
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
            self.positional_encoding = PositionalEncoding(embed_dim, dropout)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.text_projection = nn.Linear(embed_dim, output_dim)
    
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        if self.use_cnn:
            # text shape: (batch_size, seq_len, embed_dim)
            text_features = self.text_encoding(text)
            text_output = self.text_projection(text_features)
        else:
            # text shape: (batch_size, seq_len) - token indices
            x = self.token_embedding(text) * math.sqrt(self.embed_dim)
            x = self.positional_encoding(x)
            x = self.transformer(x)
            # Use the last token representation
            text_output = self.text_projection(x[:, -1, :])
        
        return F.normalize(text_output, p=2, dim=-1)


class CLIPImageEncoder(nn.Module):
    """CLIP-style image encoder."""
    
    def __init__(self, 
                 input_dim: int = 512,
                 hidden_dim: int = 256,
                 output_dim: int = 512,
                 dropout: float = 0.1):
        super(CLIPImageEncoder, self).__init__()
        
        self.image_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image_output = self.image_encoder(image)
        return F.normalize(image_output, p=2, dim=-1)


class CLIP(BaseModel):
    """CLIP model for cross-modal contrastive learning."""
    
    def __init__(self, config: Dict[str, Any]):
        super(CLIP, self).__init__(config)
        
        # Extract configuration
        self.vocab_size = config.get('vocab_size', 49408)
        self.text_embed_dim = config.get('text_embed_dim', 768)
        self.image_input_dim = config.get('image_input_dim', 512)
        self.output_dim = config.get('output_dim', 512)
        self.temperature = config.get('temperature', 0.07)
        
        # Text encoder configuration
        text_config = config.get('text_encoder', {})
        self.text_encoder = CLIPTextEncoder(
            vocab_size=self.vocab_size,
            embed_dim=self.text_embed_dim,
            num_heads=text_config.get('num_heads', 8),
            num_layers=text_config.get('num_layers', 12),
            hidden_dim=text_config.get('hidden_dim', 2048),
            dropout=text_config.get('dropout', 0.1),
            use_cnn=text_config.get('use_cnn', True),
            cnn_channel=text_config.get('cnn_channel', 32),
            cnn_kernel_size=text_config.get('cnn_kernel_size', (1, 2, 4, 8)),
            output_dim=self.output_dim
        )
        
        # Image encoder configuration
        image_config = config.get('image_encoder', {})
        self.image_encoder = CLIPImageEncoder(
            input_dim=self.image_input_dim,
            hidden_dim=image_config.get('hidden_dim', 256),
            output_dim=self.output_dim,
            dropout=image_config.get('dropout', 0.1)
        )
        
        # Classification heads for fake news detection
        self.text_classifier = nn.Linear(self.output_dim, config.get('num_classes', 2))
        self.image_classifier = nn.Linear(self.output_dim, config.get('num_classes', 2))
        self.multimodal_classifier = nn.Sequential(
            nn.Linear(self.output_dim * 2, self.output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.output_dim, config.get('num_classes', 2))
        )
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / self.temperature))
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text input."""
        return self.text_encoder(text)
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image input."""
        return self.image_encoder(image)
    
    def forward(self, text: torch.Tensor, image: torch.Tensor, 
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of CLIP model.
        
        Args:
            text: Text input tensor
            image: Image input tensor
            return_features: Whether to return encoded features
            
        Returns:
            Dictionary containing logits and optionally features
        """
        # Encode modalities
        text_features = self.encode_text(text)
        image_features = self.encode_image(image)
        
        # Classification predictions
        text_logits = self.text_classifier(text_features)
        image_logits = self.image_classifier(image_features)
        
        # Multimodal fusion for classification
        multimodal_features = torch.cat([text_features, image_features], dim=-1)
        multimodal_logits = self.multimodal_classifier(multimodal_features)
        
        # Contrastive learning logits
        logit_scale = self.logit_scale.exp()
        contrastive_logits = logit_scale * text_features @ image_features.T
        
        outputs = {
            'text_logits': text_logits,
            'image_logits': image_logits,
            'multimodal_logits': multimodal_logits,
            'contrastive_logits': contrastive_logits,
            'logit_scale': logit_scale
        }
        
        if return_features:
            outputs.update({
                'text_features': text_features,
                'image_features': image_features,
                'multimodal_features': multimodal_features
            })
        
        return outputs
    
    def compute_contrastive_loss(self, text: torch.Tensor, 
                               image: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between text and image."""
        outputs = self.forward(text, image)
        contrastive_logits = outputs['contrastive_logits']
        
        batch_size = contrastive_logits.size(0)
        labels = torch.arange(batch_size, device=contrastive_logits.device)
        
        loss_text_to_image = F.cross_entropy(contrastive_logits, labels)
        loss_image_to_text = F.cross_entropy(contrastive_logits.T, labels)
        
        return (loss_text_to_image + loss_image_to_text) / 2
    
    def compute_classification_loss(self, text: torch.Tensor, image: torch.Tensor, 
                                  labels: torch.Tensor, 
                                  loss_weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """Compute classification loss."""
        if loss_weights is None:
            loss_weights = {'text': 1.0, 'image': 1.0, 'multimodal': 1.0}
        
        outputs = self.forward(text, image)
        
        text_loss = F.cross_entropy(outputs['text_logits'], labels)
        image_loss = F.cross_entropy(outputs['image_logits'], labels)
        multimodal_loss = F.cross_entropy(outputs['multimodal_logits'], labels)
        
        total_loss = (
            loss_weights['text'] * text_loss +
            loss_weights['image'] * image_loss +
            loss_weights['multimodal'] * multimodal_loss
        )
        
        return total_loss
    
    def get_similarity_scores(self, text: torch.Tensor, 
                            image: torch.Tensor) -> torch.Tensor:
        """Get similarity scores between text and image pairs."""
        text_features = self.encode_text(text)
        image_features = self.encode_image(image)
        
        # Compute cosine similarity
        similarity_scores = torch.sum(text_features * image_features, dim=-1)
        return similarity_scores
