import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


class BaseModel(nn.Module, ABC):
    """Base class for all fake news detection models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass of the model."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return self.config
    
    def save_pretrained(self, save_directory: str):
        """Save model weights and configuration."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, f"{save_directory}/model.pt")
    
    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load model from saved weights and configuration."""
        checkpoint = torch.load(f"{load_directory}/model.pt")
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


class MultimodalModel(BaseModel):
    """Base class for multimodal fake news detection models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.text_dim = config.get('text_dim', 768)
        self.image_dim = config.get('image_dim', 512)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_classes = config.get('num_classes', 2)
        
    @abstractmethod
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text input."""
        pass
    
    @abstractmethod
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image input."""
        pass
    
    @abstractmethod
    def fuse_modalities(self, text_features: torch.Tensor, 
                       image_features: torch.Tensor) -> torch.Tensor:
        """Fuse text and image features."""
        pass


class TextEncoder(nn.Module):
    """Base text encoder using CNN or Transformer."""
    
    def __init__(self, input_dim: int = 768, output_dim: int = 128, 
                 encoder_type: str = 'cnn'):
        super().__init__()
        self.encoder_type = encoder_type
        
        if encoder_type == 'cnn':
            self.encoder = FastCNN(output_dim // 4)
            self.projection = nn.Sequential(
                nn.Linear(output_dim, output_dim // 2),
                nn.BatchNorm1d(output_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim // 2, output_dim)
            )
        else:
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=input_dim,
                    nhead=8,
                    dim_feedforward=2048,
                    dropout=0.1
                ),
                num_layers=6
            )
            self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == 'cnn':
            x = self.encoder(x)
            x = self.projection(x)
        else:
            x = self.encoder(x)
            x = x.mean(dim=1)  # Global average pooling
            x = self.projection(x)
        return x


class ImageEncoder(nn.Module):
    """Base image encoder."""
    
    def __init__(self, input_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class FastCNN(nn.Module):
    """Fast CNN for text encoding with multiple kernel sizes."""
    
    def __init__(self, channel: int = 32, kernel_size: Tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        self.fast_cnn = nn.ModuleList()
        for kernel in kernel_size:
            self.fast_cnn.append(
                nn.Sequential(
                    nn.Conv1d(200, channel, kernel_size=kernel),
                    nn.BatchNorm1d(channel),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.AdaptiveMaxPool1d(1)
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (N, seq_len, embed_dim) -> (N, embed_dim, seq_len)
        x_out = []
        for module in self.fast_cnn:
            x_out.append(module(x).squeeze(-1))
        x_out = torch.cat(x_out, 1)
        return x_out


class AttentionFusion(nn.Module):
    """Attention-based fusion for multimodal features."""
    
    def __init__(self, text_dim: int, image_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1)
        
    def forward(self, text_features: torch.Tensor, 
                image_features: torch.Tensor) -> torch.Tensor:
        text_proj = self.text_proj(text_features).unsqueeze(0)
        image_proj = self.image_proj(image_features).unsqueeze(0)
        
        # Cross-attention
        fused_features, _ = self.attention(text_proj, image_proj, image_proj)
        return fused_features.squeeze(0)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for cross-modal learning."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, text_features: torch.Tensor, 
                image_features: torch.Tensor) -> torch.Tensor:
        # Normalize features
        text_features = F.normalize(text_features, p=2, dim=-1)
        image_features = F.normalize(image_features, p=2, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(text_features, image_features.T) / self.temperature
        
        # Create labels (positive pairs are on the diagonal)
        batch_size = text_features.size(0)
        labels = torch.arange(batch_size, device=text_features.device)
        
        # Compute contrastive loss
        loss_text_to_image = F.cross_entropy(similarity_matrix, labels)
        loss_image_to_text = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_text_to_image + loss_image_to_text) / 2
