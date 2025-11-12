import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
from typing import Dict, Any, Tuple, Optional

from .base import MultimodalModel, FastCNN
from .senet import SEAttentionModule


class EncodingPart(nn.Module):
    """Shared encoding module for text and image features."""
    
    def __init__(self,
                 cnn_channel: int = 32,
                 cnn_kernel_size: Tuple[int, ...] = (1, 2, 4, 8),
                 shared_image_dim: int = 128,
                 shared_text_dim: int = 128):
        super(EncodingPart, self).__init__()
        
        # Text encoding
        self.shared_text_encoding = FastCNN(
            channel=cnn_channel,
            kernel_size=cnn_kernel_size
        )
        self.shared_text_linear = nn.Sequential(
            nn.Linear(cnn_channel * len(cnn_kernel_size), 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, shared_text_dim),
            nn.BatchNorm1d(shared_text_dim),
            nn.ReLU()
        )
        
        # Image encoding
        self.shared_image = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, shared_image_dim),
            nn.BatchNorm1d(shared_image_dim),
            nn.ReLU()
        )

    def forward(self, text: torch.Tensor, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        text_encoding = self.shared_text_encoding(text)
        text_shared = self.shared_text_linear(text_encoding)
        image_shared = self.shared_image(image)
        return text_shared, image_shared


class SimilarityModule(nn.Module):
    """Module for computing cross-modal similarity."""
    
    def __init__(self, shared_dim: int = 128, sim_dim: int = 64):
        super(SimilarityModule, self).__init__()
        
        self.encoding = EncodingPart()
        
        # Alignment networks
        self.text_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        
        self.image_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        
        # Similarity classifier
        self.sim_classifier_dim = sim_dim * 2
        self.sim_classifier = nn.Sequential(
            nn.BatchNorm1d(self.sim_classifier_dim),
            nn.Linear(self.sim_classifier_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, text: torch.Tensor, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_encoding, image_encoding = self.encoding(text, image)
        text_aligned = self.text_aligner(text_encoding)
        image_aligned = self.image_aligner(image_encoding)
        
        # Concatenate for similarity prediction
        sim_feature = torch.cat([text_aligned, image_aligned], 1)
        pred_similarity = self.sim_classifier(sim_feature)
        
        return text_aligned, image_aligned, pred_similarity


class Encoder(nn.Module):
    """Variational encoder for ambiguity learning."""
    
    def __init__(self, input_dim: int = 64, z_dim: int = 2):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(True),
            nn.Linear(input_dim, z_dim * 2),
        )

    def forward(self, x: torch.Tensor) -> Independent:
        params = self.net(x)
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7
        return Independent(Normal(loc=mu, scale=sigma), 1)


class AmbiguityLearning(nn.Module):
    """Module for learning cross-modal ambiguity using variational inference."""
    
    def __init__(self, input_dim: int = 64):
        super(AmbiguityLearning, self).__init__()
        self.encoder_text = Encoder(input_dim)
        self.encoder_image = Encoder(input_dim)

    def forward(self, text_encoding: torch.Tensor, image_encoding: torch.Tensor) -> torch.Tensor:
        # Get variational distributions
        p_z1_given_text = self.encoder_text(text_encoding)
        p_z2_given_image = self.encoder_image(image_encoding)
        
        # Sample from distributions
        z1 = p_z1_given_text.rsample()
        z2 = p_z2_given_image.rsample()
        
        # Compute symmetric KL divergence
        kl_1_2 = p_z1_given_text.log_prob(z1) - p_z2_given_image.log_prob(z1)
        kl_2_1 = p_z2_given_image.log_prob(z2) - p_z1_given_text.log_prob(z2)
        skl = (kl_1_2 + kl_2_1) / 2.0
        skl = torch.sigmoid(skl)
        
        return skl


class UnimodalDetection(nn.Module):
    """Module for unimodal feature extraction."""
    
    def __init__(self, shared_dim: int = 128, prime_dim: int = 16):
        super(UnimodalDetection, self).__init__()
        
        self.text_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU()
        )
        
        self.image_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU()
        )

    def forward(self, text_encoding: torch.Tensor, 
                image_encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        text_prime = self.text_uni(text_encoding)
        image_prime = self.image_uni(image_encoding)
        return text_prime, image_prime


class CrossModule4Batch(nn.Module):
    """Cross-modal correlation module."""
    
    def __init__(self, text_in_dim: int = 64, image_in_dim: int = 64, corre_out_dim: int = 64):
        super(CrossModule4Batch, self).__init__()
        
        self.softmax = nn.Softmax(-1)
        self.corre_dim = 64
        self.pooling = nn.AdaptiveMaxPool1d(1)
        
        self.c_specific_2 = nn.Sequential(
            nn.Linear(self.corre_dim, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim),
            nn.ReLU()
        )

    def forward(self, text: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        text_in = text.unsqueeze(2)  # (batch_size, text_dim, 1)
        image_in = image.unsqueeze(1)  # (batch_size, 1, image_dim)
        
        corre_dim = text.shape[1]
        similarity = torch.matmul(text_in, image_in) / math.sqrt(corre_dim)
        correlation = self.softmax(similarity)
        correlation_p = self.pooling(correlation).squeeze()
        correlation_out = self.c_specific_2(correlation_p)
        
        return correlation_out


class DetectionModule(nn.Module):
    """Main detection module combining all components."""
    
    def __init__(self, feature_dim: int = 96, h_dim: int = 64):  # 64 + 16 + 16
        super(DetectionModule, self).__init__()
        
        self.encoding = EncodingPart()
        self.ambiguity_module = AmbiguityLearning()
        self.uni_repre = UnimodalDetection()
        self.uni_se = UnimodalDetection(prime_dim=64)
        self.cross_module = CrossModule4Batch()
        
        # SE attention module
        self.senet = SEAttentionModule(
            text_dim=64, 
            image_dim=64, 
            correlation_dim=64
        )
        
        # Final classifier
        self.classifier_corre = nn.Sequential(
            nn.Linear(feature_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 2)
        )

    def forward(self, text_raw: torch.Tensor, image_raw: torch.Tensor, 
                text: torch.Tensor, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Get shared representations
        text_prime, image_prime = self.encoding(text_raw, image_raw)
        
        # Get SE features and unimodal representations
        text_se, image_se = self.uni_se(text_prime, image_prime)
        text_prime, image_prime = self.uni_repre(text_prime, image_prime)
        
        # Cross-modal correlation
        correlation = self.cross_module(text, image)
        
        # SE attention weights
        attention_score = self.senet(text_se, image_se, correlation)
        
        # Apply attention weights
        text_final = text_prime * attention_score[:, 0].unsqueeze(1)
        img_final = image_prime * attention_score[:, 1].unsqueeze(1)
        corre_final = correlation * attention_score[:, 2].unsqueeze(1)
        
        # Final feature concatenation
        final_corre = torch.cat([text_final, img_final, corre_final], 1)
        pre_label = self.classifier_corre(final_corre)
        
        # Ambiguity learning
        skl = self.ambiguity_module(text, image)
        weight_uni = (1 - skl).unsqueeze(1)
        weight_corre = skl.unsqueeze(1)
        skl_score = torch.cat([weight_uni, weight_uni, weight_corre], 1)
        
        return pre_label, attention_score, skl_score


class COOLANT(MultimodalModel):
    """
    COOLANT: Cross-modal Contrastive Learning for Multimodal Fake News Detection
    
    This model implements the COOLANT architecture with:
    - Cross-modal contrastive learning
    - Ambiguity learning with variational inference
    - SE attention mechanism
    - Multimodal fusion for fake news detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(COOLANT, self).__init__(config)
        
        # Model components
        self.similarity_module = SimilarityModule(
            shared_dim=config.get('shared_dim', 128),
            sim_dim=config.get('sim_dim', 64)
        )
        
        self.detection_module = DetectionModule(
            feature_dim=config.get('feature_dim', 96),
            h_dim=config.get('h_dim', 64)
        )
        
        # Loss weights
        self.contrastive_weight = config.get('contrastive_weight', 1.0)
        self.classification_weight = config.get('classification_weight', 1.0)
        self.similarity_weight = config.get('similarity_weight', 0.5)
        
        # Temperature for contrastive learning
        self.temperature = config.get('temperature', 0.07)
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text input."""
        text_shared, _ = self.similarity_module.encoding(text, torch.zeros(text.size(0), 512, device=text.device))
        return text_shared
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image input."""
        _, image_shared = self.similarity_module.encoding(torch.zeros(image.size(0), 30, 200, device=image.device), image)
        return image_shared
    
    def fuse_modalities(self, text_features: torch.Tensor, 
                       image_features: torch.Tensor) -> torch.Tensor:
        """Fuse text and image features."""
        return torch.cat([text_features, image_features], dim=-1)
    
    def forward(self, text_raw: torch.Tensor, image_raw: torch.Tensor,
                text_aligned: Optional[torch.Tensor] = None,
                image_aligned: Optional[torch.Tensor] = None,
                return_all: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of COOLANT model.
        
        Args:
            text_raw: Raw text features
            image_raw: Raw image features  
            text_aligned: Pre-aligned text features (optional)
            image_aligned: Pre-aligned image features (optional)
            return_all: Whether to return all intermediate outputs
            
        Returns:
            Dictionary containing model outputs
        """
        # Get similarity features
        if text_aligned is None or image_aligned is None:
            text_aligned, image_aligned, similarity_pred = self.similarity_module(text_raw, image_raw)
        else:
            similarity_pred = None
        
        # Detection module
        detection_logits, attention_weights, ambiguity_weights = self.detection_module(
            text_raw, image_raw, text_aligned, image_aligned
        )
        
        outputs = {
            'logits': detection_logits,
            'attention_weights': attention_weights,
            'ambiguity_weights': ambiguity_weights
        }
        
        if similarity_pred is not None:
            outputs['similarity_pred'] = similarity_pred
        
        if return_all:
            outputs.update({
                'text_features': text_aligned,
                'image_features': image_aligned,
                'text_raw': text_raw,
                'image_raw': image_raw
            })
        
        return outputs
    
    def compute_contrastive_loss(self, text_features: torch.Tensor, 
                               image_features: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between text and image features."""
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
    
    def compute_total_loss(self, text_raw: torch.Tensor, image_raw: torch.Tensor,
                          labels: torch.Tensor, 
                          similarity_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute total loss including all components."""
        outputs = self.forward(text_raw, image_raw, return_all=True)
        
        # Classification loss
        classification_loss = F.cross_entropy(outputs['logits'], labels)
        
        # Contrastive loss
        contrastive_loss = self.compute_contrastive_loss(
            outputs['text_features'], 
            outputs['image_features']
        )
        
        # Similarity loss (if labels provided)
        similarity_loss = torch.tensor(0.0, device=labels.device)
        if similarity_labels is not None and 'similarity_pred' in outputs:
            similarity_loss = F.cross_entropy(outputs['similarity_pred'], similarity_labels)
        
        # Total loss
        total_loss = (
            self.classification_weight * classification_loss +
            self.contrastive_weight * contrastive_loss +
            self.similarity_weight * similarity_loss
        )
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'contrastive_loss': contrastive_loss,
            'similarity_loss': similarity_loss
        }
    
    def predict(self, text_raw: torch.Tensor, image_raw: torch.Tensor) -> torch.Tensor:
        """Make predictions for fake news detection."""
        with torch.no_grad():
            outputs = self.forward(text_raw, image_raw)
            predictions = F.softmax(outputs['logits'], dim=-1)
        return predictions
