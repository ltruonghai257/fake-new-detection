#!/usr/bin/env python3
"""
COOLANT Official Implementation

This implementation follows the official COOLANT repository:
https://github.com/wishever/COOLANT/tree/main/twitter

Key differences from previous implementation:
1. Added CLIP module for additional contrastive learning
2. Separate training tasks with individual optimizers
3. DetectionModule uses CLIP-aligned features, not similarity-aligned
4. Proper loss computation matching official repository
"""

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
            nn.Linear(128, 64),  # Note: Official uses 128, not cnn_channel * len(cnn_kernel_size)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),  # Official uses default dropout
            nn.Linear(64, shared_text_dim),
            nn.BatchNorm1d(shared_text_dim),
            nn.ReLU()
        )
        
        # Image encoding
        self.shared_image = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
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
    """Module for computing cross-modal similarity (Task 1)."""
    
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


class CLIP(nn.Module):
    """CLIP module for additional contrastive learning (Task 1)."""
    
    def __init__(self, embed_dim: int = 64):
        super(CLIP, self).__init__()
        self.embed_dim = embed_dim
        
        # Text projection
        self.text_projection = nn.Sequential(
            nn.Linear(768, 256),  # Assuming text features are 768-dim
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
        
        # Image projection  
        self.image_projection = nn.Sequential(
            nn.Linear(512, 256),  # Image features are 512-dim
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
        
        # Temperature parameter (learnable)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
    def forward(self, image: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project features to shared embedding space
        text_embed = self.text_projection(text)  # (B, embed_dim)
        image_embed = self.image_projection(image)  # (B, embed_dim)
        
        # Normalize embeddings
        text_embed = F.normalize(text_embed, dim=-1)
        image_embed = F.normalize(image_embed, dim=-1)
        
        return image_embed, text_embed


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
        self.encoding = EncodingPart()  # Official includes encoding here
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
    """Main detection module combining all components (Task 2)."""
    
    def __init__(self, feature_dim: int = 64+16+16, h_dim: int = 64):
        super(DetectionModule, self).__init__()
        
        self.encoding = EncodingPart()
        self.ambiguity_module = AmbiguityLearning()
        self.uni_repre = UnimodalDetection()
        self.uni_se = UnimodalDetection(prime_dim=64)
        self.cross_module = CrossModule4Batch()
        
        # SE attention module (using official SENet)
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
        
        # Get shared representations from raw inputs
        text_prime, image_prime = self.encoding(text_raw, image_raw)
        
        # Get SE features and unimodal representations
        text_se, image_se = self.uni_se(text_prime, image_prime)
        text_prime, image_prime = self.uni_repre(text_prime, image_prime)
        
        # Cross-modal correlation (uses CLIP-aligned features)
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
        
        # Ambiguity learning (uses CLIP-aligned features)
        skl = self.ambiguity_module(text, image)
        weight_uni = (1 - skl).unsqueeze(1)
        weight_corre = skl.unsqueeze(1)
        skl_score = torch.cat([weight_uni, weight_uni, weight_corre], 1)
        
        return pre_label, attention_score, skl_score


class COOLANT_Official(MultimodalModel):
    """
    Official COOLANT Implementation
    
    This follows the official repository architecture with:
    - SimilarityModule (Task 1: Similarity learning)
    - CLIP module (Task 1: Additional contrastive learning)
    - DetectionModule (Task 2: Detection with ambiguity learning)
    - Separate training tasks and optimizers
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(COOLANT_Official, self).__init__(config)
        
        # Model components
        self.similarity_module = SimilarityModule(
            shared_dim=config.get('shared_dim', 128),
            sim_dim=config.get('sim_dim', 64)
        )
        
        self.clip_module = CLIP(
            embed_dim=config.get('clip_embed_dim', 64)
        )
        
        self.detection_module = DetectionModule(
            feature_dim=config.get('feature_dim', 96),  # 64 + 16 + 16
            h_dim=config.get('h_dim', 64)
        )
        
        # Loss weights
        self.contrastive_weight = config.get('contrastive_weight', 1.0)
        self.classification_weight = config.get('classification_weight', 1.0)
        self.similarity_weight = config.get('similarity_weight', 0.5)
        self.clip_weight = config.get('clip_weight', 0.2)
        
        # Temperature for contrastive learning
        self.temperature = config.get('temperature', 0.07)
    
    def forward(self, text_raw: torch.Tensor, image_raw: torch.Tensor,
                return_all: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass following official repository architecture.
        
        Args:
            text_raw: Raw text features (B, 30, 200)
            image_raw: Raw image features (B, 512)
            return_all: Whether to return all intermediate outputs
            
        Returns:
            Dictionary containing model outputs
        """
        # Task 1: Similarity learning
        text_aligned_sim, image_aligned_sim, similarity_pred = self.similarity_module(text_raw, image_raw)
        
        # Task 1: CLIP contrastive learning
        image_aligned_clip, text_aligned_clip = self.clip_module(image_raw, text_raw)
        
        # Task 2: Detection (uses CLIP-aligned features)
        detection_logits, attention_weights, ambiguity_weights = self.detection_module(
            text_raw, image_raw, text_aligned_clip, image_aligned_clip
        )
        
        outputs = {
            'similarity_pred': similarity_pred,
            'detection_logits': detection_logits,
            'attention_weights': attention_weights,
            'ambiguity_weights': ambiguity_weights,
            'text_aligned_clip': text_aligned_clip,
            'image_aligned_clip': image_aligned_clip
        }
        
        if return_all:
            outputs.update({
                'text_aligned_sim': text_aligned_sim,
                'image_aligned_sim': image_aligned_sim,
                'text_raw': text_raw,
                'image_raw': image_raw
            })
        
        return outputs
    
    def compute_similarity_loss(self, text_aligned: torch.Tensor, 
                               image_aligned: torch.Tensor, 
                               similarity_labels: torch.Tensor) -> torch.Tensor:
        """Compute similarity learning loss."""
        # Cosine embedding loss
        loss_func_similarity = torch.nn.CosineEmbeddingLoss(margin=0.2)
        return loss_func_similarity(text_aligned, image_aligned, similarity_labels)
    
    def compute_clip_loss(self, text_features: torch.Tensor, 
                         image_features: torch.Tensor) -> torch.Tensor:
        """Compute CLIP contrastive loss."""
        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.T) * torch.exp(self.temperature)
        
        # Create labels (positive pairs are on the diagonal)
        batch_size = text_features.size(0)
        labels = torch.arange(batch_size, device=text_features.device)
        
        # Compute contrastive loss
        loss_image_to_text = F.cross_entropy(logits, labels)
        loss_text_to_image = F.cross_entropy(logits.T, labels)
        
        return (loss_image_to_text + loss_text_to_image) / 2
    
    def compute_detection_loss(self, detection_logits: torch.Tensor, 
                               labels: torch.Tensor,
                               attention_weights: torch.Tensor,
                               ambiguity_weights: torch.Tensor) -> torch.Tensor:
        """Compute detection loss with ambiguity learning."""
        # Classification loss
        classification_loss = F.cross_entropy(detection_logits, labels)
        
        # Ambiguity loss (KL divergence)
        loss_func_skl = torch.nn.KLDivLoss(reduction='batchmean')
        ambiguity_loss = loss_func_skl(F.log_softmax(attention_weights, dim=1), 
                                      F.softmax(ambiguity_weights, dim=1))
        
        return classification_loss + 0.5 * ambiguity_loss
    
    def predict(self, text_raw: torch.Tensor, image_raw: torch.Tensor) -> torch.Tensor:
        """Make predictions for fake news detection."""
        with torch.no_grad():
            outputs = self.forward(text_raw, image_raw)
            predictions = F.softmax(outputs['detection_logits'], dim=-1)
        return predictions
