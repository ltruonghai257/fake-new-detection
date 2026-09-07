#!/usr/bin/env python3
"""
COOLANT Official Implementation

Follows the official COOLANT paper (ACM MM '23, arXiv:2302.14057):
  §3.2.1  Consistency Learning  → L_ITM  (CosineEmbeddingLoss, margin=0.2)
  §3.2.2  Contrastive Learning  → L_ITC  (symmetric InfoNCE)
  §3.2.4  Semantic Matching     → L_SEM  (soft distillation ITM→ITC, Eq. 6)
  §3.4    Cross-modal Detection → L_DET  (L_CE + 0.5·L_KL)

  Total: L_CL = L_ITC + λ·L_SEM   (Task 1, use_itc=True)
         L     = L_CL + L_ITM + L_DET

Module→paper notation:
  SimilarityModule outputs  →  e_s^t, e_s^v  (shared embeddings for ITM & soft targets)
  CLIPModule outputs        →  m^t,   m^v    (aligned representations for CrossModule)

Key modification from paper:
  GatedMLP (SwiGLU) replaces all standard Linear→ReLU throughout.
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


# ── SwiGLU Gated MLP ───────────────────────────────────────────────────────
class GatedMLP(nn.Module):
    """SwiGLU-activated MLP: silu(gate(x)) * up(x) → down(x).

    Replaces Linear→ReLU→Linear with a gated variant used in LLaMA/PaLM.
    Same parameter count: in_dim * hidden * 3 + hidden * out_dim
    vs standard: in_dim * hidden + hidden + hidden * out_dim + out_dim
    """

    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0
    ):
        super().__init__()
        self.gate_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.dropout(x)
        return self.down_proj(x)


class FastCNNEmbedFirst(FastCNN):
    """FastCNN variant for data already in (B, embed_dim, seq_len) format.

    CoolantPairDataset returns caption as (B, embed, seq) — the base
    FastCNN.forward would permute it back to (B, seq, embed) and break
    Conv1d channel order. This subclass skips the permute.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = []
        for module in self.fast_cnn:
            x_out.append(module(x).squeeze(-1))
        return torch.cat(x_out, 1)


class EncodingPart(nn.Module):
    """Shared encoding module for text and image features."""

    def __init__(
        self,
        cnn_channel: int = 32,
        cnn_kernel_size: Tuple[int, ...] = (1, 2, 4, 8),
        shared_image_dim: int = 128,
        shared_text_dim: int = 128,
        text_input_dim: int = 200,
        image_input_dim: int = 512,
    ):
        super(EncodingPart, self).__init__()

        # Text encoding — input is (B, embed_dim, seq_len), no permute needed
        self.shared_text_encoding = FastCNNEmbedFirst(
            input_dim=text_input_dim, channel=cnn_channel, kernel_size=cnn_kernel_size
        )
        self.shared_text_linear = nn.Sequential(
            nn.Linear(
                128, 64
            ),  # Note: Official uses 128, not cnn_channel * len(cnn_kernel_size)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),  # Official uses default dropout
            nn.Linear(64, shared_text_dim),
            nn.BatchNorm1d(shared_text_dim),
            nn.ReLU(),
        )

        # Image encoding
        self.shared_image = nn.Sequential(
            nn.Linear(image_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_image_dim),
            nn.BatchNorm1d(shared_image_dim),
            nn.ReLU(),
        )

    def forward(
        self, text: torch.Tensor, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_encoding = self.shared_text_encoding(text)
        text_shared = self.shared_text_linear(text_encoding)
        image_shared = self.shared_image(image)
        return text_shared, image_shared


class SimilarityModule(nn.Module):
    """Module for computing cross-modal similarity (Task 1)."""

    def __init__(
        self,
        shared_dim: int = 128,
        sim_dim: int = 64,
        text_input_dim: int = 200,
        image_input_dim: int = 512,
    ):
        super(SimilarityModule, self).__init__()

        self.encoding = EncodingPart(
            text_input_dim=text_input_dim, image_input_dim=image_input_dim
        )

        # Alignment networks (SwiGLU)
        self.text_aligner = nn.Sequential(
            GatedMLP(shared_dim, shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
        )

        self.image_aligner = nn.Sequential(
            GatedMLP(shared_dim, shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
        )

        # Similarity classifier (SwiGLU)
        self.sim_classifier_dim = sim_dim * 2
        self.sim_classifier = nn.Sequential(
            nn.BatchNorm1d(self.sim_classifier_dim),
            GatedMLP(self.sim_classifier_dim, 64, 2),
        )

    def forward(
        self, text: torch.Tensor, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_encoding, image_encoding = self.encoding(text, image)
        text_aligned = self.text_aligner(text_encoding)
        image_aligned = self.image_aligner(image_encoding)

        # Concatenate for similarity prediction
        sim_feature = torch.cat([text_aligned, image_aligned], 1)
        pred_similarity = self.sim_classifier(sim_feature)

        return text_aligned, image_aligned, pred_similarity


class CLIPModule(nn.Module):
    """Optional CLIP-style InfoNCE contrastive module (use_itc=True)."""

    def __init__(
        self, embed_dim: int = 64, text_input_dim: int = 200, image_input_dim: int = 512
    ):
        super(CLIPModule, self).__init__()
        self.text_projection = GatedMLP(text_input_dim, 256, embed_dim)
        self.image_projection = GatedMLP(image_input_dim, 256, embed_dim)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(
        self, text: torch.Tensor, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if text.dim() == 3:
            text = text.mean(dim=2)
        m_t = F.normalize(self.text_projection(text), dim=-1)
        m_v = F.normalize(self.image_projection(image), dim=-1)
        return m_t, m_v


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
        mu, sigma = params[:, : self.z_dim], params[:, self.z_dim :]
        sigma = softplus(sigma) + 1e-7
        return Independent(Normal(loc=mu, scale=sigma), 1)


class AmbiguityLearning(nn.Module):
    """Module for learning cross-modal ambiguity using variational inference."""

    def __init__(
        self, input_dim: int = 64, text_input_dim: int = 200, image_input_dim: int = 512
    ):
        super(AmbiguityLearning, self).__init__()
        self.encoding = EncodingPart(
            text_input_dim=text_input_dim, image_input_dim=image_input_dim
        )  # Official includes encoding here
        self.encoder_text = Encoder(input_dim)
        self.encoder_image = Encoder(input_dim)

    def forward(
        self, text_encoding: torch.Tensor, image_encoding: torch.Tensor
    ) -> torch.Tensor:
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
            GatedMLP(shared_dim, shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
        )

        self.image_uni = nn.Sequential(
            GatedMLP(shared_dim, shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
        )

    def forward(
        self, text_encoding: torch.Tensor, image_encoding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_prime = self.text_uni(text_encoding)
        image_prime = self.image_uni(image_encoding)
        return text_prime, image_prime


class CrossModule4Batch(nn.Module):
    """Cross-modal correlation module."""

    def __init__(
        self, text_in_dim: int = 64, image_in_dim: int = 64, corre_out_dim: int = 64
    ):
        super(CrossModule4Batch, self).__init__()

        self.softmax = nn.Softmax(-1)
        self.corre_dim = 64
        self.pooling = nn.AdaptiveMaxPool1d(1)

        self.c_specific_2 = nn.Sequential(
            GatedMLP(self.corre_dim, corre_out_dim, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim),
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

    def __init__(
        self,
        feature_dim: int = 64 + 16 + 16,
        h_dim: int = 64,
        text_input_dim: int = 200,
        image_input_dim: int = 512,
    ):
        super(DetectionModule, self).__init__()

        self.encoding = EncodingPart(
            text_input_dim=text_input_dim, image_input_dim=image_input_dim
        )
        self.ambiguity_module = AmbiguityLearning(
            text_input_dim=text_input_dim, image_input_dim=image_input_dim
        )
        self.uni_repre = UnimodalDetection()
        self.uni_se = UnimodalDetection(prime_dim=64)
        self.cross_module = CrossModule4Batch()

        # SE attention module (using official SENet)
        self.senet = SEAttentionModule(text_dim=64, image_dim=64, correlation_dim=64)

        # Final classifier (SwiGLU)
        self.classifier_corre = nn.Sequential(
            GatedMLP(feature_dim, h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            GatedMLP(h_dim, h_dim, 2),
        )

    def forward(
        self,
        text_raw: torch.Tensor,
        image_raw: torch.Tensor,
        text: torch.Tensor,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Get shared representations from raw inputs
        text_prime, image_prime = self.encoding(text_raw, image_raw)

        # Get SE features and unimodal representations
        text_se, image_se = self.uni_se(text_prime, image_prime)
        text_prime, image_prime = self.uni_repre(text_prime, image_prime)

        # Cross-modal correlation (uses similarity-aligned features m^t, m^v)
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

        # Ambiguity learning (uses similarity-aligned features m^t, m^v)
        skl = self.ambiguity_module(text, image)
        weight_uni = (1 - skl).unsqueeze(1)
        weight_corre = skl.unsqueeze(1)
        skl_score = torch.cat([weight_uni, weight_uni, weight_corre], 1)

        return pre_label, attention_score, skl_score


class COOLANT_Official(MultimodalModel):
    """
    Official COOLANT implementation (ACM MM '23).

    Modules:
      SimilarityModule  — consistency learning; produces e_s^t, e_s^v (§3.2.1)
      CLIPModule        — contrastive learning;  produces m^t,   m^v   (§3.2.2, use_itc=True)
      DetectionModule   — detection + ambiguity;  consumes m^t,  m^v   (§3.3–3.4)

    Loss notation (paper §3.2):
      L_ITM  — CosineEmbeddingLoss on shared embeddings  (SimilarityModule)
      L_ITC  — symmetric InfoNCE                         (CLIPModule, use_itc=True)
      L_SEM  — soft distillation ITM→ITC (Eq. 6)        (use_itc=True)
      L_DET  — L_CE + 0.5·L_KL                          (DetectionModule)
      L_CL   — L_ITC + λ·L_SEM                          (total Task-1 contrastive)

    Modification vs paper: GatedMLP (SwiGLU) replaces all standard MLPs.
    """

    def __init__(self, config: Dict[str, Any]):
        super(COOLANT_Official, self).__init__(config)

        # Get input dimensions from config
        self.text_input_dim = config.get("text_input_dim", 200)
        self.image_input_dim = config.get("image_input_dim", 512)
        self.text_seq_len = config.get("text_seq_len", 30)
        text_input_dim = self.text_input_dim
        image_input_dim = self.image_input_dim

        # Model components
        self.similarity_module = SimilarityModule(
            shared_dim=config.get("shared_dim", 128),
            sim_dim=config.get("sim_dim", 64),
            text_input_dim=text_input_dim,
            image_input_dim=image_input_dim,
        )

        # Optional InfoNCE contrastive module (use_itc=True to enable)
        self.use_itc = config.get("use_itc", False)
        if self.use_itc:
            self.clip_module = CLIPModule(
                embed_dim=config.get("clip_embed_dim", 64),
                text_input_dim=text_input_dim,
                image_input_dim=image_input_dim,
            )

        self.detection_module = DetectionModule(
            feature_dim=config.get("feature_dim", 96),  # 64 + 16 + 16
            h_dim=config.get("h_dim", 64),
            text_input_dim=text_input_dim,
            image_input_dim=image_input_dim,
        )

        # Loss weights
        self.classification_weight = config.get("classification_weight", 1.0)
        self.itm_weight = config.get("itm_weight", 0.5)  # weight for L_ITM
        self.sem_weight = config.get("sem_weight", 1.0)  # λ in L_CL = L_ITC + λ·L_SEM

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text to shared space via SimilarityModule encoder."""
        dummy_image = torch.zeros(
            text.size(0), self.image_input_dim, device=text.device
        )
        text_shared, _ = self.similarity_module.encoding(text, dummy_image)
        return text_shared

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to shared space via SimilarityModule encoder."""
        dummy_text = torch.zeros(
            image.size(0), self.text_input_dim, self.text_seq_len, device=image.device
        )
        _, image_shared = self.similarity_module.encoding(dummy_text, image)
        return image_shared

    def fuse_modalities(
        self, text_features: torch.Tensor, image_features: torch.Tensor
    ) -> torch.Tensor:
        """Fuse via concatenation."""
        return torch.cat([text_features, image_features], dim=-1)

    def forward(
        self, text_raw: torch.Tensor, image_raw: torch.Tensor, return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass following official COOLANT architecture (§3).

        Args:
            text_raw:   Raw text  features (B, embed_dim, seq_len) for Conv1d
            image_raw:  Raw image features (B, image_input_dim)
            return_all: Include intermediate shared embeddings e_s_t, e_s_v in output.

        Returns dict keys (always):
            similarity_pred     — ITM classifier logits from SimilarityModule
            detection_logits    — fake/real classification logits
            attention_weights   — SE attention over [text, image, cross] features
            ambiguity_weights   — VAE-based ambiguity scores (for L_KL guidance)
            m_t, m_v            — aligned representations fed to CrossModule
                                  (CLIPModule output if use_itc, else SimilarityModule output)

        Additional keys when return_all=True:
            e_s_t, e_s_v        — shared embeddings from SimilarityModule
                                  (used to build soft targets for L_SEM, §3.2.3)
        """
        # §3.2.1 Consistency Learning — shared embeddings e_s^t, e_s^v
        e_s_t, e_s_v, similarity_pred = self.similarity_module(text_raw, image_raw)

        # §3.2.2 Contrastive Learning — aligned m^t, m^v for CrossModule
        if self.use_itc:
            m_t, m_v = self.clip_module(text_raw, image_raw)
        else:
            # Ablation / simplified: use shared embeddings directly
            m_t, m_v = e_s_t, e_s_v

        # §3.3–3.4 Cross-modal Fusion + Aggregation + Detection
        detection_logits, attention_weights, ambiguity_weights = self.detection_module(
            text_raw, image_raw, m_t, m_v
        )

        outputs = {
            "similarity_pred": similarity_pred,
            "detection_logits": detection_logits,
            "attention_weights": attention_weights,
            "ambiguity_weights": ambiguity_weights,
            "m_t": m_t,
            "m_v": m_v,
        }
        if return_all:
            outputs.update({"e_s_t": e_s_t, "e_s_v": e_s_v})

        return outputs

    # ── Loss helpers (paper §3.2 notation) ─────────────────────────────────

    def compute_loss_itm(
        self,
        e_s_t: torch.Tensor,
        e_s_v: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """L_ITM (§3.2.1): CosineEmbeddingLoss on shared embeddings.

        labels: +1 for matched pairs, -1 for unmatched pairs.
        """
        return F.cosine_embedding_loss(e_s_t, e_s_v, labels, margin=0.2)

    def compute_loss_itc(self, m_t: torch.Tensor, m_v: torch.Tensor) -> torch.Tensor:
        """L_ITC (§3.2.2): symmetric InfoNCE (use_itc=True only).

        Hard one-hot targets: diagonal = positive pair.
        """
        temp = torch.exp(self.clip_module.temperature)
        logits = torch.matmul(m_v, m_t.T) * temp
        labels = torch.arange(m_t.size(0), device=m_t.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    def compute_loss_sem(
        self,
        m_t: torch.Tensor,
        m_v: torch.Tensor,
        e_s_t: torch.Tensor,
        e_s_v: torch.Tensor,
    ) -> torch.Tensor:
        """L_SEM (§3.2.4, Eq. 6): soft distillation from SimilarityModule → CLIPModule.

        Builds soft targets S from shared embeddings (e_s_t, e_s_v),
        then computes cross-entropy against ITC predictions (m_t, m_v).
        use_itc=True required.
        """
        temp = torch.exp(self.clip_module.temperature)
        # Soft targets from SimilarityModule (detached — teacher signal)
        with torch.no_grad():
            soft_v2t = F.softmax(torch.matmul(e_s_v, e_s_t.T) * temp, dim=1)
            soft_t2v = F.softmax(torch.matmul(e_s_t, e_s_v.T) * temp, dim=1)
        # Log-predictions from CLIPModule (student)
        log_v2t = F.log_softmax(torch.matmul(m_v, m_t.T) * temp, dim=1)
        log_t2v = F.log_softmax(torch.matmul(m_t, m_v.T) * temp, dim=1)
        l_v2t = -(soft_v2t * log_v2t).sum(dim=1).mean()
        l_t2v = -(soft_t2v * log_t2v).sum(dim=1).mean()
        return (l_v2t + l_t2v) / 2

    def compute_loss_det(
        self,
        detection_logits: torch.Tensor,
        labels: torch.Tensor,
        attention_weights: torch.Tensor,
        ambiguity_weights: torch.Tensor,
    ) -> torch.Tensor:
        """L_DET = L_CE + 0.5 * L_KL (detection + ambiguity, Task 2)."""
        # Classification loss
        l_ce = F.cross_entropy(detection_logits, labels)

        # Ambiguity loss (symmetric KL divergence)
        loss_func_skl = torch.nn.KLDivLoss(reduction="batchmean")
        l_kl = loss_func_skl(
            F.log_softmax(attention_weights, dim=1), F.softmax(ambiguity_weights, dim=1)
        )

        return l_ce + 0.5 * l_kl

    def predict(self, text_raw: torch.Tensor, image_raw: torch.Tensor) -> torch.Tensor:
        """Make predictions for fake news detection."""
        with torch.no_grad():
            outputs = self.forward(text_raw, image_raw)
            predictions = F.softmax(outputs["detection_logits"], dim=-1)
        return predictions
