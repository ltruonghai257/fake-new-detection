"""
AnchoredCOOLANT: COOLANT + ViFactCheck PhoBERT semantic anchor injection.

Direction 1 — Semantic Anchor Injection:
  Inject Stage 03.9 PhoBERT CLS [B,768] into COOLANT's DetectionModule as a
  semantic anchor. The anchor is projected to anchor_proj_dim (default 64),
  concatenated to final_corre before classifier_corre.
  New feature_dim = base_feature_dim + anchor_proj_dim = 96 + 64 = 160.

Direction 2 — NEI-Ambiguity Coupling:
  Auxiliary loss that pushes COOLANT's ambiguity scalar (skl) high when
  Stage 03.9 predicts NEI (class 2). Use nei_ambiguity_loss() alongside
  the standard detection loss during training.
"""

from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coolant_official import DetectionModule as OfficialDetectionModule
from .resnet_coolant import PatchedCOOLANT, _apply_all_patches


class AnchoredDetectionModule(OfficialDetectionModule):
    """
    DetectionModule extended with a semantic anchor (PhoBERT CLS) injection.

    The anchor is projected to anchor_proj_dim and concatenated to final_corre
    (text_final ++ img_final ++ corre_final) before the final classifier.
    classifier_corre input: feature_dim + anchor_proj_dim.

    If phobert_cls is None at inference time, the anchor slot is zero-filled so
    the model still runs without Stage 03.9.
    """

    def __init__(
        self,
        anchor_dim: int = 768,
        anchor_proj_dim: int = 64,
        feature_dim: int = 96,
        h_dim: int = 64,
        text_input_dim: int = 200,
        image_input_dim: int = 512,
    ):
        # parent builds classifier_corre for (feature_dim + anchor_proj_dim) input
        super().__init__(
            feature_dim=feature_dim + anchor_proj_dim,
            h_dim=h_dim,
            text_input_dim=text_input_dim,
            image_input_dim=image_input_dim,
        )
        self.anchor_proj_dim = anchor_proj_dim
        self.anchor_proj = nn.Sequential(
            nn.Linear(anchor_dim, anchor_proj_dim),
            nn.BatchNorm1d(anchor_proj_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        text_raw: torch.Tensor,
        image_raw: torch.Tensor,
        text: torch.Tensor,
        image: torch.Tensor,
        phobert_cls: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Shared encoding
        text_prime, image_prime = self.encoding(text_raw, image_raw)

        # SE features and unimodal representations
        text_se, image_se = self.uni_se(text_prime, image_prime)
        text_prime, image_prime = self.uni_repre(text_prime, image_prime)

        # Cross-modal correlation using CLIP-aligned features
        correlation = self.cross_module(text, image)

        # SE attention weights
        attention_score = self.senet(text_se, image_se, correlation)

        # Attention-weighted combination
        text_final = text_prime * attention_score[:, 0].unsqueeze(1)
        img_final = image_prime * attention_score[:, 1].unsqueeze(1)
        corre_final = correlation * attention_score[:, 2].unsqueeze(1)

        final_corre = torch.cat([text_final, img_final, corre_final], dim=1)  # [B, 96]

        # Semantic anchor injection
        if phobert_cls is not None:
            anchor = self.anchor_proj(phobert_cls)  # [B, anchor_proj_dim]
        else:
            anchor = torch.zeros(
                final_corre.size(0), self.anchor_proj_dim, device=final_corre.device
            )
        final_corre = torch.cat([final_corre, anchor], dim=1)  # [B, 160]

        pre_label = self.classifier_corre(final_corre)

        # Ambiguity learning
        skl = self.ambiguity_module(text, image)
        weight_uni = (1 - skl).unsqueeze(1)
        weight_corre = skl.unsqueeze(1)
        skl_score = torch.cat([weight_uni, weight_uni, weight_corre], dim=1)

        return pre_label, attention_score, skl_score


class AnchoredCOOLANT(PatchedCOOLANT):
    """
    COOLANT + ViFactCheck semantic anchor.

    Extends PatchedCOOLANT by replacing DetectionModule with AnchoredDetectionModule.
    All PatchedCOOLANT patches (ResNet/PhoBERT dimension adaptation) still apply
    because AnchoredDetectionModule inherits from OfficialDetectionModule and exposes
    the same .encoding and .ambiguity_module.encoding attributes.

    forward() accepts an optional phobert_cls [B,768] kwarg.
    When absent (e.g. pure inference), the anchor slot is zero-filled.

    Config extras (on top of standard COOLANT config):
        anchor_dim       int  PhoBERT CLS dimension (default 768)
        anchor_proj_dim  int  projected anchor dimension (default 64)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            config
        )  # builds full PatchedCOOLANT with standard detection_module

        # Replace detection_module with anchor-aware variant
        anchor_dim = config.get("anchor_dim", 768)
        anchor_proj_dim = config.get("anchor_proj_dim", 64)
        self.detection_module = AnchoredDetectionModule(
            anchor_dim=anchor_dim,
            anchor_proj_dim=anchor_proj_dim,
            feature_dim=config.get("feature_dim", 96),
            h_dim=config.get("h_dim", 64),
            text_input_dim=config.get("text_input_dim", 200),
            image_input_dim=config.get("image_input_dim", 512),
        )

    @classmethod
    def from_config(cls, model_cfg: dict, device: str = "cpu") -> "AnchoredCOOLANT":
        """
        Build AnchoredCOOLANT with all dimension patches applied.

        Same interface as PatchedCOOLANT.from_config but forwards anchor_dim /
        anchor_proj_dim through to the constructor.
        """
        _PATCH_KEYS = {"image_dim", "text_embed_dim", "text_dim", "variant"}
        inner_cfg = {k: v for k, v in model_cfg.items() if k not in _PATCH_KEYS}
        model = cls(inner_cfg)
        _apply_all_patches(
            model,
            image_dim=model_cfg["image_dim"],
            text_dim=model_cfg.get("text_embed_dim", model_cfg.get("text_dim", 768)),
            dropout=model_cfg.get("dropout", 0.1),
        )
        return model.to(device)

    def forward(  # type: ignore[override]
        self,
        text_raw: torch.Tensor,
        image_raw: torch.Tensor,
        phobert_cls: Optional[torch.Tensor] = None,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            text_raw:    PhoBERT token embeddings [B, text_seq_len, text_embed_dim]
            image_raw:   ResNet50 features [B, image_dim]
            phobert_cls: PhoBERT [CLS] from Stage 03.9 [B, 768] (optional)
            return_all:  include intermediate features in output dict

        Returns dict with:
            detection_logits  [B, 2]
            attention_weights [B, 3]
            ambiguity_weights [B, 3]
            similarity_pred   [B, 2]
            text_aligned_clip [B, clip_embed_dim]
            image_aligned_clip[B, clip_embed_dim]
        """
        # Task 1: similarity + CLIP
        text_aligned_sim, image_aligned_sim, similarity_pred = self.similarity_module(
            text_raw, image_raw
        )
        image_aligned_clip, text_aligned_clip = self.clip_module(image_raw, text_raw)

        # Task 2: detection with semantic anchor
        detection_logits, attention_weights, ambiguity_weights = self.detection_module(
            text_raw,
            image_raw,
            text_aligned_clip,
            image_aligned_clip,
            phobert_cls=phobert_cls,
        )

        outputs = {
            "similarity_pred": similarity_pred,
            "detection_logits": detection_logits,
            "attention_weights": attention_weights,
            "ambiguity_weights": ambiguity_weights,
            "text_aligned_clip": text_aligned_clip,
            "image_aligned_clip": image_aligned_clip,
        }

        if return_all:
            outputs.update(
                {
                    "text_aligned_sim": text_aligned_sim,
                    "image_aligned_sim": image_aligned_sim,
                    "text_raw": text_raw,
                    "image_raw": image_raw,
                }
            )

        return outputs

    def compute_detection_loss(
        self,
        detection_logits: torch.Tensor,
        labels: torch.Tensor,
        attention_weights: torch.Tensor,
        ambiguity_weights: torch.Tensor,
        factcheck_logits: Optional[torch.Tensor] = None,
        nei_weight: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Detection loss with optional Direction-2 NEI-ambiguity coupling.

        Args:
            factcheck_logits: Stage 03.9 raw logits [B, 3] (support/refute/NEI).
                              When provided, adds nei_ambiguity_loss.
            nei_weight:       Weight for the NEI-ambiguity auxiliary loss.
        """
        classification_loss = F.cross_entropy(detection_logits, labels)

        loss_func_skl = nn.KLDivLoss(reduction="batchmean")
        ambiguity_loss = loss_func_skl(
            F.log_softmax(attention_weights, dim=1),
            F.softmax(ambiguity_weights, dim=1),
        )

        total = classification_loss + 0.5 * ambiguity_loss
        result = {
            "classification_loss": classification_loss,
            "ambiguity_loss": ambiguity_loss,
        }

        if factcheck_logits is not None:
            nei_loss = nei_ambiguity_loss(ambiguity_weights, factcheck_logits)
            total = total + nei_weight * nei_loss
            result["nei_ambiguity_loss"] = nei_loss

        result["detection_loss"] = total
        return result

    def predict(
        self,
        text_raw: torch.Tensor,
        image_raw: torch.Tensor,
        phobert_cls: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.forward(text_raw, image_raw, phobert_cls=phobert_cls)
            return F.softmax(outputs["detection_logits"], dim=-1)


def nei_ambiguity_loss(
    ambiguity_weights: torch.Tensor,
    factcheck_logits: torch.Tensor,
    nei_class: int = 2,
) -> torch.Tensor:
    """
    Direction 2 auxiliary loss: push COOLANT ambiguity high for NEI examples.

    ambiguity_weights [B,3] = [1-skl, 1-skl, skl].
    skl ([:,2]) is the correlation weight — high skl means cross-modal ambiguity.
    For NEI (not enough information), high skl is semantically correct.

    Loss = mean(P(NEI) * (1 - skl))
         = mean(P(NEI) * ambiguity_weights[:,0])  [since ambiguity_weights[:,0] = 1-skl]

    Args:
        ambiguity_weights: [B, 3] from DetectionModule.forward skl_score
        factcheck_logits:  [B, 3] raw logits from ViFactCheck (support=0/refute=1/NEI=2)
        nei_class:         index of NEI class (default 2)
    """
    nei_probs = F.softmax(factcheck_logits, dim=1)[:, nei_class]  # [B]
    skl = ambiguity_weights[:, 2]  # [B]  == skl from AmbiguityLearning
    return (nei_probs * (1.0 - skl)).mean()


# Convenience alias
ResNetAnchoredCOOLANT = AnchoredCOOLANT
