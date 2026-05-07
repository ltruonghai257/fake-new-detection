"""
PatchedCOOLANT: Adapt COOLANT for arbitrary image and text feature dimensions.

This module provides a wrapper class that adapts the COOLANT_Official model
to work with pre-extracted features from any image encoder (ResNet50, CLIP ViT,
SigLIP, etc.) and any text encoder (PhoBERT, ViSoBERT, etc.).

The patch_* functions below handle dimension adaptation at init time.
"""

import torch
import torch.nn as nn
from .coolant_official import COOLANT_Official


class PatchedCOOLANT(COOLANT_Official):
    """
    COOLANT wrapper that patches layer dimensions for arbitrary feature extractors.

    The base COOLANT implementation expects 512-dim image and 200-dim text features.
    This wrapper, combined with the patch_* functions, adapts the model to accept
    any feature dimensions (e.g., 2048 for ResNet50, 1024 for CLIP ViT-L/14,
    768 for SigLIP or PhoBERT).

    Args:
        config: Configuration dictionary with model hyperparameters
    """

    def __init__(self, config):
        super().__init__(config)

    def encode_text(self, text):
        """
        Encode text features using the similarity module.

        Args:
            text: Text features [B, 768, 512] (BERT embeddings)

        Returns:
            Encoded text features
        """
        # Create dummy image for encoding (will be ignored by shared_text_encoding)
        dummy_img = torch.zeros(text.size(0), 2048, device=text.device)
        t, _ = self.similarity_module.encoding(text, dummy_img)
        return t

    def encode_image(self, image):
        """
        Encode image features using the similarity module.

        Args:
            image: Image features [B, 2048] (ResNet50 features)

        Returns:
            Encoded image features
        """
        # Create dummy text for encoding (will be ignored by shared_image)
        dummy_txt = torch.zeros(
            image.size(0), 768, 512, device=image.device
        )
        _, i = self.similarity_module.encoding(dummy_txt, image)
        return i

    def fuse_modalities(self, text_f, image_f):
        """
        Fuse text and image features.

        Args:
            text_f: Encoded text features
            image_f: Encoded image features

        Returns:
            Concatenated features
        """
        return torch.cat([text_f, image_f], dim=-1)


def patch_encoding(enc, image_dim=2048):
    """
    Patch the shared_image layer to accept 2048-dim input instead of 512-dim.

    Args:
        enc: Encoding module to patch
        image_dim: Target image feature dimension (default: 2048 for ResNet50)
    """
    layers, done = [], False
    for l in enc.shared_image:
        if isinstance(l, nn.Linear) and not done:
            # Replace first Linear layer with correct input dimension
            layers.append(nn.Linear(image_dim, l.out_features))
            done = True
        else:
            layers.append(l)
    enc.shared_image = nn.Sequential(*layers)


def patch_clip_projection(clip_module, target_dim, is_image=True):
    """
    Patch CLIP projection layer to accept different input dimensions.

    Args:
        clip_module: CLIP module to patch
        target_dim: Target input dimension
        is_image: True for image projection, False for text projection
    """
    proj = clip_module.image_projection if is_image else clip_module.text_projection

    layers, done = [], False
    for l in proj:
        if isinstance(l, nn.Linear) and not done:
            # Replace first Linear layer with correct input dimension
            layers.append(nn.Linear(target_dim, l.out_features))
            done = True
        else:
            layers.append(l)

    new_proj = nn.Sequential(*layers)
    if is_image:
        clip_module.image_projection = new_proj
    else:
        clip_module.text_projection = new_proj


def patch_cnn_with_dropout(m, input_dim, dropout=0.1):
    """
    Recreate FastCNN with specified dropout rate.

    Args:
        m: FastCNN module to patch
        input_dim: Input dimension (e.g., 768 for BERT)
        dropout: Dropout rate (default: 0.1)
    """
    channel = 32
    kernel_size = (1, 2, 4, 8)

    new_cnn = nn.ModuleList()
    for kernel in kernel_size:
        new_cnn.append(
            nn.Sequential(
                nn.Conv1d(input_dim, channel, kernel_size=kernel),
                nn.BatchNorm1d(channel),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.AdaptiveMaxPool1d(1),
            )
        )
    m.fast_cnn = new_cnn

    # Patch forward method to handle [B, 768, 512] input format
    def patched_forward(self, x):
        """Forward without permute - data already [B, 768, 512]."""
        x_out = []
        for module in self.fast_cnn:
            x_out.append(module(x).squeeze(-1))
        return torch.cat(x_out, 1)

    import types
    m.forward = types.MethodType(patched_forward, m)


# Backward-compatible alias — old notebooks import this name
ResNetCOOLANT = PatchedCOOLANT
