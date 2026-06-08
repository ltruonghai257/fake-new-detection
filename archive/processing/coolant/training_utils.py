"""
COOLANT training utilities for matched/unmatched pair construction.

Following the COOLANT paper (arXiv:2302.14057):
- Matched pair (caption_i, image_i) = Real (label 0)
- Unmatched pair (caption_i, image_j where i!=j) = Fake (label 1)
"""

import torch
import torch.nn.functional as F
import random


def make_coolant_pairs(caption, image, shift=3):
    """
    Construct matched and unmatched pairs for COOLANT similarity learning.

    Args:
        caption: [B, embed_dim, seq_len] caption features
        image: [B, 2048] image features
        shift: How many positions to roll for creating unmatched pairs

    Returns:
        caption_anchor: [B, embed_dim, seq_len] (same as input)
        image_matched: [B, 2048] (same image, Real pair)
        image_unmatched: [B, 2048] (different image, Fake pair)
    """
    caption_anchor = caption.clone()
    image_matched = image.clone()
    image_unmatched = image.roll(shifts=shift, dims=0)

    return caption_anchor, image_matched, image_unmatched


def make_detection_batch(caption, image, shift=3):
    """
    Construct a balanced detection batch with matched + unmatched pairs.

    Returns:
        combined_caption: [2B, embed_dim, seq_len]
        combined_image: [2B, 2048]
        labels: [2B] (0=Real/matched, 1=Fake/unmatched)
    """
    B = caption.size(0)
    device = caption.device

    # Matched (Real) pairs
    matched_caption = caption
    matched_image = image

    # Unmatched (Fake) pairs — roll images
    unmatched_caption = caption.clone()
    unmatched_image = image.roll(shifts=shift, dims=0)

    # Combine
    combined_caption = torch.cat([matched_caption, unmatched_caption], dim=0)
    combined_image = torch.cat([matched_image, unmatched_image], dim=0)
    labels = torch.cat([
        torch.zeros(B, dtype=torch.long, device=device),  # Real
        torch.ones(B, dtype=torch.long, device=device),   # Fake
    ])

    # Shuffle to avoid the model learning position
    perm = torch.randperm(2 * B, device=device)
    combined_caption = combined_caption[perm]
    combined_image = combined_image[perm]
    labels = labels[perm]

    return combined_caption, combined_image, labels


def soft_cross_entropy(logits, soft_target):
    """Soft cross-entropy loss for CLIP distillation."""
    return -(soft_target * F.log_softmax(logits, 1)).sum() / logits.size(0)
