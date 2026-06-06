"""
Shared pipeline configuration for all four pipeline notebooks.

Usage in any notebook:
    from pipeline_config import make_config
    CONFIG = make_config(PROJECT_ROOT)
    # Override only what this notebook changes, e.g.:
    # CONFIG["training"]["max_epochs"] = 10

Notebooks may extend the returned dict freely; this file owns only
the values that are shared across multiple notebooks.
"""

import os
from pathlib import Path


def make_config(project_root: Path, data_root: Path = None) -> dict:
    """
    Return the base CONFIG dict with all shared paths and model dimensions.

    Args:
        project_root: Absolute path to the repo root (contains src/, notebooks/, etc.)
        data_root: Root directory for all data output (processed_data/, training/,
                   data/json, data/jpg, mlruns/).  Defaults to the DATA_ROOT env var,
                   or project_root if neither is set.  Set this to an external drive
                   path to keep large files outside the git repo.
    """
    if data_root is None:
        env_val = os.environ.get("DATA_ROOT")
        data_root = Path(env_val) if env_val else project_root

    return {
        # ── Shared feature dimensions ──────────────────────────────────────
        # These must match the Phase 2 feature extractors and COOLANT Stage 1.
        # Changing them requires re-running ALL pipeline stages from scratch.
        "feature_dims": {
            "image_dim":      2048,   # ResNet50 output
            "text_embed_dim": 768,    # PhoBERT-base-v2 hidden size
            "text_seq_len":   128,    # max_length passed to TextPreprocessor
            "clip_dim":       128,    # COOLANT aligned feature dim (Stage 1 output)
        },

        # ── Shared data paths ──────────────────────────────────────────────
        # All paths below live under data_root so they can be placed on an
        # external drive without touching the git-tracked source tree.
        "paths": {
            "data_root":         data_root,
            "hdf5_dir":          data_root / "processed_data" / "hdf5",
            "json_dir":          data_root / "data" / "json",
            "jpg_dir":           data_root / "data" / "jpg",
            "stage2_features":   data_root / "training" / "stage2_features",
            "stage1_checkpoints": data_root / "training" / "checkpoints_coolant",
            "stage2_checkpoints": data_root / "training" / "checkpoints_stage2",
            "stage2_results":    data_root / "training" / "stage2_results",
            "mlflow_dir":        data_root / "mlruns",
        },

        # ── Shared model architecture ──────────────────────────────────────
        # These are the COOLANT_Official constructor params + patching dims.
        # If you change image_dim / text_embed_dim here, also re-run Phase 2.
        "model": {
            "variant":        "ResNetCOOLANT",
            "image_dim":      2048,
            "text_embed_dim": 768,
            "text_seq_len":   128,
            "shared_dim":     128,
            "sim_dim":        64,
            "clip_embed_dim": 64,
            "feature_dim":    96,
            "h_dim":          64,
            "lr":             1e-4,
            "weight_decay":   1e-5,
            "dropout":        0.1,
        },

        # ── Shared safety / debug flags ────────────────────────────────────
        "safety": {
            "smoke_test":         False,
            "smoke_batches":      2,
            "auto_install_deps":  False,
            "seed":               42,
        },
    }
