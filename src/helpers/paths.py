import os
from pathlib import Path


def get_data_root() -> Path:
    """Return effective data root: DATA_ROOT env var → repo root."""
    env_val = os.environ.get("DATA_ROOT")
    if env_val:
        return Path(env_val)
    # This file lives at src/helpers/paths.py → parents[2] = repo root
    return Path(__file__).resolve().parents[2]
