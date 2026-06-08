"""Environment utilities for cross-platform data root resolution."""

import os
from pathlib import Path


def get_data_root() -> Path:
    """
    Return the effective data root path for the current environment.

    Reads DATA_ROOT from the loaded .env file, converts it to the correct
    format for the running OS using pathlib.Path, and validates it exists.

    Returns:
        Path: The resolved data root directory.

    Raises:
        EnvironmentError: If DATA_ROOT is not set or the path does not exist.

    Examples:
        >>> from src.utils.env_utils import get_data_root
        >>> data_root = get_data_root()
        >>> print(data_root)
        PosixPath('/workspace/fake-news-data-for-thesis')
    """
    env_val = os.environ.get("DATA_ROOT")

    if not env_val:
        raise EnvironmentError(
            "DATA_ROOT environment variable is not set.\n"
            "Please create a .env file from the appropriate example:\n"
            "  - Vast.ai: cp .env.vastai.example .env\n"
            "  - Colab:   cp .env.colab.example .env\n"
            "  - Windows: cp .env.windows.example .env\n"
            "  - macOS:   cp .env.mac.example .env\n"
            "Then update the DATA_ROOT path for your environment."
        )

    # Convert to Path - handles both forward slashes and OS-specific separators
    # On Windows: / separators work, backslashes work with raw strings
    # On Linux/macOS: / separators work as expected
    data_root = Path(env_val)

    # Validate the path exists
    if not data_root.exists():
        raise EnvironmentError(
            f"DATA_ROOT path does not exist: {data_root}\n"
            f"Please verify the path is correct in your .env file.\n"
            f"For Google Colab: ensure you've mounted Drive with:\n"
            f"  from google.colab import drive; drive.mount('/content/drive')"
        )

    return data_root
