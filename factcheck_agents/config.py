"""Runtime configuration for the fact-checking module.

All settings come from environment variables (loaded from the project ``.env``
if present). Nothing here imports torch or heavy deps so it is cheap to load
from the CLI / MCP server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:  # optional; .env is convenient but not required
    from dotenv import load_dotenv

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
except Exception:  # pragma: no cover
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _root() -> Path:
    return Path(os.environ["DATA_ROOT"]) if os.environ.get("DATA_ROOT") else _PROJECT_ROOT


@dataclass
class Settings:
    # ── LLM (agent reasoning) ────────────────────────────────────────────
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    llm_model: str = field(default_factory=lambda: os.getenv("FACTCHECK_LLM_MODEL", "gpt-4o-mini"))
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("FACTCHECK_LLM_TEMPERATURE", "0.1")))

    # ── Web search providers ─────────────────────────────────────────────
    tavily_api_key: Optional[str] = field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    google_cse_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_CSE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )
    google_cse_id: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_CSE_ID"))
    max_results: int = field(default_factory=lambda: int(os.getenv("FACTCHECK_MAX_RESULTS", "6")))
    max_queries: int = field(default_factory=lambda: int(os.getenv("FACTCHECK_MAX_QUERIES", "3")))

    # ── Model checkpoints (validation-stage, may be absent) ──────────────
    data_root: Path = field(default_factory=_root)
    # explicit overrides win over auto-detection under data_root/training/...
    phobert_ckpt_dir: Optional[str] = field(default_factory=lambda: os.getenv("VIFACTCHECK_CKPT_DIR"))
    coolant_ckpt_path: Optional[str] = field(default_factory=lambda: os.getenv("COOLANT_CKPT_PATH"))
    device: str = field(default_factory=lambda: os.getenv("FACTCHECK_DEVICE", "auto"))

    def phobert_search_root(self) -> Path:
        return self.data_root / "training" / "checkpoints_vifactcheck"

    def coolant_search_root(self) -> Path:
        return self.data_root / "training" / "checkpoints_coolant"

    def has_llm(self) -> bool:
        return bool(self.openai_api_key)

    def has_search(self) -> bool:
        return bool(self.tavily_api_key) or bool(self.google_cse_api_key and self.google_cse_id)


settings = Settings()
