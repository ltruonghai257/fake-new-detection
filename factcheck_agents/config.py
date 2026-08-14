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
    return (
        Path(os.environ["DATA_ROOT"]) if os.environ.get("DATA_ROOT") else _PROJECT_ROOT
    )


@dataclass
class Settings:
    # ── LLM (agent reasoning) ────────────────────────────────────────────
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("FACTCHECK_LLM_MODEL", "gpt-4o-mini")
    )
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("FACTCHECK_LLM_TEMPERATURE", "0.1"))
    )

    # ── Web search providers ─────────────────────────────────────────────
    tavily_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY")
    )
    google_cse_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_CSE_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    google_cse_id: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_CSE_ID")
    )
    max_results: int = field(
        default_factory=lambda: int(os.getenv("FACTCHECK_MAX_RESULTS", "6"))
    )
    max_queries: int = field(
        default_factory=lambda: int(os.getenv("FACTCHECK_MAX_QUERIES", "3"))
    )

    # ── Model checkpoints (validation-stage, may be absent) ──────────────
    data_root: Path = field(default_factory=_root)
    # explicit overrides win over auto-detection under data_root/training/...
    phobert_ckpt_dir: Optional[str] = field(
        default_factory=lambda: os.getenv("VIFACTCHECK_CKPT_DIR")
    )
    phobert_original_ckpt_dir: Optional[str] = field(
        default_factory=lambda: os.getenv("VIFACTCHECK_ORIGINAL_CKPT_DIR")
    )
    coolant_ckpt_path: Optional[str] = field(
        default_factory=lambda: os.getenv("COOLANT_CKPT_PATH")
    )
    device: str = field(default_factory=lambda: os.getenv("FACTCHECK_DEVICE", "auto"))

    # ── Source tier & reliability ─────────────────────────────────────────────
    trusted_domains: str = field(
        default_factory=lambda: os.getenv(
            "FACTCHECK_TRUSTED_DOMAINS",
            "vnexpress.net,thanhnien.vn,dantri.com.vn,tuoitre.vn",
        )
    )
    flagged_domains: str = field(
        default_factory=lambda: os.getenv("FACTCHECK_FLAGGED_DOMAINS", "kenh14.vn")
    )
    reliability_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("FACTCHECK_RELIABILITY_THRESHOLD", "0.5")
        )
    )
    debate_rounds: int = field(
        default_factory=lambda: int(os.getenv("FACTCHECK_DEBATE_ROUNDS", "0"))
    )

    # ── Debate pipeline (Phase 1 / M2) ────────────────────────────────────────
    agreement_threshold: float = field(
        default_factory=lambda: float(os.getenv("FACTCHECK_AGREEMENT_THRESHOLD", "0.7"))
    )
    max_debate_rounds: int = field(
        default_factory=lambda: int(os.getenv("FACTCHECK_MAX_DEBATE_ROUNDS", "10"))
    )
    google_factcheck_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_FACTCHECK_API_KEY")
    )
    social_loop_min_count: int = field(
        default_factory=lambda: int(os.getenv("FACTCHECK_SOCIAL_LOOP_MIN_COUNT", "3"))
    )
    social_loop_min_credibility: float = field(
        default_factory=lambda: float(
            os.getenv("FACTCHECK_SOCIAL_LOOP_MIN_CREDIBILITY", "0.6")
        )
    )

    # ── Debate advocate prompts (override via env; empty string = use default) ─
    real_advocate_prompt: str = field(
        default_factory=lambda: os.getenv("FACTCHECK_REAL_ADVOCATE_PROMPT", "")
    )
    fake_advocate_prompt: str = field(
        default_factory=lambda: os.getenv("FACTCHECK_FAKE_ADVOCATE_PROMPT", "")
    )

    # ── LangGraph checkpoint ─────────────────────────────────────────────────
    checkpoint_db: str = field(
        default_factory=lambda: os.getenv(
            "FACTCHECK_CHECKPOINT_DB", ".factcheck_checkpoints.db"
        )
    )

    # ── A2A agent server ports (9001–9010) ───────────────────────────────────
    a2a_port_search: int = field(
        default_factory=lambda: int(os.getenv("A2A_PORT_SEARCH", "9001"))
    )
    a2a_port_evaluate: int = field(
        default_factory=lambda: int(os.getenv("A2A_PORT_EVALUATE", "9002"))
    )
    a2a_port_real_source: int = field(
        default_factory=lambda: int(os.getenv("A2A_PORT_REAL_SOURCE", "9003"))
    )
    a2a_port_fake_source: int = field(
        default_factory=lambda: int(os.getenv("A2A_PORT_FAKE_SOURCE", "9004"))
    )
    a2a_port_social_loop: int = field(
        default_factory=lambda: int(os.getenv("A2A_PORT_SOCIAL_LOOP", "9005"))
    )
    a2a_port_agreement_gate: int = field(
        default_factory=lambda: int(os.getenv("A2A_PORT_AGREEMENT_GATE", "9006"))
    )
    a2a_port_real_advocate: int = field(
        default_factory=lambda: int(os.getenv("A2A_PORT_REAL_ADVOCATE", "9007"))
    )
    a2a_port_fake_advocate: int = field(
        default_factory=lambda: int(os.getenv("A2A_PORT_FAKE_ADVOCATE", "9008"))
    )
    a2a_port_judge: int = field(
        default_factory=lambda: int(os.getenv("A2A_PORT_JUDGE", "9009"))
    )
    a2a_port_conclusion: int = field(
        default_factory=lambda: int(os.getenv("A2A_PORT_CONCLUSION", "9010"))
    )

    def phobert_search_root(self) -> Path:
        return self.data_root / "training" / "checkpoints_vifactcheck"

    def phobert_original_search_root(self) -> Path:
        return self.data_root / "training" / "checkpoints_vifactcheck_original"

    def coolant_search_root(self) -> Path:
        return self.data_root / "training" / "checkpoints_coolant"

    def has_llm(self) -> bool:
        return bool(self.openai_api_key)

    def has_search(self) -> bool:
        return bool(self.tavily_api_key) or bool(
            self.google_cse_api_key and self.google_cse_id
        )


settings = Settings()


def a2a_ports() -> dict[str, int]:
    """Map agent name → A2A server port (contract with start_agents.sh / Phase 4)."""
    return {
        "search_agent": settings.a2a_port_search,
        "evaluate_agent": settings.a2a_port_evaluate,
        "real_source_agent": settings.a2a_port_real_source,
        "fake_source_agent": settings.a2a_port_fake_source,
        "social_loop_agent": settings.a2a_port_social_loop,
        "agreement_gate": settings.a2a_port_agreement_gate,
        "real_advocate": settings.a2a_port_real_advocate,
        "fake_advocate": settings.a2a_port_fake_advocate,
        "judge_agent": settings.a2a_port_judge,
        "conclusion_agent": settings.a2a_port_conclusion,
    }
