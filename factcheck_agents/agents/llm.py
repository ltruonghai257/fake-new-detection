"""Shared LLM helpers for the agents."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Optional

from ..config import settings


@lru_cache(maxsize=1)
def get_llm():
    """Return a cached ChatOpenAI instance, or None if no key is configured."""
    if not settings.has_llm():
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )


def parse_json(text: str) -> Optional[dict]:
    """Best-effort JSON extraction from an LLM response."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None
