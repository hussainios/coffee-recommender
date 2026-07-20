from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_CHAT_MODEL = "gpt-5.4-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

_client: OpenAI | None = None


def reset_openai_client_cache() -> None:
    global _client
    _client = None


def get_openai_client(purpose: str) -> OpenAI:
    global _client
    if _client is None:
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"OPENAI_API_KEY is required for {purpose}. "
                "Add it to .env or export it before running this workflow."
            )
        _client = OpenAI(api_key=api_key)
    return _client
