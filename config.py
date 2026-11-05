"""
config.py — Central configuration loader for the College Admission Agent.
Reads from .env (or .env.example fallback) and exposes a single Settings object.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ── locate the project root (parent of this file) ───────────────────────────
PROJECT_ROOT = Path(__file__).parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

# Load .env if it exists, otherwise fall back to .env.example so the app
# can run in demo mode without requiring manual setup.
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
elif ENV_EXAMPLE.exists():
    load_dotenv(ENV_EXAMPLE)


class Settings:
    """Typed access to all environment-driven configuration values."""

    # ── LLM ─────────────────────────────────────────────────────────────────
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").lower()
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

    # ── Embeddings ───────────────────────────────────────────────────────────
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local").lower()

    # ── Storage ──────────────────────────────────────────────────────────────
    VECTOR_STORE_PATH: Path = PROJECT_ROOT / os.getenv(
        "VECTOR_STORE_PATH", "data/vector_store"
    )
    KNOWLEDGE_BASE_DIR: Path = PROJECT_ROOT / os.getenv(
        "KNOWLEDGE_BASE_DIR", "data/knowledge_base"
    )

    # ── Agent behaviour ──────────────────────────────────────────────────────
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "10"))

    # ── Chunking ─────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    @classmethod
    def validate(cls) -> None:
        """Raise a clear error early if the chosen provider needs a missing key."""
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise EnvironmentError(
                "LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set. "
                "Add it to your .env file or switch LLM_PROVIDER to 'ollama'."
            )
        if cls.EMBEDDING_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise EnvironmentError(
                "EMBEDDING_PROVIDER is 'openai' but OPENAI_API_KEY is not set."
            )

    @classmethod
    def display(cls) -> str:
        return (
            f"  LLM provider    : {cls.LLM_PROVIDER} "
            f"({'model=' + cls.OPENAI_MODEL if cls.LLM_PROVIDER == 'openai' else 'model=' + cls.OLLAMA_MODEL})\n"
            f"  Embedding       : {cls.EMBEDDING_PROVIDER}\n"
            f"  Knowledge base  : {cls.KNOWLEDGE_BASE_DIR}\n"
            f"  Vector store    : {cls.VECTOR_STORE_PATH}\n"
            f"  Top-K retrieval : {cls.TOP_K_RESULTS}"
        )


settings = Settings()
