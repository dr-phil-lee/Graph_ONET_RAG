"""Path setup and constants for the O*NET GraphRAG chatbot.

Import this module FIRST in every Chatbot script: it puts the vendored
src/ on sys.path so `import onetrag` works from anywhere, and loads .env
for settings (OPENROUTER_API_KEY, ...). The repo is self-contained: the
onetrag retrieval package, the knowledge graph (data/graph/*.parquet), and
the O*NET level-anchor file (data/onet/) all live inside it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CHATBOT_DIR = Path(__file__).resolve().parent
DATA_DIR = CHATBOT_DIR / "data"

_src = str(CHATBOT_DIR / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

try:
    from dotenv import load_dotenv

    load_dotenv(CHATBOT_DIR / ".env")
except ImportError:  # dotenv is in requirements; tolerate a bare interpreter
    pass

# --- OpenRouter (the only LLM backend) --------------------------------------
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_DEFAULT_MODEL = "anthropic/claude-haiku-4.5"

# --- generation / retrieval -------------------------------------------------
# In-memory override of cfg["budget"]["context_tokens"]; Research/config.yaml
# itself must never be edited (its hash keys the research LLM cache).
DEFAULT_CONTEXT_BUDGET = 3000
# Generous output ceiling: list answers with many citations were getting cut
# mid-JSON at 700 tokens, which broke parsing.
GEN_MAX_TOKENS = 2000
GEN_TEMPERATURE = 0.2
