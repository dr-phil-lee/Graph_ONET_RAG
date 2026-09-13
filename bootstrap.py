"""Path setup and constants for the O*NET GraphRAG chatbot.

Import this module FIRST in every Chatbot script: it puts Research/src on
sys.path so `import onetrag` works from anywhere, and loads Chatbot/.env
for optional settings (OPENROUTER_API_KEY, ONET_CHATBOT_MODEL_DIR, ...).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CHATBOT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = CHATBOT_DIR.parent  # ...\ONET\Research

_src = str(RESEARCH_DIR / "src")
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
