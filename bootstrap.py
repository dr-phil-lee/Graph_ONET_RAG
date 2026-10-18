"""Path setup and constants for the O*NET GraphRAG chatbot.

Import this module FIRST in every Chatbot script: it puts the vendored
src/ on sys.path so `import onetrag` works from anywhere, and loads .env
for settings (OPENROUTER_API_KEY, ...). The repo is self-contained: the
onetrag retrieval package, the knowledge graph (data/graph/*.parquet), and
the O*NET level-anchor file (data/onet/) all live inside it.

Environment variables (all optional, read from Chatbot/.env or the shell):
  OPENROUTER_API_KEY   default generator key (UI / API callers may override)
  OPENROUTER_MODEL     default OpenRouter model id
  CHATBOT_API_TOKEN    if set, the REST API requires `Authorization: Bearer`
  CHATBOT_CORS_ORIGINS comma-separated origins allowed by the REST API ("*")
  CHATBOT_SESSION_DIR  if set, API conversation memory is also saved there as JSON
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
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_DEFAULT_MODEL = (os.getenv("OPENROUTER_MODEL", "").strip()
                            or "anthropic/claude-haiku-4.5")
# Optional attribution headers OpenRouter shows in its dashboard.
OPENROUTER_APP_TITLE = "O*NET Graph RAG"

# --- generation / retrieval -------------------------------------------------
# In-memory override of cfg["budget"]["context_tokens"]; config.yaml itself
# must never be edited (its hash keys the research LLM cache).
DEFAULT_CONTEXT_BUDGET = 3000
MIN_CONTEXT_BUDGET = 500
MAX_CONTEXT_BUDGET = 8000
# Generous output ceiling: advisory answers (career change, comparisons) with
# inline citations were getting cut mid-JSON at 2000 tokens. The prompt also
# caps answer length, and a cut reply is repaired and flagged "truncated".
GEN_MAX_TOKENS = 4000
GEN_TEMPERATURE = 0.2
LLM_TIMEOUT_S = 120

# --- REST API ---------------------------------------------------------------
API_TOKEN = os.getenv("CHATBOT_API_TOKEN", "").strip()
API_CORS_ORIGINS = [o.strip() for o in
                    os.getenv("CHATBOT_CORS_ORIGINS", "*").split(",") if o.strip()]
