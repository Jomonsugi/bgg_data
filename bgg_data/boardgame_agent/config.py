"""Central configuration for the boardgame rules agent.

API keys are read from .env or environment variables (e.g. exported in .zshrc).
Everything else is a plain Python constant — edit this file to change defaults.
"""

from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

# ── LLM (Together API) ────────────────────────────────────────────────────────
# API key comes from environment; model is selected via the sidebar dropdown.
TOGETHER_API_KEY: str | None = os.getenv("TOGETHER_API_KEY")

# Default model (first in the list below is used until the user picks another).
TOGETHER_MODEL_NAME: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# Models shown in the sidebar dropdown — add or remove freely.
TOGETHER_MODEL_OPTIONS: list[str] = [
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "deepseek-ai/DeepSeek-V3",
    "openai/gpt-oss-120b",
]

# ── Embeddings ────────────────────────────────────────────────────────────────
# Changing this requires clicking "Rebuild index" in the sidebar.
EMBED_MODEL_NAME: str = "BAAI/bge-base-en-v1.5"

# ── Retrieval ─────────────────────────────────────────────────────────────────
# Default number of pages retrieved per query. Adjustable in the sidebar.
RETRIEVAL_TOP_K: int = 5

# ── Web Search (Tavily) ───────────────────────────────────────────────────────
TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")

# ── Hardware ──────────────────────────────────────────────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch

    DEVICE: str = "mps" if torch.backends.mps.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).parent
DATA_DIR: Path = BASE_DIR / "data"
QDRANT_PATH: Path = DATA_DIR / "qdrant"
GAMES_DB_PATH: Path = DATA_DIR / "games.db"
CHECKPOINTS_DB_PATH: Path = DATA_DIR / "agent_checkpoints.db"
COLLECTION_NAME: str = "rulebook_pages"

# Create data directories on import so nothing downstream needs to mkdir.
DATA_DIR.mkdir(parents=True, exist_ok=True)
QDRANT_PATH.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "games").mkdir(exist_ok=True)
