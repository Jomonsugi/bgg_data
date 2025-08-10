"""
Rulebooks module (agentic) for finding and downloading game rulebooks.

This package handles:
- LLM-based rulebook detection from website screenshots
- Web scraping and interaction
- Rulebook download and validation
- Web search fallback for hard-to-find rulebooks
- Agentic orchestration (LangGraph)
"""

from .agentic_fetcher import AgenticRulebookFetcher
from ..models import Game, FetchResult

__all__ = [
    "AgenticRulebookFetcher",
    "Game",
    "FetchResult",
]
