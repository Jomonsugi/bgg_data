"""
Rulebooks module for finding and downloading game rulebooks.

This package handles:
- LLM-based rulebook detection from website screenshots
- Web scraping and interaction
- Rulebook download and validation
- Web search fallback for hard-to-find rulebooks
- Orchestrated flow (LangGraph)
"""

from .agentic_fetcher import AgenticRulebookFetcher as RulebookOrchestrator
# Backwards-compat export
AgenticRulebookFetcher = RulebookOrchestrator
from ..models import Game, FetchResult

__all__ = [
    "RulebookOrchestrator",
    "AgenticRulebookFetcher",
    "Game",
    "FetchResult",
]
