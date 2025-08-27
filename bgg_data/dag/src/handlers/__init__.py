"""
Handlers for different aspects of rulebook fetching.

This module contains specialized handlers for:
- LLM vision analysis
- Web page interaction
- File downloads
- Web search fallback
- Fallback strategy orchestration
"""

from .llm import LLMHandler, PDFAssessment
from .web import WebPageHandler
from .download import DownloadHandler
from .search import WebSearchFallback, SearchResult
from .fallback_strategy import FallbackOrchestrator, FallbackContext, FallbackStrategy

__all__ = [
    "LLMHandler",
    "PDFAssessment",
    "WebPageHandler", 
    "DownloadHandler",
    "WebSearchFallback",
    "SearchResult",
    "FallbackOrchestrator",
    "FallbackContext",
    "FallbackStrategy",
]
