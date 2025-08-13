"""
BGG Data Package - BoardGameGeek data collection and rulebook fetching.

This package provides two main functionalities:
1. Fetching and maintaining BoardGameGeek game data in a database
2. Finding and downloading official English rulebooks for board games
"""

__version__ = "0.1.0"
__author__ = "BGG Data Team"

# Main package imports for convenience
from .database import BGGDatabase
from .rulebooks import RulebookOrchestrator
from .models import Game, FetchResult
from .logging_config import setup_logging

__all__ = [
    "BGGDatabase", 
    "RulebookOrchestrator",
    "Game",
    "FetchResult", 
    "setup_logging",
]
