"""
Shared configuration and utility functions for the rulebook agent.
"""

from pathlib import Path


def default_db_path() -> str:
    """
    Get the default path to the BGG games database.
    
    Returns:
        Path to bgg_games.db in the project root.
    """
    # bgg_data/bgg_data/rule_book_agent/config.py -> project root is parents[2]
    return str(Path(__file__).resolve().parents[2] / "bgg_games.db")
