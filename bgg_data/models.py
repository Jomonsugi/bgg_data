"""
Shared data models for the BGG Data package.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Game:
    """Unified game data model."""
    name: str
    url: str
    id: Optional[str] = None
    rank: Optional[int] = None
    publisher: Optional[str] = None
    year_published: Optional[int] = None


@dataclass 
class FetchResult:
    """Result of a rulebook fetch attempt."""
    game_name: str
    success: bool
    rulebook_url: Optional[str] = None
    filename: Optional[str] = None
    file_path: Optional[str] = None
    method: str = "unknown"
    error_message: Optional[str] = None
    processing_time: float = 0.0
