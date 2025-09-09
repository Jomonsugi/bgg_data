"""
Shared domain models for the BGG Data project.

This module contains neutral data transfer objects used across packages
(`database`, `dag`, etc.) to avoid circular dependencies.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Game:
    """Unified game data model shared across the project."""
    name: str
    url: str
    id: Optional[str] = None
    rank: Optional[int] = None
    publisher: Optional[str] = None
    year_published: Optional[int] = None


