"""
Database module for BGG data collection and management.

This module handles:
- BGG API data collection
- Database schema and operations
- Game data storage and retrieval
"""

from .operations import BGGDatabase
from ..models import Game

__all__ = [
    "Game", 
    "BGGDatabase",
]
