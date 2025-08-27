"""
Command-line interface for the BGG Data package.

This module provides CLI commands for:
- BGG data collection
- Rulebook fetching
- Database management
"""

from .main import main

__all__ = [
    "main",
]
