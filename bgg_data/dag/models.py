"""
Data models specific to the DAG rulebook fetching layer.
"""

from dataclasses import dataclass
from typing import Optional


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
