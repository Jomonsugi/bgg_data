"""
Database operations for BGG game data.

This module provides high-level database operations for querying and managing
BGG game data, including integration with the rulebook fetcher.
"""

import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

from ..models import Game
from .models import create_database

logger = logging.getLogger(__name__)


class BGGDatabase:
    """
    High-level database operations for BGG game data.
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize the BGG database handler.
        
        Args:
            db_path: Path to the BGG database
        """
        self.db_path = db_path
        
        # Create database if it doesn't exist
        if not self.db_path.exists():
            logger.info(f"Database not found at {self.db_path}, creating it...")
            create_database(str(self.db_path))
    
    def get_games(self, limit: Optional[int] = None, rank_from: Optional[int] = None, 
                  rank_to: Optional[int] = None) -> List[Game]:
        """
        Get games from the BGG database.
        
        Args:
            limit: Maximum number of games to return
            rank_from: Minimum rank to include (lower number = higher rank)
            rank_to: Maximum rank to include
            
        Returns:
            List of BGG games
        """
        try:
            if not self.db_path.exists():
                logger.info(f"Database not found at {self.db_path}, creating it...")
                create_database(str(self.db_path))
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if games table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
            table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                logger.info("Games table doesn't exist, creating database...")
                conn.close()
                create_database(str(self.db_path))
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
            
            # Build query
            query = """
                SELECT bgg_id, name, rank, url, publisher, year_published
                FROM games 
                WHERE 1=1
            """
            params = []
            
            # Rank range filters
            if rank_from is not None:
                query += " AND rank >= ?"
                params.append(rank_from)
            if rank_to is not None:
                query += " AND rank <= ?"
                params.append(rank_to)
            
            query += " ORDER BY rank ASC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            games = []
            for row in rows:
                game = Game(
                    id=str(row[0]),  # Convert bgg_id to string
                    name=row[1],
                    rank=row[2],
                    url=row[3] if row[3] else self._generate_bgg_url(str(row[0]), row[1]),
                    publisher=row[4],  # publisher
                    year_published=row[5]  # year_published
                )
                games.append(game)
            
            conn.close()
            logger.info(f"Retrieved {len(games)} games from database")
            return games
            
        except Exception as e:
            logger.error(f"Error retrieving games from database: {e}")
            return []
    
    def _generate_bgg_url(self, game_id: str, game_name: str) -> str:
        """
        Generate BGG URL for a game if not stored in database.
        
        Args:
            game_id: BGG game ID
            game_name: Game name
            
        Returns:
            Generated BGG URL
        """
        # Convert game name to URL-friendly format
        url_name = game_name.lower().replace(' ', '-').replace(':', '').replace("'", '')
        return f"https://boardgamegeek.com/boardgame/{game_id}/{url_name}"
    
    def get_statistics(self) -> dict:
        """
        Get statistics about games in the database.
        
        Returns:
            Dictionary with statistics
        """
        try:
            # Ensure database exists
            if not self.db_path.exists():
                logger.info(f"Database not found at {self.db_path}, creating it...")
                create_database(str(self.db_path))
            
            # Get total games from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if games table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
            table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                logger.info("Games table doesn't exist, creating database...")
                conn.close()
                create_database(str(self.db_path))
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM games")
            total_games = cursor.fetchone()[0]
            conn.close()
            
            stats = {
                'total_games_in_db': total_games,
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    

