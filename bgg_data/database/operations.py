"""
Database operations for BGG game data.

This module provides high-level database operations for querying and managing
BGG game data, including integration with the rulebook fetcher.
"""

import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

from .models import Game, create_database

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
        Retrieve games from the database with optional filtering.
        
        Args:
            limit: Maximum number of games to return
            rank_from: Minimum rank (inclusive)
            rank_to: Maximum rank (inclusive)
            
        Returns:
            List of Game objects
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Check if games table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='games'
            """)
            if not cursor.fetchone():
                logger.info("Games table doesn't exist, creating database...")
                create_database(str(self.db_path))
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
            
            # Build query based on parameters
            query = "SELECT bgg_id, name, rank, url, publisher, year_published FROM games"
            params = []
            conditions = []
            
            if rank_from is not None:
                conditions.append("rank >= ?")
                params.append(rank_from)
            
            if rank_to is not None:
                conditions.append("rank <= ?")
                params.append(rank_to)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY rank ASC"
            
            if limit is not None:
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
            BGG URL for the game
        """
        # Clean game name for URL
        clean_name = game_name.replace(" ", "+").replace(":", "%3A")
        return f"https://boardgamegeek.com/boardgame/{game_id}/{clean_name}"
    
    def get_game_by_name(self, game_name: str) -> Optional[Game]:
        """
        Retrieve a specific game by name.
        
        Args:
            game_name: Game name
            
        Returns:
            Game object if found, None otherwise
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Check if games table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='games'
            """)
            if not cursor.fetchone():
                logger.info("Games table doesn't exist, creating database...")
                create_database(str(self.db_path))
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
            
            cursor.execute("""
                SELECT bgg_id, name, rank, url, publisher, year_published 
                FROM games 
                WHERE name = ?
            """, (game_name,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return Game(
                    id=str(row[0]),
                    name=row[1],
                    rank=row[2],
                    url=row[3] if row[3] else self._generate_bgg_url(str(row[0]), row[1]),
                    publisher=row[4],
                    year_published=row[5]
                )
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving game by name from database: {e}")
            return None