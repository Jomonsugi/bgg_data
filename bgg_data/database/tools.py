"""
Utility functions for finding and adding games to the database.
These can be called directly by the graph (deterministic) or exposed as tools for a chatbot.
"""

import re
import logging
import sqlite3
from typing import Optional
from pathlib import Path

from .collector import BGGDataCollector
from .operations import BGGDatabase
from .models import Game

logger = logging.getLogger(__name__)


def find_bgg_game_by_search(game_name: str) -> dict:
    """
    Search for a game on BoardGameGeek using Tavily web search.
    
    Returns dict with:
    - 'ok': bool
    - 'bgg_id': str | None
    - 'game_name': str
    - 'bgg_url': str | None
    - 'error': str | None
    """
    import os
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "bgg_id": None,
            "game_name": game_name,
            "bgg_url": None,
            "error": "TAVILY_API_KEY not set"
        }
    
    try:
        from tavily import TavilyClient
    except Exception as e:
        return {
            "ok": False,
            "bgg_id": None,
            "game_name": game_name,
            "bgg_url": None,
            "error": f"Tavily client import failed: {e}"
        }
    
    try:
        client = TavilyClient(api_key=api_key)
        # Search for the game on BGG
        query = f'"{game_name}" site:boardgamegeek.com/boardgame'
        resp = client.search(query=query, max_results=5)
        
        results = resp.get("results", [])
        if not results:
            return {
                "ok": False,
                "bgg_id": None,
                "game_name": game_name,
                "bgg_url": None,
                "error": "No BGG results found"
            }
        
        # Extract BGG ID from the first result URL
        # BGG URLs look like: https://boardgamegeek.com/boardgame/123456/game-name
        for result in results:
            url = result.get("url", "")
            if not url:
                continue
            
            # Match BGG boardgame URL pattern
            match = re.search(r'/boardgame/(\d+)/', url)
            if match:
                bgg_id = match.group(1)
                return {
                    "ok": True,
                    "bgg_id": bgg_id,
                    "game_name": game_name,
                    "bgg_url": url,
                    "error": None
                }
        
        return {
            "ok": False,
            "bgg_id": None,
            "game_name": game_name,
            "bgg_url": None,
            "error": "Could not extract BGG ID from search results"
        }
        
    except Exception as e:
        logger.error(f"Error searching for game '{game_name}': {e}")
        return {
            "ok": False,
            "bgg_id": None,
            "game_name": game_name,
            "bgg_url": None,
            "error": str(e)
        }


def find_and_add_game_to_db(
    game_name: str,
    db_path: Optional[str] = None,
    bgg_id: Optional[str] = None
) -> dict:
    """
    Find a game on BGG via web search (or use provided BGG ID), fetch its details, and add it to the database.
    
    If bgg_id is provided, skips search and fetches directly from BGG XML API.
    Otherwise, searches for the game using Tavily.
    
    This function is called directly by the graph (deterministic step).
    It can also be exposed as a tool for a chatbot in the future.
    
    Returns:
        dict with:
        - 'ok': bool
        - 'game': Game object dict | None
        - 'bgg_id': str | None
        - 'error': str | None
    """
    if not db_path:
        # Default to project root bgg_games.db
        db_path = str(Path(__file__).resolve().parents[2] / "bgg_games.db")
    
    # If BGG ID is provided, use it directly (no search needed)
    if bgg_id and bgg_id != "0":
        # Step 1: Check if game already exists in DB
        db = BGGDatabase(Path(db_path))
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(
            "SELECT bgg_id, name, rank, url, publisher, year_published FROM games WHERE bgg_id = ?",
            (int(bgg_id),),
        )
        row = cur.fetchone()
        conn.close()
        
        if row:
            # Game already exists
            existing = Game(
                id=str(row[0]),
                name=row[1],
                rank=row[2],
                url=row[3],
                publisher=row[4],
                year_published=row[5],
            )
            logger.info(f"Game with BGG ID {bgg_id} already exists in database")
            return {
                "ok": True,
                "game": {
                    "id": existing.id,
                    "name": existing.name,
                    "rank": existing.rank or 0,
                    "url": existing.url or "",
                    "publisher": existing.publisher or "",
                    "year_published": existing.year_published or 0,
                },
                "bgg_id": existing.id,
                "error": None
            }
        
        # Step 2: Fetch game details from BGG XML API using the provided ID
        collector = BGGDataCollector(db_path=db_path)
        game_details = collector.get_game_details(bgg_id)
        
        if not game_details:
            return {
                "ok": False,
                "game": None,
                "bgg_id": bgg_id,
                "error": f"Could not fetch game details from BGG API for ID {bgg_id}"
            }
        
        # Step 3: Save to database
        collector.save_game_to_db(game_details)
        
        # Step 4: Return Game object
        saved_game = db.get_game_by_name(game_details['name'])
        if not saved_game:
            return {
                "ok": False,
                "game": None,
                "bgg_id": bgg_id,
                "error": "Game saved but could not retrieve from database"
            }
        
        return {
            "ok": True,
            "game": {
                "id": saved_game.id,
                "name": saved_game.name,
                "rank": saved_game.rank or 0,
                "url": saved_game.url or "",
                "publisher": saved_game.publisher or "",
                "year_published": saved_game.year_published or 0,
            },
            "bgg_id": saved_game.id,
            "error": None
        }
    
    # No BGG ID provided - use search-based approach
    # Step 1: Search for the game on BGG
    search_result = find_bgg_game_by_search(game_name)
    if not search_result.get("ok"):
        return {
            "ok": False,
            "game": None,
            "bgg_id": None,
            "error": search_result.get("error", "Search failed")
        }
    
    found_bgg_id = search_result["bgg_id"]
    if not found_bgg_id:
        return {
            "ok": False,
            "game": None,
            "bgg_id": None,
            "error": "BGG ID not found in search results"
        }
    
    # Step 2: Check if game already exists in DB
    db = BGGDatabase(Path(db_path))
    existing = db.get_game_by_name(game_name)
    if existing:
        logger.info(f"Game '{game_name}' already exists in database")
        return {
            "ok": True,
            "game": {
                "id": existing.id,
                "name": existing.name,
                "rank": existing.rank or 0,
                "url": existing.url or "",
                "publisher": existing.publisher or "",
                "year_published": existing.year_published or 0,
            },
            "bgg_id": existing.id,
            "error": None
        }
    
    # Step 3: Fetch game details from BGG XML API
    collector = BGGDataCollector(db_path=db_path)
    game_details = collector.get_game_details(found_bgg_id)
    
    if not game_details:
        return {
            "ok": False,
            "game": None,
            "bgg_id": found_bgg_id,
            "error": f"Could not fetch game details from BGG API for ID {found_bgg_id}"
        }
    
    # Step 4: Save to database
    collector.save_game_to_db(game_details)
    
    # Step 5: Return Game object
    saved_game = db.get_game_by_name(game_name)
    if not saved_game:
        return {
            "ok": False,
            "game": None,
            "bgg_id": found_bgg_id,
            "error": "Game saved but could not retrieve from database"
        }
    
    return {
        "ok": True,
        "game": {
            "id": saved_game.id,
            "name": saved_game.name,
            "rank": saved_game.rank or 0,
            "url": saved_game.url or "",
            "publisher": saved_game.publisher or "",
            "year_published": saved_game.year_published or 0,
        },
        "bgg_id": saved_game.id,
        "error": None
    }
