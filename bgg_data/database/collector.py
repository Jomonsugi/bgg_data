import requests
import sqlite3
import time
import xml.etree.ElementTree as ET
import logging
from bs4 import BeautifulSoup
import re
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BGGDataCollector:
    def __init__(self, db_path="bgg_games.db"):
        self.db_path = db_path
        self.base_url = "https://boardgamegeek.com/xmlapi2"
        
    def get_top_games(self, limit=100, start_rank=1):
        """Get top games by rank from BGG rankings page."""
        games = []
        
        # Calculate which page to start from (each page has 100 games)
        start_page = ((start_rank - 1) // 100) + 1
        
        # Calculate how many games we need to collect from the start page
        games_on_start_page = 100 - ((start_rank - 1) % 100)
        remaining_games = limit
        
        # Calculate total pages needed
        if remaining_games <= games_on_start_page:
            pages_needed = 1
        else:
            additional_games = remaining_games - games_on_start_page
            additional_pages = (additional_games + 99) // 100  # Ceiling division
            pages_needed = 1 + additional_pages
        
        logger.info(f"Need to scrape {pages_needed} page(s) starting from page {start_page} to get {limit} games starting from rank {start_rank}")
        
        for page_offset in range(pages_needed):
            page = start_page + page_offset
            url = f"https://boardgamegeek.com/browse/boardgame"
            if page > 1:
                url += f"/page/{page}"
            
            try:
                logger.info(f"Scraping rankings page {page}...")
                response = requests.get(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                })
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find the rankings table
                table = soup.find('table', {'class': 'collection_table'})
                if not table:
                    logger.warning(f"No rankings table found on page {page}")
                    break
                
                # Extract games from the table - skip the header row (first row)
                rows = table.find_all('tr')[1:]  # Skip header row
                
                for row in rows:
                    if len(games) >= limit:
                        break
                    
                    # Extract game ID from the first game link in the row
                    title_link = row.find('a', href=re.compile(r'/boardgame/\d+/'))
                    if not title_link:
                        continue
                    
                    # Extract game ID from href (e.g., "/boardgame/224517/brass-birmingham" -> "224517")
                    href = title_link.get('href', '')
                    game_id_match = re.search(r'/boardgame/(\d+)/', href)
                    if not game_id_match:
                        continue
                    
                    game_id = game_id_match.group(1)
                    name = title_link.get_text(strip=True)
                    
                    # Extract rank from the first column
                    cells = row.find_all('td')
                    if not cells:
                        continue
                    
                    rank_cell = cells[0]  # First column contains rank
                    rank_text = rank_cell.get_text(strip=True)
                    try:
                        rank = int(rank_text)
                    except ValueError:
                        continue
                    
                    # Only include games within our desired rank range
                    if rank >= start_rank and rank < start_rank + limit:
                        games.append({
                            'id': game_id,
                            'name': name,
                            'rank': rank
                        })
                
                # Add a small delay to be respectful
                if page < pages_needed:
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error scraping page {page}: {e}")
                break
        
        logger.info(f"Scraped {len(games)} top-ranked games from {pages_needed} page(s)")
        return games
    
    def get_game_details(self, game_id):
        """Get detailed information for a specific game."""
        url = f"{self.base_url}/thing"
        params = {
            'id': game_id,
            'stats': 1
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            item = root.find('.//item')
            
            if item is None:
                return None
                
            # Extract basic game info
            name = item.find('.//name[@type="primary"]')
            name = name.get('value') if name is not None else 'Unknown'
            
            description = item.find('.//description')
            description = description.text if description is not None else ''
            
            # Extract player count
            min_players = item.find('.//minplayers')
            min_players = int(min_players.get('value')) if min_players is not None else None
            
            max_players = item.find('.//maxplayers')
            max_players = int(max_players.get('value')) if max_players is not None else None
            
            # Extract playing time
            playing_time = item.find('.//playingtime')
            playing_time = int(playing_time.get('value')) if playing_time is not None else None
            
            min_playing_time = item.find('.//minplaytime')
            min_playing_time = int(min_playing_time.get('value')) if min_playing_time is not None else None
            
            max_playing_time = item.find('.//maxplaytime')
            max_playing_time = int(max_playing_time.get('value')) if max_playing_time is not None else None
            
            # Extract year published
            year_published = item.find('.//yearpublished')
            year_published = int(year_published.get('value')) if year_published is not None else None
            
            # Extract ratings
            stats = item.find('.//statistics/ratings')
            average_rating = stats.find('.//average')
            average_rating = float(average_rating.get('value')) if average_rating is not None else None
            
            # Extract complexity
            complexity_weight = stats.find('.//averageweight')
            complexity_weight = float(complexity_weight.get('value')) if complexity_weight is not None else None
            
            # Extract suggested age
            suggested_age = item.find('.//minage')
            suggested_age = int(suggested_age.get('value')) if suggested_age is not None else None
            
            # Extract best player count (this is often subjective, taking the 'numvotes' for 'best' poll result)
            best_player_count = None
            poll_results = item.findall('.//poll[@name="suggested_numplayers"]/results')
            max_votes = -1
            for result in poll_results:
                num_players_str = result.get('numplayers')
                if num_players_str and '-' not in num_players_str: # Only consider single player counts for 'best'
                    best_vote = result.find('.//result[@value="Best"]')
                    if best_vote:
                        votes = int(best_vote.get('numvotes'))
                        if votes > max_votes:
                            max_votes = votes
                            best_player_count = int(num_players_str)
            
            # Extract designers, categories, mechanics
            designers = [link.get('value') for link in item.findall('.//link[@type="boardgamedesigner"]')]
            categories = [link.get('value') for link in item.findall('.//link[@type="boardgamecategory"]')]
            mechanics = [link.get('value') for link in item.findall('.//link[@type="boardgamemechanic"]')]
            
            # Extract publisher information
            publishers = [link.get('value') for link in item.findall('.//link[@type="boardgamepublisher"]')]
            # Use the first publisher as the primary publisher
            publisher = publishers[0] if publishers else None
            
            return {
                'bgg_id': game_id,
                'name': name,
                'description': description,
                'min_players': min_players,
                'max_players': max_players,
                'playing_time': playing_time,
                'min_playing_time': min_playing_time,
                'max_playing_time': max_playing_time,
                'year_published': year_published,
                'average_rating': average_rating,
                'complexity_weight': complexity_weight,
                'suggested_age': suggested_age,
                'best_player_count': best_player_count,
                'publisher': publisher,
                'designers': designers,
                'categories': categories,
                'mechanics': mechanics
            }
            
        except Exception as e:
            logger.error(f"Error fetching game details for {game_id}: {e}")
            return None
    
    def save_game_to_db(self, game_data):
        """Save game data to the database, only updating if there are changes."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if game already exists
            cursor.execute("SELECT * FROM games WHERE bgg_id = ?", (game_data['bgg_id'],))
            existing_game = cursor.fetchone()
            
            if existing_game:
                # Game exists, check if we need to update
                if self._has_changes(existing_game, game_data):
                    logger.info(f"Updating existing game: {game_data['name']} (changes detected)")
                    self._update_game(cursor, game_data)
                    conn.commit()
                else:
                    logger.info(f"Skipping {game_data['name']} - no changes detected")
            else:
                # New game, insert it
                logger.info(f"Inserting new game: {game_data['name']}")
                self._insert_game(cursor, game_data)
                conn.commit()
            
        except Exception as e:
            logger.error(f"Error saving game to database: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def _has_changes(self, existing_game, new_game_data):
        """Check if there are meaningful changes between existing and new game data."""
        # Convert lists to JSON strings for comparison
        designers_json = json.dumps(new_game_data['designers'])
        categories_json = json.dumps(new_game_data['categories'])
        mechanics_json = json.dumps(new_game_data['mechanics'])
        
        # Compare key fields (excluding timestamp fields)
        existing_fields = [
            existing_game[2],  # name
            existing_game[3],  # description
            existing_game[4],  # min_players
            existing_game[5],  # max_players
            existing_game[6],  # playing_time
            existing_game[7],  # min_playing_time
            existing_game[8],  # max_playing_time
            existing_game[9],  # year_published
            existing_game[10], # average_rating
            existing_game[11], # complexity_weight
            existing_game[12], # suggested_age
            existing_game[13], # best_player_count
            existing_game[14], # rank
            existing_game[15], # publisher
            existing_game[16], # designers
            existing_game[17], # categories
            existing_game[18]  # mechanics
        ]
        
        new_fields = [
            new_game_data['name'],
            new_game_data['description'],
            new_game_data['min_players'],
            new_game_data['max_players'],
            new_game_data['playing_time'],
            new_game_data['min_playing_time'],
            new_game_data['max_playing_time'],
            new_game_data['year_published'],
            new_game_data['average_rating'],
            new_game_data['complexity_weight'],
            new_game_data['suggested_age'],
            new_game_data['best_player_count'],
            new_game_data.get('rank'),
            new_game_data.get('publisher'),
            designers_json,
            categories_json,
            mechanics_json
        ]
        
        # Normalize None values to empty strings for comparison
        existing_fields = ['' if f is None else f for f in existing_fields]
        new_fields = ['' if f is None else f for f in new_fields]
        
        return existing_fields != new_fields
    
    def _insert_game(self, cursor, game_data):
        """Insert a new game into the database."""
        designers_json = json.dumps(game_data['designers'])
        categories_json = json.dumps(game_data['categories'])
        mechanics_json = json.dumps(game_data['mechanics'])
        
        cursor.execute("""
            INSERT INTO games 
            (bgg_id, name, description, min_players, max_players, playing_time,
             min_playing_time, max_playing_time, year_published, average_rating,
             complexity_weight, suggested_age, best_player_count, rank,
             publisher, designers, categories, mechanics, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            game_data['bgg_id'], game_data['name'], game_data['description'],
            game_data['min_players'], game_data['max_players'], game_data['playing_time'],
            game_data['min_playing_time'], game_data['max_playing_time'],
            game_data['year_published'], game_data['average_rating'],
            game_data['complexity_weight'], game_data['suggested_age'],
            game_data['best_player_count'], game_data.get('rank'),
            game_data.get('publisher'), designers_json, categories_json, mechanics_json
        ))
    
    def _update_game(self, cursor, game_data):
        """Update an existing game in the database."""
        designers_json = json.dumps(game_data['designers'])
        categories_json = json.dumps(game_data['categories'])
        mechanics_json = json.dumps(game_data['mechanics'])
        
        cursor.execute("""
            UPDATE games SET
                name = ?, description = ?, min_players = ?, max_players = ?,
                playing_time = ?, min_playing_time = ?, max_playing_time = ?,
                year_published = ?, average_rating = ?, complexity_weight = ?,
                suggested_age = ?, best_player_count = ?, rank = ?,
                publisher = ?, designers = ?, categories = ?, mechanics = ?, last_updated = datetime('now')
            WHERE bgg_id = ?
        """, (
            game_data['name'], game_data['description'],
            game_data['min_players'], game_data['max_players'], game_data['playing_time'],
            game_data['min_playing_time'], game_data['max_playing_time'],
            game_data['year_published'], game_data['average_rating'],
            game_data['complexity_weight'], game_data['suggested_age'],
            game_data['best_player_count'], game_data.get('rank'),
            game_data.get('publisher'), designers_json, categories_json, mechanics_json,
            game_data['bgg_id']
        ))
    
    def collect_top_games(self, limit=100, delay_seconds=3, start_rank=1):
        """Collect data for top games with rate limiting."""
        end_rank = start_rank + limit - 1
        logger.info(f"Starting collection of {limit} games (ranks {start_rank}-{end_rank}) with {delay_seconds}s delay")
        
        # Get top games by rank
        top_games = self.get_top_games(limit, start_rank)
        
        if not top_games:
            logger.error("No games found to collect")
            return
        
        # Show the games we're about to collect
        logger.info(f"\nTop {len(top_games)} games by rank:")
        for game in top_games:
            logger.info(f"#{game['rank']}: {game['name']} (ID: {game['id']})")
        
        # Collect details for each game
        for i, game in enumerate(top_games, 1):
            logger.info(f"\nCollecting data for game {i}/{len(top_games)}: #{game['rank']} {game['name']}")
            
            game_details = self.get_game_details(game['id'])
            if game_details:
                # Add rank information to the game details
                game_details['rank'] = game['rank']
                self.save_game_to_db(game_details)
                logger.info(f"Successfully saved #{game['rank']} {game['name']}")
            else:
                logger.error(f"Failed to get details for #{game['rank']} {game['name']}")
            
            # Rate limiting
            if i < len(top_games):  # Don't delay after the last game
                logger.info(f"Waiting {delay_seconds} seconds before next request...")
                time.sleep(delay_seconds)
        
        logger.info("Data collection completed!")
