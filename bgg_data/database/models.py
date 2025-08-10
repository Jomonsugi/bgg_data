import sqlite3
import os

def create_database(db_path="bgg_games.db"):
    """Create the database and tables for BGG game data."""
    
    # Ensure database directory exists (only if path contains directory)
    db_dir = os.path.dirname(db_path)
    if db_dir:  # Only create directory if there is one
        os.makedirs(db_dir, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create games table with JSON arrays for designers, categories, and mechanics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY,
            bgg_id INTEGER UNIQUE NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            min_players INTEGER,
            max_players INTEGER,
            playing_time INTEGER,
            min_playing_time INTEGER,
            max_playing_time INTEGER,
            year_published INTEGER,
            average_rating REAL,
            complexity_weight REAL,
            suggested_age INTEGER,
            best_player_count INTEGER,
            rank INTEGER,
            url TEXT,  -- BGG URL for the game
            publisher TEXT,  -- Publisher name
            designers TEXT,  -- JSON array of designer names
            categories TEXT, -- JSON array of category names
            mechanics TEXT,  -- JSON array of mechanic names
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Add missing columns to existing databases if they don't exist
    columns_to_add = [
        ("last_updated", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
        ("url", "TEXT"),
        ("publisher", "TEXT")
    ]
    
    for column_name, column_def in columns_to_add:
        try:
            cursor.execute(f"ALTER TABLE games ADD COLUMN {column_name} {column_def}")
            print(f"Added {column_name} column to existing database")
        except sqlite3.OperationalError:
            # Column already exists, ignore
            pass
    
    conn.commit()
    conn.close()
    print(f"Database created successfully at {db_path}")

if __name__ == "__main__":
    create_database()
