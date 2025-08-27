# Database Module

Shared database functionality for BoardGameGeek (BGG) game data collection and management across all rulebook fetching approaches.

## Overview

The database module provides a centralized SQLite database for storing and managing BGG game information. This database is shared across all experimental rulebook fetching frameworks (DAG, HF Agent, etc.) to avoid duplicate data collection and enable consistent game metadata access.

## Features

- **BGG Data Collection**: Fetches game data from BoardGameGeek API
- **SQLite Storage**: Local database for fast queries and offline access
- **Game Metadata**: Stores names, IDs, ranks, publishers, and other game info
- **Shared Access**: Used by all rulebook fetching approaches
- **Incremental Updates**: Efficiently updates existing data

## Database Schema

The main `games` table includes:

```sql
CREATE TABLE games (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    rank INTEGER,
    year_published INTEGER,
    min_players INTEGER,
    max_players INTEGER,
    playing_time INTEGER,
    min_age INTEGER,
    publisher TEXT,
    designer TEXT,
    artist TEXT,
    description TEXT,
    image_url TEXT,
    thumbnail_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Quick Start

### Collect BGG Data

```bash
# Collect top 100 games
uv run python -m bgg_data.database.collect_data --limit 100

# Collect specific rank range
uv run python -m bgg_data.database.collect_data --start-rank 51 --limit 50

# Custom database path
uv run python -m bgg_data.database.collect_data --db custom_path.db --limit 20
```

### Command Options

- `--start-rank N`: Starting rank (default: 1)
- `--limit N`: Number of games to collect (default: 100)  
- `--delay N`: Seconds between API calls (default: 3)
- `--db PATH`: Custom database path (default: `bgg_games.db`)

## Programmatic Usage

### Basic Operations

```python
from bgg_data.database import BGGDatabase, Game

# Initialize database
db = BGGDatabase()

# Get all games
games = db.get_all_games()

# Get games by rank range
top_20 = db.get_games_by_rank_range(1, 20)

# Get specific game
game = db.get_game_by_id(174430)  # Gloomhaven

# Search games
results = db.search_games("Pandemic")
```

### Integration with Rulebook Fetchers

```python
from bgg_data.database import BGGDatabase
from dag import RulebookOrchestrator

# Get games that need rulebooks
db = BGGDatabase()
games = db.get_games_by_rank_range(1, 50)

# Filter for games without rulebooks
missing_rulebooks = [g for g in games if not has_rulebook(g)]

# Use with any fetching approach
fetcher = RulebookOrchestrator()
results = fetcher.fetch_rulebooks_for_games(missing_rulebooks)
```

## Database Management

### Location
- Default: `bgg_games.db` in project root
- Configurable via `DATABASE_PATH` in config files

### Backup
```bash
# Simple backup
cp bgg_games.db bgg_games_backup.db

# With timestamp
cp bgg_games.db "bgg_games_$(date +%Y%m%d_%H%M%S).db"
```

### Reset/Rebuild
```bash
# Remove existing database
rm bgg_games.db

# Recollect data
uv run python -m bgg_data.database.collect_data --limit 100
```

## BGG API Integration

### Rate Limiting
- Default 3-second delay between API calls
- Respects BGG API guidelines
- Configurable via `--delay` parameter

### Data Freshness
- Games are updated if they already exist
- New games are added to the database
- Timestamps track creation and update times

### Error Handling
- Retries failed API calls
- Logs errors for debugging
- Continues processing on individual failures

## Architecture

```
database/
├── __init__.py          # Main exports (BGGDatabase, Game)
├── models.py           # Database schema and table creation
├── operations.py       # BGGDatabase class with all operations
├── collect_data.py     # CLI script for data collection
└── collector.py        # BGG API integration logic
```

## Shared Usage Patterns

### Cross-Framework Compatibility
All rulebook fetching approaches use the same database:

```python
# DAG approach
from dag.cli import BGGIntegration
integration = BGGIntegration()
games = integration.get_games_in_range(1, 20)

# HF Agent approach  
from hf_agent import SomeHFApproach
agent = SomeHFApproach()
agent.process_games(games)  # Same game objects
```

### Game Model
```python
@dataclass
class Game:
    id: int
    name: str
    rank: Optional[int] = None
    year_published: Optional[int] = None
    min_players: Optional[int] = None
    max_players: Optional[int] = None
    # ... other fields
```

## Performance

### Indexing
- Primary key on `id` for fast lookups
- Index on `rank` for range queries
- Index on `name` for text searches

### Query Optimization
- Use rank ranges for efficient filtering
- Batch operations when possible
- Connection pooling for concurrent access

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Increase `--delay` if getting 429 errors
2. **Network Issues**: Check internet connectivity for BGG API
3. **Database Locked**: Ensure no other processes are using the database
4. **Missing Games**: Some games may not have complete BGG data

### Debug Information
```bash
# Verbose logging
PYTHONPATH=. python -m bgg_data.database.collect_data --limit 10 -v
```

### Database Inspection
```bash
# Using sqlite3 CLI
sqlite3 bgg_games.db
.tables
.schema games
SELECT COUNT(*) FROM games;
SELECT name, rank FROM games ORDER BY rank LIMIT 10;
```

## Future Enhancements

- **Multi-language Support**: Store game names in multiple languages
- **Advanced Filtering**: More sophisticated game queries
- **Data Validation**: Enhanced validation of BGG data
- **Caching**: Redis or similar for frequently accessed data
- **Analytics**: Game statistics and trend analysis
