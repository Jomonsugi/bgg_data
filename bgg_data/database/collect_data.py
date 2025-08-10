import argparse
import logging
from .models import create_database
from .collector import BGGDataCollector

def main():
    parser = argparse.ArgumentParser(description='Collect BGG game data')
    parser.add_argument('--db', default='bgg_games.db', help='Database file path')
    parser.add_argument('--limit', type=int, default=100, help='Number of games to collect')
    parser.add_argument('--start-rank', type=int, default=1, help='Starting rank (default: 1)')
    parser.add_argument('--delay', type=int, default=3, help='Seconds to wait between API requests')
    
    args = parser.parse_args()
    
    # Create database if it doesn't exist
    create_database(args.db)
    
    # Collect data
    collector = BGGDataCollector(args.db)
    collector.collect_top_games(args.limit, args.delay, args.start_rank)

if __name__ == "__main__":
    main()
