"""
Main CLI entry point for BGG Data package.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from ..config import DATABASE_PATH, RULEBOOKS_DIR
from ..database import BGGDatabase, Game
from ..rulebooks import AgenticRulebookFetcher
from ..rulebooks.utils import is_rulebook_already_downloaded
from ..logging_config import setup_logging

logger = logging.getLogger(__name__)


class BGGIntegration:
    """
    Integrates the rulebook fetcher with BGG database data.
    """
    
    def __init__(self, db_path: Path = DATABASE_PATH, rulebooks_dir: Path = RULEBOOKS_DIR):
        """
        Initialize the BGG integration.
        
        Args:
            db_path: Path to the BGG database
            rulebooks_dir: Directory for rulebooks
        """
        self.db_path = db_path
        self.rulebooks_dir = rulebooks_dir
        self.db = BGGDatabase(db_path)
        self.fetcher = None
    
    def filter_games_without_rulebooks(self, games: List[Game]) -> List[Game]:
        """
        Filter out games that already have rulebooks.
        
        Args:
            games: List of games to filter
            
        Returns:
            List of games without rulebooks
        """
        filtered_games = []
        
        for game in games:
            if not is_rulebook_already_downloaded(game.name, self.rulebooks_dir):
                filtered_games.append(game)
            else:
                logger.debug(f"Rulebook already exists for '{game.name}', skipping")
        
        logger.info(f"Filtered to {len(filtered_games)} games without rulebooks (from {len(games)} total)")
        return filtered_games
    

    
    def fetch_rulebooks_for_bgg_games(self, 
                                     limit: Optional[int] = None, 
                                     rank_from: Optional[int] = None,
                                     rank_to: Optional[int] = None,
                                     delay_between_games: float = 3.0,
                                     save_screenshots: bool = False):
        """
        Fetch rulebooks for BGG games from the database.
        
        Args:
            limit: Maximum number of games to process
            rank_from: Minimum rank to include
            rank_to: Maximum rank to include  
            delay_between_games: Delay between processing games
            save_screenshots: Whether to save screenshots for debugging
            
        Returns:
            List of fetch results
        """
        try:
            logger.info("Starting BGG rulebook fetch integration...")
            
            # Get games from database
            bgg_games = self.db.get_games(limit, rank_from, rank_to)
            if not bgg_games:
                logger.warning("No games found in database")
                return []
            
            # Filter games without rulebooks
            games_to_process = self.filter_games_without_rulebooks(bgg_games)
            if not games_to_process:
                logger.info("All games already have rulebooks")
                return []
            
            # Initialize and run the fetcher
            self.fetcher = RulebookFetcher(
                rulebooks_dir=self.rulebooks_dir,
                save_screenshots=save_screenshots
            )
            
            # Fetch rulebooks
            results = self.fetcher.fetch_rulebooks_for_games(
                games_to_process, 
                delay_between_games=delay_between_games
            )
            
            logger.info(f"BGG integration completed. Processed {len(results)} games.")
            return results
            
        except Exception as e:
            logger.error(f"Error in BGG integration: {e}")
            return []
        finally:
            if self.fetcher:
                self.fetcher.cleanup()
    
    def get_statistics(self) -> dict:
        """
        Get statistics about rulebook coverage.
        
        Returns:
            Dictionary with statistics
        """
        try:
            # Get total games from database
            db_stats = self.db.get_statistics()
            total_games = db_stats.get('total_games_in_db', 0)
            
            # Count existing rulebooks (PDF and HTML)
            pdf_count = len(list(self.rulebooks_dir.glob("*.pdf")))
            html_count = len(list(self.rulebooks_dir.glob("*.html"))) + len(list(self.rulebooks_dir.glob("*.htm")))
            existing_rulebooks_total = pdf_count + html_count
            
            # Calculate coverage (PDF + HTML)
            coverage_percentage = (existing_rulebooks_total / total_games * 100) if total_games > 0 else 0
            
            stats = {
                'total_games_in_db': total_games,
                'existing_rulebooks_pdf': pdf_count,
                'existing_rulebooks_html': html_count,
                'existing_rulebooks_total': existing_rulebooks_total,
                'missing_rulebooks': total_games - existing_rulebooks_total,
                'coverage_percentage': round(coverage_percentage, 1)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}


def main():
    """Main CLI entry point for BGG integration."""
    try:
        # Set up logging first
        setup_logging()
        
        # Simple CLI args: --rank-from, --rank-to, --limit, --screenshots
        parser = argparse.ArgumentParser(description="Fetch rulebooks for BGG games from DB")
        parser.add_argument("--limit", type=int, default=None, help="Max number of games to process")
        parser.add_argument("--rank-from", type=int, default=None, help="Inclusive lower bound rank (e.g., 6)")
        parser.add_argument("--rank-to", type=int, default=None, help="Inclusive upper bound rank (e.g., 10)")
        parser.add_argument("--screenshots", action="store_true", help="Save screenshots for debugging")

        args = parser.parse_args()

        # Initialize integration
        integration = BGGIntegration()
        

        # Show current statistics
        stats = integration.get_statistics()
        print("\n" + "="*60)
        print("CURRENT RULEBOOK COVERAGE")
        print("="*60)
        print(f"Total games in database: {stats.get('total_games_in_db', 0)}")
        print(f"Existing rulebooks: {stats.get('existing_rulebooks_total', stats.get('existing_rulebooks', 0))}")
        print(f"  - PDFs: {stats.get('existing_rulebooks_pdf', 0)}  |  HTML: {stats.get('existing_rulebooks_html', 0)}")
        print(f"Missing rulebooks: {stats.get('missing_rulebooks', 0)}")
        print(f"Coverage: {stats.get('coverage_percentage', 0)}%")
        print("="*60)
        
        # If database is empty, suggest collecting real BGG data
        if stats.get('total_games_in_db', 0) == 0:
            print("\n💡 Database is empty! To get started, collect real BGG data first:")
            print("   python -m bgg_data.database.collect_data --limit 100")
            print("   This will fetch real game data from the BGG API.")
            return

        # Fetch rulebooks
        print("\nStarting rulebook fetch...")
        from ..rulebooks.agentic_fetcher import AgenticRulebookFetcher
        fetcher = AgenticRulebookFetcher(save_screenshots=args.screenshots)
        games = integration.db.get_games(args.limit, args.rank_from, args.rank_to)
        results = fetcher.fetch_rulebooks_for_games(games, delay_between_games=3.0)

        # Show results
        print("\n" + "="*60)
        print("FETCH RESULTS")
        print("="*60)
        for result in results:
            status = "✓ SUCCESS" if result.success else "✗ FAILED"
            print(f"{status} | {result.game_name} | {result.method}")
            if result.success:
                print(f"  └─ Downloaded: {result.filename}")
            else:
                print(f"  └─ Error: {result.error_message}")
        
        # Show updated statistics
        updated_stats = integration.get_statistics()
        print("\n" + "="*60)
        print("UPDATED RULEBOOK COVERAGE")
        print("="*60)
        print(f"Total games in database: {updated_stats.get('total_games_in_db', 0)}")
        print(f"Existing rulebooks: {updated_stats.get('existing_rulebooks_total', updated_stats.get('existing_rulebooks', 0))}")
        print(f"  - PDFs: {updated_stats.get('existing_rulebooks_pdf', 0)}  |  HTML: {updated_stats.get('existing_rulebooks_html', 0)}")
        print(f"Missing rulebooks: {updated_stats.get('missing_rulebooks', 0)}")
        print(f"Coverage: {updated_stats.get('coverage_percentage', 0)}%")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        raise


if __name__ == "__main__":
    main()
