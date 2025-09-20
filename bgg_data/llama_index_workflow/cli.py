from __future__ import annotations

import argparse
from pathlib import Path
import asyncio
from dotenv import load_dotenv

from .workflow import RulebookWorkflow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch board game rulebooks via workflow")
    p.add_argument("--rank-from", type=int, default=1, help="Minimum BGG rank (inclusive)")
    p.add_argument("--rank-to", type=int, default=5, help="Maximum BGG rank (inclusive)")
    p.add_argument("--db-path", type=Path, default=Path("bgg_games.db"), help="Path to SQLite DB")
    default_out = Path(__file__).resolve().parent / "rulebooks"
    p.add_argument("--out-dir", type=Path, default=default_out, help="Download directory")
    p.add_argument("--limit", type=int, default=None, help="Optional max number of games")
    p.add_argument("--verbose", action="store_true", help="Enable workflow verbose logs")
    # Removed --model-config; model_config.json bundled with the package is always used
    return p.parse_args()


async def _amain(args: argparse.Namespace) -> None:
    # Load .env to pick up HUGGING_FACE_HUB_TOKEN etc.
    load_dotenv()

    w = RulebookWorkflow(timeout=180, verbose=args.verbose)
    result = await w.run(
        rank_from=args.rank_from,
        rank_to=args.rank_to,
        db_path=str(args.db_path),
        out_dir=str(args.out_dir),
        limit=args.limit,
        model_config=None,
    )
    print(result)


def main() -> None:
    args = parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()


