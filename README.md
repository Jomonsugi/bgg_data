# BGG Data Package

A Python package for collecting BoardGameGeek game data and downloading official rulebooks.

## Quick Start

- Requirements:
  - Python 3.11+
  - uv (package manager)
  - Chrome + ChromeDriver (for web automation)
  - Tavily API key in your shell (`TAVILY_API_KEY`)
  - Optional: local MLX vision model (no external LLM needed)

Install deps:

```bash
uv sync
```

## Update Database (BGG Games)

Collect BGG games into the local SQLite DB `bgg_games.db`:

```bash
# Top 100 games
uv run python -m bgg_data.database.collect_data --limit 100

# A specific window (e.g., ranks 51–100)
uv run python -m bgg_data.database.collect_data --start-rank 51 --limit 50
```

- Options:
  - `--start-rank N` (default 1)
  - `--limit N` (default 100)
  - `--delay N` seconds between API calls (default 3)
  - `--db PATH` custom DB path

## Fetch Rulebooks (Agentic)

The rulebook fetcher is agentic by default:
- Tries BGG official website → HTML quick scan → MLX/Together vision → Tavily search → verification
- Prefers PDFs; will save HTML if no PDF is available and try to upgrade HTML→PDF when possible.

Run for your DB:

```bash
# Process all games missing rulebooks
uv run python -m bgg_data.cli.main

# Only ranks 1–20
uv run python -m bgg_data.cli.main --rank-from 1 --rank-to 20

# Limit to 5 games and save screenshots
uv run python -m bgg_data.cli.main --limit 5 --screenshots
```

### Vision Backend

- Default: Together.ai (requires `TOGETHER_API_KEY`)
- Local: MLX vision (no external API). Example:

```bash
VISION_BACKEND=mlx \
MLX_VLM_MODEL=mlx-community/Llama-3.2-11B-Vision-Instruct-4bit \
uv run python -m bgg_data.cli.main --rank-from 1 --rank-to 20
```

To install MLX vision support, the dependency is already in this project (`mlx-vlm`). Model reference: `mlx-community/Llama-3.2-11B-Vision-Instruct-4bit` on Hugging Face.

### Coverage Output

Coverage counts both PDFs and HTML and prints a breakdown, for example:

```
Total games in database: 20
Existing rulebooks: 20
  - PDFs: 19  |  HTML: 1
Missing rulebooks: 0
Coverage: 100.0%
```

Note: counts reflect files present in `rulebooks/`. If numbers look off, clean or review that folder.

## Outputs

- Database: `bgg_games.db`
- Rulebooks: `rulebooks/` (standardized filenames like `Game-Name_rules.pdf`)
- Screenshots: `screenshots/Game_Name/` (if `--screenshots`)
- Logs: `bgg_data.log`

## Notes

- Web search uses Tavily (via `TAVILY_API_KEY`) with fallback; you can run without the vision backend in many cases.
- If official sites block direct downloads, the agent falls back to reliable sources when possible and verifies files.