# DAG - Rule-Based Rulebook Fetcher

A rule-based approach to fetching board game rulebooks using LLM vision models and structured workflows.

## Overview

The DAG (Directed Acyclic Graph) approach uses a structured, rule-based workflow to find and download board game rulebooks. It combines:

- **LLM Vision Models** - Analyzes webpage screenshots to identify download links
- **Web Scraping** - Automated browser interaction with Selenium
- **Structured Fallbacks** - Multiple strategies when initial approaches fail
- **Search Integration** - Tavily API for web search when direct methods fail

## Architecture

```
dag/
├── src/                    # Main rulebook fetching modules (renamed from rulebooks/)
│   ├── agentic_fetcher.py  # Main orchestrator class
│   ├── handlers/           # Specialized handlers for different tasks
│   │   ├── web.py         # Web scraping and browser automation  
│   │   ├── llm.py         # LLM vision model integration
│   │   ├── download.py    # PDF download and validation
│   │   ├── search.py      # Web search fallback
│   │   └── fallback_strategy.py  # Fallback coordination
│   └── utils.py           # Utility functions
├── cli/                   # Command-line interface
├── rulebooks/             # Downloaded PDF files
├── screenshots/           # Debug screenshots (optional)
└── config.py             # Configuration settings
```

## Quick Start

### Prerequisites

- Python 3.11+
- Chrome + ChromeDriver
- API Keys (optional but recommended):
  - `TOGETHER_API_KEY` for vision models
  - `TAVILY_API_KEY` for web search

### Basic Usage

```bash
# Process all games missing rulebooks
uv run python -m dag.cli.main

# Process specific rank range
uv run python -m dag.cli.main --rank-from 1 --rank-to 20

# Process with debugging screenshots
uv run python -m dag.cli.main --limit 5 --screenshots

# List games missing rulebooks (no processing)
uv run python -m dag.cli.main --list-missing
```

### Vision Backend Options

**Together.ai (Default - Cloud)**
```bash
TOGETHER_API_KEY="your_key" uv run python -m dag.cli.main
```

**MLX (Local - No API needed)**
```bash
VISION_BACKEND=mlx \
MLX_VLM_MODEL=mlx-community/Llama-3.2-11B-Vision-Instruct-4bit \
uv run python -m dag.cli.main --rank-from 1 --rank-to 20
```

## How It Works

### 1. **Official Website First**
- Navigates to the game's official website
- Takes screenshots of relevant pages
- Uses LLM vision to identify download links

### 2. **LLM Vision Analysis**
- Analyzes screenshots using vision models (Together.ai or local MLX)
- Identifies PDF download buttons, links, and file icons
- Extracts URLs for potential rulebook downloads

### 3. **Download & Validation**
- Attempts to download identified files
- Validates PDF format and content
- Handles redirects, authentication, and edge cases
- Falls back to HTML if PDF unavailable

### 4. **Search Fallback**
- Uses Tavily API for web search if direct methods fail
- Searches for "game_name rulebook PDF download"
- Analyzes search results for official sources

### 5. **Verification**
- Ensures downloaded files are valid PDFs
- Checks file size and content
- Removes invalid downloads

## Configuration

Key settings in `config.py`:

```python
# Vision Model Settings
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
VISION_BACKEND = "together"  # or "mlx"

# Browser Settings  
HEADLESS_BROWSER = True
BROWSER_TIMEOUT = 30
SCREENSHOT_DELAY = 3

# File Paths
RULEBOOKS_DIR = PROJECT_ROOT / "rulebooks"
```

## Outputs

- **Rulebooks**: `dag/rulebooks/Game-Name_ID_rules.pdf`
- **Screenshots**: `dag/screenshots/Game_Name/` (if `--screenshots`)
- **Logs**: `dag/bgg_data.log`

## Advanced Usage

### Custom Log Files
```bash
uv run python -m dag.cli.main --log-file custom_run.log
```

### Integration with Database
The DAG approach integrates with the shared database module to:
- Query games that need rulebooks
- Track download status and metadata
- Avoid duplicate processing

### Programmatic Usage
```python
from dag import RulebookOrchestrator

fetcher = RulebookOrchestrator(save_screenshots=True)
results = fetcher.fetch_rulebooks_for_games(games, delay_between_games=3.0)
```

## Troubleshooting

### Common Issues

1. **Vision Model Errors**: Ensure API keys are set correctly
2. **Browser Issues**: Make sure Chrome and ChromeDriver are installed
3. **Download Failures**: Check network connectivity and site accessibility
4. **Empty Results**: Some sites may block automated access

### Debug Mode
```bash
# Enable screenshots and detailed logging
uv run python -m dag.cli.main --screenshots --limit 1
```

### MLX Setup Issues
```bash
# Install MLX vision dependencies
pip install mlx-vlm
```

## Contributing

This is an experimental framework for testing rule-based approaches to rulebook fetching. The structured workflow makes it easy to:

- Add new handlers for specific sites
- Modify the LLM prompts for better accuracy  
- Implement new fallback strategies
- Debug and analyze the decision process

The rule-based nature provides predictable behavior and clear debugging paths compared to more dynamic agentic approaches.
