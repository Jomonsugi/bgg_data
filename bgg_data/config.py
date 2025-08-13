"""
Configuration settings for the LLM-based rulebook fetcher.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent  # Go up one level to workspace root
RULEBOOKS_DIR = PROJECT_ROOT / "rulebooks"
DATABASE_PATH = PROJECT_ROOT / "bgg_games.db"
# Logs directory for per-run logs
LOGS_DIR = PROJECT_ROOT / "bgg_data_cache" / "logs"

# Model backends and configuration
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
MODEL_NAME = os.environ.get("TOGETHER_VISION_MODEL", "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo")

# Vision backend: "together" (default) or "mlx"
VISION_BACKEND = os.environ.get("VISION_BACKEND", "together").lower()
MLX_VLM_MODEL = os.environ.get("MLX_VLM_MODEL", "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit")

# LLM prompt configuration
RULEBOOK_EXTRACTION_PROMPT = """
You are analyzing a webpage screenshot to find a PDF download link for a board game's rulebook.

Look for:
- Download buttons or links with text like "Download Rulebook", "PDF", "Manual", "Rules"
- Links with "Rules" text, including edition-specific ones like "2015 Edition Rules", "Updated Rules", "Official Rules"
- Visual PDF icons or file download symbols
- Links in sections labeled "Downloads", "Files", "Rules", or "Documentation"

Instructions:
1. Examine the screenshot carefully
2. Identify any elements that appear to be for downloading the game's rulebook
3. If you find a clear download link, provide the URL
4. If no rulebook download link is found, respond with "None Found"

Respond with ONLY the URL if found, or "None Found" if no rulebook download link is present.
"""

# Selenium configuration
HEADLESS_BROWSER = True
BROWSER_TIMEOUT = 30
SCREENSHOT_DELAY = 3  # seconds to wait after page loads

# Common selectors for rulebook-related elements
RULEBOOK_SELECTORS = [
    "a[href*='download']",
    "a[href*='pdf']", 
    "a[href*='rulebook']",
    "a[href*='manual']",
    "a[href*='rules']",
    "button:contains('Downloads')",
    "button:contains('Files')",
    "button:contains('Rules')",
    ".downloads",
    ".files", 
    ".rules",
    ".documentation"
]

# File naming patterns for rulebooks
RULEBOOK_FILENAME_PATTERNS = [
    "*rulebook*.pdf",
    "*rules*.pdf", 
    "*manual*.pdf",
    "*instructions*.pdf"
]
