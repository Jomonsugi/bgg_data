"""
Utility functions for the LLM-based rulebook fetcher.
"""

import os
import re
import requests
from pathlib import Path
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def check_existing_rulebooks(rulebooks_dir: Path) -> List[str]:
    """
    Scan the rulebooks directory and return a list of existing rulebook filenames.
    
    Args:
        rulebooks_dir: Path to the rulebooks directory
        
    Returns:
        List of existing rulebook filenames (without path)
    """
    if not rulebooks_dir.exists():
        logger.info(f"Rulebooks directory {rulebooks_dir} does not exist, creating it")
        rulebooks_dir.mkdir(parents=True, exist_ok=True)
        return []
    
    existing_files = [f.name for f in rulebooks_dir.glob("*.pdf")]
    logger.info(f"Found {len(existing_files)} existing rulebooks in {rulebooks_dir}")
    return existing_files

def extract_game_name_from_filename(filename: str) -> str:
    """
    Extract the game name from a rulebook filename.
    
    Args:
        filename: Rulebook filename (e.g., "Brass-Birmingham_rules.pdf")
        
    Returns:
        Extracted game name (e.g., "Brass-Birmingham")
    """
    # Remove file extension
    name_without_ext = filename.rsplit('.', 1)[0]
    
    # Handle optional id segment like _123456 before _rules
    m = re.match(r"^(.*)_\d+_(rules|rulebook|manual|instructions)$", name_without_ext, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    
    # Remove common suffixes
    suffixes_to_remove = ['_rules', '_rulebook', '_manual', '_instructions', '_spanish', '_german', '_french']
    for suffix in suffixes_to_remove:
        if name_without_ext.endswith(suffix):
            name_without_ext = name_without_ext[:-len(suffix)]
    
    return name_without_ext

def is_rulebook_already_downloaded(game_name: str, rulebooks_dir: Path, game_id: Optional[str] = None) -> bool:
    """
    Check if a rulebook for the given game already exists.
    Uses exact filename matching to avoid false positives.
    
    Args:
        game_name: Name of the game to check
        rulebooks_dir: Directory containing rulebooks
        
    Returns:
        True if rulebook already exists, False otherwise
    """
    if not rulebooks_dir.exists():
        return False
    
    # Create expected filename patterns for this specific game
    sanitized_name = sanitize_filename(game_name.replace(' ', '-').replace(':', ''))
    # Prefer id-appended filenames when id is available
    expected_patterns = []
    if game_id:
        expected_patterns.extend([
            f"{sanitized_name}_{game_id}_rules.pdf",
            f"{sanitized_name}_{game_id}_rules.html",
        ])
    expected_patterns.extend([
        f"{sanitized_name}_rules.pdf",
        f"{sanitized_name}_rules.html",
        f"{sanitized_name}_rulebook.pdf",
        f"{sanitized_name}_manual.pdf",
    ])
    
    # Check for exact matches only
    for pattern in expected_patterns:
        if (rulebooks_dir / pattern).exists():
            logger.info(f"Rulebook for '{game_name}' already exists: {pattern}")
            return True
    
    return False

def validate_url(url: str) -> bool:
    """
    Validate if a URL is accessible and points to a rulebook resource (PDF or HTML).
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is valid and accessible, False otherwise
    """
    if not url or url == "None Found":
        return False
    
    try:
        # Check if URL is well-formed
        if not url.startswith(('http://', 'https://')):
            return False

        # First try HEAD
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '').lower()
                if any(x in content_type for x in ['pdf', 'html', 'application/octet-stream']):
                    return True
        except Exception:
            pass

        # Fallback GET (some servers block HEAD or omit content-type)
        try:
            response = requests.get(url, timeout=15, allow_redirects=True, stream=True)
            if 200 <= response.status_code < 400:
                content_type = response.headers.get('content-type', '').lower()
                if any(x in content_type for x in ['pdf', 'html', 'application/octet-stream']):
                    return True
                # If unknown content type, allow and let downloader validate bytes
                return True
        except Exception:
            pass

        return False
    except Exception as e:
        logger.warning(f"Error validating URL {url}: {e}")
        return False

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed"
    
    return filename

def create_rulebook_filename(game_name: str, url: str, game_id: Optional[str] = None) -> str:
    """
    Create a standardized filename for a rulebook.
    
    Args:
        game_name: Name of the game
        url: URL of the rulebook
        
    Returns:
        Standardized filename
    """
    # Create clean base name
    clean_name = sanitize_filename(game_name.replace(' ', '-').replace(':', '').replace("'", ''))
    
    # Determine extension from URL
    extension = 'pdf' if '.pdf' in url.lower() else 'html'
    
    # Append BGG id when provided (before _rules)
    if game_id:
        return f"{clean_name}_{game_id}_rules.{extension}"
    return f"{clean_name}_rules.{extension}"


def is_pdf_or_html_path(path: Optional[str]) -> bool:
    """
    Check if a filesystem path string refers to a PDF or HTML file.
    """
    if not path:
        return False
    p = path.lower()
    return p.endswith('.pdf') or p.endswith('.html') or p.endswith('.htm')


def is_likely_english(url: str, text: str = "") -> bool:
    """
    Heuristic: determine if a resource is likely English from URL/text hints.
    """
    u = (url or "").lower()
    t = (text or "").lower()

    english_hints = ["_en", "-en", "/en/", "english", "en-us", "en_gb", "en-uk", "us", "uk"]
    non_english_hints = [
        "_de", "-de", "/de/", "german", "deutsch",
        "_fr", "-fr", "/fr/", "french", "francais",
        "_es", "-es", "/es/", "spanish", "espanol",
        "_it", "-it", "/it/", "italian",
        "_pt", "-pt", "/pt/", "portuguese",
        "_ru", "-ru", "/ru/", "russian",
        "_pl", "-pl", "/pl/", "polish",
        "_zh", "-zh", "/zh/", "chinese", "_cn", "-cn", "/cn/",
        "_ja", "-ja", "/ja/", "japanese", "_jp", "-jp", "/jp/",
        "_ko", "-ko", "/ko/", "korean",
    ]

    eng_score = sum(1 for h in english_hints if h in u or h in t)
    non_eng_score = sum(1 for h in non_english_hints if h in u or h in t)

    if eng_score > non_eng_score:
        return True
    if non_eng_score > eng_score:
        return False
    # Default unknown to True when nothing indicates non-English
    return True

def is_rulebook_already_downloaded(game_name: str, rulebooks_dir: Path, game_id: Optional[str] = None) -> bool:
    """
    Check if a rulebook for the given game already exists.
    Uses exact filename matching to avoid false positives.
    
    Args:
        game_name: Name of the game to check
        rulebooks_dir: Directory containing rulebooks
        
    Returns:
        True if rulebook already exists, False otherwise
    """
    if not rulebooks_dir.exists():
        return False
    
    # Create expected filename patterns for this specific game
    sanitized_name = sanitize_filename(game_name.replace(' ', '-').replace(':', ''))
    expected_patterns = []
    if game_id:
        expected_patterns.extend([
            f"{sanitized_name}_{game_id}_rules.pdf",
            f"{sanitized_name}_{game_id}_rules.html",
        ])
    expected_patterns.extend([
        f"{sanitized_name}_rules.pdf",
        f"{sanitized_name}_rules.html",
        f"{sanitized_name}_rulebook.pdf",
        f"{sanitized_name}_manual.pdf",
    ])
    
    # Check for exact matches only
    for pattern in expected_patterns:
        if (rulebooks_dir / pattern).exists():
            logger.info(f"Rulebook for '{game_name}' already exists: {pattern}")
            return True
    
    return False

def log_download_attempt(game_name: str, url: str, success: bool, error: Optional[str] = None):
    """
    Log download attempt details.
    
    Args:
        game_name: Name of the game
        url: URL that was attempted
        success: Whether download was successful
        error: Error message if download failed
    """
    if success:
        logger.info(f"Successfully downloaded rulebook for '{game_name}' from {url}")
    else:
        if error:
            logger.error(f"Failed to download rulebook for '{game_name}' from {url}: {error}")
        else:
            logger.warning(f"No rulebook found for '{game_name}' at {url}")
