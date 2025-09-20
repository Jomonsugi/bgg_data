from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import List, Optional

import requests
import fitz  # PyMuPDF
from tavily import TavilyClient
from workflows.events import Event
import json
from langdetect import detect, DetectorFactory
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# Set seed for consistent results
DetectorFactory.seed = 0

# --- Model configuration (JSON only) ---

def load_model_config(config_path: Optional[str]) -> dict:
    """Load the model configuration JSON from the given path.

    The config maps tasks (e.g., exploration LLM, official VLM) to model IDs
    and providers so we can change models without editing code.
    """
    # If no path provided, fall back to bundled model_config.json next to this file
    if not config_path:
        p = Path(__file__).resolve().parent / "model_config.json"
    else:
        p = Path(config_path)
    if not p.exists():
        raise RuntimeError(f"model_config.json not found at: {p}")
    with open(p, "r") as f:
        return json.load(f)

# --- Data types ---

class GamesFound(Event):
    """Event carrying the games to process and runtime settings."""
    games: List[dict]
    out_dir: str

# Removed unused event classes

# --- Simple Tools ---

def query_games_by_rank(
    db_path: Path, rank_from: int, rank_to: int, limit: Optional[int] = None
) -> List[dict]:
    """Return games in the given rank range from the SQLite DB as dicts."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    query = (
        "SELECT bgg_id, name, rank, url, publisher, year_published FROM games "
        "WHERE rank >= ? AND rank <= ? ORDER BY rank ASC"
    )
    params: List[object] = [rank_from, rank_to]
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    cur.execute(query, params)
    games = []
    for row in cur.fetchall():
        games.append(
            {
                "bgg_id": row[0],
                "name": row[1],
                "rank": row[2],
                "url": row[3],
                "publisher": row[4],
                "year_published": row[5],
            }
        )
    
    conn.close()
    return games

def _clean_url(url: str) -> str:
    """Trim trailing punctuation that sometimes clings to scraped URLs."""
    url = url.strip()
    while url and url[-1] in ")].,'\"":
        url = url[:-1]
    return url

# Removed unused _extract_urls_from_results function (replaced by Tavily)

def search_rulebook_urls(
    game_name: str,
    publisher: str | None = None,
    prefer_english: bool = False,
) -> List[str]:
    """Search the web and return an ordered list of candidate URLs.

    The list is de-duplicated and ordered with PDFs first, then likely
    official pages, then other pages. Set prefer_english to add English-
    leaning queries; validation still happens later.
    """
    # Initialize Tavily client
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    # Base queries
    queries = [
        f"{game_name} rulebook filetype:pdf",
        f"{game_name} \"rulebook pdf\"",
        f"{game_name} \"rules pdf\"",
        f"{game_name} rulebook site:boardgamegeek.com filetype:pdf",
        f"{game_name} official site rulebook pdf",
    ]
    
    # Add publisher-specific queries if available
    if publisher:
        # Clean publisher name for search
        clean_publisher = publisher.replace("Games", "").replace("Game", "").strip()
        queries.extend([
            f"{game_name} {publisher} official rulebook",
            f"{publisher} {game_name} rulebook pdf",
            f"{game_name} {publisher} resources",
        ])

    # Optional pass to bias towards English PDFs (generic, non-heuristic)
    if prefer_english:
        queries.extend([
            f"{game_name} rulebook english filetype:pdf",
            f"{game_name} \"english rulebook pdf\"",
            f"{game_name} \"english rules pdf\"",
        ])
    
    all_urls: List[str] = []
    for q in queries:
        try:
            # Use Tavily search
            response = tavily.search(query=q, search_depth="basic", max_results=5)
            results = response.get("results", [])
            
            # Extract URLs from Tavily results
            for result in results:
                url = result.get("url", "")
                if url:
                    all_urls.append(url)
                    
        except Exception as e:
            print(f"  ⚠️  Tavily search failed for '{q}': {e}")
            continue
    
    # Remove duplicates and prioritize PDF URLs first (agent will validate),
    # then official publisher websites, then other URLs
    seen = set()
    pdf_urls: List[str] = []
    official_urls: List[str] = []
    other_urls: List[str] = []
    
    for url in all_urls:
        if url not in seen:
            seen.add(url)
            
            # Check if this looks like an official publisher website
            is_official = False
            if publisher:
                publisher_clean = publisher.lower().replace(" ", "").replace(",", "").replace("llc", "").replace("inc", "")
                url_clean = url.lower().replace("www.", "").replace("https://", "").replace("http://", "").split("/")[0]
                is_official = publisher_clean in url_clean or any(part in url_clean for part in publisher_clean.split())
            
            if url.lower().endswith(".pdf"):
                pdf_urls.append(url)
            elif is_official:
                official_urls.append(url)
            else:
                other_urls.append(url)
    
    # Return PDFs first (so the agent can validate quickly), then official pages, then others
    return pdf_urls + official_urls + other_urls[:10]

def download_pdf(url: str, out_dir: Path, filename_stem: str) -> Optional[Path]:
    """Download a PDF (or an HTML page linking to one) to out_dir.

    Returns the path to the saved PDF, or None if download failed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0 Safari/537.36",
            "Accept": "application/pdf, text/html;q=0.9,*/*;q=0.8",
        }
    )

    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename_stem).strip("._-") or "rulebook"
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    path = out_dir / filename

    try:
        resp = session.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "").lower()
        data = resp.content
        if (b"%PDF" in data[:8]) or ("pdf" in content_type):
            with open(path, "wb") as f:
                f.write(data)
            return path
        if data.strip().lower().startswith(b"<"):
            html = data.decode("utf-8", errors="ignore")
            pdf_links = re.findall(r'href=["\']([^"\']+\.pdf)["\']', html, re.IGNORECASE)
            for link in pdf_links:
                if not link.startswith("http"):
                    from urllib.parse import urljoin
                    link = urljoin(url, link)
                try:
                    resp2 = session.get(link, timeout=30)
                    resp2.raise_for_status()
                    if b"%PDF" in resp2.content[:8]:
                        with open(path, "wb") as f:
                            f.write(resp2.content)
                        return path
                except Exception:
                    continue
    except Exception:
        return None
    finally:
        session.close()

    return None

def explore_site_for_pdfs(url: str, game_name: str, config: dict) -> List[str]:
    """Navigate a site (headless) and return any PDF links found.

    Uses a lightweight LLM prompt to suggest what to click, and also
    collects direct PDF links present in the HTML.
    """
    try:
        # Check if Chrome is available
        try:
            # Set up Chrome driver
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36")
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(30)
        except Exception as e:
            print(f"  ⚠️  Chrome not available, falling back to simple scraping: {e}")
            return _scrape_website_for_pdfs(url, game_name)
        
        try:
            driver.get(url)
            
            # Get page content for LLM analysis
            page_source = driver.page_source
            page_text = driver.find_element(By.TAG_NAME, "body").text[:2000]  # First 2000 chars
            
            # Ask the LLM what to click to reach a rulebook
            llm_model_id, llm_provider = get_exploration_llm_config(config)
            
            prompt = f"""
            You are exploring a website to find a rulebook for the board game "{game_name}".
            
            Current page: {url}
            Page content preview: {page_text}
            
            Look for:
            1. Direct PDF links to rulebooks
            2. Buttons or links that might lead to rulebooks (like "Resources", "Downloads", "Rules", "Rulebook")
            3. Navigation menus that might contain rulebook links
            
            Respond with a JSON list of actions to take:
            [
                {{"action": "click", "text": "button text to click", "reason": "why this might lead to rulebook"}},
                {{"action": "extract_pdfs", "reason": "found direct PDF links"}},
                {{"action": "stop", "reason": "no promising paths found"}}
            ]
            
            Only suggest clicking on elements that are likely to lead to rulebooks. Be specific about the text to click.
            """
            
            response = generate_text(prompt, llm_model_id, llm_provider)
            
            pdf_links = []
            
            # Extract any direct PDF links
            pdf_matches = re.findall(r'href=["\']([^"\']*\.pdf)["\']', page_source, re.IGNORECASE)
            for link in pdf_matches:
                if not link.startswith("http"):
                    from urllib.parse import urljoin
                    link = urljoin(url, link)
                pdf_links.append(link)
            
            # Handle Google Drive links
            gdrive_matches = re.findall(r'href=["\']([^"\']*drive\.google\.com[^"\']*)["\']', page_source, re.IGNORECASE)
            for link in gdrive_matches:
                if not link.startswith("http"):
                    from urllib.parse import urljoin
                    link = urljoin(url, link)
                if "/file/d/" in link and "/view" in link:
                    file_id = re.search(r'/file/d/([a-zA-Z0-9_-]+)', link)
                    if file_id:
                        direct_link = f"https://drive.google.com/uc?export=download&id={file_id.group(1)}"
                        pdf_links.append(direct_link)
            
            # If found, return now
            if pdf_links:
                return list(dict.fromkeys(pdf_links))
            
            # Try to parse LLM response for click actions
            try:
                import json
                actions = json.loads(response)
                
                for action in actions:
                    if action.get("action") == "click":
                        button_text = action.get("text", "")
                        if button_text:
                            try:
                                # Find and click the button
                                button = driver.find_element(By.XPATH, f"//*[contains(text(), '{button_text}')]")
                                button.click()
                                
                                # Wait for page to load
                                import time
                                time.sleep(2)  # Simple wait instead of WebDriverWait
                                
                                # Extract PDFs from new page
                                new_page_source = driver.page_source
                                new_pdf_matches = re.findall(r'href=["\']([^"\']*\.pdf)["\']', new_page_source, re.IGNORECASE)
                                for link in new_pdf_matches:
                                    if not link.startswith("http"):
                                        from urllib.parse import urljoin
                                        link = urljoin(url, link)
                                    pdf_links.append(link)
                                
                                # If we found PDFs, return them
                                if pdf_links:
                                    return list(dict.fromkeys(pdf_links))
                                    
                            except Exception:
                                continue
                                
            except (json.JSONDecodeError, KeyError):
                pass
            
            return list(dict.fromkeys(pdf_links))
            
        finally:
            driver.quit()
            
    except Exception as e:
        print(f"  ⚠️  Agentic exploration failed: {e}")
        return []

# Removed _scrape_website_for_pdfs function (replaced by agentic exploration)

def find_pdf_for_game(game_name: str, publisher: str = None, config: dict = None) -> dict:
    """Find one candidate URL for a game's rulebook.

    Uses search first (preferring PDFs), then explores likely sites to
    extract PDF links. The workflow will validate and iterate candidates.
    """
    log: dict = {"queries": [], "candidates": [], "selected_from": "none"}
    
    # Get search candidates using publisher information
    candidates = search_rulebook_urls(game_name, publisher)
    log["queries"] = [
        f"{game_name} rulebook filetype:pdf",
        f"{game_name} \"rulebook pdf\"",
        f"{game_name} \"rules pdf\"",
        f"{game_name} rulebook site:boardgamegeek.com filetype:pdf",
        f"{game_name} official site rulebook pdf",
    ]
    if publisher:
        log["queries"].extend([
            f"{game_name} rulebook site:{publisher.lower().replace(' ', '')}.com",
            f"{game_name} {publisher} official rulebook",
            f"{publisher} {game_name} rulebook pdf",
        ])
    log["candidates"] = candidates[:10]

    # Try direct PDF links first (validation happens later)
    for url in candidates:
        if url.lower().endswith(".pdf"):
            log["selected_from"] = "direct"
            return {"url": url, "log": log}

    # Explore likely sites and try PDFs found there
    if config:
        for url in candidates[:3]:  # Try first 3 candidates (should be official sites)
            if url and not url.lower().endswith(".pdf") and "boardgamegeek.com" not in url.lower():
                try:
                    print(f"  🤖 Exploring site: {url}")
                    pdf_links = explore_site_for_pdfs(url, game_name, config)
                    if pdf_links:
                        log["selected_from"] = "agentic_exploration"
                        log["candidates"].extend(pdf_links[:3])  # Add found PDFs to candidates
                        return {"url": pdf_links[0], "log": log}
                except Exception as e:
                    print(f"  ⚠️  Agentic exploration failed: {e}")
                    continue

    # Fallback: try simple PDF extraction from remaining websites
    for url in candidates[:3]:  # Try first 3 candidates
        if url and not url.lower().endswith(".pdf") and "boardgamegeek.com" not in url.lower():
            try:
                print(f"  🔍 Simple extraction from: {url}")
                session = requests.Session()
                session.headers.update({
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
                })
                
                response = session.get(url, timeout=15)
                response.raise_for_status()
                html = response.text
                
                # Find PDF links
                pdf_links = []
                pdf_matches = re.findall(r'href=["\']([^"\']*\.pdf)["\']', html, re.IGNORECASE)
                for link in pdf_matches:
                    if not link.startswith("http"):
                        from urllib.parse import urljoin
                        link = urljoin(url, link)
                    pdf_links.append(link)
                
                if pdf_links:
                    log["selected_from"] = "simple_extraction"
                    log["candidates"].extend(pdf_links[:3])
                    return {"url": pdf_links[0], "log": log}
            except Exception:
                continue

    # If still no PDFs, try the first candidate (might be a page with PDF links)
    if candidates:
        log["selected_from"] = "first_candidate"
        return {"url": candidates[0], "log": log}
    
    return {"url": None, "log": log}

# --- Validation helpers ---

def extract_first_pages_text(pdf_path: Path, max_pages: int = 2, max_chars: int = 1000) -> str:
    """Extract up to max_pages of text (capped to max_chars) from a PDF."""
    from PyPDF2 import PdfReader

    text_parts: List[str] = []
    try:
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages[:max_pages]):
            try:
                text = page.extract_text() or ""
                text_parts.append(text)
                if sum(len(t) for t in text_parts) >= max_chars:
                    break
            except Exception:
                continue
    except Exception:
        return ""
    text = "\n".join(text_parts)
    return text[:max_chars]

def render_first_page_image(pdf_path: Path, dpi: int = 150) -> Optional[Path]:
    """Render the first page of a PDF to a PNG image and return its path."""
    try:
        doc = fitz.open(str(pdf_path))
        if doc.page_count == 0:
            return None
        page = doc.load_page(0)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_path = pdf_path.with_suffix(".page1.png")
        pix.save(str(img_path))
        return img_path
    except Exception:
        return None

def is_english_text(text: str) -> dict:
    """Return a result dict indicating whether the given text is English."""
    try:
        # Clean the text for better detection
        cleaned = re.sub(r'[^\w\s]', ' ', text)
        cleaned = ' '.join(cleaned.split())
        
        if len(cleaned) < 10:
            return {"ok": False, "reason": "Text too short for language detection", "method": "langdetect"}
        
        detected_lang = detect(cleaned)
        is_english = detected_lang == 'en'
        
        return {
            "ok": is_english, 
            "reason": f"Detected language: {detected_lang}", 
            "method": "langdetect"
        }
    except Exception as e:
        return {"ok": False, "reason": f"Language detection failed: {e}", "method": "langdetect"}

def looks_like_official_rulebook(image_path: Path, game_name: str, config_dict: dict) -> dict:
    """Use a VLM to judge if the page image resembles a rulebook page."""
    try:
        vlm_model_id, vlm_provider = get_official_vlm_config(config_dict)
        
        # Simple, direct prompt with explicit format instruction
        prompt = f"You are a board game rulebook classifier. Look at this image and determine if it shows a board game rulebook page. Only respond with one word: 'YES' or 'NO'."
        
        # Use VLM to analyze the image
        response = classify_image_with_vlm(image_path, prompt, vlm_model_id, vlm_provider)
        
        # Simple parsing - look for YES or NO
        response_upper = response.upper().strip()
        is_rulebook = "YES" in response_upper
        
        return {
            "ok": is_rulebook,
            "reason": f"VLM response: {response[:100]}...",
            "method": "vlm_vision"
        }
    except Exception as e:
        return {
            "ok": False,
            "reason": f"VLM failed: {e}",
            "method": "vlm_vision"
        }

def get_official_vlm_config(config: dict) -> tuple[str, Optional[str]]:
    """Return (model_id, provider) for the VLM used in officialness checks."""
    entry = config.get("official_vlm")
    if not entry or not entry.get("model_id"):
        raise RuntimeError("model_config.json missing official_vlm.model_id")
    return entry["model_id"], entry.get("provider")

def get_exploration_llm_config(config: dict) -> tuple[str, Optional[str]]:
    """Return (model_id, provider) for the LLM used to guide navigation."""
    entry = config.get("exploration_llm")
    if not entry or not entry.get("model_id"):
        # Fallback to text_language model
        entry = config.get("text_language")
    if not entry or not entry.get("model_id"):
        raise RuntimeError("model_config.json missing exploration_llm.model_id")
    return entry["model_id"], entry.get("provider")

def generate_text(prompt: str, model_id: str, provider: Optional[str]) -> str:
    """Call a chat LLM with the prompt and return the raw string result."""
    if not model_id:
        raise RuntimeError("LLM model_id is required")
    provider = provider or ""
    if provider.lower() == "local":
        raise RuntimeError("Local LLM provider not implemented. Set exploration_llm.provider to a remote provider (e.g., auto).")
    from smolagents import InferenceClientModel

    model = InferenceClientModel(model_id=model_id, provider=provider)
    result = model.generate([{"role": "user", "content": prompt}])
    return str(result)

def classify_image_with_vlm(image_path: Path, prompt: str, model_id: str, provider: Optional[str]) -> str:
    """Call a VLM on the image + prompt and return the raw string result."""
    if not model_id:
        raise RuntimeError("VLM model_id is required")
    provider = provider or ""
    if provider.lower() == "local":
        raise RuntimeError("Local VLM provider not implemented. Set official_vlm.provider to a remote provider (e.g., together).")
    from smolagents import InferenceClientModel

    model = InferenceClientModel(model_id=model_id, provider=provider)
    result = model.generate([{"role": "user", "content": f"{prompt}\nImage path: {image_path}"}])
    return str(result)

# --- Helper function for workflow ---

def do_query_games(ev: Event) -> GamesFound:
    """Load games from the DB based on input rank range and return an event."""
    get = getattr(ev, "get", None)
    def _g(key: str, default=None):
        if get is not None:
            val = get(key)
            return default if val is None else val
        return getattr(ev, key, default)

    db_path = Path(_g("db_path", "bgg_games.db"))
    rank_from = int(_g("rank_from", 1))
    rank_to = int(_g("rank_to", 5))
    limit = _g("limit", None)
    default_out = Path(__file__).resolve().parent / "rulebooks"
    out_dir = str(_g("out_dir", str(default_out)))
    games = query_games_by_rank(db_path, rank_from, rank_to, limit)
    return GamesFound(games=games, out_dir=out_dir)