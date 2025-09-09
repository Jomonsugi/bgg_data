"""
Comprehensive toolset for LlamaIndex workflow to find official rulebooks.

All tools are available to the workflow agent, which can pick and choose
based on success patterns learned through the Context object.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import urllib.parse

from bgg_data.database.operations import BGGDatabase
from bgg_data.models import Game

# Optional imports with fallbacks
try:
    from llama_parse import LlamaParse
except Exception:
    LlamaParse = None  # type: ignore

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except Exception:
    SELENIUM_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---- DB Query Tool ----

def query_games(db_path: Path, rank_from: Optional[int], rank_to: Optional[int]) -> List[Game]:
    """Query games from the local SQLite DB using shared DB operations.

    Returns a list of `Game` with URL synthesized if missing.
    """
    db = BGGDatabase(db_path)
    return db.get_games(rank_from=rank_from, rank_to=rank_to)


# ---- Tavily Web Search Tool ----

@dataclass
class TavilyResult:
    url: str
    title: Optional[str] = None
    score: Optional[float] = None


def tavily_search(query: str, max_results: int = 5) -> List[TavilyResult]:
    """Search using Tavily official client (proven working from dag implementation)."""
    # Check for mock mode for testing
    if os.environ.get("MOCK_TAVILY") == "1":
        logger.info(f"Mock Tavily search for: {query}")
        # Return realistic mock results for testing based on query type
        if "pdf" in query.lower():
            # For PDF-specific queries, return direct PDF links
            mock_results = [
                TavilyResult(url="https://cdn.example.com/games/brass-birmingham-rules.pdf", title="Brass Birmingham Official Rules PDF"),
                TavilyResult(url="https://files.boardgame.com/rules/brass_rules_v2.pdf", title="Brass Birmingham Rulebook"),
            ]
        else:
            # For website queries, return publisher/official sites
            mock_results = [
                TavilyResult(url="https://roxleygames.com/brass-birmingham", title="Brass Birmingham - Roxley Games"),
                TavilyResult(url="https://boardgamegeek.com/boardgame/224517/brass-birmingham", title="BGG - Brass Birmingham"),
            ]
        return mock_results[:max_results]
    
    try:
        from tavily import TavilyClient
    except ImportError:
        logger.warning("tavily-python not available; returning empty list")
        return []
    
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set; tavily_search returning empty list")
        return []
    
    try:
        client = TavilyClient(api_key=api_key)
        resp = client.search(query, max_results=max_results, include_answer=False)
        results = []
        for item in resp.get("results", [])[:max_results]:
            url = item.get("url") or ""
            title = item.get("title") or item.get("url") or ""
            if url and url.startswith("http"):
                results.append(TavilyResult(url=url, title=title))
        return results
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return []


# ---- BGG Official Links Tool ----

def extract_official_link_from_bgg(game_bgg_url: str) -> Optional[str]:
    """Extract an official website link from a BGG game page.

    Parses the "Official Links" section when available.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        resp = requests.get(game_bgg_url, headers=headers, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # BGG structure can vary; look for link sections titled Official
        link_sections = soup.find_all("div", class_="game-page__links")
        for section in link_sections:
            header = section.find(["h2", "h3"]) or section
            if header and "Official" in header.get_text(" ", strip=True):
                anchors = section.find_all("a", href=True)
                for a in anchors:
                    href = a["href"].strip()
                    if href.startswith("http"):
                        return href
        # Fallback: scan all anchor text for 'Official'
        for a in soup.find_all("a", href=True):
            text = a.get_text(" ", strip=True)
            if "Official" in text:
                href = a["href"].strip()
                if href.startswith("http"):
                    return href
    except Exception as e:
        logger.warning(f"Failed to parse BGG official link: {e}")
    return None


# ---- LlamaParse PDF Assessment Tool ----

def assess_pdf_official_llamaparse(pdf_path: Path, game_name: str) -> Tuple[bool, bool, str]:
    """Use LlamaParse (Llama Cloud) to extract text and heuristically assess
    whether the PDF is an official English rulebook.

    Returns: (is_official_like, is_english, rationale)

    Note: This is a simplified local implementation that does not call the SDK directly.
    Users must set LLAMA_CLOUD_API_KEY if they want to wire up the real client later.
    """
    # Prefer real llama-parse if available and API key is set
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
    if LlamaParse and api_key:
        try:
            # Per official examples, use parser to load and get text
            parser = LlamaParse(api_key=api_key)
            docs = parser.load_data(str(pdf_path))
            # `docs` is a list of PartitionedElements/doc objects; concatenate text
            text = "\n".join(getattr(d, "text", "") for d in docs)
            lower = text.lower()
            is_english = any(k in lower for k in ["setup", "components", "rules", "game", "player"]) and not any(
                k in lower for k in ["regel", "regelwerk", "reglas", "règles", "regole", "regeln"]
            )
            official_signals = ["©", "copyright", "all rights reserved", "published by", "official rules"]
            is_official_like = any(sig.lower() in lower for sig in official_signals)
            rationale = "llama-parse signals: " + ", ".join(sig for sig in official_signals if sig.lower() in lower)
            return is_official_like, is_english, rationale
        except Exception as e:
            logger.warning(f"llama-parse failed, falling back to heuristic: {e}")

    # Fallback: heuristic on PDF bytes only
    try:
        with open(pdf_path, "rb") as f:
            head = f.read(200_000)
        text_sample = head.decode("latin-1", errors="ignore")
        lower = text_sample.lower()
        is_english = any(k in lower for k in ["setup", "components", "rules", "game", "player"]) and not any(
            k in lower for k in ["regel", "regelwerk", "reglas", "règles", "regole", "regeln"]
        )
        official_signals = ["©", "copyright", "all rights reserved", "published by", "official rules"]
        is_official_like = any(s in lower for s in (sig.lower() for sig in official_signals))
        rationale = "Heuristic signals: " + ", ".join(sig for sig in official_signals if sig.lower() in lower)
        return is_official_like, is_english, rationale
    except Exception as e:
        return False, False, f"assessment_failed: {e}"


# ---- Simple Browser Probe Tool (HTTP only) ----

def probe_direct_pdf(url: str, referer: Optional[str] = None, timeout: int = 20, max_retries: int = 0) -> Tuple[bool, Optional[bytes]]:
    """Attempt to fetch a PDF directly, following redirects. Returns (success, content).

    Retries are intentionally kept low to avoid long blocking loops on bad hosts.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/pdf, text/html;q=0.9,*/*;q=0.8",
        }
        if referer:
            headers["Referer"] = referer
            headers["Origin"] = referer
        with requests.Session() as s:
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            retry_cfg = Retry(total=max_retries, backoff_factor=0.4, status_forcelist=[429, 500, 502, 503, 504])
            s.mount("http://", HTTPAdapter(max_retries=retry_cfg))
            s.mount("https://", HTTPAdapter(max_retries=retry_cfg))
            r = s.get(url, timeout=timeout, allow_redirects=True, headers=headers, stream=True)
            r.raise_for_status()
            content = r.content
            if content and content[:4] == b"%PDF":
                return True, content
            return False, None
    except Exception as e:
        # Name resolution / DNS errors should abort quickly
        logger.warning(f"probe_direct_pdf failed for {url}: {e}")
        return False, None


def likely_non_english_url(url: str) -> bool:
    """Heuristic: return True if URL likely points to a non-English document."""
    try:
        path = urllib.parse.urlparse(url).path.lower()
        filename = path.split('/')[-1]
        markers = ["/de/", "/fr/", "/es/", "/it/", "-de-", "_de.", "-fr-", "_fr.", "-es-", "_es.", "-it-", "_it."]
        return any(m in path or m in filename for m in markers)
    except Exception:
        return False


# ---- Comprehensive Selenium Tools ----

class SeleniumAgent:
    """
    Selenium-based agent implementing the proven strategy from web_search_agent.py
    """
    
    def __init__(self, headless: bool = True, timeout: int = 10):
        self.driver = None
        self.headless = headless
        self.timeout = timeout
        self.session = requests.Session()
        self._setup_session()
    
    def _setup_session(self):
        """Setup requests session with retry logic"""
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf, text/html;q=0.9,*/*;q=0.8'
        })
        
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def __enter__(self):
        self.start_driver()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.quit()
    
    def start_driver(self):
        """Initialize Chrome driver"""
        if self.driver or not SELENIUM_AVAILABLE:
            return
        
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--force-device-scale-factor=1")
        chrome_options.add_argument("--window-size=1200,1000")
        chrome_options.add_argument("--disable-pdf-viewer")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        except Exception as e:
            logger.error(f"Failed to start Chrome driver: {e}")
            raise
    
    def quit(self):
        """Clean shutdown"""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
        if hasattr(self, 'session'):
            self.session.close()
    
    def close_popups(self):
        """Close popups using proven strategies"""
        try:
            webdriver.ActionChains(self.driver).send_keys(Keys.ESCAPE).perform()
            time.sleep(0.5)
            
            close_selectors = [
                "button[aria-label*='close']", "button[aria-label*='Close']",
                ".close-button", ".modal-close", "[data-dismiss='modal']", ".popup-close"
            ]
            
            for selector in close_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if element.is_displayed() and element.is_enabled():
                            element.click()
                            time.sleep(0.3)
                            break
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"Popup closing failed: {e}")
    
    def search_duckduckgo(self, query: str) -> List[str]:
        """Search DuckDuckGo and return URLs"""
        try:
            search_url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}"
            self.driver.get(search_url)
            time.sleep(2)
            
            # Handle consent
            try:
                consent_button = WebDriverWait(self.driver, 3).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept')]"))
                )
                consent_button.click()
                time.sleep(1)
            except TimeoutException:
                pass
            
            self.close_popups()
            
            # Extract results
            results = []
            result_elements = self.driver.find_elements(By.CSS_SELECTOR, "[data-result] h2 a")
            for element in result_elements[:5]:
                href = element.get_attribute("href")
                if href and href.startswith("http"):
                    results.append(href)
            
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []
    
    def navigate_to_url(self, url: str) -> bool:
        """Navigate to URL with error handling"""
        try:
            logger.info(f"Navigating to: {url}")
            self.driver.get(url)
            time.sleep(2)
            self.close_popups()
            return True
        except Exception as e:
            logger.warning(f"Failed to navigate to {url}: {e}")
            return False
    
    def find_pdf_links_on_page(self) -> List[str]:
        """Find PDF links on current page"""
        pdf_links = set()
        
        try:
            # Direct PDF links
            pdf_elements = self.driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf')]")
            for element in pdf_elements:
                href = element.get_attribute("href")
                if href:
                    pdf_links.add(self._resolve_url(href))
            
            # Links with PDF-related text
            pdf_text_patterns = ["rulebook", "rules", "manual", "instructions", "download", "PDF", "guide"]
            
            for pattern in pdf_text_patterns:
                elements = self.driver.find_elements(
                    By.XPATH, 
                    f"//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{pattern.lower()}')]"
                )
                for element in elements:
                    href = element.get_attribute("href")
                    if href:
                        resolved_url = self._resolve_url(href)
                        if any(keyword in resolved_url.lower() for keyword in ["pdf", "download", "file", "doc"]):
                            pdf_links.add(resolved_url)
            
            # Look in page source for PDF URLs
            page_source = self.driver.page_source
            pdf_urls_in_source = re.findall(r'https?://[^\s<>"\']+\.pdf', page_source, re.IGNORECASE)
            for url in pdf_urls_in_source:
                pdf_links.add(url)
                
        except Exception as e:
            logger.warning(f"Error finding PDF links: {e}")
        
        return list(pdf_links)
    
    def _resolve_url(self, url: str) -> str:
        """Resolve relative URLs to absolute"""
        if url.startswith("http"):
            return url
        return urljoin(self.driver.current_url, url)
    
    def download_pdf(self, url: str, filename: str, download_dir: Path) -> Tuple[bool, Optional[str]]:
        """Download PDF with retry logic from web_search_agent.py"""
        try:
            download_dir.mkdir(parents=True, exist_ok=True)
            
            # Handle special URL patterns
            original_url = url
            if 'dropbox.com' in url and ('?dl=0' in url or '&dl=0' in url):
                url = url.replace('?dl=0', '?dl=1').replace('&dl=0', '&dl=1')
            
            if 'drive.google.com' in url and '/file/d/' in url:
                match = re.search(r"/file/d/([a-zA-Z0-9_-]+)/", url)
                if match:
                    file_id = match.group(1)
                    url = f"https://drive.usercontent.google.com/uc?id={file_id}&export=download"
            
            if not filename.lower().endswith('.pdf'):
                filename += '.pdf'
            
            # Download with retries
            success = False
            content = None
            
            for attempt in range(3):
                try:
                    response = self.session.get(url, timeout=30, stream=True)
                    response.raise_for_status()
                    content = response.content
                    
                    if content and len(content) > 0:
                        success = True
                        break
                except Exception as e:
                    logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        time.sleep(1.0 * (2 ** attempt))
            
            # Try with browser context
            if not success and self.driver:
                try:
                    cookies = self.driver.get_cookies()
                    for cookie in cookies:
                        self.session.cookies.set(
                            cookie['name'], cookie['value'],
                            domain=cookie.get('domain'), path=cookie.get('path', '/')
                        )
                    
                    headers = {"Referer": self.driver.current_url}
                    response = self.session.get(url, headers=headers, timeout=30, stream=True)
                    response.raise_for_status()
                    content = response.content
                    
                    if content and len(content) > 0:
                        success = True
                except Exception as e:
                    logger.warning(f"Browser context download failed: {e}")
            
            if not success or not content:
                return False, None
            
            # Validate PDF
            if not content.strip()[:4] == b'%PDF':
                logger.warning("Downloaded content is not a valid PDF")
                return False, None
            
            # Save file
            file_path = download_dir / filename
            with open(file_path, 'wb') as f:
                f.write(content)
            
            logger.info(f"Successfully saved PDF: {file_path}")
            return True, str(file_path)
            
        except Exception as e:
            logger.error(f"PDF download failed: {e}")
            return False, None


def selenium_comprehensive_search(game_name: str, download_dir: Path, headless: bool = True) -> Tuple[bool, Optional[str], str]:
    """
    Comprehensive Selenium-based search implementing the proven web_search_agent.py strategy.
    
    This tool replicates the successful approach:
    1. Search for "game official website" (not "rulebook")
    2. Navigate to results and find PDF links
    3. Download and validate PDFs
    
    Returns: (success, file_path, method_used)
    """
    # Check for demo mode to show successful workflow
    if os.environ.get("DEMO_SUCCESS") == "1":
        logger.info(f"DEMO MODE: Creating mock rulebook for {game_name}")
        download_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple mock PDF
        filename = f"{game_name.replace(' ', '_').replace(':', '')}_rulebook_demo.pdf"
        file_path = download_dir / filename
        
        # Create a minimal valid PDF
        mock_pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Official Rulebook for """ + game_name.encode() + b""") Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000010 00000 n 
0000000053 00000 n 
0000000110 00000 n 
0000000181 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
275
%%EOF"""
        
        with open(file_path, 'wb') as f:
            f.write(mock_pdf_content)
        
        logger.info(f"Created demo rulebook: {file_path}")
        return True, str(file_path), "demo_success"
    
    if not SELENIUM_AVAILABLE:
        logger.warning("Selenium not available, skipping comprehensive search")
        return False, None, "selenium_not_available"
    
    try:
        with SeleniumAgent(headless=headless) as agent:
            # Step 1: Search for official website (proven strategy)
            query = f"{game_name} official website"
            logger.info(f"Selenium comprehensive search: {query}")
            
            search_results = agent.search_duckduckgo(query)
            if not search_results:
                return False, None, "no_search_results"
            
            # Step 2: Check each result for PDF links
            for i, url in enumerate(search_results[:3]):
                logger.info(f"Checking search result {i+1}: {url}")
                
                if not agent.navigate_to_url(url):
                    continue
                
                pdf_links = agent.find_pdf_links_on_page()
                logger.info(f"Found {len(pdf_links)} potential PDF links")
                
                # Step 3: Try to download each PDF
                for j, pdf_url in enumerate(pdf_links[:3]):
                    filename = f"{game_name.replace(' ', '_').replace(':', '')}_rulebook_selenium_{i+1}_{j+1}.pdf"
                    
                    success, file_path = agent.download_pdf(pdf_url, filename, download_dir)
                    if success:
                        return True, file_path, f"selenium_comprehensive_result_{i+1}_pdf_{j+1}"
            
            return False, None, "no_valid_pdfs_found"
            
    except Exception as e:
        logger.error(f"Selenium comprehensive search failed: {e}")
        return False, None, f"selenium_error: {str(e)}"


def selenium_simple_probe(url: str, base_url: Optional[str] = None) -> List[str]:
    """
    Simple Selenium probe to find PDF links on a specific page.
    Used as a lighter-weight tool when you already have a target URL.
    
    Returns: List of PDF URLs found on the page
    """
    if not SELENIUM_AVAILABLE:
        logger.warning("Selenium not available")
        return []
    
    try:
        with SeleniumAgent(headless=True) as agent:
            if not agent.navigate_to_url(url):
                return []
            
            return agent.find_pdf_links_on_page()
            
    except Exception as e:
        logger.error(f"Selenium simple probe failed for {url}: {e}")
        return []


