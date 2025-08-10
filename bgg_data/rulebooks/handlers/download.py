"""
Download handler for rulebook PDFs in the LLM-based rulebook fetcher.
"""

import requests
import logging
from pathlib import Path
from typing import Optional, Tuple
import time

from ..utils import validate_url, create_rulebook_filename, log_download_attempt, is_likely_english
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class DownloadHandler:
    """
    Handles downloading and saving rulebook PDFs.
    """
    
    def __init__(self, rulebooks_dir: Path, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the download handler.
        
        Args:
            rulebooks_dir: Directory to save rulebooks
            max_retries: Maximum number of download retries
            retry_delay: Delay between retries in seconds
        """
        self.rulebooks_dir = rulebooks_dir
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Ensure rulebooks directory exists
        self.rulebooks_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up session for better performance
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def download_rulebook(self, url: str, game_name: str, save_screenshots: bool = False, screenshot_path: Optional[Path] = None, web_handler=None) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Download a rulebook PDF from a URL.
        
        Args:
            url: URL of the rulebook to download
            game_name: Name of the game (for filename creation)
            save_screenshots: Whether to save screenshots for debugging
            screenshot_path: Path to save screenshot if save_screenshots is True
            
        Returns:
            Tuple of (success, filename_or_error, file_path_or_error)
        """
        try:
            logger.info(f"Attempting to download rulebook for '{game_name}' from {url}")
            
            # Convert Dropbox sharing links to direct download links
            if 'dropbox.com' in url and ('?dl=0' in url or '&dl=0' in url):
                url = url.replace('?dl=0', '?dl=1').replace('&dl=0', '&dl=1')
                logger.info(f"Converted Dropbox link to direct download: {url}")
            
            # Validate URL first
            if not validate_url(url):
                error_msg = f"Invalid or inaccessible URL: {url}"
                log_download_attempt(game_name, url, False, error_msg)
                return False, error_msg, None
            
            # Create filename using game name for consistent naming (may be updated later if URL changes)
            filename = create_rulebook_filename(game_name, url)
            file_path = self.rulebooks_dir / filename
            
            # If similarly-named files already exist, prefer existing PDF; do NOT early-return on existing HTML
            safe_base = game_name.replace(' ', '-').replace(':', '').replace("'", '')
            existing_pdf = self.rulebooks_dir / f"{safe_base}_rules.pdf"
            existing_html = self.rulebooks_dir / f"{safe_base}_rules.html"
            if existing_pdf.exists():
                logger.info(f"Rulebook already exists: {existing_pdf}")
                return True, existing_pdf.name, str(existing_pdf)
            # If computed target path already exists and is a PDF, return it
            if file_path.exists() and str(file_path).lower().endswith('.pdf'):
                logger.info(f"Rulebook already exists: {file_path}")
                return True, file_path.name, str(file_path)
            # If only HTML exists, continue to attempt fetching a PDF; keep track for fallback
            existing_html_path = existing_html if existing_html.exists() else None
            
            # Download the file with retries
            success, content = self._download_with_retries(url)
            if not success:
                # Try browser-based download if web_handler is available
                if web_handler:
                    logger.info("Direct download failed, trying browser-based download...")
                    browser_success, browser_content = self._download_with_browser(url, web_handler)
                    if browser_success:
                        success, content = True, browser_content
                    else:
                        error_msg = f"Failed to download after {self.max_retries} retries (including browser fallback)"
                        log_download_attempt(game_name, url, False, error_msg)
                        return False, error_msg, None
                else:
                    error_msg = f"Failed to download after {self.max_retries} retries"
                    log_download_attempt(game_name, url, False, error_msg)
                    return False, error_msg, None
            
            # Validate content: prefer PDFs; if HTML, try to extract a PDF link
            content_type = ''
            try:
                head = self.session.head(url, timeout=10, allow_redirects=True)
                content_type = head.headers.get('content-type', '').lower()
            except Exception:
                pass

            is_pdf_like = self._is_valid_pdf(content)
            if not is_pdf_like:
                # If HTML page, attempt to find a PDF link within and download that instead
                looks_like_html = ('html' in content_type) or (content.strip()[:15].lower().startswith(b'<!doctype html') or content.strip().lower().startswith(b'<html'))
                if looks_like_html:
                    pdf_from_html = self._extract_pdf_link_from_html_content(content, url)
                    if pdf_from_html:
                        logger.info(f"Found PDF link inside HTML page, switching to: {pdf_from_html}")
                        # Attempt direct PDF download from discovered link
                        success_pdf, pdf_bytes = self._download_with_retries(pdf_from_html)
                        if success_pdf and pdf_bytes and self._is_valid_pdf(pdf_bytes):
                            # Override original url and content with PDF
                            url = pdf_from_html
                            content = pdf_bytes
                            is_pdf_like = True
                            logger.info("Successfully fetched PDF via HTML indirection")
                            # Since we now have a PDF, we should not treat this as HTML anymore
                            looks_like_html = False
                            # Recompute filename and path now that URL is a PDF
                            new_filename = create_rulebook_filename(game_name, url)
                            if new_filename != filename:
                                filename = new_filename
                                file_path = self.rulebooks_dir / filename
                        else:
                            logger.warning("Failed to download valid PDF from link found in HTML; will save HTML as fallback")

                # Language guard: only apply when we STILL only have HTML (no PDF found)
                if (not is_pdf_like) and looks_like_html and not is_likely_english(url):
                    logger.warning("HTML page appears non-English; not saving as primary rulebook")
                    # If we previously had an HTML saved copy, return that as existing instead of failing
                    if existing_html_path and existing_html_path.exists():
                        return True, existing_html_path.name, str(existing_html_path)
                    return False, "Non-English HTML page", None
                
            if not is_pdf_like and not ('html' in content_type or (content.strip()[:15].lower().startswith(b'<!doctype html') or content.strip().lower().startswith(b'<html'))):
                error_msg = "Downloaded content is neither valid PDF nor HTML"
                # Fallback to existing HTML if present
                if existing_html_path and existing_html_path.exists():
                    logger.info("Falling back to existing HTML rulebook copy")
                    return True, existing_html_path.name, str(existing_html_path)
                log_download_attempt(game_name, url, False, error_msg)
                return False, error_msg, None
            
            # Save the file (use .pdf for PDFs, .html for HTML)
            try:
                if is_pdf_like:
                    with open(file_path, 'wb') as f:
                        f.write(content)
                else:
                    # ensure .html extension
                    if not str(file_path).lower().endswith('.html'):
                        file_path = file_path.with_suffix('.html')
                    with open(file_path, 'wb') as f:
                        f.write(content)
                
                # Verify file was saved correctly
                if file_path.exists() and file_path.stat().st_size > 0:
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    logger.info(f"Rulebook saved successfully: {file_path} ({file_size_mb:.2f}MB)")
                    
                    # Log successful download
                    log_download_attempt(game_name, url, True)
                    
                    # Save screenshot if requested
                    if save_screenshots and screenshot_path:
                        self._save_screenshot_for_debugging(screenshot_path, game_name)
                    
                    return True, filename, str(file_path)
                else:
                    error_msg = "File was not saved correctly"
                    log_download_attempt(game_name, url, False, error_msg)
                    return False, error_msg, None
                    
            except Exception as e:
                error_msg = f"Error saving file: {e}"
                log_download_attempt(game_name, url, False, error_msg)
                return False, error_msg, None
                
        except Exception as e:
            error_msg = f"Unexpected error during download: {e}"
            log_download_attempt(game_name, url, False, error_msg)
            return False, error_msg, None

    def _extract_pdf_link_from_html_content(self, html_content: bytes, base_url: str) -> Optional[str]:
        """
        Parse HTML content to find a likely PDF rulebook link.
        Returns an absolute URL if found.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            candidates = []
            html_str = html_content.decode('utf-8', errors='ignore')
            rule_keywords = [
                'rulebook', 'rules', 'manual', 'guide', 'reference', 'learn to play', 'how to play', 'playbook', 'instructions'
            ]
            for a in soup.find_all('a', href=True):
                href = a.get('href') or ''
                text = (a.get_text(strip=True) or '').lower()
                href_l = href.lower()
                # Accept direct PDFs
                if '.pdf' in href_l:
                    score = 0
                    if any(k in text for k in rule_keywords):
                        score += 10
                    if any(k in href_l for k in rule_keywords):
                        score += 8
                    # Prefer files that look like rulebook-ish names
                    for strong in ['rulebook', 'rules', 'reference']:
                        if strong in href_l:
                            score += 5
                    candidates.append((score, urljoin(base_url, href)))

                # Accept Google Drive direct download links without .pdf
                if ('export=download' in href_l) or ('drive.usercontent.google.com/uc' in href_l):
                    score = 18
                    if any(k in text for k in rule_keywords):
                        score += 6
                    candidates.append((score, urljoin(base_url, href)))

            # Look for Google Drive IDs and construct direct download link
            m = re.search(r"https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)/", html_str)
            if m:
                file_id = m.group(1)
                gdrive_direct = f"https://drive.usercontent.google.com/uc?id={file_id}&export=download"
                candidates.append((22, gdrive_direct))

            # Regex for embedded direct usercontent link in scripts
            m2 = re.search(r"https?://drive\.usercontent\.google\.com/uc\?id=[^\"'&\s]+&export=download", html_str)
            if m2:
                candidates.append((24, m2.group(0)))

            # Look for JavaScript redirects (common in search engine redirect pages)
            js_redirect = re.search(r'(?:window\.location\.(?:href|replace)\s*=\s*|var\s+u\s*=\s*)["\']([^"\']+\.pdf)["\']', html_str, re.IGNORECASE)
            if js_redirect:
                pdf_url = js_redirect.group(1)
                if pdf_url.startswith('http'):
                    candidates.append((25, pdf_url))
                else:
                    candidates.append((25, urljoin(base_url, pdf_url)))

            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                return candidates[0][1]
            return None
        except Exception:
            return None
    
    def _download_with_retries(self, url: str) -> Tuple[bool, Optional[bytes]]:
        """
        Download content with retry logic.
        
        Args:
            url: URL to download
            
        Returns:
            Tuple of (success, content_or_none)
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Download attempt {attempt + 1}/{self.max_retries}")
                
                response = self.session.get(url, timeout=30, stream=True)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' not in content_type and 'application/octet-stream' not in content_type:
                    logger.warning(f"Unexpected content type: {content_type}")
                
                # Download content
                content = response.content
                
                if content and len(content) > 0:
                    logger.info(f"Download successful: {len(content)} bytes")
                    return True, content
                else:
                    logger.warning("Download returned empty content")
                    return False, None
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All download attempts failed for {url}")
                    return False, None
            except Exception as e:
                logger.error(f"Unexpected error during download attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return False, None
        
        return False, None
    
    def _is_valid_pdf(self, content: bytes) -> bool:
        """
        Check if downloaded content is a valid PDF.
        
        Args:
            content: Downloaded content bytes
            
        Returns:
            True if content appears to be a valid PDF
        """
        try:
            # Check PDF magic number
            if len(content) < 4:
                return False
            
            # PDF files start with "%PDF"
            if content[:4] != b'%PDF':
                logger.warning("Content does not start with PDF magic number")
                return False
            
            # Check file size (should be reasonable for a PDF)
            size_mb = len(content) / (1024 * 1024)
            if size_mb < 0.01:  # Less than 10KB
                logger.warning(f"File too small to be a valid PDF: {size_mb:.2f}MB")
                return False
            
            if size_mb > 100:  # More than 100MB
                logger.warning(f"File suspiciously large for a PDF: {size_mb:.2f}MB")
                # Don't reject, just warn
            
            logger.info(f"Content appears to be a valid PDF ({size_mb:.2f}MB)")
            return True
            
        except Exception as e:
            logger.warning(f"Error validating PDF content: {e}")
            return False
    
    def _save_screenshot_for_debugging(self, screenshot_path: Path, game_name: str) -> None:
        """
        Save screenshot for debugging purposes.
        
        Args:
            screenshot_path: Path to the screenshot
            game_name: Name of the game
        """
        try:
            if screenshot_path and screenshot_path.exists():
                # Create debug directory
                debug_dir = self.rulebooks_dir / "debug_screenshots"
                debug_dir.mkdir(exist_ok=True)
                
                # Create debug filename
                debug_filename = f"{game_name.replace(' ', '_')}_screenshot.png"
                debug_path = debug_dir / debug_filename
                
                # Copy screenshot to debug directory
                import shutil
                shutil.copy2(screenshot_path, debug_path)
                logger.info(f"Debug screenshot saved: {debug_path}")
                
        except Exception as e:
            logger.warning(f"Could not save debug screenshot: {e}")
    
    def _download_with_browser(self, url: str, web_handler) -> Tuple[bool, Optional[bytes]]:
        """
        Download a file using the browser when direct requests fail.
        
        Args:
            url: URL to download
            web_handler: WebPageHandler instance with active browser
            
        Returns:
            Tuple of (success, content_bytes)
        """
        try:
            logger.info(f"Attempting browser-based download from: {url}")
            
            # Navigate to the URL using the existing browser
            if not web_handler.navigate_to_page(url):
                logger.warning("Failed to navigate to download URL with browser")
                return False, None
            
            # Get the page content
            page_source = web_handler.driver.page_source
            
            # Check if we got a PDF by looking at the content
            if page_source.startswith('%PDF'):
                # Direct PDF content
                content_bytes = page_source.encode('latin1')  # Preserve binary data
                logger.info(f"Browser downloaded PDF content: {len(content_bytes)} bytes")
                return True, content_bytes
            
            # Try to get the response using browser's network capabilities
            try:
                # Use browser to download the file content
                content_bytes = web_handler.driver.execute_script("""
                    return new Promise((resolve) => {
                        fetch(arguments[0])
                            .then(response => response.arrayBuffer())
                            .then(buffer => {
                                const bytes = new Uint8Array(buffer);
                                resolve(Array.from(bytes));
                            })
                            .catch(() => resolve(null));
                    });
                """, url)
                
                if content_bytes:
                    content_bytes = bytes(content_bytes)
                    logger.info(f"Browser fetch downloaded: {len(content_bytes)} bytes")
                    return True, content_bytes
                    
            except Exception as e:
                logger.debug(f"Browser fetch failed: {e}")
            
            logger.warning("Browser-based download did not retrieve PDF content")
            return False, None
            
        except Exception as e:
            logger.error(f"Error in browser-based download: {e}")
            return False, None
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            self.session.close()
            logger.info("Download session closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
