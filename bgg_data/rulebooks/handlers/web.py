"""
Web page handling and screenshot capture for the LLM-based rulebook fetcher.
"""

import time
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from PIL import Image
import io
import base64
import json

from ...config import HEADLESS_BROWSER, BROWSER_TIMEOUT, SCREENSHOT_DELAY, RULEBOOK_SELECTORS

logger = logging.getLogger(__name__)

class WebPageHandler:
    """
    Handles web page automation, interaction, and screenshot capture.
    """
    
    def __init__(self, headless: bool = HEADLESS_BROWSER, timeout: int = BROWSER_TIMEOUT):
        """
        Initialize the web page handler.
        
        Args:
            headless: Whether to run browser in headless mode
            timeout: Browser timeout in seconds
        """
        self.headless = headless
        self.timeout = timeout
        self.driver = None
        self.wait = None
        
    def setup_driver(self) -> None:
        """Set up the Chrome WebDriver with appropriate options."""
        try:
            chrome_options = Options()
            
            if self.headless:
                chrome_options.add_argument("--headless")
            
            # Add options for better performance and stability
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            
            # Anti-bot detection options
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            chrome_options.add_argument("--disable-images")  # Faster loading
            
            # Enable performance logging to capture network events
            try:
                chrome_options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
            except Exception:
                pass
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(self.timeout)
            self.wait = WebDriverWait(self.driver, self.timeout)
            try:
                # Enable CDP Network domain for response body access
                self.driver.execute_cdp_cmd("Network.enable", {})
            except Exception:
                pass
            
            logger.info("Chrome WebDriver initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Chrome WebDriver: {e}")
            raise
    
    def navigate_to_page(self, url: str) -> bool:
        """
        Navigate to a specific URL.
        
        Args:
            url: URL to navigate to
            
        Returns:
            True if navigation successful, False otherwise
        """
        try:
            logger.info(f"Navigating to: {url}")
            self.driver.get(url)
            
            # Wait for page to load
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            # Execute stealth script to hide automation markers
            try:
                self.driver.execute_script("""
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined,
                    });
                    
                    // Mock plugins
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => [1, 2, 3, 4, 5],
                    });
                    
                    // Mock languages
                    Object.defineProperty(navigator, 'languages', {
                        get: () => ['en-US', 'en'],
                    });
                """)
            except Exception:
                pass
            
            # Additional delay for dynamic content
            time.sleep(SCREENSHOT_DELAY)
            
            # Attempt to dismiss all popups immediately after load
            try:
                if self.dismiss_all_popups():
                    logger.info("Popups dismissed after navigation")
            except Exception:
                pass

            logger.info("Page loaded successfully")
            return True
            
        except TimeoutException:
            logger.warning(f"Page load timeout for {url}")
            return False
        except Exception as e:
            logger.error(f"Error navigating to {url}: {e}")
            return False

    def capture_pdf_via_network(self, wait_seconds: float = 3.0) -> Tuple[Optional[str], Optional[bytes]]:
        """
        Inspect performance logs and CDP to capture any PDF response and body.
        Returns (url, content_bytes) if a PDF response was observed.
        """
        try:
            end_time = time.time() + max(0.5, wait_seconds)
            seen_ids = set()
            while time.time() < end_time:
                try:
                    entries = self.driver.get_log('performance')
                except Exception:
                    time.sleep(0.2)
                    continue
                for entry in entries:
                    try:
                        msg = json.loads(entry.get('message', '{}')).get('message', {})
                        if msg.get('method') != 'Network.responseReceived':
                            continue
                        params = msg.get('params', {})
                        response = params.get('response', {})
                        mime = (response.get('mimeType') or '').lower()
                        if 'pdf' not in mime:
                            continue
                        request_id = params.get('requestId')
                        if not request_id or request_id in seen_ids:
                            continue
                        seen_ids.add(request_id)
                        url = response.get('url') or ''
                        # Fetch response body via CDP
                        try:
                            body = self.driver.execute_cdp_cmd('Network.getResponseBody', {'requestId': request_id})
                            data = body.get('body')
                            if data is None:
                                continue
                            if body.get('base64Encoded'):
                                content_bytes = base64.b64decode(data)
                            else:
                                content_bytes = data.encode('latin1')
                            if content_bytes and content_bytes[:4] == b'%PDF':
                                logger.info(f"Captured PDF via network: {url} ({len(content_bytes)} bytes)")
                                return url, content_bytes
                        except Exception as e:
                            logger.debug(f"CDP getResponseBody failed: {e}")
                    except Exception:
                        continue
                time.sleep(0.2)
        except Exception as e:
            logger.debug(f"Network capture error: {e}")
        return None, None

    def dismiss_all_popups(self) -> bool:
        """
        Try to dismiss all types of popups including cookie banners, modals, overlays, etc.
        Returns True if any dismissal was attempted.
        """
        dismissed = False
        try:
            # First dismiss cookie banners
            if self.dismiss_common_cookie_banners():
                dismissed = True
            
            # Then dismiss other types of popups
            if self.dismiss_modal_popups():
                dismissed = True
                
            if self.dismiss_overlay_popups():
                dismissed = True
                
            return dismissed
        except Exception as e:
            logger.debug(f"Error dismissing popups: {e}")
            return dismissed
    
    def dismiss_common_cookie_banners(self) -> bool:
        """
        Try to dismiss cookie consent banners/popups.
        Returns True if a dismissal was attempted.
        """
        dismissed = False
        try:
            # Some consent managers render inside iframes; try within each iframe too
            def try_dismiss_in_context() -> bool:
                local_dismissed = False
                # Generic reject/decline/deny/disagree buttons (button or anchor)
                xpaths = [
                    "//button[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'reject')]",
                    "//button[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'reject all')]",
                    "//button[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'decline')]",
                    "//button[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'deny')]",
                    "//button[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'disagree')]",
                    "//button[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'use necessary')]",
                    "//button[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'necessary only')]",
                    "//a[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'reject')]",
                    "//a[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'decline')]",
                    "//a[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'deny')]",
                    "//a[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'disagree')]",
                    # Exact phrase commonly used on Fantasy Flight Games
                    "//button[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'disagree and close')]",
                ]
                for xp in xpaths:
                    try:
                        elements = self.driver.find_elements(By.XPATH, xp)
                        for el in elements:
                            if el.is_displayed() and el.is_enabled():
                                el.click()
                                time.sleep(0.5)
                                logger.info("Dismissed cookie banner via reject option")
                                return True
                    except Exception:
                        pass

                # OneTrust common selector
                try:
                    reject_all = self.driver.find_elements(By.CSS_SELECTOR, "#onetrust-reject-all-handler, button#onetrust-reject-all-handler")
                    for el in reject_all:
                        if el.is_displayed() and el.is_enabled():
                            el.click()
                            time.sleep(0.5)
                            logger.info("Dismissed cookie banner via OneTrust reject all")
                            return True
                except Exception:
                    pass

                # Didomi common selector
                try:
                    didomi_buttons = self.driver.find_elements(By.CSS_SELECTOR, "button.didomi-components-button--secondary, .didomi-continue-without-agreeing")
                    for el in didomi_buttons:
                        text = (el.text or "").lower()
                        if el.is_displayed() and el.is_enabled() and any(t in text for t in ["reject", "decline", "deny", "disagree", "without"]):
                            el.click()
                            time.sleep(0.5)
                            logger.info("Dismissed cookie banner via Didomi")
                            return True
                except Exception:
                    pass

                return local_dismissed

            # Generic reject/decline/deny/disagree buttons (button or anchor)
            # First try in the top-level document
            if try_dismiss_in_context():
                return True

            # Then try inside likely consent iframes
            try:
                iframes = self.driver.find_elements(By.CSS_SELECTOR, "iframe[id*='consent'], iframe[src*='consent'], iframe[id*='sp_message_iframe'], iframe[id*='ot-sdk'], iframe[id*='onetrust']")
                original = self.driver.switch_to
                for frame in iframes:
                    try:
                        self.driver.switch_to.frame(frame)
                        if try_dismiss_in_context():
                            # Return to default content before exiting
                            self.driver.switch_to.default_content()
                            return True
                    except Exception:
                        pass
                    finally:
                        try:
                            self.driver.switch_to.default_content()
                        except Exception:
                            pass
            except Exception:
                pass

            return dismissed
        except Exception as e:
            logger.debug(f"Error dismissing cookie banners: {e}")
            return dismissed
    
    def dismiss_modal_popups(self) -> bool:
        """
        Try to dismiss modal popups (newsletters, promotions, etc.).
        Returns True if a dismissal was attempted.
        """
        dismissed = False
        try:
            # Common modal close patterns
            close_selectors = [
                # Generic close buttons
                ".modal-close", ".close", ".close-button", ".btn-close",
                "[data-dismiss='modal']", "[data-bs-dismiss='modal']",
                # X buttons
                "button[aria-label='Close']", "button[title='Close']",
                ".fa-times", ".fa-close", ".icon-close", ".icon-x",
                # Modal overlay areas (clicking outside modal)
                ".modal-backdrop", ".modal-overlay", ".overlay",
                # Specific close text
                "button:contains('×')", "button:contains('Close')", 
                "button:contains('No Thanks')", "button:contains('Skip')",
                "a:contains('Close')", "a:contains('×')"
            ]
            
            for selector in close_selectors:
                try:
                    if ':contains(' in selector:
                        # Handle text-based selectors with XPath
                        text = selector.split(':contains(')[1].rstrip(')')
                        xpath = f"//button[contains(text(), {text})] | //a[contains(text(), {text})]"
                        elements = self.driver.find_elements(By.XPATH, xpath)
                    else:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    
                    for element in elements:
                        if element.is_displayed() and element.is_enabled():
                            element.click()
                            time.sleep(0.5)
                            logger.info(f"Dismissed modal via {selector}")
                            dismissed = True
                            break
                except Exception:
                    continue
            
            # Try pressing Escape key to close modals
            try:
                from selenium.webdriver.common.keys import Keys
                self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
                time.sleep(0.5)
                logger.info("Sent Escape key to dismiss modals")
                dismissed = True
            except Exception:
                pass
                
        except Exception as e:
            logger.debug(f"Error dismissing modal popups: {e}")
        
        return dismissed
    
    def dismiss_overlay_popups(self) -> bool:
        """
        Try to dismiss overlay popups, promotional banners, etc.
        Returns True if a dismissal was attempted.
        """
        dismissed = False
        try:
            # Common overlay patterns
            overlay_selectors = [
                # Generic overlays
                ".popup", ".popup-overlay", ".lightbox", ".dialog",
                ".notification", ".banner", ".promo-banner",
                ".newsletter-popup", ".email-signup", ".subscription-popup",
                # Age verification
                ".age-gate", ".age-verification", ".age-popup",
                # Survey/feedback
                ".survey-popup", ".feedback-popup", ".review-popup",
                # Mobile app promotion
                ".app-banner", ".mobile-app-banner", ".download-app",
                # Region/language selection
                ".region-popup", ".language-popup", ".country-selector",
                # GDPR/Privacy beyond cookies
                ".privacy-popup", ".gdpr-popup", ".consent-popup"
            ]
            
            for selector in overlay_selectors:
                try:
                    overlays = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for overlay in overlays:
                        if overlay.is_displayed():
                            # Try to find close button within overlay
                            close_buttons = overlay.find_elements(By.CSS_SELECTOR, 
                                ".close, .btn-close, [aria-label='Close'], button[title='Close']")
                            
                            for close_btn in close_buttons:
                                if close_btn.is_displayed() and close_btn.is_enabled():
                                    close_btn.click()
                                    time.sleep(0.5)
                                    logger.info(f"Dismissed overlay via close button in {selector}")
                                    dismissed = True
                                    break
                            
                            # If no close button found, try clicking the overlay itself to dismiss
                            if not dismissed:
                                try:
                                    overlay.click()
                                    time.sleep(0.5)
                                    logger.info(f"Dismissed overlay by clicking {selector}")
                                    dismissed = True
                                except Exception:
                                    pass
                                    
                except Exception:
                    continue
            
            # Try to dismiss elements with high z-index (likely popups)
            try:
                high_z_elements = self.driver.execute_script("""
                    var elements = document.querySelectorAll('*');
                    var highZElements = [];
                    for (var i = 0; i < elements.length; i++) {
                        var zIndex = window.getComputedStyle(elements[i]).zIndex;
                        if (zIndex && parseInt(zIndex) > 1000) {
                            highZElements.push(elements[i]);
                        }
                    }
                    return highZElements;
                """)
                
                for element in high_z_elements[:3]:  # Limit to first 3 to avoid issues
                    try:
                        if element.is_displayed():
                            # Look for close button in high z-index element
                            close_buttons = element.find_elements(By.CSS_SELECTOR, 
                                ".close, .btn-close, [aria-label='Close']")
                            for close_btn in close_buttons:
                                if close_btn.is_displayed() and close_btn.is_enabled():
                                    close_btn.click()
                                    time.sleep(0.5)
                                    logger.info("Dismissed high z-index popup")
                                    dismissed = True
                                    break
                    except Exception:
                        continue
                        
            except Exception:
                pass
                
        except Exception as e:
            logger.debug(f"Error dismissing overlay popups: {e}")
        
        return dismissed
    
    def prepare_page_for_rulebook(self) -> bool:
        """
        Prepare the page by clicking common elements that might reveal rulebook downloads.
        
        Returns:
            True if any preparation was done, False otherwise
        """
        try:
            logger.info("Preparing page for rulebook search...")
            
            # Try to find and click common rulebook-related elements
            elements_clicked = False
            
            # Look for download links first
            download_links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='download'], a[href*='pdf'], a[href*='rulebook']")
            for link in download_links:
                try:
                    if link.is_displayed() and link.is_enabled():
                        link.click()
                        time.sleep(1)
                        elements_clicked = True
                        logger.info(f"Clicked download link: {link.text or link.get_attribute('href')}")
                except Exception as e:
                    logger.debug(f"Could not click download link: {e}")
            
            # Look for buttons or sections that might contain rulebook information
            button_selectors = [
                "button:contains('Downloads')",
                "button:contains('Files')", 
                "button:contains('Rules')",
                "button:contains('Documentation')"
            ]
            
            for selector in button_selectors:
                try:
                    # Try to find buttons by text content
                    buttons = self.driver.find_elements(By.XPATH, f"//button[contains(text(), '{selector.split(':contains(')[1].rstrip(')')}')]")
                    for button in buttons:
                        if button.is_displayed() and button.is_enabled():
                            button.click()
                            time.sleep(1)
                            elements_clicked = True
                            logger.info(f"Clicked button: {button.text}")
                except Exception as e:
                    logger.debug(f"Could not click button with selector {selector}: {e}")
            
            # Look for expandable sections
            section_selectors = [".downloads", ".files", ".rules", ".documentation"]
            for selector in section_selectors:
                try:
                    sections = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for section in sections:
                        if section.is_displayed():
                            # Try to click if it's clickable
                            try:
                                section.click()
                                time.sleep(1)
                                elements_clicked = True
                                logger.info(f"Clicked section: {selector}")
                            except:
                                # If not clickable, just note it's visible
                                logger.debug(f"Found visible section: {selector}")
                except Exception as e:
                    logger.debug(f"Could not interact with section {selector}: {e}")
            
            if elements_clicked:
                # Wait for any dynamic content to load
                time.sleep(2)
                logger.info("Page preparation completed")
            else:
                logger.info("No interactive elements found during preparation")

            # Attempt to scroll to reveal rulebook-related links (generalized)
            try:
                found_links = []
                last_height = 0
                for _ in range(30):
                    links = self.driver.find_elements(
                        By.XPATH,
                        "|".join([
                            "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'rulebook')]",
                            "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'rules')]",
                            "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'manual')]",
                            "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'guide')]",
                            "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'how to play')]",
                            "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'learn to play')]",
                            "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'download')]",
                            "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'support')]",
                            "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'resources')]",
                            # Common site pattern like "Download the <Game> board game rules PDF"
                            "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'rules pdf')]",
                            "//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'board game rules pdf')]",
                            # Icon-only buttons/links where aria-label or title contains download/rules
                            "//*[@aria-label and contains(translate(@aria-label, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'download')]",
                            "//*[@title and contains(translate(@title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'download')]",
                        ])
                    )
                    visible = [l for l in links if l.is_displayed()]
                    if visible:
                        found_links = visible
                        logger.info("Found rulebook-related link text on page")
                        break
                    self.driver.execute_script("window.scrollBy(0, Math.floor(window.innerHeight*0.9));")
                    time.sleep(0.5)
                    # Stop if no further scroll growth (bottom reached)
                    try:
                        current_height = int(self.driver.execute_script("return document.body.scrollHeight"))
                        if current_height <= last_height:
                            break
                        last_height = current_height
                    except Exception:
                        pass

                if found_links:
                    try:
                        link = found_links[0]
                        if link.is_enabled():
                            link.click()
                            time.sleep(1)
                            elements_clicked = True
                            logger.info("Clicked rulebook-related link after scrolling")
                    except Exception as e:
                        logger.debug(f"Could not click rulebook link: {e}")
            except Exception:
                pass
            
            return elements_clicked
            
        except Exception as e:
            logger.warning(f"Error during page preparation: {e}")
            return False
    
    def take_screenshot(self, save_path: Optional[Path] = None, game_name: str = "", site_type: str = "bgg") -> Tuple[bool, Optional[bytes]]:
        """
        Take a screenshot of the current page.
        
        Args:
            save_path: Optional path to save the screenshot
            game_name: Name of the game for filename
            site_type: Type of site ("bgg" or "official")
            
        Returns:
            Tuple of (success, image_bytes)
        """
        try:
            logger.info("Taking page screenshot...")

            # Try to capture a full-page screenshot by resizing window to full height
            try:
                total_height = int(self.driver.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight, document.body.offsetHeight, document.documentElement.offsetHeight, document.body.clientHeight, document.documentElement.clientHeight)"))
                self.driver.set_window_size(1920, max(1080, min(total_height, 8000)))
                time.sleep(0.2)
            except Exception:
                pass

            # Capture screenshot (viewport or full page depending on support)
            screenshot_bytes = self.driver.get_screenshot_as_png()
            
            # If save path is provided, save the screenshot
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'wb') as f:
                    f.write(screenshot_bytes)
                logger.info(f"Screenshot saved to: {save_path}")
            
            # Also save to screenshots/<game_name>/ directory if game_name is provided
            if game_name:
                from pathlib import Path
                screenshots_root = Path("screenshots")
                screenshots_root.mkdir(exist_ok=True)

                # Create game-specific subdirectory
                safe_name = game_name.replace(' ', '_').replace(':', '').replace("'", '')
                game_dir = screenshots_root / safe_name
                game_dir.mkdir(parents=True, exist_ok=True)

                # Create filename with timestamp
                import time
                timestamp = int(time.time())
                filename = f"{safe_name}_{site_type}_{timestamp}.png"
                screenshot_path = game_dir / filename

                with open(screenshot_path, 'wb') as f:
                    f.write(screenshot_bytes)
                logger.info(f"Screenshot saved: {screenshot_path}")
            
            logger.info(f"Screenshot captured successfully ({len(screenshot_bytes)} bytes)")
            return True, screenshot_bytes
            
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return False, None
    
    def extract_official_website_from_bgg(self) -> Optional[str]:
        """
        Extract the official website URL from a BGG game page.
        
        Returns:
            Official website URL if found, None otherwise
        """
        try:
            logger.info("Extracting official website from BGG page...")
            
            # Look for embedded JavaScript data containing official website
            script_tags = self.driver.find_elements(By.TAG_NAME, "script")
            for script in script_tags:
                script_content = script.get_attribute('innerHTML') or ""
                if 'GEEK.geekitemPreload' in script_content:
                    # Use regex to find the website URL
                    import re
                    website_match = re.search(r'"website":\s*{\s*"url":\s*"([^"]+)"', script_content)
                    if website_match:
                        website_url = website_match.group(1).replace('\\/', '/')
                        logger.info(f"Found official website: {website_url}")
                        return website_url
            
            # Fallback: look for external links that might be official websites
            external_links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='http']")
            for link in external_links:
                href = link.get_attribute('href')
                text = link.text.lower()
                if href and any(word in text for word in ['website', 'official', 'homepage', 'home']):
                    logger.info(f"Found potential official website via link text: {href}")
                    return href
            
            logger.info("No official website found on BGG page")
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting official website: {e}")
            return None

    def quick_html_check(self) -> Optional[str]:
        """
        Perform a quick HTML check for obvious PDF links before using LLM.
        
        Returns:
            PDF URL if found, None otherwise
        """
        try:
            logger.info("Performing quick HTML check for rulebook links...")
            # Ensure all popups are closed before scanning
            self.dismiss_all_popups()

            indicators = [
                'rulebook', 'rules', 'manual', 'instructions', 'guide',
                'learn to play', 'how to play', 'reference', 'rules reference',
                'rule reference', 'playbook', 'download'
            ]
            english_hints = ['english', 'en', 'us', 'uk']
            non_english_hints = ['de', 'german', 'fr', 'french', 'es', 'spanish', 'it', 'italian', 'pt', 'portuguese', 'ru', 'russian', 'pl', 'polish', 'zh', 'cn', 'jp', 'ja', 'japanese', 'ko', 'korean']

            def collect_pdf_candidates_in_context() -> list:
                local_candidates = []
                links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='.pdf']")
                for link in links:
                    href = (link.get_attribute('href') or '').strip()
                    text = (link.text or '').lower()
                    if not href:
                        continue
                    score = 0
                    href_l = href.lower()
                    if 'rulebook' in text or 'rulebook' in href_l:
                        score += 20
                    elif any(ind in text for ind in indicators) or any(ind in href_l for ind in indicators):
                        score += 10
                    if any(h in href_l for h in ['_en', '-en', '/en/']) or any(h in text for h in english_hints):
                        score += 15
                    if any(h in href_l for h in ['_de', '-de', '/de/', '_fr', '-fr', '/fr/', '_es', '-es', '/es/']) or any(h in text for h in non_english_hints):
                        score -= 12
                    for strong in ['rules-reference', 'rules_reference', 'reference', 'quick-start', 'quickstart', 'learn-to-play', 'how-to-play']:
                        if strong in href_l:
                            score -= 5
                    local_candidates.append((score, href))
                return local_candidates

            candidates = collect_pdf_candidates_in_context()

            # Also consider icon-only or JS-triggered downloads
            icon_candidates = self.driver.find_elements(By.XPATH,
                "|".join([
                    "//*[@aria-label and contains(translate(@aria-label, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'download')]",
                    "//*[@title and contains(translate(@title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'download')]",
                    "//button[contains(translate(@aria-label, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'download')]",
                    "//button[contains(translate(@title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'download')]",
                ])
            )
            for el in icon_candidates:
                # Attempt to extract a URL from common attributes or onclick handlers
                for attr in ["href", "data-href", "data-url", "data-file", "data-download", "onclick"]:
                    val = (el.get_attribute(attr) or '').strip()
                    if not val:
                        continue
                    val_l = val.lower()
                    if '.pdf' in val_l:
                        candidates.append((12, val))
                        break

            # Parse inline onclick for window.open('...pdf')
            onclick_links = self.driver.find_elements(By.XPATH, "//*[@onclick]")
            for el in onclick_links:
                onclick = (el.get_attribute('onclick') or '').lower()
                if '.pdf' in onclick:
                    # crude extraction
                    import re
                    m = re.search(r"(https?:[^'\"]+\.pdf)", onclick)
                    if m:
                        candidates.append((12, m.group(1)))

            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                best = candidates[0][1]
                logger.info(f"Found rulebook PDF via HTML check (best-scored): {best}")
                return best

            # Scan inside iframes as well
            try:
                frames = self.driver.find_elements(By.CSS_SELECTOR, "iframe")
                for frame in frames[:5]:
                    try:
                        self.driver.switch_to.frame(frame)
                        inner = collect_pdf_candidates_in_context()
                        if inner:
                            inner.sort(key=lambda x: x[0], reverse=True)
                            best_inner = inner[0][1]
                            logger.info(f"Found rulebook PDF inside iframe via HTML check: {best_inner}")
                            return best_inner
                    except Exception:
                        pass
                    finally:
                        try:
                            self.driver.switch_to.default_content()
                        except Exception:
                            pass
            except Exception:
                pass

            # Fallback: any anchor whose text suggests rules/rulebook (can be HTML page)
            xpath_text = " or ".join([
                f"contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{q}')" for q in indicators
            ])
            candidate_links = self.driver.find_elements(By.XPATH, f"//a[{xpath_text}]")
            for link in candidate_links:
                href = (link.get_attribute('href') or '').strip()
                if href:
                    logger.info(f"Found candidate rulebook page via HTML check: {href}")
                    return href

            logger.info("No rulebook-related links found via quick HTML check")
            return None
            
        except Exception as e:
            logger.warning(f"Error during HTML check: {e}")
            return None
    
    def collect_candidate_links(self, max_candidates: int = 6) -> List[Dict[str, str]]:
        """
        Collect top candidate links/buttons that might lead to a rulebook.
        Returns a list of dicts: {"text": ..., "href": ...}
        """
        try:
            indicators = [
                'rulebook', 'rules', 'manual', 'instructions', 'guide',
                'learn to play', 'how to play', 'reference', 'rules reference',
                'rule reference', 'playbook', 'download', 'support', 'resources'
            ]
            english_hints = ['english', 'en', 'us', 'uk']
            non_english_hints = ['de', 'german', 'fr', 'french', 'es', 'spanish', 'it', 'italian', 'pt', 'portuguese', 'ru', 'russian', 'pl', 'polish', 'zh', 'cn', 'jp', 'ja', 'japanese', 'ko', 'korean']

            def score_link(text: str, href: str) -> int:
                t = (text or '').lower()
                h = (href or '').lower()
                s = 0
                if '.pdf' in h:
                    s += 25
                if 'rulebook' in t or 'rulebook' in h:
                    s += 20
                elif any(ind in t for ind in indicators) or any(ind in h for ind in indicators):
                    s += 10
                if any(hh in h for hh in ['_en', '-en', '/en/']) or any(hh in t for hh in english_hints):
                    s += 12
                if any(hh in h for hh in ['_de', '-de', '/de/', '_fr', '-fr', '/fr/', '_es', '-es', '/es/']) or any(hh in t for hh in non_english_hints):
                    s -= 10
                for demote in ['quickstart', 'quick-start', 'reference', 'faq']:
                    if demote in h:
                        s -= 5
                return s

            seen = set()
            cands: List[Tuple[int, str, str]] = []
            # anchors
            for a in self.driver.find_elements(By.CSS_SELECTOR, "a[href]"):
                try:
                    href = (a.get_attribute('href') or '').strip()
                    if not href or href in seen:
                        continue
                    seen.add(href)
                    text = (a.text or '').strip()
                    cands.append((score_link(text, href), text, href))
                except Exception:
                    continue
            # buttons with data-url/data-href
            for b in self.driver.find_elements(By.CSS_SELECTOR, "button, [role='button']"):
                try:
                    href = ''
                    for attr in ['data-url', 'data-href', 'data-download', 'onclick']:
                        val = (b.get_attribute(attr) or '').strip()
                        if val and ('http' in val or '.pdf' in val):
                            href = val
                            break
                    if not href or href in seen:
                        continue
                    seen.add(href)
                    text = (b.text or b.get_attribute('aria-label') or b.get_attribute('title') or '').strip()
                    cands.append((score_link(text, href), text, href))
                except Exception:
                    continue
            # sort and return top
            cands.sort(key=lambda x: x[0], reverse=True)
            top = []
            for s, t, h in cands[:max_candidates * 2]:
                if len(top) >= max_candidates:
                    break
                top.append({"text": t, "href": h})
            return top
        except Exception:
            return []

    def take_element_screenshots_for_candidates(self, candidates: List[Dict[str, str]], max_images: int = 2) -> List[bytes]:
        """
        Capture screenshots focused on the top candidate link/button elements.
        Returns a list of PNG bytes for up to max_images candidates.
        """
        shots: List[bytes] = []
        if not candidates:
            return shots
        try:
            for cand in candidates[: max_images * 2]:
                href = (cand.get('href') or '').strip()
                if not href:
                    continue
                el = None
                # Try exact anchor match
                try:
                    el = self.driver.find_element(By.CSS_SELECTOR, f"a[href='{href}']")
                except Exception:
                    el = None
                # Try partial match when exact fails
                if el is None:
                    try:
                        el = self.driver.find_element(By.XPATH, f"//a[contains(@href, '{href[:40]}')]")
                    except Exception:
                        el = None
                # Try button-like elements with data-url
                if el is None:
                    try:
                        el = self.driver.find_element(By.XPATH, f"//*[@data-url='{href}' or contains(@onclick, '{href}')]")
                    except Exception:
                        el = None
                if el is None:
                    continue
                try:
                    # Scroll into view and add a small margin around it using JS
                    self.driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", el)
                    time.sleep(0.2)
                    # Prefer element-native screenshot when supported
                    png = el.screenshot_as_png
                    if png:
                        shots.append(png)
                    if len(shots) >= max_images:
                        break
                except Exception:
                    continue
        except Exception:
            pass
        return shots
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("WebDriver closed successfully")
            except Exception as e:
                logger.warning(f"Error closing WebDriver: {e}")
            finally:
                self.driver = None
                self.wait = None
    
    def __enter__(self):
        """Context manager entry."""
        self.setup_driver()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
