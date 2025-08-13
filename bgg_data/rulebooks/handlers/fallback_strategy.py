"""
Fallback strategy orchestrator for rulebook fetching.
This module centralizes all fallback logic to reduce code duplication in the main fetcher.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import json

from .search import WebSearchFallback, SearchResult
from .web import WebPageHandler
from .llm import LLMHandler
from .download import DownloadHandler
from ...models import Game, FetchResult

logger = logging.getLogger(__name__)


@dataclass
class FallbackContext:
    """Context for fallback operations."""
    game: Game
    web_handler: WebPageHandler
    llm_handler: LLMHandler
    download_handler: DownloadHandler
    save_screenshots: bool
    rulebooks_dir: Path


class FallbackStrategy:
    """
    Orchestrates fallback strategies for rulebook fetching.
    Reduces code duplication and centralizes fallback logic.
    """
    
    def __init__(self, search_fallback: WebSearchFallback):
        self.search_fallback = search_fallback
    
    def try_official_website_strategy(self, context: FallbackContext) -> Optional[FetchResult]:
        """
        Try to find rulebooks by searching for the game's official website first.
        This is the primary fallback strategy.
        """
        logger.info(f"Trying official website strategy for '{context.game.name}'...")
        
        # Search for official website candidates
        candidates = self.search_fallback.search_official_website(
            context.game.name, 
            publisher=context.game.publisher
        )
        
        if not candidates:
            logger.warning(f"No official website candidates found for '{context.game.name}'")
            return None
        
        logger.info(f"Found {len(candidates)} potential official websites via web search")
        
        # Try each candidate website with the same LLM strategy
        for candidate in candidates:
            candidate_url = candidate.url
            candidate_score = candidate.score
            logger.info(f"Trying official website candidate (score: {candidate_score}): {candidate_url}")
            
            result = self._try_candidate_website(context, candidate)
            if result and result.success:
                return result
        
        logger.warning("All official website candidates failed")
        return None
    
    def try_direct_rulebook_search(self, context: FallbackContext) -> Optional[FetchResult]:
        """
        Try direct rulebook search as a last resort.
        This is the tertiary fallback strategy.
        """
        logger.info(f"Trying direct rulebook search for '{context.game.name}'...")
        
        candidates = self.search_fallback.search_official_rulebook(
            context.game.name, 
            publisher=context.game.publisher,
            relaxed=True
        )
        
        if not candidates:
            logger.warning(f"No rulebook candidates found for '{context.game.name}'")
            return None
        
        # Try top 3 candidates
        for candidate in candidates[:3]:
            url_cand = candidate.url
            logger.info(f"Trying direct rulebook search candidate: {url_cand}")
            
            success, filename, file_path = context.download_handler.download_rulebook(
                url_cand, context.game.name, context.save_screenshots, web_handler=context.web_handler, game_id=context.game.id
            )
            
            if success:
                # LLM verification: Check if the file is official English rulebook
                if self._verify_downloaded_file(context, file_path, url_cand):
                    # Record success for the domain
                    try:
                        # On success, promote domain into known_publishers map for this publisher
                        self.search_fallback.scorer.record_success(url_cand, context.game.publisher)
                    except Exception:
                        pass
                    return FetchResult(
                        game_name=context.game.name,
                        success=True,
                        rulebook_url=url_cand,
                        filename=filename,
                        file_path=file_path,
                        method="web_search_direct"
                    )
                else:
                    logger.warning(f"Downloaded file failed LLM verification, trying next candidate")
                    # Delete the unsuitable file
                    if file_path and Path(file_path).exists():
                        Path(file_path).unlink()
                    # Record failure for the domain
                    try:
                        self.search_fallback.scorer.record_failure(url_cand, context.game.publisher)
                    except Exception:
                        pass
                    continue

            # If candidate appears to be a BGG thread, open it and scan for PDF link once
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url_cand)
                if 'boardgamegeek.com' in parsed.netloc and '/thread/' in parsed.path:
                    if context.web_handler.navigate_to_page(url_cand):
                        pdf_url = context.web_handler.quick_html_check()
                        if pdf_url:
                            logger.info(f"Found PDF via BGG thread page: {pdf_url}")
                            success2, filename2, file_path2 = context.download_handler.download_rulebook(
                                pdf_url, context.game.name, context.save_screenshots, web_handler=context.web_handler, game_id=context.game.id
                            )
                            if success2 and self._verify_downloaded_file(context, file_path2, pdf_url):
                                return FetchResult(
                                    game_name=context.game.name,
                                    success=True,
                                    rulebook_url=pdf_url,
                                    filename=filename2,
                                    file_path=file_path2,
                                    method="web_search_bgg_thread"
                                )
            except Exception:
                pass
        
        logger.warning("All direct rulebook search candidates failed")
        return None
    
    def try_web_search_fallback(self, context: FallbackContext, original_url: str) -> Optional[FetchResult]:
        """
        Try web search fallback when the original strategy fails.
        This is used when we have a downloaded file but it fails LLM verification.
        """
        logger.info(f"Trying web search fallback for '{context.game.name}'...")
        
        candidates = self.search_fallback.search_official_rulebook(
            context.game.name, 
            publisher=context.game.publisher
        )
        
        for candidate in candidates:
            url_cand = candidate.url
            logger.info(f"Trying web search fallback candidate: {url_cand}")
            
            success, filename, file_path = context.download_handler.download_rulebook(
                url_cand, context.game.name, context.save_screenshots, web_handler=context.web_handler, game_id=context.game.id
            )
            
            if success:
                return FetchResult(
                    game_name=context.game.name,
                    success=True,
                    rulebook_url=url_cand,
                    filename=filename,
                    file_path=file_path,
                    method="web_search_fallback"
                )
        
        logger.warning("Web search fallback failed")
        return None
    
    def _try_candidate_website(self, context: FallbackContext, candidate: SearchResult) -> Optional[FetchResult]:
        """
        Try a single candidate website using the same strategy as official websites.
        """
        candidate_url = candidate.url
        
        # Navigate to the candidate website
        if not context.web_handler.navigate_to_page(candidate_url):
            logger.warning(f"Failed to navigate to candidate website: {candidate_url}")
            return None
        
        # Take screenshot for debugging if enabled
        if context.save_screenshots:
            context.web_handler.take_screenshot(
                game_name=context.game.name, 
                site_type="web_search_official"
            )
        
        # Try HTML check first (cost-effective)
        pdf_url = context.web_handler.quick_html_check()
        
        if pdf_url:
            logger.info(f"Found rulebook via HTML check on web search candidate: {pdf_url}")
            return self._handle_html_check_success(context, pdf_url, "web_search_official_website")
        
        # If HTML check failed, try LLM vision analysis
        logger.info("HTML check failed on web search candidate, trying LLM vision analysis...")
        # Attempt to capture direct PDF responses via network (covers JS/new-tab downloads)
        try:
            url_net, content = context.web_handler.capture_pdf_via_network(wait_seconds=3.0)
            if url_net and content and len(content) > 0:
                # Save through download handler path by short-circuiting retries
                success, filename, file_path = context.download_handler.download_rulebook(
                    url_net, context.game.name, context.save_screenshots, web_handler=context.web_handler
                )
                if success:
                    if self._verify_downloaded_file(context, file_path, url_net):
                        return FetchResult(
                            game_name=context.game.name,
                            success=True,
                            rulebook_url=url_net,
                            filename=filename,
                            file_path=file_path,
                            method="web_search_network_capture"
                        )
        except Exception:
            pass

        return self._try_llm_vision_on_candidate(context, candidate_url, "web_search_official_website_llm")
    
    def _try_llm_vision_on_candidate(self, context: FallbackContext, candidate_url: str, method: str) -> Optional[FetchResult]:
        """
        Try LLM vision analysis on a candidate website.
        """
        # Prepare page for rulebook (click buttons, expand sections)
        context.web_handler.prepare_page_for_rulebook()
        
        # Take screenshot for LLM analysis
        screenshot_success, screenshot_bytes = context.web_handler.take_screenshot(
            game_name=context.game.name, site_type="web_search_official_llm"
        )
        
        if not screenshot_success:
            logger.warning(f"Failed to capture screenshot for LLM analysis on candidate: {candidate_url}")
            return None
        
        # Use LLM to extract rulebook URL from candidate website
        cands = context.web_handler.collect_candidate_links(max_candidates=6)
        elem_shots = context.web_handler.take_element_screenshots_for_candidates(cands, max_images=2)
        llm_success, llm_result, confidence = context.llm_handler.extract_rulebook_url(
            screenshot_bytes, context.game.name, candidates=cands, extra_images=elem_shots
        )
        
        if not llm_success:
            logger.info(f"LLM vision failed on candidate: {candidate_url}")
            return None
        
        logger.info(f"LLM vision successful on web search candidate, found: {llm_result}")
        
        # Resolve relative URL if necessary
        try:
            from urllib.parse import urljoin
            current = context.web_handler.driver.current_url if hasattr(context.web_handler, 'driver') else candidate_url
            resolved_url = llm_result if (llm_result and llm_result.lower().startswith(('http://', 'https://'))) else urljoin(current, llm_result)
        except Exception:
            resolved_url = llm_result
        
        # Try to download the rulebook
        success, filename, file_path = context.download_handler.download_rulebook(
            resolved_url, context.game.name, context.save_screenshots, web_handler=context.web_handler, game_id=context.game.id
        )
        
        if not success:
            logger.warning(f"LLM found URL but download failed: {filename}")
            return None
        
        # Verify the downloaded file (require both official AND English)
        if self._verify_downloaded_file(context, file_path, llm_result):
            try:
                self.search_fallback.scorer.record_success(resolved_url)
            except Exception:
                pass
            return FetchResult(
                game_name=context.game.name,
                success=True,
                rulebook_url=resolved_url,
                filename=filename,
                file_path=file_path,
                method=method
            )
        
        try:
            self.search_fallback.scorer.record_failure(resolved_url)
        except Exception:
            pass
        return None
    
    def _handle_html_check_success(self, context: FallbackContext, pdf_url: str, method: str) -> Optional[FetchResult]:
        """
        Handle successful HTML check by downloading and verifying the file.
        """
        # Try to download
        success, filename, file_path = context.download_handler.download_rulebook(
            pdf_url, context.game.name, context.save_screenshots, web_handler=context.web_handler, game_id=context.game.id
        )
        
        if not success:
            logger.warning(f"HTML check found URL but download failed: {filename}")
            return None
        
        # Verify the downloaded file (require both official AND English)
        if self._verify_downloaded_file(context, file_path, pdf_url):
            return FetchResult(
                game_name=context.game.name,
                success=True,
                rulebook_url=pdf_url,
                filename=filename,
                file_path=file_path,
                method=method
            )
        # If HTML saved and not verified, remove it so we don't keep non-official HTML
        try:
            if file_path and Path(file_path).exists():
                Path(file_path).unlink()
        except Exception:
            pass
        return None
    
    def _verify_downloaded_file(self, context: FallbackContext, file_path: str, source_url: str) -> bool:
        """
        Verify that a downloaded file is an official English rulebook using LLM assessment.
        Captures debug information including screenshots when assessment fails.
        """
        try:
            if file_path and (file_path.lower().endswith('.pdf') or file_path.lower().endswith('.html')):
                is_official, is_english, rationale = context.llm_handler.assess_file_official_rulebook(
                    Path(file_path), context.game.name
                )
                logger.info(f"LLM assessment: is_official={is_official}, is_english={is_english}")

                # Policy: PDFs are accepted if English, even if "official" is uncertain; HTML remains strict
                if file_path.lower().endswith('.pdf'):
                    if is_english:
                        return True
                    # Not English → reject
                    logger.warning(f"English check failed for PDF; rejecting. Rationale: {rationale}")
                    if context.save_screenshots:
                        self._save_debug_screenshot(context, file_path, source_url, is_official, is_english, rationale)
                    return False

                # For HTML, require both signals
                if is_official and is_english:
                    return True

                logger.warning(f"Downloaded HTML failed LLM verification. Rationale: {rationale}")
                if context.save_screenshots:
                    self._save_debug_screenshot(context, file_path, source_url, is_official, is_english, rationale)
                return False
            else:
                # For non-PDF/HTML files, assume they're valid
                return True
        except Exception as e:
            logger.error(f"LLM assessment failed: {e}")
            # Be conservative: do NOT accept when LLM verification fails
            return False
    
    def _save_debug_screenshot(self, context: FallbackContext, file_path: str, source_url: str, 
                              is_official: bool, is_english: bool, rationale: str) -> None:
        """
        Save a debug screenshot when LLM assessment fails.
        This helps visualize what the LLM was assessing.
        """
        try:
            from datetime import datetime
            
            # Create debug directory if it doesn't exist
            debug_dir = Path("bgg_data/debug/llm_assessments")
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a unique filename for this debug screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_game_name = "".join(c for c in context.game.name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_game_name = safe_game_name.replace(' ', '_')
            debug_filename = f"{timestamp}_{safe_game_name}_failed_assessment.png"
            debug_path = debug_dir / debug_filename
            
            # Take a screenshot of the current page (if we're on a webpage)
            if context.web_handler:
                try:
                    screenshot_success, screenshot_bytes = context.web_handler.take_screenshot(
                        game_name=context.game.name, site_type="llm_assessment_failed"
                    )
                    
                    if screenshot_success and screenshot_bytes:
                        # Save the screenshot
                        with open(debug_path, 'wb') as f:
                            f.write(screenshot_bytes)
                        
                        # Save additional debug metadata
                        metadata_path = debug_path.with_suffix('.json')
                        metadata = {
                            "timestamp": timestamp,
                            "game_name": context.game.name,
                            "file_path": file_path,
                            "source_url": source_url,
                            "llm_assessment": {
                                "is_official": is_official,
                                "is_english": is_english,
                                "rationale": rationale
                            },
                            "screenshot_path": str(debug_path),
                            "current_page_url": context.web_handler.driver.current_url if hasattr(context.web_handler, 'driver') else "unknown"
                        }
                        
                        with open(metadata_path, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)
                        
                        logger.info(f"Saved debug screenshot to: {debug_path}")
                        logger.info(f"Saved debug metadata to: {metadata_path}")
                    else:
                        logger.warning("Failed to capture debug screenshot")
                        
                except Exception as e:
                    logger.error(f"Error capturing debug screenshot: {e}")
            else:
                logger.warning("No web handler available for debug screenshot")
                
        except Exception as e:
            logger.error(f"Failed to save debug screenshot: {e}")


class FallbackOrchestrator:
    """
    Main orchestrator for fallback strategies.
    Provides a clean interface for the main fetcher to use.
    """
    
    def __init__(self, search_fallback: WebSearchFallback):
        self.strategy = FallbackStrategy(search_fallback)
    
    def execute_fallback_strategy(self, context: FallbackContext, original_strategy_failed: bool = False) -> Optional[FetchResult]:
        """
        Execute the complete fallback strategy for a game.
        
        Args:
            context: Fallback context with all necessary handlers
            original_strategy_failed: Whether the original strategy already failed
            
        Returns:
            FetchResult if successful, None if all strategies failed
        """
        logger.info(f"Executing fallback strategy for '{context.game.name}'")
        
        # Strategy 1: Try to find official website via web search
        result = self.strategy.try_official_website_strategy(context)
        if result and result.success:
            return result
        
        # Strategy 2: Try direct rulebook search as last resort
        result = self.strategy.try_direct_rulebook_search(context)
        if result and result.success:
            return result
        
        # All strategies failed
        logger.error(f"All fallback strategies failed for '{context.game.name}'")
        return None
    
    def try_web_search_fallback(self, context: FallbackContext, original_url: str) -> Optional[FetchResult]:
        """
        Try web search fallback when original download fails verification.
        """
        return self.strategy.try_web_search_fallback(context, original_url)
