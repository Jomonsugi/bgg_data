"""
Agentic rulebook fetcher using LangGraph to orchestrate the overall flow.

This wraps the existing handlers and fallback logic in a clear, inspectable
graph so the order of operations and fallbacks are explicit and extensible.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, List, TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END

from .handlers import (
    WebPageHandler,
    LLMHandler,
    DownloadHandler,
    WebSearchFallback,
    FallbackOrchestrator,
    FallbackContext,
)
from ..models import Game, FetchResult
from ..config import RULEBOOKS_DIR
from .utils import is_rulebook_already_downloaded


logger = logging.getLogger(__name__)


class FetchState(TypedDict, total=False):
    game: Game
    official_website: Optional[str]
    pdf_url: Optional[str]
    last_error: Optional[str]
    result: Optional[FetchResult]
    save_screenshots: bool
    attempts: int
    max_attempts: int
    # Accumulated processing time
    total_time_s: float


class AgenticRulebookFetcher:
    """Agentic orchestrator driven by LangGraph."""

    def __init__(self, rulebooks_dir: Path = RULEBOOKS_DIR, save_screenshots: bool = False):
        self.rulebooks_dir = rulebooks_dir
        self.save_screenshots = save_screenshots
        self.rulebooks_dir.mkdir(parents=True, exist_ok=True)

        # Handlers reused across nodes
        self.llm_handler = LLMHandler()
        self.download_handler = DownloadHandler(self.rulebooks_dir)
        self.web_search = WebSearchFallback()
        self.fallback_orchestrator = FallbackOrchestrator(self.web_search)

        # Pre-test LLM connectivity (non-fatal)
        try:
            if not self.llm_handler.test_connection():
                logger.warning("LLM handler connection test failed; proceeding anyway")
        except Exception:
            logger.warning("LLM handler test_connection raised; proceeding anyway")

        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(FetchState)

        def extract_official_from_bgg(state: FetchState) -> FetchState:
            game = state["game"]
            with WebPageHandler() as web:
                if not web.navigate_to_page(game.url):
                    return {"last_error": "Failed to navigate to BGG page"}
                official = web.extract_official_website_from_bgg()
                if official:
                    return {"official_website": official}
                return {"official_website": None}

        def try_official_html(state: FetchState) -> FetchState:
            game = state["game"]
            official = state.get("official_website")
            if not official:
                return {}
            with WebPageHandler() as web:
                if not web.navigate_to_page(official):
                    return {}
                if self.save_screenshots:
                    web.take_screenshot(game_name=game.name, site_type="official")
                pdf_url = web.quick_html_check()
                if not pdf_url:
                    return {}
                success, filename, file_path = self.download_handler.download_rulebook(
                    pdf_url, game.name, self.save_screenshots, web_handler=web
                )
                if not success:
                    return {}
                # Verify with LLM if PDF/HTML
                try:
                    if file_path and (file_path.lower().endswith('.pdf') or file_path.lower().endswith('.html')):
                        is_official, is_english, _ = self.llm_handler.assess_file_official_rulebook(
                            Path(file_path), game.name
                        )
                        if not (is_official and is_english):
                            return {}
                except Exception:
                    # If verification fails, accept the download to avoid blocking
                    pass
                return {
                    "pdf_url": pdf_url,
                    "result": FetchResult(
                        game_name=game.name,
                        success=True,
                        rulebook_url=pdf_url,
                        filename=filename,
                        file_path=file_path,
                        method="html_check",
                    ),
                }

        def try_official_llm(state: FetchState) -> FetchState:
            game = state["game"]
            official = state.get("official_website")
            if not official:
                return {}
            with WebPageHandler() as web:
                if not web.navigate_to_page(official):
                    return {}
                web.prepare_page_for_rulebook()
                ok, screenshot = web.take_screenshot(game_name=game.name, site_type="official_llm")
                if not ok or not screenshot:
                    return {}
                success, url, _conf = self.llm_handler.extract_rulebook_url(screenshot, game.name)
                if not success or not url:
                    return {}
                dl_success, filename, file_path = self.download_handler.download_rulebook(
                    url, game.name, self.save_screenshots, web_handler=web
                )
                if not dl_success:
                    return {}
                # Optional verification
                try:
                    if file_path and (file_path.lower().endswith('.pdf') or file_path.lower().endswith('.html')):
                        is_official, is_english, _ = self.llm_handler.assess_file_official_rulebook(
                            Path(file_path), game.name
                        )
                        if not (is_official and is_english):
                            return {}
                except Exception:
                    pass
                return {
                    "pdf_url": url,
                    "result": FetchResult(
                        game_name=game.name,
                        success=True,
                        rulebook_url=url,
                        filename=filename,
                        file_path=file_path,
                        method="llm_vision",
                    ),
                }

        def try_agentic_websearch(state: FetchState) -> FetchState:
            game = state["game"]
            # Reuse our existing fallback orchestrator which now uses Tavily underneath
            with WebPageHandler() as web:
                ctx = FallbackContext(
                    game=game,
                    web_handler=web,
                    llm_handler=self.llm_handler,
                    download_handler=self.download_handler,
                    save_screenshots=self.save_screenshots,
                    rulebooks_dir=self.rulebooks_dir,
                )
                res = self.fallback_orchestrator.execute_fallback_strategy(ctx)
                if res and res.success:
                    return {"result": res}
                return {}

        def is_done(state: FetchState) -> str:
            if state.get("result"):
                return END
            return "try_agentic_websearch"

        graph.add_node("extract_official_from_bgg", extract_official_from_bgg)
        graph.add_node("try_official_html", try_official_html)
        graph.add_node("try_official_llm", try_official_llm)
        graph.add_node("try_agentic_websearch", try_agentic_websearch)

        graph.add_edge("extract_official_from_bgg", "try_official_html")
        graph.add_edge("try_official_html", "try_official_llm")
        graph.add_conditional_edges("try_official_llm", is_done, {END: END, "try_agentic_websearch": "try_agentic_websearch"})
        graph.add_edge("try_agentic_websearch", END)

        graph.set_entry_point("extract_official_from_bgg")
        return graph.compile()

    def fetch_rulebooks_for_games(self, games: List[Game], delay_between_games: float = 2.0) -> List[FetchResult]:
        results: List[FetchResult] = []
        for i, game in enumerate(games):
            logger.info(f"[Agentic] Processing game {i+1}/{len(games)}: {game.name}")
            start = time.time()

            # Skip if rulebook already exists
            try:
                if is_rulebook_already_downloaded(game.name, self.rulebooks_dir):
                    elapsed = time.time() - start
                    results.append(FetchResult(
                        game_name=game.name,
                        success=True,
                        method="already_exists",
                        error_message="Rulebook already downloaded",
                        processing_time=elapsed,
                    ))
                    if i < len(games) - 1 and delay_between_games > 0:
                        time.sleep(delay_between_games)
                    continue
            except Exception:
                # If check fails, continue with normal flow
                pass
            state: FetchState = {
                "game": game,
                "official_website": None,
                "pdf_url": None,
                "save_screenshots": self.save_screenshots,
                "attempts": 0,
                "max_attempts": 1,
                "total_time_s": 0.0,
            }
            final: FetchState = self.graph.invoke(state)
            elapsed = time.time() - start
            res = final.get("result") or FetchResult(
                game_name=game.name,
                success=False,
                method="error",
                error_message="Agentic flow failed to find rulebook",
            )
            res.processing_time = elapsed
            results.append(res)
            if i < len(games) - 1 and delay_between_games > 0:
                time.sleep(delay_between_games)
        return results


