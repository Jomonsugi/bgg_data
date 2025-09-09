"""
LlamaIndex Workflow orchestration for rulebook discovery and download.

Root agent chooses tools based on Context and model strategy (e.g., local MLX).
Context is serialized per model strategy to retain learnings about which tools
work best for different publishers or sites.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from bgg_data.models import Game

from .tools import (
    query_games,
    tavily_search,
    extract_official_link_from_bgg,
    probe_direct_pdf,
    assess_pdf_official_llamaparse,  # Legacy
    agent_choose_pdf_assessment,     # Agentic choice
    likely_non_english_url,
    selenium_comprehensive_search,
    selenium_simple_probe,
)
from .agents import get_strategy_order
from .events import (
    ProcessNextGameEvent,
    PlanStrategiesEvent,
    StrategyNextEvent,
    TryBGGOfficialLinkEvent,
    TryTavilyPdfSearchEvent,
    TryWebsiteProbeEvent,
    TryComprehensiveSeleniumEvent,
    TryDirectPdfCandidateEvent,
    TryDirectPdfDownloadEvent,
)

# LlamaIndex workflow primitives
from llama_index.core.workflow import (
    Workflow,
    step,
    StartEvent,
    StopEvent,
    Context,
    Event,
)

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    db_path: Path
    rulebooks_dir: Path
    model_strategy: str  # e.g., "mlx-llm-8b", "mlx-vision-11b", etc.
    context_dir: Path
    log_dir: Path
    save_html_when_pdf_missing: bool = True


def _safe_filename(game: Game) -> str:
    safe = "".join(c for c in game.name if c.isalnum() or c in (" ", "-", "_")).rstrip()
    safe = safe.replace(" ", "_")
    return f"{safe}_official.pdf"


# Legacy inline Event classes removed; using events.py instead




class RulebookWorkflow(Workflow):
    def __init__(self, config: RunConfig):
        super().__init__()
        self.config = config
        self.config.rulebooks_dir.mkdir(parents=True, exist_ok=True)
        self.config.context_dir.mkdir(parents=True, exist_ok=True)
        self.config.log_dir.mkdir(parents=True, exist_ok=True)

    @step()
    async def start(self, ctx: Context, ev: StartEvent) -> ProcessNextGameEvent | StopEvent:
        args = ev.payload or {}
        rank_from: Optional[int] = args.get("rank_from")
        rank_to: Optional[int] = args.get("rank_to")

        games: List[Game] = query_games(self.config.db_path, rank_from, rank_to)
        logger.info(f"Loaded {len(games)} games from DB for ranks {rank_from}-{rank_to}")

        await ctx.store.set("games_remaining", [g.__dict__ for g in games])
        # Initialize required state keys unconditionally to avoid missing path errors
        await ctx.store.set("results", [])
        await ctx.store.set("tried_pdf_candidates", [])
        await ctx.store.set("official_url", None)
        await ctx.store.set("pdf_candidate", None)
        await ctx.store.set("total_attempts", 0)

        if not games:
            return StopEvent(result={"results": [], "summary": "no_games"})

        # Kick off processing the first game
        return ProcessNextGameEvent()
    @step()
    async def process_next_game(self, ctx: Context, ev: ProcessNextGameEvent) -> PlanStrategiesEvent | ProcessNextGameEvent | StopEvent:
        remaining = await ctx.store.get("games_remaining") or []
        if not remaining:
            # All done
            return StopEvent(result={"results": (await ctx.store.get("results")) or []})
        
        game_dict = remaining.pop(0)
        await ctx.store.set("games_remaining", remaining)
        await ctx.store.set("current_game", game_dict)
        
        game = Game(**game_dict)  # type: ignore[arg-type]
        
        # Reset attempt counters for new game
        await ctx.store.set("total_attempts", 0)
        await ctx.store.set("tried_pdf_candidates", [])
        
        # Check if rulebook already exists (deduplication)
        expected_filename = f"{game.name.replace(' ', '_').replace(':', '')}_official.pdf"
        expected_path = self.config.rulebooks_dir / expected_filename
        
        if expected_path.exists():
            logger.info(f"[Workflow] Rulebook already exists for {game.name}: {expected_path}")
            await self._record_result(ctx, game, success=True, method="already_downloaded", file_path=str(expected_path))
            # Process next game
            return ProcessNextGameEvent()
        
        logger.info(f"[Workflow] Processing {game.name} - no existing rulebook found")
        return PlanStrategiesEvent()

    @step()
    async def plan_strategies(self, ctx: Context, ev: PlanStrategiesEvent) -> StrategyNextEvent:
        game = Game(**(await ctx.store.get("current_game") or {}))  # type: ignore[arg-type]
        context_summary = await self._get_context_summary(ctx)
        order = get_strategy_order(game, context_summary)
        await ctx.store.set("strategy_order", order)
        await ctx.store.set("strategy_idx", 0)
        return StrategyNextEvent()

    @step()
    async def strategy_next(self, ctx: Context, ev: StrategyNextEvent) -> TryBGGOfficialLinkEvent | TryTavilyPdfSearchEvent | TryWebsiteProbeEvent | TryComprehensiveSeleniumEvent | ProcessNextGameEvent:
        from .events import (
            TryBGGOfficialLinkEvent,
            TryTavilyPdfSearchEvent,
            TryWebsiteProbeEvent,
            TryComprehensiveSeleniumEvent,
        )
        game = Game(**(await ctx.store.get("current_game") or {}))  # type: ignore[arg-type]
        order = await ctx.store.get("strategy_order") or []
        idx = await ctx.store.get("strategy_idx") or 0
        if idx >= len(order):
            logger.warning(f"All strategies exhausted for {game.name} - moving to next game")
            await self._record_result(ctx, game, success=False, method="all_strategies_exhausted")
            from .events import ProcessNextGameEvent as _Next
            return _Next()
        current = order[idx]
        await ctx.store.set("current_strategy", current)
        logger.info(f"[Workflow] Using strategy {idx+1}/{len(order)} for {game.name}: {current}")
        if current == "bgg_official":
            return TryBGGOfficialLinkEvent()
        if current == "tavily_pdf_search":
            return TryTavilyPdfSearchEvent()
        if current == "website_probe":
            return TryWebsiteProbeEvent()
        return TryComprehensiveSeleniumEvent()

    async def _advance_strategy(self, ctx: Context) -> None:
        idx = await ctx.store.get("strategy_idx") or 0
        await ctx.store.set("strategy_idx", idx + 1)

    @step()
    async def try_bgg_official_link(self, ctx: Context, ev: TryBGGOfficialLinkEvent) -> TryDirectPdfDownloadEvent | StrategyNextEvent:
        game = Game(**(await ctx.store.get("current_game") or {}))  # type: ignore[arg-type]
        logger.info(f"[Workflow] Trying BGG official link for {game.name}")
        official = extract_official_link_from_bgg(game.url)
        if official:
            await ctx.store.set("official_url", official)
            return TryDirectPdfDownloadEvent()
        # No official link; advance to next strategy
        await self._advance_strategy(ctx)
        return StrategyNextEvent()

    @step()
    async def try_tavily_pdf_search(self, ctx: Context, ev: TryTavilyPdfSearchEvent) -> TryDirectPdfCandidateEvent | StrategyNextEvent:
        game = Game(**(await ctx.store.get("current_game") or {}))  # type: ignore[arg-type]
        
        # Track what we've already tried to avoid repeating failed approaches
        tried_candidates = await ctx.store.get("tried_pdf_candidates") or []
        
        # Circuit breaker: limit PDF candidates and total attempts
        total_attempts = await ctx.store.get("total_attempts") or 0
        
        if len(tried_candidates) > 3:
            logger.warning(f"Circuit breaker: tried {len(tried_candidates)} PDF candidates for {game.name}, giving up")
            await ctx.store.set("tried_pdf_candidates", [])
            await ctx.store.set("total_attempts", 0)
            await self._record_result(ctx, game, success=False, method="max_pdf_candidates_exceeded")
            return ProcessNextGameEvent()
        
        if total_attempts > 12:
            logger.warning(f"Circuit breaker: made {total_attempts} total attempts for {game.name}, giving up")
            await ctx.store.set("tried_pdf_candidates", [])
            await ctx.store.set("total_attempts", 0)
            await self._record_result(ctx, game, success=False, method="max_total_attempts_exceeded")
            return ProcessNextGameEvent()
        
        # Increment attempt counter
        await ctx.store.set("total_attempts", total_attempts + 1)
        
        # Try multiple search strategies for finding English PDFs
        search_strategies = [
            f"{game.name} official rulebook pdf english",
            f"{game.name} official rules pdf",
            f"{game.name} rulebook download english",
            f"{game.name} official website rules",
        ]
        
        # Try direct PDF search with multiple strategies
        for strategy in search_strategies:
            logger.info(f"Trying Tavily strategy: {strategy}")
            rb_results = tavily_search(strategy, max_results=5)
            for r in rb_results:
                if r.url and r.url.lower().endswith(".pdf") and r.url not in tried_candidates:
                    # Avoid non-English URLs
                    if likely_non_english_url(r.url):
                        logger.info(f"Skipping likely non-English PDF: {r.url}")
                        continue
                    logger.info(f"Found PDF candidate from Tavily: {r.url}")
                    await ctx.store.set("pdf_candidate", r.url)
                    # Track that we tried this candidate
                    tried_candidates.append(r.url)
                    await ctx.store.set("tried_pdf_candidates", tried_candidates)
                    return TryDirectPdfCandidateEvent()
        
        # Try website search and look for PDF links via Selenium (limited attempts)
        website_queries = [f"{game.name} official website"]
        
        publisher = getattr(game, "publisher", None)
        if publisher:
            website_queries.insert(0, f"{game.name} {publisher} official website")
        
        for query in website_queries[:2]:  # Limit to 2 website queries max
            logger.info(f"Trying website search: {query}")
            results = tavily_search(query, max_results=2)  # Reduced from 3 to 2
            
            for r in results[:2]:  # Only try first 2 results
                if r.url and r.url.startswith("http"):
                    logger.info(f"Probing Tavily website result: {r.url}")
                    # Use simple Selenium probe to find PDF links on this page
                    pdf_links = selenium_simple_probe(r.url)
                    for pdf_url in pdf_links[:2]:  # Reduced from 5 to 2 PDF links per page
                        if pdf_url in tried_candidates:
                            continue
                        if likely_non_english_url(pdf_url):
                            logger.info(f"Skipping likely non-English PDF: {pdf_url}")
                            continue
                        logger.info(f"Found PDF link via Selenium probe: {pdf_url}")
                        await ctx.store.set("pdf_candidate", pdf_url)
                        # Track that we tried this candidate
                        tried_candidates.append(pdf_url)
                        await ctx.store.set("tried_pdf_candidates", tried_candidates)
                        return TryDirectPdfCandidateEvent()
        
        # Final attempt: comprehensive Selenium search
        logger.info(f"[Workflow] Trying comprehensive Selenium search for {game.name}")
        
        success, file_path, method = selenium_comprehensive_search(
            game.name, 
            self.config.rulebooks_dir, 
            headless=True
        )
        
        if success and file_path:
            # Use truly agentic assessment - let agent reason about context
            context_summary = await self._get_context_summary(ctx)
            is_official, is_english, rationale = agent_choose_pdf_assessment(
                Path(file_path), game.name, self.config.model_strategy, context_summary
            )
            logger.info(f"Selenium PDF assessment: official={is_official}, english={is_english}, rationale={rationale}")
            
            if is_english and is_official:
                await self._record_result(ctx, game, success=True, method=method, rulebook_url=None, file_path=file_path)
                return ProcessNextGameEvent()
            else:
                # Even comprehensive search didn't find English version
                logger.info(f"Comprehensive search found non-English PDF: {rationale}")
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception:
                    pass
        
        # All strategies exhausted - clear tried candidates for next game
        await ctx.store.set("tried_pdf_candidates", [])
        logger.warning(f"All strategies exhausted for {game.name} - no English official rulebook found")
        await self._record_result(ctx, game, success=False, method="all_strategies_exhausted")
        return ProcessNextGameEvent()

    @step()
    async def try_direct_pdf_candidate(self, ctx: Context, ev: TryDirectPdfCandidateEvent) -> ProcessNextGameEvent | StopEvent:
        game = Game(**(await ctx.store.get("current_game") or {}))  # type: ignore[arg-type]
        url = await ctx.store.get("pdf_candidate")
        if not url:
            logger.warning(f"No PDF candidate set for {game.name}")
            await self._record_result(ctx, game, success=False, method="no_pdf_candidate")
            return ProcessNextGameEvent()
        
        logger.info(f"Testing PDF candidate: {url}")
        ok, content = probe_direct_pdf(url, timeout=15, max_retries=1)
        if ok and content:
            file_path = self._save_pdf(game, content)
            # Use truly agentic assessment - let agent reason about context
            context_summary = await self._get_context_summary(ctx)
            is_official, is_english, rationale = agent_choose_pdf_assessment(
                file_path, game.name, self.config.model_strategy, context_summary
            )
            logger.info(f"PDF assessment for {game.name}: official={is_official}, english={is_english}, rationale={rationale}")
            
            if is_english and is_official:
                await self._record_result(ctx, game, success=True, method="pdf_candidate_success", rulebook_url=url, file_path=str(file_path))
                return ProcessNextGameEvent()
            else:
                # Reject this PDF but CONTINUE TRYING OTHER TOOLS
                logger.info(f"Rejecting PDF for {game.name}: {rationale} - CONTINUING search with other tools")
                try:
                    file_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
                except Exception:
                    pass
                # Don't return here - continue to try other strategies
        else:
            logger.warning(f"Failed to download PDF candidate for {game.name}: {url}")
        
        # PDF candidate failed - but DON'T give up yet! Try other strategies
        # Clear the failed candidate and let other tools try
        await ctx.store.set("pdf_candidate", None)
        
        # If we have an official URL, try other approaches on it
        official_url = await ctx.store.get("official_url")
        if official_url:
            logger.info(f"PDF candidate failed, trying other approaches on official URL: {official_url}")
            return TryDirectPdfDownloadEvent()
        
        # Otherwise try comprehensive Selenium search as final attempt
        logger.info(f"No official URL available, advancing strategy for {game.name}")
        await self._advance_strategy(ctx)
        return StrategyNextEvent()

    @step()
    async def try_website_probe(self, ctx: Context, ev: TryWebsiteProbeEvent) -> TryDirectPdfCandidateEvent | StrategyNextEvent:
        game = Game(**(await ctx.store.get("current_game") or {}))  # type: ignore[arg-type]
        tried_candidates = await ctx.store.get("tried_pdf_candidates") or []
        website_queries = [f"{game.name} official website"]
        publisher = getattr(game, "publisher", None)
        if publisher:
            website_queries.insert(0, f"{game.name} {publisher} official website")
        for query in website_queries[:2]:
            logger.info(f"Trying website search: {query}")
            results = tavily_search(query, max_results=2)
            for r in results[:2]:
                if r.url and r.url.startswith("http"):
                    logger.info(f"Probing Tavily website result: {r.url}")
                    pdf_links = selenium_simple_probe(r.url)
                    for pdf_url in pdf_links[:2]:
                        if pdf_url in tried_candidates:
                            continue
                        if likely_non_english_url(pdf_url):
                            logger.info(f"Skipping likely non-English PDF: {pdf_url}")
                            continue
                        logger.info(f"Found PDF link via Selenium probe: {pdf_url}")
                        await ctx.store.set("pdf_candidate", pdf_url)
                        tried_candidates.append(pdf_url)
                        await ctx.store.set("tried_pdf_candidates", tried_candidates)
                        return TryDirectPdfCandidateEvent()
        await self._advance_strategy(ctx)
        return StrategyNextEvent()

    @step()
    async def try_comprehensive(self, ctx: Context, ev: TryComprehensiveSeleniumEvent) -> ProcessNextGameEvent | StrategyNextEvent:
        game = Game(**(await ctx.store.get("current_game") or {}))  # type: ignore[arg-type]
        logger.info(f"[Workflow] Trying comprehensive Selenium search for {game.name}")
        success, file_path, method = selenium_comprehensive_search(
            game.name, self.config.rulebooks_dir, headless=True
        )
        if success and file_path:
            context_summary = await self._get_context_summary(ctx)
            is_official, is_english, rationale = agent_choose_pdf_assessment(
                Path(file_path), game.name, self.config.model_strategy, context_summary
            )
            logger.info(f"Selenium PDF assessment: official={is_official}, english={is_english}, rationale={rationale}")
            if is_english and is_official:
                await self._record_result(ctx, game, success=True, method=method, rulebook_url=None, file_path=file_path)
                return ProcessNextGameEvent()
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass
        await self._advance_strategy(ctx)
        return StrategyNextEvent()

    @step()
    async def try_direct_pdf_download(self, ctx: Context, ev: TryDirectPdfDownloadEvent) -> ProcessNextGameEvent | StrategyNextEvent:
        game = Game(**(await ctx.store.get("current_game") or {}))  # type: ignore[arg-type]
        official = await ctx.store.get("official_url")
        if not official:
            await self._advance_strategy(ctx)
            return StrategyNextEvent()
        # Heuristic: try common rulebook paths under official site (quick attempts only)
        candidates = [
            f"{official.rstrip('/')}/rulebook.pdf",
            f"{official.rstrip('/')}/rules.pdf",
            f"{official.rstrip('/')}/downloads/rulebook.pdf",
        ]
        for url in candidates:
            if likely_non_english_url(url):
                continue
            logger.info(f"Trying direct PDF: {url}")
            ok, content = probe_direct_pdf(url, referer=official, max_retries=0, timeout=10)
            if ok and content:
                file_path = self._save_pdf(game, content)
                is_official, is_english, rationale = assess_pdf_official_llamaparse(file_path, game.name)
                if is_english and is_official:
                    await self._record_result(ctx, game, success=True, method="official_url_guess", rulebook_url=url, file_path=str(file_path))
                    await self._learn_success(ctx, game, source_url=url)
                    return ProcessNextGameEvent()
                try:
                    file_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
                except Exception:
                    pass
        # If guesses failed, use Selenium to surface JS-bound links
        try:
            pdf_links = selenium_simple_probe(official)
            for url in pdf_links[:5]:
                if likely_non_english_url(url):
                    continue
                ok, content = probe_direct_pdf(url, referer=official, max_retries=1)
                if ok and content:
                    file_path = self._save_pdf(game, content)
                    is_official, is_english, rationale = assess_pdf_official_llamaparse(file_path, game.name)
                    if is_english and is_official:
                        await self._record_result(ctx, game, success=True, method="selenium_pdf_link", rulebook_url=url, file_path=str(file_path))
                        await self._learn_success(ctx, game, source_url=url)
                        return ProcessNextGameEvent()
                    try:
                        file_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Selenium probe failed for {official}: {e}")
        # Fall back to comprehensive Selenium search directly
        logger.info(f"[Workflow] Official URL strategies exhausted, trying comprehensive Selenium search for {game.name}")
        
        # Use the comprehensive Selenium tool that replicates web_search_agent.py success
        success, file_path, method = selenium_comprehensive_search(
            game.name, 
            self.config.rulebooks_dir, 
            headless=True
        )
        
        if success and file_path:
            # Assess the downloaded PDF
            is_official, is_english, rationale = assess_pdf_official_llamaparse(Path(file_path), game.name)
            logger.info(f"Selenium PDF assessment: official={is_official}, english={is_english}, rationale={rationale}")
            
            if is_english and is_official:
                await self._record_result(ctx, game, success=True, method=method, rulebook_url=None, file_path=file_path)
                await self._learn_success(ctx, game, source_url="selenium_comprehensive")
                return ProcessNextGameEvent()
            else:
                # Reject non-official or non-English PDF
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception:
                    pass
        
        # All strategies exhausted
        await self._record_result(ctx, game, success=False, method="all_strategies_exhausted")
        return ProcessNextGameEvent()

    def _save_pdf(self, game: Game, content: bytes) -> Path:
        file_name = _safe_filename(game)
        out_path = self.config.rulebooks_dir / file_name
        with open(out_path, "wb") as f:
            f.write(content)
        return out_path

    async def _record_result(self, ctx: Context, game: Game, success: bool, method: str, rulebook_url: Optional[str] = None, file_path: Optional[str] = None) -> None:
        result = {
            "game_name": game.name,
            "success": success,
            "rulebook_url": rulebook_url,
            "file_path": file_path,
            "method": method,
        }
        results = await ctx.store.get("results") or []
        results.append(result)
        await ctx.store.set("results", results)


    async def _get_context_summary(self, ctx: Context) -> str:
        """
        Get a summary of the execution context for agentic decision making.
        
        This extracts relevant patterns from the Context execution history
        for the agent to reason about when making decisions.
        """
        results = await ctx.store.get("results") or []
        
        # Create a summary of what's happened so far
        summary_parts = [
            f"Model strategy: {self.config.model_strategy}",
            f"Results so far: {len(results)} games processed",
        ]
        
        # Add info about successful/failed attempts
        successes = [r for r in results if r.get("success")]
        failures = [r for r in results if not r.get("success")]
        
        if successes:
            successful_methods = [r.get("method") for r in successes]
            summary_parts.append(f"Successful methods: {successful_methods}")
        
        if failures:
            failed_methods = [r.get("method") for r in failures]
            summary_parts.append(f"Failed methods: {failed_methods}")
        
        # In a full implementation, this would be much richer context analysis
        # The agent would analyze patterns, extract insights, and provide
        # strategic guidance based on execution history
        return " | ".join(summary_parts)




from typing import Optional as _Optional


def load_or_init_context(workflow: Workflow, ctx_dir: Path, model_strategy: str) -> _Optional[Context]:
    from llama_index.core.workflow import JsonSerializer
    ctx_path = ctx_dir / f"context_{model_strategy}.json"
    if ctx_path.exists():
        try:
            with open(ctx_path, "r", encoding="utf-8") as f:
                ctx_dict = json.load(f)
            return Context.from_dict(workflow, ctx_dict, serializer=JsonSerializer())
        except Exception:
            pass
    # If no saved context, let the workflow create a fresh one on run
    return None


def save_context(ctx: Context, ctx_dir: Path, model_strategy: str) -> None:
    from llama_index.core.workflow import JsonSerializer
    ctx_path = ctx_dir / f"context_{model_strategy}.json"
    try:
        with open(ctx_path, "w", encoding="utf-8") as f:
            json.dump(ctx.to_dict(serializer=JsonSerializer()), f)
    except Exception as e:
        logger.warning(f"Failed to save context: {e}")


