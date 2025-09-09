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
    assess_pdf_official_llamaparse,
    likely_non_english_url,
    selenium_comprehensive_search,
    selenium_simple_probe,
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


class ProcessNextGameEvent(Event):
    pass


class TryBGGOfficialLinkEvent(Event):
    pass


class TryTavilyStrategyEvent(Event):
    pass


class TryDirectPdfCandidateEvent(Event):
    pass


class TryDirectPdfDownloadEvent(Event):
    pass




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
        await ctx.store.set("learned_strategies", {})
        await ctx.store.set("results", [])

        if not games:
            return StopEvent(result={"results": [], "summary": "no_games"})

        # Kick off processing the first game
        return ProcessNextGameEvent()
    @step()
    async def process_next_game(self, ctx: Context, ev: ProcessNextGameEvent) -> TryBGGOfficialLinkEvent | StopEvent:
        remaining = await ctx.store.get("games_remaining") or []
        if not remaining:
            # All done
            return StopEvent(result={"results": (await ctx.store.get("results")) or []})
        game_dict = remaining.pop(0)
        await ctx.store.set("games_remaining", remaining)
        await ctx.store.set("current_game", game_dict)
        return TryBGGOfficialLinkEvent()

    @step()
    async def try_bgg_official_link(self, ctx: Context, ev: TryBGGOfficialLinkEvent) -> TryDirectPdfDownloadEvent | TryTavilyStrategyEvent | StopEvent:
        game = Game(**(await ctx.store.get("current_game") or {}))  # type: ignore[arg-type]
        logger.info(f"[Workflow] Trying BGG official link for {game.name}")
        official = extract_official_link_from_bgg(game.url)
        if official:
            await ctx.store.set("official_url", official)
            return TryDirectPdfDownloadEvent()
        # No official link; try Tavily-based discovery
        return TryTavilyStrategyEvent()

    @step()
    async def try_tavily_strategy(self, ctx: Context, ev: TryTavilyStrategyEvent) -> TryDirectPdfCandidateEvent | ProcessNextGameEvent | StopEvent:
        game = Game(**(await ctx.store.get("current_game") or {}))  # type: ignore[arg-type]
        
        # Try direct PDF search first
        rb_results = tavily_search(f"{game.name} official rulebook pdf", max_results=5)
        for r in rb_results:
            if r.url and r.url.lower().endswith(".pdf"):
                logger.info(f"Found PDF candidate from Tavily: {r.url}")
                await ctx.store.set("pdf_candidate", r.url)
                return TryDirectPdfCandidateEvent()
        
        # Try website search and look for PDF links via Selenium
        query = f"{game.name} official website"
        publisher = getattr(game, "publisher", None)
        if publisher:
            query += f" {publisher}"
        results = tavily_search(query, max_results=3)
        
        for r in results:
            if r.url and r.url.startswith("http"):
                logger.info(f"Probing Tavily website result: {r.url}")
                # Use simple Selenium probe to find PDF links on this page
                pdf_links = selenium_simple_probe(r.url)
                for pdf_url in pdf_links[:3]:  # Try first few PDF links
                    if likely_non_english_url(pdf_url):
                        continue
                    logger.info(f"Found PDF link via Selenium probe: {pdf_url}")
                    await ctx.store.set("pdf_candidate", pdf_url)
                    return TryDirectPdfCandidateEvent()
        
        # Nothing found via Tavily - try comprehensive Selenium search directly
        logger.info(f"[Workflow] Trying comprehensive Selenium search for {game.name}")
        
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
            is_official, is_english, rationale = assess_pdf_official_llamaparse(file_path, game.name)
            logger.info(f"PDF assessment for {game.name}: official={is_official}, english={is_english}, rationale={rationale}")
            
            if is_english and is_official:
                await self._record_result(ctx, game, success=True, method="pdf_candidate_success", rulebook_url=url, file_path=str(file_path))
                await self._learn_success(ctx, game, source_url=url)
                return ProcessNextGameEvent()
            else:
                # Reject this PDF and clean up
                logger.info(f"Rejecting PDF for {game.name}: not official or not English")
                try:
                    file_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
                except Exception:
                    pass
        else:
            logger.warning(f"Failed to download PDF candidate for {game.name}: {url}")
        
        # PDF candidate failed - mark as unsuccessful and move on
        await self._record_result(ctx, game, success=False, method="pdf_candidate_failed")
        return ProcessNextGameEvent()

    @step()
    async def try_direct_pdf_download(self, ctx: Context, ev: TryDirectPdfDownloadEvent) -> ProcessNextGameEvent | TryTavilyStrategyEvent | StopEvent:
        game = Game(**(await ctx.store.get("current_game") or {}))  # type: ignore[arg-type]
        official = await ctx.store.get("official_url")
        if not official:
            return TryTavilyStrategyEvent()
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


    async def _learn_success(self, ctx: Context, game: Game, source_url: str) -> None:
        """Very simple learning: record domain->success counts per publisher under the current model strategy."""
        from urllib.parse import urlparse
        
        # Handle special cases for learning
        if source_url == "selenium_comprehensive":
            domain = "selenium_comprehensive"
        else:
            domain = urlparse(source_url).netloc
            
        publisher = getattr(game, "publisher", "unknown") or "unknown"
        learned: Dict[str, Dict[str, int]] = await ctx.store.get("learned_strategies") or {}
        pub_map = learned.get(publisher) or {}
        pub_map[domain] = pub_map.get(domain, 0) + 1
        learned[publisher] = pub_map
        await ctx.store.set("learned_strategies", learned)




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


