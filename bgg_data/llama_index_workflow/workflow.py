from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import urlparse
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent

from .tools import (
    GamesFound,
    do_query_games,
    find_pdf_for_game,
    search_rulebook_urls,
    download_pdf,
    extract_first_pages_text,
    is_english_text,
    looks_like_official_rulebook,
    render_first_page_image,
    load_model_config,
    explore_site_for_pdfs,
)

# Selection labels (for logging consistency)
SELECT_DIRECT = "direct_initial"
SELECT_EN_BIAS = "direct_english_bias"
SELECT_AGENT = "agentic_exploration"


@dataclass
class GameCtx:
    """Per-game context to keep helper signatures short and state isolated."""
    name: str
    publisher: str
    out_dir: str
    config: dict
    tried_urls: set
    language_log: dict
    official_log: dict
    pdf_find: dict

class RulebookWorkflow(Workflow):
    """Query games, search for rulebook PDFs, validate, and save.

    Steps:
    1) Query DB for games in the requested rank range.
    2) For each game missing a rulebook, search for candidate URLs.
    3) Try PDFs first; if needed, bias search or explore sites to find PDFs.
    4) Validate each candidate (English text + rulebook-looking first page).
    5) Save on success; move on otherwise.
    """

    @step()
    async def query(self, ev: StartEvent) -> GamesFound:  # type: ignore[override]
        """Load games from the DB and return them as an event."""
        return do_query_games(ev)

    @step()
    async def process(self, ev: GamesFound) -> StopEvent:  # type: ignore[override]
        """Process each game: search, validate, and save a rulebook PDF if found."""
        results = []
        config = load_model_config(None)
        
        # Check which rulebooks already exist
        existing_rulebooks = self._check_existing_rulebooks(ev.games, ev.out_dir)
        total_games = len(ev.games)
        existing_count = len(existing_rulebooks)
        
        print(f"\n📊 Progress: {existing_count}/{total_games} rulebooks already exist")
        print(f"\n🔍 Processing {total_games - existing_count} games that need rulebooks...")
        
        # Filter out games that already have rulebooks
        games_to_process = [game for game in ev.games if game.get("name") not in existing_rulebooks]
        
        for i, game in enumerate(games_to_process, 1):
            name = game.get("name")
            rank = game.get("rank")

            print(f"\n[{i}/{len(games_to_process)}] Processing: {name} (Rank {rank})")

            # Get publisher information
            publisher = game.get("publisher", "")
            if publisher:
                print(f"  🏢 Publisher: {publisher}")

            # Agentic retry loop - try different URLs from search results; adapt queries; explore sites
            file_path = None

            # Initialize logs and search context
            language_log = {"ok": False, "reason": "", "method": ""}
            official_log = {"ok": False, "reason": "", "method": ""}
            pdf_find = find_pdf_for_game(name, publisher, config)
            all_candidates = pdf_find.get("log", {}).get("candidates", [])
            pdf_candidates = [url for url in all_candidates if url.lower().endswith(".pdf")]
            tried_urls = set()

            ctx = GameCtx(
                name=name,
                publisher=publisher,
                out_dir=ev.out_dir,
                config=config,
                tried_urls=tried_urls,
                language_log=language_log,
                official_log=official_log,
                pdf_find=pdf_find,
            )

            # 1) Try direct PDFs from initial search (ordered)
            file_path = self._strategy_direct_initial(pdf_candidates, ctx)
            
            # 2) If not found, run an English-biased search and try new PDFs
            if not file_path:
                file_path = self._strategy_english_bias(ctx)
            
            # 3) If still not found, explore top website candidates and try discovered PDFs
            if not file_path:
                file_path = await self._strategy_agentic_exploration(all_candidates, ctx)
            
            if not file_path:
                print(f"  ❌ Failed to find valid rulebook after trying available strategies")

            results.append({
                "game": name,
                "rank": rank,
                "pdf_url": ctx.pdf_find.get("url") or "",
                "file_path": str(file_path) if file_path else "",
                "log": {
                    **ctx.pdf_find.get("log", {}),
                    "language_check": ctx.language_log,
                    "official_check": ctx.official_log,
                    "decision": "saved" if file_path else "skipped",
                },
            })
        
        # Final summary
        final_existing = self._check_existing_rulebooks(ev.games, ev.out_dir)
        final_count = len(final_existing)
        not_found = [game["name"] for game in ev.games if game["name"] not in final_existing]
        
        print(f"\n📊 Final Results: {final_count}/{total_games} rulebooks exist")
        if not_found:
            print(f"❌ Rulebooks not found: {', '.join(not_found)}")
        else:
            print(f"🎉 All rulebooks found successfully!")
            
        return StopEvent(result=results)
    
    def _strategy_direct_initial(self, pdf_candidates, ctx: GameCtx) -> Optional[Path]:
        """Try direct PDFs from initial search (ordered)."""
        for pdf_url in pdf_candidates:
            ctx.pdf_find["url"] = pdf_url
            ctx.pdf_find["log"]["selected_from"] = SELECT_DIRECT
            ok, saved_path = self._try_validate_pdf(
                pdf_url,
                ctx.out_dir,
                ctx.name,
                ctx.publisher,
                ctx.config,
                ctx.tried_urls,
                ctx.language_log,
                ctx.official_log,
            )
            if ok:
                return saved_path
        return None

    def _strategy_english_bias(self, ctx: GameCtx) -> Optional[Path]:
        """Run an English-biased search and try new PDFs."""
        try:
            eng_candidates = search_rulebook_urls(ctx.name, ctx.publisher, prefer_english=True)
            new_pdf_candidates = [u for u in eng_candidates if u.lower().endswith(".pdf") and u not in ctx.tried_urls]
            for pdf_url in new_pdf_candidates:
                ctx.pdf_find["url"] = pdf_url
                ctx.pdf_find["log"]["selected_from"] = SELECT_EN_BIAS
                ok, saved_path = self._try_validate_pdf(
                    pdf_url,
                    ctx.out_dir,
                    ctx.name,
                    ctx.publisher,
                    ctx.config,
                    ctx.tried_urls,
                    ctx.language_log,
                    ctx.official_log,
                )
                if ok:
                    return saved_path
        except Exception:
            pass
        return None

    async def _strategy_agentic_exploration(self, all_candidates, ctx: GameCtx) -> Optional[Path]:
        """Explore top website candidates and try discovered PDFs."""
        site_candidates = [u for u in all_candidates if not u.lower().endswith(".pdf")]
        for site_url in site_candidates[:3]:
            try:
                # Try Context cache per-domain to avoid repeating exploration
                domain = urlparse(site_url).netloc or site_url
                cache_key = f"explore:{domain}"

                cached_links = await self._ctx_get(cache_key, default=None)
                if cached_links:
                    pdf_links = cached_links
                else:
                    print(f"  🤖 Exploring site: {site_url}")
                    pdf_links = explore_site_for_pdfs(site_url, ctx.name, ctx.config)
                    if pdf_links:
                        # Cache a bounded list to keep Context small
                        await self._ctx_set(cache_key, pdf_links[:10])
                if pdf_links:
                    # Extend candidates for logging and try these PDFs
                    ctx.pdf_find["log"]["candidates"].extend(pdf_links[:3])
                    for pdf_url in pdf_links:
                        if pdf_url in ctx.tried_urls:
                            continue
                        ctx.pdf_find["url"] = pdf_url
                        ctx.pdf_find["log"]["selected_from"] = SELECT_AGENT
                        ok, saved_path = self._try_validate_pdf(
                            pdf_url,
                            ctx.out_dir,
                            ctx.name,
                            ctx.publisher,
                            ctx.config,
                            ctx.tried_urls,
                            ctx.language_log,
                            ctx.official_log,
                        )
                        if ok:
                            return saved_path
            except Exception:
                continue
        return None

    async def _ctx_get(self, key: str, default=None):
        """Safe Context getter; returns default if Context not available."""
        ctx = getattr(self, "ctx", None)
        if ctx is None:
            return default
        try:
            return await ctx.get(key, default=default)
        except Exception:
            return default

    async def _ctx_set(self, key: str, value) -> None:
        """Safe Context setter; no-op if Context not available."""
        ctx = getattr(self, "ctx", None)
        if ctx is None:
            return
        try:
            await ctx.set(key, value)
        except Exception:
            return

    def _try_validate_pdf(self, pdf_url: str, out_dir: str, name: str, publisher: str, config: dict, tried_urls: set, language_log: dict, official_log: dict) -> tuple[bool, Optional[Path]]:
        """Download, validate, and save a PDF if it passes all checks.

        Returns (ok: bool, saved_path: Optional[Path]).
        """

        if not pdf_url or pdf_url in tried_urls:
            return False, None
        tried_urls.add(pdf_url)

        print(f"  🔗 Found PDF URL: {pdf_url}")
        print(f"  📥 Downloading...")

        # Download to temp (always)
        tmp = download_pdf(pdf_url, Path(out_dir), f"{name}_rulebook.tmp")
        if not tmp or not tmp.exists():
            print(f"  ❌ Download failed")
            return False, None

        print(f"  ✅ Downloaded successfully")

        # Language check using langdetect
        print(f"  🌐 Checking language...")
        text = extract_first_pages_text(tmp)
        # If very short text, try reading more pages before detection
        if len(text) < 100:
            print(f"  ℹ️  Text short on first pages; extracting more pages for language check...")
            text = extract_first_pages_text(tmp, max_pages=5, max_chars=2000)

        # preserve external reference by clearing and updating
        language_log.clear()
        language_log.update(is_english_text(text))

        lang_too_short = not language_log.get("ok") and "Text too short" in language_log.get("reason", "")
        if not language_log.get("ok") and not lang_too_short:
            print(f"  ❌ Language check failed: {language_log['reason']}")
            # Clean up temp file
            try:
                tmp.unlink()
            except Exception:
                pass
            return False, None

        if language_log.get("ok"):
            print(f"  ✅ Language check passed: {language_log['reason']}")
        else:
            # Only case here: lang_too_short == True
            print(f"  ⚠️  Language check inconclusive (too short). Proceeding with VLM officialness check...")

        # Quick acceptance via title + domain/filename trust before VLM
        try:
            from urllib.parse import urlparse
            parsed = urlparse(pdf_url)
            host = (parsed.netloc or "").lower().replace("www.", "")
            filename = (parsed.path.rsplit("/", 1)[-1] if parsed.path else "").lower()
        except Exception:
            host, filename = "", ""

        # Normalize game title tokens (length >= 3) and check they appear in text
        import re as _re
        norm_text = _re.sub(r"[^a-z0-9 ]", " ", text.lower())
        title_tokens = [t for t in _re.sub(r"[^a-z0-9 ]", " ", (name or "").lower()).split() if len(t) >= 3]
        title_ok = all(tok in norm_text for tok in title_tokens) if title_tokens else False

        # Domain trust based on publisher tokens
        pub_tokens = [t for t in _re.sub(r"[^a-z0-9]", "", (publisher or "").lower()).split() if t]
        domain_ok = any(t and t in host for t in pub_tokens) if pub_tokens else False

        # Filename signals base rules
        fname_ok = ("rulebook" in filename or "rules" in filename) and not any(bad in filename for bad in ("exp", "expansion", "learn", "reference"))

        if title_ok and (domain_ok or fname_ok):
            clean_name = self._clean_name_for_file(name)
            final = Path(out_dir) / f"{clean_name}_rulebook.pdf"
            try:
                tmp.replace(final)
                print(f"  ✅ Rulebook saved by title/domain/filename trust: {final}")
                return True, final
            except Exception as e:
                print(f"  ❌ Failed to rename file: {e}")
                try:
                    tmp.unlink()
                except Exception:
                    pass
                return False, None

        # Officialness check using VLM on first page image
        print(f"  🖼️  Rendering first page image...")
        img_path = render_first_page_image(tmp, dpi=300)
        if not img_path:
            official_log.clear()
            official_log.update({"ok": False, "reason": "Failed to render first page image", "method": "vlm_vision"})
            print(f"  ❌ Failed to render first page image")
            # Clean up temp file
            try:
                tmp.unlink()
            except Exception:
                pass
            return False, None

        print(f"  🤖 Checking if official rulebook...")
        official_log.clear()
        official_log.update(looks_like_official_rulebook(img_path, name, config))
        print(f"  {'✅' if official_log.get('ok') else '❌'} Official check: {official_log.get('reason', 'Unknown')}")

        # Clean up image file
        try:
            img_path.unlink()
        except Exception:
            pass

        if not official_log.get("ok"):
            print(f"  ❌ Rulebook rejected: {official_log.get('reason', 'Unknown reason')}")
            # Clean up temp file
            try:
                tmp.unlink()
            except Exception:
                pass
            return False, None
        # If VLM says official but language was too short, accept based on VLM
        # (no extra action needed; fall through to save)

        # Success! Move tmp to final (rename and remove .tmp)
        clean_name = self._clean_name_for_file(name)
        final = Path(out_dir) / f"{clean_name}_rulebook.pdf"
        try:
            tmp.replace(final)
            print(f"  ✅ Rulebook saved: {final}")
            return True, final
        except Exception as e:
            print(f"  ❌ Failed to rename file: {e}")
            # Clean up temp file
            try:
                tmp.unlink()
            except Exception:
                pass
            return False, None

    def _check_existing_rulebooks(self, games: list, out_dir: str) -> list:
        """Check which rulebooks already exist in the output directory."""
        existing = []
        out_path = Path(out_dir)
        for game in games:
            name = game.get("name")
            if name:
                # Check for both .pdf and .tmp.pdf files
                clean_name = self._clean_name_for_file(name)
                pdf_file = out_path / f"{clean_name}_rulebook.pdf"
                tmp_file = out_path / f"{clean_name}_rulebook.tmp.pdf"
                if pdf_file.exists() or tmp_file.exists():
                    existing.append(name)
        return existing

    def _clean_name_for_file(self, name: str) -> str:
        """Normalize a game name for filesystem usage (preserves prior behavior)."""
        return name.replace(" ", "_").replace(":", "_")