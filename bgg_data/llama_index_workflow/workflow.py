from __future__ import annotations

from pathlib import Path
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
        config = load_model_config(ev.model_config_path)
        
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
            language_log = {"ok": False, "reason": "", "method": ""}
            official_log = {"ok": False, "reason": "", "method": ""}
            
            # Get all search candidates once
            pdf_find = find_pdf_for_game(name, publisher, config)
            all_candidates = pdf_find.get("log", {}).get("candidates", [])
            pdf_candidates = [url for url in all_candidates if url.lower().endswith(".pdf")]
            tried_urls = set()
            
            def _try_validate_pdf(pdf_url: str) -> bool:
                nonlocal file_path, language_log, official_log
                if not pdf_url or pdf_url in tried_urls:
                    return False
                tried_urls.add(pdf_url)
                
                print(f"  🔗 Found PDF URL: {pdf_url}")
                print(f"  📥 Downloading...")
                
                # Download to temp (always)
                tmp = download_pdf(pdf_url, Path(ev.out_dir), f"{name}_rulebook.tmp")
                if not tmp or not tmp.exists():
                    print(f"  ❌ Download failed")
                    return False
                
                print(f"  ✅ Downloaded successfully")
                
                # Language check using langdetect
                print(f"  🌐 Checking language...")
                text = extract_first_pages_text(tmp)
                language_log = is_english_text(text)
                
                if not language_log.get("ok"):
                    print(f"  ❌ Language check failed: {language_log['reason']}")
                    # Clean up temp file
                    try:
                        tmp.unlink()
                    except Exception:
                        pass
                    return False
                
                print(f"  ✅ Language check passed: {language_log['reason']}")
                
                # Officialness check using VLM on first page image
                print(f"  🖼️  Rendering first page image...")
                img_path = render_first_page_image(tmp)
                if not img_path:
                    official_log = {"ok": False, "reason": "Failed to render first page image", "method": "vlm_vision"}
                    print(f"  ❌ Failed to render first page image")
                    # Clean up temp file
                    try:
                        tmp.unlink()
                    except Exception:
                        pass
                    return False
                
                print(f"  🤖 Checking if official rulebook...")
                official_log = looks_like_official_rulebook(img_path, name, config)
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
                    return False
                
                # Success! Move tmp to final (rename and remove .tmp)
                clean_name = name.replace(" ", "_").replace(":", "_")
                final = Path(ev.out_dir) / f"{clean_name}_rulebook.pdf"
                try:
                    tmp.replace(final)
                    file_path = final
                    print(f"  ✅ Rulebook saved: {file_path}")
                    return True
                except Exception as e:
                    print(f"  ❌ Failed to rename file: {e}")
                    # Clean up temp file
                    try:
                        tmp.unlink()
                    except Exception:
                        pass
                    return False

            # 1) Try direct PDFs from initial search (ordered)
            for pdf_url in pdf_candidates:
                pdf_find["url"] = pdf_url
                pdf_find["log"]["selected_from"] = "direct_initial"
                if _try_validate_pdf(pdf_url):
                    break
            
            # 2) If not found, run an English-biased search and try new PDFs
            if not file_path:
                try:
                    eng_candidates = search_rulebook_urls(name, publisher, prefer_english=True)
                    new_pdf_candidates = [u for u in eng_candidates if u.lower().endswith(".pdf") and u not in tried_urls]
                    for pdf_url in new_pdf_candidates:
                        pdf_find["url"] = pdf_url
                        pdf_find["log"]["selected_from"] = "direct_english_bias"
                        if _try_validate_pdf(pdf_url):
                            break
                except Exception:
                    pass
            
            # 3) If still not found, explore top website candidates and try discovered PDFs
            if not file_path:
                site_candidates = [u for u in all_candidates if not u.lower().endswith(".pdf")]
                for site_url in site_candidates[:3]:
                    try:
                        print(f"  🤖 Exploring site: {site_url}")
                        pdf_links = explore_site_for_pdfs(site_url, name, config)
                        if pdf_links:
                            # Extend candidates for logging and try these PDFs
                            pdf_find["log"]["candidates"].extend(pdf_links[:3])
                            for pdf_url in pdf_links:
                                if pdf_url in tried_urls:
                                    continue
                                pdf_find["url"] = pdf_url
                                pdf_find["log"]["selected_from"] = "agentic_exploration"
                                if _try_validate_pdf(pdf_url):
                                    break
                    except Exception:
                        continue
                    if file_path:
                        break
            
            if not file_path:
                print(f"  ❌ Failed to find valid rulebook after trying available strategies")

            results.append({
                "game": name,
                "rank": rank,
                "pdf_url": pdf_find.get("url") or "",
                "file_path": str(file_path) if file_path else "",
                "log": {
                    **pdf_find.get("log", {}),
                    "language_check": language_log,
                    "official_check": official_log,
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
    
    def _check_existing_rulebooks(self, games: list, out_dir: str) -> list:
        """Check which rulebooks already exist in the output directory."""
        existing = []
        out_path = Path(out_dir)
        for game in games:
            name = game.get("name")
            if name:
                # Check for both .pdf and .tmp.pdf files
                clean_name = name.replace(" ", "_").replace(":", "_")
                pdf_file = out_path / f"{clean_name}_rulebook.pdf"
                tmp_file = out_path / f"{clean_name}_rulebook.tmp.pdf"
                if pdf_file.exists() or tmp_file.exists():
                    existing.append(name)
        return existing