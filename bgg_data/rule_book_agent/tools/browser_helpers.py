from __future__ import annotations

import re
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from . import browser_primitives as bp


class BrowserClosePopupsIn(BaseModel):
    session_id: str


def browser_close_popups(session_id: str) -> dict:
    """
    Attempt to close common cookie banners / modal popups.

    This tool is intentionally best-effort and returns what it tried.
    """
    out = bp._call_worker("close_popups", session_id=session_id)
    if out.get("ok"):
        return {"closed_any": bool(out.get("closed_any")), "actions": out.get("actions", [])}
    return {"closed_any": False, "actions": [], "reason": out.get("reason", "unknown session_id")}


class BrowserFindLinksIn(BaseModel):
    session_id: str
    include_patterns: list[str] = Field(default_factory=list, description="Regex patterns; include if any matches URL or text")
    exclude_patterns: list[str] = Field(default_factory=list, description="Regex patterns; exclude if any matches URL or text")
    max_matches: int = Field(default=50)


def browser_find_links(
    session_id: str,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    max_matches: int = 50,
) -> dict:
    """
    Find links on the current page matching include/exclude regex patterns.

    This tool helps avoid the LLM doing brittle regex across huge page text.
    """
    out = bp._call_worker(
        "find_links",
        session_id=session_id,
        include_patterns=include_patterns or [],
        exclude_patterns=exclude_patterns or [],
        max_matches=int(max_matches),
    )
    if out.get("ok"):
        return {"matches": out.get("matches", [])}
    return {"matches": [], "reason": out.get("reason", "unknown session_id")}


class BrowserGetBggOfficialLinkIn(BaseModel):
    bgg_url: str = Field(..., description="BGG page URL (e.g., from game.url)")
    run_id: str = Field(..., description="Run id for artifact storage")
    navigate: bool = Field(default=True, description="If True, navigate to the official link after finding it")
    headless: bool = Field(default=True, description="Whether to run browser headless")


def browser_get_bgg_official_link(bgg_url: str, run_id: str, navigate: bool = True, headless: bool = True) -> dict:
    """
    Open a BGG game page and click through to the game's official site.

    Simple behavior (agent-friendly):
    - Open the provided BGG game page
    - Locate the "Official Links" section
    - Extract the first external URL in that section (the "official site" link)
    - Optionally navigate to it

    The agent can then use other tools (snapshot/find_links/click/download/validate)
    on the official site to locate the rulebook.
    """
    from .browser_primitives import browser_open

    tool_version = 2

    if not bgg_url.startswith(("http://", "https://")):
        return {"ok": False, "official_link": None, "reason": "bgg_url must start with http:// or https://", "tool_version": tool_version}

    # Open the BGG page
    open_result = browser_open(bgg_url, run_id, headless=headless)
    if not open_result.get("session_id"):
        return {
            "ok": False,
            "official_link": None,
            "reason": open_result.get("reason", "failed to open BGG page"),
            "tool_version": tool_version,
        }

    session_id = open_result["session_id"]
    # Close any popups first
    try:
        browser_close_popups(session_id)
    except Exception:
        pass

    # BGG pages often hydrate sections client-side; wait briefly for "Official Links" to appear.
    try:
        bp._call_worker("wait_for_selector", session_id=session_id, selector="text=Official Links", timeout_ms=6000)
    except Exception:
        pass

    try:
        # Find the "Official Links" section and extract external URLs from it.
        # BGG structure is fairly stable:
        # - The section is rendered inside an `OFFICIAL-LINKS-MODULE` element
        # - The heading is `h3.panel-title` with text "Official Links"
        # We should scope extraction to that module to avoid accidentally picking up
        # unrelated external links elsewhere on the page (e.g., BGG Store, ads, etc.).
        script = """
            () => {
              const norm = (s) => (s || "").replace(/\\s+/g, " ").trim().toLowerCase();
              const isExternal = (u) => !!u && /^https?:\\/\\//i.test(u) && !/boardgamegeek\\.com/i.test(u);

              // Find the "Official Links" heading (prefer the panel title)
              const headings = Array.from(document.querySelectorAll("h3.panel-title, h1,h2,h3,h4,h5,h6"));
              const heading = headings.find(h => norm(h.textContent) === "official links") || null;

              if (!heading) {
                return {
                  external_links: [],
                  found_heading: false,
                  heading_text: null,
                  found_module: false,
                };
              }

              const seen = new Set();
              const external = [];

              // Strategy 1 (preferred): scope to OFFICIAL-LINKS-MODULE
              const module = heading.closest("official-links-module");
              if (module) {
                const links = Array.from(module.querySelectorAll("a[href]"));
                for (const a of links) {
                  const url = a.href;
                  if (!url || seen.has(url)) continue;
                  seen.add(url);
                  if (isExternal(url)) {
                    external.push({
                      text: (a.innerText || a.textContent || "").trim().slice(0, 200),
                      url: url
                    });
                  }
                }
              }

              // Strategy 2: Check next siblings of the heading (typically a list with links)
              if (external.length === 0) {
                let sibling = heading.nextElementSibling;
                let sibDepth = 0;
                while (sibling && sibDepth < 3) {
                  const links = Array.from(sibling.querySelectorAll("a[href]"));
                  for (const a of links) {
                    const url = a.href;
                    if (!url || seen.has(url)) continue;
                    seen.add(url);
                    if (isExternal(url)) {
                      external.push({
                        text: (a.innerText || a.textContent || "").trim().slice(0, 200),
                        url: url
                      });
                    }
                  }
                  if (external.length > 0) break;
                  sibling = sibling.nextElementSibling;
                  sibDepth++;
                }
              }

              return {
                external_links: external.slice(0, 10),
                found_heading: true,
                heading_text: (heading.textContent || "").trim().slice(0, 120),
                found_module: !!module,
              };
            }
            """
        data_out = bp._call_worker("eval", session_id=session_id, script=script)
        if not data_out.get("ok"):
            return {
                "ok": False,
                "official_link": None,
                "official_links": [],
                "session_id": session_id,
                "current_url": data_out.get("current_url"),
                "reason": f"Failed to evaluate page: {data_out.get('reason')}",
                "tool_version": tool_version,
            }
        data = data_out.get("value") or {}

        external_links = (data or {}).get("external_links", []) or []
        official_link = (external_links[0].get("url") if external_links else None)

        if not official_link:
            return {
                "ok": False,
                "official_link": None,
                "official_links": external_links,
                "debug": {"found_heading": (data or {}).get("found_heading"), "heading_text": (data or {}).get("heading_text")},
                "session_id": session_id,
                "current_url": bp._call_worker("snapshot", session_id=session_id, max_text_chars=1).get("current_url"),
                "reason": "Could not find an external link in the 'Official Links' section on the BGG page.",
                "tool_version": tool_version,
            }

        if navigate:
            try:
                goto_out = bp._call_worker("goto", session_id=session_id, url=official_link)
                if not goto_out.get("ok"):
                    raise RuntimeError(goto_out.get("reason", "goto failed"))
                try:
                    browser_close_popups(session_id)
                except Exception:
                    pass
            except Exception as e:
                return {
                    "ok": True,
                    "official_link": official_link,
                    "official_links": external_links,
                    "session_id": session_id,
                    "current_url": bp._call_worker("snapshot", session_id=session_id, max_text_chars=1).get("current_url"),
                    "navigated": False,
                    "reason": f"Found official link but failed to navigate: {e}",
                    "tool_version": tool_version,
                }

        return {
            "ok": True,
            "official_link": official_link,
            "official_links": external_links,
            "debug": {"found_heading": (data or {}).get("found_heading"), "heading_text": (data or {}).get("heading_text")},
            "session_id": session_id,
            "current_url": bp._call_worker("snapshot", session_id=session_id, max_text_chars=1).get("current_url"),
            "navigated": bool(navigate),
            "tool_version": tool_version,
        }

    except Exception as e:
        return {
            "ok": False,
            "official_link": None,
            "session_id": session_id,
            "current_url": None,
            "reason": f"{e}",
            "tool_version": tool_version,
        }


def build_browser_helper_tools():
    return [
        StructuredTool.from_function(
            func=browser_close_popups,
            name="browser_close_popups",
            description="Best-effort attempt to close cookie banners/popups/modals; returns what it tried.",
            args_schema=BrowserClosePopupsIn,
        ),
        StructuredTool.from_function(
            func=browser_find_links,
            name="browser_find_links",
            description="Find links on the current page matching include/exclude regex patterns (URL/text).",
            args_schema=BrowserFindLinksIn,
        ),
        StructuredTool.from_function(
            func=browser_get_bgg_official_link,
            name="browser_get_bgg_official_link",
            description="Open a BGG game page, find the official website link, and optionally navigate to it. This is often a good first step since many publishers host rulebooks on their official sites.",
            args_schema=BrowserGetBggOfficialLinkIn,
        ),
    ]


