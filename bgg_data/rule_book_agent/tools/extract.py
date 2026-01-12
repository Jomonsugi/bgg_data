from __future__ import annotations

import re
from urllib.parse import urlparse

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class ExtractCandidatesIn(BaseModel):
    game_name: str
    page_text: str = Field(..., description="Visible page text (possibly truncated)")
    links: list[dict] = Field(default_factory=list, description="List of {text,url} links from snapshot")
    max_candidates: int = Field(default=15)


_KIND_PATTERNS: list[tuple[str, str]] = [
    ("direct_pdf", r"\.pdf(\?|$)"),
    ("downloads_page", r"download|downloads|resources|support|help|assets"),
    ("support_page", r"support|faq|customer|service"),
    ("drive", r"drive\.google\.com|usercontent\.google\.com"),
    ("dropbox", r"dropbox\.com"),
    ("bgg_filepage", r"boardgamegeek\.com.*\/file\/"),
]


def _classify(url: str) -> str:
    u = url.lower()
    for kind, pat in _KIND_PATTERNS:
        if re.search(pat, u):
            return kind
    return "unknown"


def extract_candidate_rulebook_links(game_name: str, page_text: str, links: list[dict], max_candidates: int = 15) -> dict:
    """
    Turn a page snapshot into a ranked list of candidate rulebook URLs.

    This is a heuristic extractor (fast, deterministic). It does not browse or download.
    """
    game_l = (game_name or "").lower()
    signals = ["rulebook", "rules", "manual", "instructions", "download", "pdf"]
    candidates = []

    def score_link(text: str, url: str) -> tuple[int, str]:
        t = (text or "").lower()
        u = (url or "").lower()
        score = 0
        why = []
        if u.endswith(".pdf") or ".pdf?" in u:
            score += 80
            why.append("pdf")
        if any(s in t for s in ["rulebook", "rules", "manual", "instructions"]):
            score += 40
            why.append("anchor_text_rule_signal")
        if any(s in u for s in ["rulebook", "rules", "manual", "instructions"]):
            score += 30
            why.append("url_rule_signal")
        if game_l and game_l[:20] and game_l[:20] in t:
            score += 10
            why.append("game_name_in_text")
        # prefer https
        if u.startswith("https://"):
            score += 3
        # penalize obvious non-docs
        if any(bad in u for bad in ["privacy", "terms", "cookie", "login", "signup", "cart"]):
            score -= 30
            why.append("penalty_non_doc")
        return score, ",".join(why)

    seen = set()
    for l in links or []:
        url = (l.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        text = (l.get("text") or "").strip()
        s, why = score_link(text, url)
        if s <= 0:
            continue
        candidates.append(
            {
                "url": url,
                "confidence": min(0.99, max(0.05, s / 120.0)),
                "rationale": why,
                "kind": _classify(url),
                "score": s,
                "text": text[:200],
            }
        )

    candidates.sort(key=lambda c: int(c.get("score", 0)), reverse=True)
    candidates = candidates[: max(1, int(max_candidates))]
    for c in candidates:
        c.pop("score", None)
    return {"candidates": candidates}


def build_extract_tools():
    return [
        StructuredTool.from_function(
            func=extract_candidate_rulebook_links,
            name="extract_candidate_rulebook_links",
            description="Given page text and links, return ranked candidate rulebook URLs with confidence and kind.",
            args_schema=ExtractCandidatesIn,
        )
    ]


