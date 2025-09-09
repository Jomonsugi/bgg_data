"""
Agent helpers for tool selection and strategy planning.

These are intentionally thin wrappers that use the local MLX LLM to
choose an order of strategies based on the current game and a brief
context summary string assembled by the workflow. No hard-coded learning
is done here; the Context history informs the agent via the prompt.
"""

from __future__ import annotations

from typing import List

from .tools import call_local_llm
from bgg_data.models import Game


def get_strategy_order(game: Game, context_summary: str) -> List[str]:
    """Ask the local MLX LLM to propose an ordered list of strategies.

    Allowed strategies:
    - bgg_official
    - tavily_pdf_search
    - website_probe
    - comprehensive_selenium

    Returns a list subset in preferred order. Fallback to a sensible default.
    """
    prompt = f"""
You are orchestrating tools to find an official English PDF rulebook for the board game "{game.name}".

Context summary (recent successes/failures):
{context_summary}

Choose and order from these strategies:
- bgg_official: Use BoardGameGeek to get the official site, then try direct download paths.
- tavily_pdf_search: Use web search to find direct PDF links.
- website_probe: Use web search to find official/publisher pages, then probe those pages for PDF links.
- comprehensive_selenium: As a last resort, do a deeper Selenium-driven search.

Return ONLY a comma-separated list using these exact tokens. Example:
bgg_official, tavily_pdf_search, website_probe, comprehensive_selenium
"""

    response = call_local_llm(prompt, max_tokens=60)
    if not response:
        return ["bgg_official", "tavily_pdf_search", "website_probe", "comprehensive_selenium"]

    text = response.lower()
    order: List[str] = []
    for token in ["bgg_official", "tavily_pdf_search", "website_probe", "comprehensive_selenium"]:
        if token in text and token not in order:
            order.append(token)

    # Ensure at least one and cap to all four in a sane order
    if not order:
        order = ["bgg_official", "tavily_pdf_search", "website_probe", "comprehensive_selenium"]

    return order


