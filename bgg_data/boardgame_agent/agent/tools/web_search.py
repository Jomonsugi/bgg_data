"""search_web tool — Tavily web search with per-game domain restrictions."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from bgg_data.boardgame_agent.config import TAVILY_API_KEY, GAMES_DB_PATH


def make_web_search_tool(game_id: str, db_path: Path = GAMES_DB_PATH):
    """Return a search_web tool that respects the game's allowed domain list."""

    @tool
    def search_web(query: str) -> str:
        """Search the web for community rulings, FAQs, or clarifications.

        Results are restricted to trusted domains configured for this game
        (default: boardgamegeek.com). If no domains are configured the search
        is unrestricted.

        Always include the source URL in your answer when citing web results.
        """
        from tavily import TavilyClient
        from bgg_data.boardgame_agent.db.games import get_search_domains

        domains = get_search_domains(game_id, db_path)

        client = TavilyClient(api_key=TAVILY_API_KEY)
        kwargs: dict = {
            "query": query,
            "max_results": 5,
            "include_answer": True,
        }
        if domains:
            kwargs["include_domains"] = domains

        response = client.search(**kwargs)

        lines: list[str] = []
        if response.get("answer"):
            lines.append(f"Summary: {response['answer']}\n")

        for result in response.get("results", []):
            lines.append(
                f"Source: {result.get('url', 'unknown')}\n"
                f"Title: {result.get('title', '')}\n"
                f"Content: {result.get('content', '')[:600]}\n"
            )

        return "\n---\n".join(lines) if lines else "No results found."

    return search_web
