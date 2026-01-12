from __future__ import annotations

import os
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class TavilySearchIn(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(default=5, description="Maximum number of results to return")


def tavily_search(query: str, max_results: int = 5) -> dict:
    """
    Search the web using Tavily.

    Returns a small, structured list of results that the agent can then open/browse.
    Requires `TAVILY_API_KEY` in the environment.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"results": [], "query_used": query, "reason": "TAVILY_API_KEY not set"}

    try:
        from tavily import TavilyClient  # type: ignore
    except Exception as e:
        return {"results": [], "query_used": query, "reason": f"Tavily client import failed: {e}"}

    try:
        client = TavilyClient(api_key=api_key)
        resp = client.search(query=query, max_results=int(max_results))
        out = []
        for r in (resp.get("results") or [])[: int(max_results)]:
            out.append(
                {
                    "url": r.get("url"),
                    "title": r.get("title") or "",
                    "score": r.get("score", 0.0),
                    "source": "tavily",
                }
            )
        return {"results": [r for r in out if r.get("url")], "query_used": query}
    except Exception as e:
        return {"results": [], "query_used": query, "reason": f"{e}"}


def build_search_tools():
    return [
        StructuredTool.from_function(
            func=tavily_search,
            name="tavily_search",
            description="Search the web via Tavily. Returns a small list of structured results (url/title/score).",
            args_schema=TavilySearchIn,
        )
    ]


