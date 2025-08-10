"""
Agentic Tavily search using a small LangGraph loop.

This module exposes an agent that iteratively queries Tavily to find
either official websites (preferred) or direct rulebook PDFs, using the
existing SimpleScorer for domain/keyword weighting.
"""

from __future__ import annotations

import os
import logging
from typing import List, TypedDict, Annotated, Set, Dict, Optional
import operator

from langgraph.graph import StateGraph, END

try:
    from tavily import TavilyClient  # type: ignore
    _TAVILY_AVAILABLE = True
except Exception:
    _TAVILY_AVAILABLE = False

from .search import SearchResult, SimpleScorer

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    query: str
    goal: str
    iteration: int
    max_iterations: int
    seen_urls: Annotated[Set[str], operator.or_]
    candidates: Annotated[List[SearchResult], operator.add]
    done: bool


class TavilyAgent:
    """A minimal agentic search loop powered by LangGraph + Tavily.

    We intentionally avoid LLM usage here; the loop expands queries
    deterministically for reliability and speed.
    """

    def __init__(self, api_key: Optional[str] = None, max_results: int = 8):
        if not _TAVILY_AVAILABLE:
            raise RuntimeError("tavily-python is not available")
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY is not set")
        self.client = TavilyClient(api_key=self.api_key)
        self.max_results = max_results
        self.scorer = SimpleScorer()

    def _run_graph(self, initial_query: str, goal: str, publisher: Optional[str]) -> List[SearchResult]:
        def do_search(state: AgentState) -> AgentState:
            q = state["query"]
            try:
                resp: Dict = self.client.search(q, max_results=self.max_results, include_answer=False)
            except Exception as e:
                logger.warning(f"Tavily agent search failed: {e}")
                return {"done": True}

            results: List[SearchResult] = []
            for item in resp.get("results", [])[: self.max_results]:
                url = item.get("url") or ""
                title = item.get("title") or url
                if not url or not url.startswith("http"):
                    continue
                if url in state.get("seen_urls", set()):
                    continue
                sr = SearchResult(url=url, title=title, source="tavily")
                sr.score = self.scorer.score(sr, publisher)
                results.append(sr)

            # Merge into state
            new_seen = set(state.get("seen_urls", set()))
            new_seen.update([r.url for r in results])
            new_candidates = list(state.get("candidates", [])) + results

            # Heuristic stop: if we have a high-score candidate or enough results
            top_score = max([r.score for r in new_candidates], default=0)
            it = state.get("iteration", 0)
            max_it = state.get("max_iterations", 3)
            done = top_score >= 100 or len(new_candidates) >= self.max_results or it + 1 >= max_it

            return {
                "seen_urls": new_seen,
                "candidates": new_candidates,
                "iteration": it + 1,
                "done": done,
            }

        def expand_query(state: AgentState) -> AgentState:
            # Deterministic query schedule by iteration and goal
            base_q = state.get("goal", "")
            it = state.get("iteration", 0)
            if it == 0:
                q = f'"{base_q}" board game official website'
            elif it == 1:
                q = f'"{base_q}" official rulebook filetype:pdf'
            else:
                q = f'"{base_q}" rules pdf OR rulebook pdf'
            return {"query": q}

        def should_continue(state: AgentState) -> str:
            return END if state.get("done") else "search"

        graph = StateGraph(AgentState)
        graph.add_node("expand_query", expand_query)
        graph.add_node("search", do_search)
        graph.add_conditional_edges("search", should_continue, {END: END, "search": "expand_query"})
        graph.set_entry_point("expand_query")
        app = graph.compile()

        final_state: AgentState = app.invoke(
            {
                "goal": initial_query,
                "query": initial_query,
                "iteration": 0,
                "max_iterations": 3,
                "seen_urls": set(),
                "candidates": [],
                "done": False,
            }
        )
        cands = sorted(final_state.get("candidates", []), key=lambda r: r.score, reverse=True)
        return cands[: self.max_results]

    def search_official_website(self, game_name: str, publisher: Optional[str], max_results: int = 5) -> List[SearchResult]:
        goal = f"{publisher} {game_name}" if publisher else game_name
        results = self._run_graph(goal, goal, publisher)
        return results[:max_results]

    def search_rulebook(self, game_name: str, publisher: Optional[str], max_results: int = 5) -> List[SearchResult]:
        goal = f"{game_name} rulebook"
        results = self._run_graph(goal, goal, publisher)
        return results[:max_results]


