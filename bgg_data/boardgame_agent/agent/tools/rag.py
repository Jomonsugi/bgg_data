"""search_rulebook tool — hybrid RAG retrieval from indexed documents."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from qdrant_client import QdrantClient

from bgg_data.boardgame_agent.rag.retriever import retrieve_pages, format_pages_for_llm


def make_rag_tool(
    game_id: str,
    qdrant_client: QdrantClient,
    config: dict[str, Any],
):
    """Return a search_rulebook tool bound to *game_id*.

    *config* is a mutable dict — ``config["top_k"]`` is read at call time
    so the sidebar slider takes effect without rebuilding the agent.
    """

    @tool
    def search_rulebook(query: str, source: str = "all") -> str:
        """Search the indexed documents for rules relevant to the query.

        Always call this tool first for any rules question. Returns page text
        and numbered bounding-box references you must use in citations.

        Args:
            query: The search query.
            source: Filter by document tag. Use a specific tag like 'rulebook'
                    or 'faq' to search only those documents, or 'all' to search
                    everything.
        """
        doc_tag = None if source == "all" else source
        points = retrieve_pages(qdrant_client, query, game_id, k=config["top_k"], doc_tag=doc_tag)
        return format_pages_for_llm(points)

    return search_rulebook
