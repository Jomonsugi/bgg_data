"""search_rulebook tool — RAG retrieval from indexed PDF documents."""

from __future__ import annotations

from langchain_core.tools import tool
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from bgg_data.boardgame_agent.rag.retriever import retrieve_pages, format_pages_for_llm


def make_rag_tool(
    game_id: str,
    qdrant_client: QdrantClient,
    text_model: TextEmbedding,
    top_k: int = 5,
):
    """Return a search_rulebook tool bound to *game_id* and *top_k*."""

    @tool
    def search_rulebook(query: str) -> str:
        """Search the indexed rulebook and supplemental PDF documents for rules
        relevant to the query.

        Always call this tool first for any rules question. Returns page text
        and numbered bounding-box references you must use in citations.
        """
        points = retrieve_pages(qdrant_client, text_model, query, game_id, k=top_k)
        return format_pages_for_llm(points)

    return search_rulebook
