"""Qdrant retrieval for rulebook pages, filtered by game_id."""

from __future__ import annotations

from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from bgg_data.boardgame_agent.config import COLLECTION_NAME, RETRIEVAL_TOP_K as _DEFAULT_K


def retrieve_pages(
    client: QdrantClient,
    text_model: TextEmbedding,
    query: str,
    game_id: str,
    k: int = _DEFAULT_K,
) -> list[Any]:
    """Return top-k Qdrant points for *query*, restricted to *game_id*."""
    query_emb = list(text_model.embed([query]))[0]

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_emb.tolist(),
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="game_id",
                    match=models.MatchValue(value=game_id),
                )
            ]
        ),
        limit=k,
        with_payload=True,
    )
    return response.points


def format_pages_for_llm(points: list[Any]) -> str:
    """Convert Qdrant points into a structured string the LLM can cite from.

    Format:
        === DOCUMENT: <doc_name> | PAGE <page_num> ===
        <page text>
        Bboxes (cite by index):
          [0] "..."
          [1] "..."
    """
    if not points:
        return "No relevant pages found in the indexed rulebooks."

    sections: list[str] = []
    for point in points:
        p = point.payload
        doc_name = p.get("doc_name", "unknown")
        page_num = p.get("page_num", "?")
        text = p.get("text", "")
        bboxes = p.get("bboxes", [])

        original_indices = p.get("original_bbox_indices", list(range(len(bboxes))))
        bbox_lines = "\n".join(
            f'  [{original_indices[i]}] "{b.get("text", "")[:200]}"'
            for i, b in enumerate(bboxes)
            if b.get("text")
        )

        sections.append(
            f"=== DOCUMENT: {doc_name} | PAGE {page_num} ===\n"
            f"{text}\n\n"
            f"Bboxes (cite by index):\n{bbox_lines}"
        )

    return "\n\n" + ("\n\n" + "─" * 60 + "\n\n").join(sections)
