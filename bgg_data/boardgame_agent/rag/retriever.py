"""Qdrant hybrid retrieval for rulebook chunks, filtered by game_id.

Uses Qdrant's native prefetch + RRF fusion to combine:
  - Dense search  (Ollama qwen3-embedding) — semantic similarity
  - Sparse search (SPLADE++) — exact term matching
"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from bgg_data.boardgame_agent.config import COLLECTION_NAME, RETRIEVAL_TOP_K as _DEFAULT_K
from bgg_data.boardgame_agent.rag.indexer import embed_dense_single, embed_sparse


def retrieve_pages(
    client: QdrantClient,
    query: str,
    game_id: str,
    k: int = _DEFAULT_K,
) -> list[Any]:
    """Return top-k Qdrant points for *query*, restricted to *game_id*.

    Runs two prefetch branches (dense + sparse) and fuses with RRF server-side.
    """
    game_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="game_id",
                match=models.MatchValue(value=game_id),
            )
        ]
    )

    # Prefetch pool is larger than final k so RRF has enough candidates.
    prefetch_limit = k * 4

    dense_emb = embed_dense_single(query)
    sparse_emb = embed_sparse([query])[0]

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_emb,
                using="dense",
                filter=game_filter,
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=sparse_emb,
                using="sparse",
                filter=game_filter,
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
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
