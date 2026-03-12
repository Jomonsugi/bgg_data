"""Qdrant indexing for rulebook pages.

build_index  — upsert a list of page dicts into the collection.
reindex_all  — rebuild the entire collection from cached Docling JSONs
               (call this after changing EMBED_MODEL_NAME in config.py).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from bgg_data.boardgame_agent.config import (
    COLLECTION_NAME,
    DATA_DIR,
    EMBED_MODEL_NAME,
    QDRANT_PATH,
)
from bgg_data.boardgame_agent.rag.extractor import chunk_by_sections


_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Return the process-wide shared QdrantClient (created once).

    Cleans up any stale .lock file left by a previous crashed process before
    opening the storage, so the app never requires manual intervention on restart.
    """
    global _qdrant_client
    if _qdrant_client is None:
        lock_file = QDRANT_PATH / ".lock"
        if lock_file.exists():
            lock_file.unlink()
        _qdrant_client = QdrantClient(path=str(QDRANT_PATH))
    return _qdrant_client


def _get_client() -> QdrantClient:
    return get_qdrant_client()


def _get_text_model() -> TextEmbedding:
    return TextEmbedding(model_name=EMBED_MODEL_NAME)


def _ensure_collection(client: QdrantClient, vector_size: int) -> None:
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )


def build_index(
    pages_data: list[dict[str, Any]],
    client: QdrantClient | None = None,
    text_model: TextEmbedding | None = None,
) -> tuple[QdrantClient, TextEmbedding]:
    """Embed *pages_data* and upsert into Qdrant. Returns (client, text_model)."""
    if client is None:
        client = _get_client()
    if text_model is None:
        text_model = _get_text_model()

    _ensure_collection(client, text_model.embedding_size)

    texts = [page["text"] for page in pages_data]
    embeddings = list(text_model.embed(texts))
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload=page,
        )
        for page, emb in zip(pages_data, embeddings)
    ]

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    return client, text_model


def remove_doc_from_index(
    doc_name: str,
    game_id: str,
    client: QdrantClient | None = None,
) -> None:
    """Delete all Qdrant points belonging to *doc_name* in *game_id*."""
    if client is None:
        client = _get_client()
    if not client.collection_exists(COLLECTION_NAME):
        return
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="game_id", match=models.MatchValue(value=game_id)
                    ),
                    models.FieldCondition(
                        key="doc_name", match=models.MatchValue(value=doc_name)
                    ),
                ]
            )
        ),
    )


def reindex_all() -> None:
    """Rebuild the entire Qdrant collection from cached Docling JSONs.

    Call this whenever EMBED_MODEL_NAME changes in config.py.
    Docling extraction is NOT re-run — only embeddings are rebuilt.
    """
    client = _get_client()
    text_model = _get_text_model()

    # Drop and recreate collection so stale vectors don't accumulate.
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    games_dir = DATA_DIR / "games"
    if not games_dir.exists():
        print("No games directory found — nothing to reindex.")
        return

    for extracted_dir in sorted(games_dir.glob("*/extracted")):
        game_id = extracted_dir.parent.name
        for json_path in sorted(extracted_dir.glob("*.json")):
            pages = json.loads(json_path.read_text(encoding="utf-8"))
            chunks = chunk_by_sections(pages)
            print(f"  Indexing {game_id}/{json_path.stem} ({len(pages)} pages → {len(chunks)} chunks) …")
            build_index(chunks, client=client, text_model=text_model)

    print("Reindex complete.")
