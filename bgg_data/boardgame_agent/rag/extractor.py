"""Docling-based PDF extraction with JSON caching.

Docling reliably parses complex rulebook PDFs (multi-column, icons, tables)
and returns per-item provenance bounding boxes that power visual citations.

The output is cached as JSON so Docling only runs once per document.
Use force=True to re-extract (e.g. if the PDF is replaced).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from bgg_data.boardgame_agent.config import DATA_DIR


def _extract_single_pdf(
    pdf_path: Path, game_id: str, doc_name: str
) -> list[dict[str, Any]]:
    """Run Docling on one PDF and return a list of per-page dicts.

    Each dict contains:
      - game_id, doc_name, page_num
      - text: full page text
      - bboxes: list of {x0, y0, x1, y1, text}  (Docling bottom-left origin, pts)
    """
    pipeline_options = PdfPipelineOptions()
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(str(pdf_path.resolve()))

    pages_data: list[dict[str, Any]] = []

    for page_num in sorted(result.document.pages.keys()):
        items_for_page = list(result.document.iterate_items(page_no=page_num))

        text_parts: list[str] = []
        bboxes: list[dict[str, Any]] = []

        for item, _ in items_for_page:
            item_text = ""
            if getattr(item, "text", None):
                item_text = str(item.text)
                text_parts.append(item_text)

            if getattr(item, "prov", None):
                for prov in item.prov:
                    if getattr(prov, "bbox", None):
                        bbox = prov.bbox
                        bboxes.append(
                            {
                                "x0": bbox.l,
                                "y0": bbox.t,
                                "x1": bbox.r,
                                "y1": bbox.b,
                                "text": item_text,
                            }
                        )

        pages_data.append(
            {
                "game_id": game_id,
                "doc_name": doc_name,
                "page_num": page_num,
                "text": "\n\n".join(text_parts),
                "bboxes": bboxes,
            }
        )

    return pages_data


def get_or_extract(
    pdf_path: Path,
    game_id: str,
    doc_name: str,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Return cached Docling output for *pdf_path*, running Docling if needed.

    Cache lives at data/games/{game_id}/extracted/{doc_name}.json.
    Pass force=True to ignore the cache and re-run Docling.
    """
    cache_path = DATA_DIR / "games" / game_id / "extracted" / f"{doc_name}.json"

    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    print(f"  Docling extracting: {pdf_path.name} …")
    pages = _extract_single_pdf(pdf_path, game_id, doc_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(pages), encoding="utf-8")
    return pages


def load_cached_pages(game_id: str, doc_name: str) -> list[dict[str, Any]] | None:
    """Load already-cached Docling output without running extraction."""
    cache_path = DATA_DIR / "games" / game_id / "extracted" / f"{doc_name}.json"
    if not cache_path.exists():
        return None
    return json.loads(cache_path.read_text(encoding="utf-8"))


def extract_source(
    source: Path,
    game_id: str,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Extract from a single PDF or every PDF in a folder.

    Returns all pages across all documents, each tagged with doc_name.
    """
    source = Path(source)
    pdf_paths = sorted(source.glob("*.pdf")) if source.is_dir() else [source]
    if not pdf_paths:
        raise ValueError(f"No PDF files found at {source}")

    all_pages: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        doc_name = pdf_path.stem
        all_pages.extend(get_or_extract(pdf_path, game_id, doc_name, force=force))
    return all_pages
