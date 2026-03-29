"""PDF rendering helpers.

Provides two complementary views:
  1. render_highlighted_page — PyMuPDF renders a single page with bbox highlights
     as a PIL Image. Used when the user clicks a citation.
  2. show_pdf_viewer — renders the full scrollable PDF using streamlit-pdf-viewer.
     Used as the persistent right-panel PDF browser.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from bgg_data.boardgame_agent.config import DATA_DIR
from bgg_data.boardgame_agent.rag.extractor import load_cached_pages


def get_pdf_path(game_id: str, doc_name: str) -> Path | None:
    """Find the PDF for a document, checking both docs/ (new) and pdfs/ (legacy)."""
    for subdir in ("docs", "pdfs"):
        p = DATA_DIR / "games" / game_id / subdir / f"{doc_name}.pdf"
        if p.exists():
            return p
    return None


def render_highlighted_page(
    game_id: str,
    doc_name: str,
    page_num: int,
    bbox_indices: list[int],
    dpi: int = 150,
) -> Image.Image | None:
    """Render *page_num* of *doc_name* with cited bboxes highlighted in yellow.

    Bbox coordinates come from the cached Docling JSON so no live DB query is
    needed. Returns None if the page or PDF cannot be found.

    Docling stores bboxes with bottom-left origin; PyMuPDF uses top-left.
    The conversion is: top_y = page_height - docling_y.
    """
    pdf_path = get_pdf_path(game_id, doc_name)
    if pdf_path is None:
        return None

    pages = load_cached_pages(game_id, doc_name)
    if pages is None:
        return None

    page_data = next((p for p in pages if p["page_num"] == page_num), None)
    if page_data is None:
        return None

    bboxes = page_data.get("bboxes", [])
    doc = fitz.open(str(pdf_path.resolve()))
    try:
        fitz_page = doc[page_num - 1]  # PyMuPDF is 0-indexed
        page_height = fitz_page.rect.height

        for idx in bbox_indices:
            if 0 <= idx < len(bboxes):
                b = bboxes[idx]
                x0, y0, x1, y1 = b["x0"], b["y0"], b["x1"], b["y1"]
                # Docling: bottom-left origin → PyMuPDF: top-left origin
                top_y0 = page_height - y1
                top_y1 = page_height - y0
                rect = fitz.Rect(min(x0, x1), min(top_y0, top_y1), max(x0, x1), max(top_y0, top_y1))
                annot = fitz_page.add_highlight_annot(rect)
                annot.set_colors(stroke=(1, 1, 0))
                annot.update()

        pix = fitz_page.get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()


def show_pdf_viewer(game_id: str, doc_name: str, scroll_to_page: int = 1) -> None:
    """Display the full scrollable PDF in the Streamlit right panel.

    Uses streamlit-pdf-viewer. Falls back gracefully if the PDF is not found.
    """
    import streamlit as st

    pdf_path = get_pdf_path(game_id, doc_name)
    if pdf_path is None:
        st.warning(f"PDF not found: {doc_name}.pdf")
        return

    try:
        from streamlit_pdf_viewer import pdf_viewer

        pdf_viewer(
            input=str(pdf_path),
            height=700,
            scroll_to_page=scroll_to_page,
        )
    except ImportError:
        st.info(
            "Install `streamlit-pdf-viewer` for the embedded viewer. "
            f"Currently showing: **{doc_name}** · Page {scroll_to_page}"
        )
        # Fallback: render the target page as an image
        img = render_highlighted_page(game_id, doc_name, scroll_to_page, [])
        if img:
            st.image(img, caption=f"{doc_name} · Page {scroll_to_page}")
