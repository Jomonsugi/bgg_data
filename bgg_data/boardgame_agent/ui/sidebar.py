"""Streamlit sidebar: game management, document management, and search domains."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import streamlit as st

from bgg_data.boardgame_agent.config import (
    DATA_DIR,
    DEFAULT_MODEL,
    EMBED_MODEL_NAME,
    MODEL_OPTIONS,
    RETRIEVAL_TOP_K,
)
from bgg_data.boardgame_agent.db.games import (
    add_search_domain,
    clear_search_domains,
    create_game,
    delete_document,
    get_all_games,
    get_documents,
    get_search_domains,
    init_db,
    register_document,
    remove_search_domain,
)
from bgg_data.boardgame_agent.rag.extractor import chunk_by_sections, get_or_extract
from bgg_data.boardgame_agent.rag.indexer import build_index, reindex_all, remove_doc_from_index


def _game_id_from_name(name: str) -> str:
    """Convert a game name to a safe identifier."""
    return re.sub(r"[^a-z0-9_]", "_", name.strip().lower()).strip("_")


def render_sidebar() -> tuple[str | None, str | None, str, int]:
    """Render the full sidebar.

    Returns (game_id, game_name, selected_model, top_k).
    game_id / game_name are None when no game is selected.
    """
    init_db()

    with st.sidebar:
        st.title("Board Game Rules")

        # ── Model settings ────────────────────────────────────────────────────
        with st.expander("Model settings", expanded=False):
            _model_list = list(MODEL_OPTIONS.keys())
            selected_model = st.selectbox(
                "LLM model",
                options=_model_list,
                index=_model_list.index(DEFAULT_MODEL) if DEFAULT_MODEL in _model_list else 0,
                key="selected_model",
            )
            top_k = st.slider(
                "Retrieval pages (top-k)",
                min_value=1,
                max_value=15,
                value=RETRIEVAL_TOP_K,
                step=1,
                key="top_k",
                help="Number of rulebook pages retrieved per query.",
            )
            st.caption(f"**Embeddings:** `{EMBED_MODEL_NAME}`")
            if st.button("Rebuild index (new embed model)", width='stretch'):
                with st.spinner("Rebuilding Qdrant index from cached Docling data…"):
                    reindex_all()
                st.success("Index rebuilt.")

        st.divider()

        # ── Game selector ─────────────────────────────────────────────────────
        games = get_all_games()
        game_names = [g["game_name"] for g in games]
        game_ids = [g["game_id"] for g in games]

        selected_game_name = None
        selected_game_id = None

        if game_names:
            idx = st.selectbox(
                "Select game",
                options=range(len(game_names)),
                format_func=lambda i: game_names[i],
                key="selected_game_idx",
            )
            selected_game_id = game_ids[idx]
            selected_game_name = game_names[idx]

        # ── Add new game ──────────────────────────────────────────────────────
        with st.expander("Add new game"):
            new_name = st.text_input("Game name", key="new_game_name")
            if st.button("Create game", key="create_game_btn") and new_name.strip():
                gid = _game_id_from_name(new_name)
                create_game(gid, new_name.strip())
                st.success(f"Created: {new_name}")
                st.rerun()

        if selected_game_id is None:
            st.info("Create a game to get started.")
            return None, None, selected_model, top_k

        st.divider()

        # ── Documents ─────────────────────────────────────────────────────────
        st.subheader("Documents")
        docs = get_documents(selected_game_id)

        if docs:
            for doc in docs:
                col1, col2 = st.columns([4, 1])
                col1.write(f"📄 {doc['doc_name']}")
                if col2.button("✕", key=f"del_doc_{doc['doc_name']}", help="Remove"):
                    _remove_document(selected_game_id, doc["doc_name"])
                    st.rerun()
        else:
            st.caption("No documents indexed yet.")

        # Upload new documents
        uploaded = st.file_uploader(
            "Add PDF(s)",
            type="pdf",
            accept_multiple_files=True,
            key="doc_uploader",
        )
        if uploaded and st.button("Index uploaded PDFs", key="index_pdfs_btn"):
            _index_uploaded_pdfs(selected_game_id, uploaded)
            st.rerun()

        # Folder path shortcut (useful for local use)
        folder_path = st.text_input(
            "Or index a folder path", placeholder="/path/to/folder", key="folder_path"
        )
        if folder_path and st.button("Index folder", key="index_folder_btn"):
            _index_folder(selected_game_id, Path(folder_path))
            st.rerun()

        st.divider()

        # ── Web search domains ────────────────────────────────────────────────
        st.subheader("Web search domains")
        st.caption("Agent searches these sites. Empty = unrestricted.")

        domains = get_search_domains(selected_game_id)
        for domain in domains:
            col1, col2 = st.columns([4, 1])
            col1.write(f"🌐 {domain}")
            if col2.button("✕", key=f"del_dom_{domain}", help="Remove"):
                remove_search_domain(selected_game_id, domain)
                st.rerun()

        new_domain = st.text_input("Add domain", placeholder="example.com", key="new_domain")
        col_a, col_b = st.columns(2)
        if col_a.button("Add", key="add_domain_btn") and new_domain.strip():
            add_search_domain(selected_game_id, new_domain.strip())
            st.rerun()
        if col_b.button("Clear all", key="clear_domains_btn"):
            clear_search_domains(selected_game_id)
            st.rerun()

    return selected_game_id, selected_game_name, selected_model, top_k


# ── Document management helpers ───────────────────────────────────────────────

def _copy_pdf_to_store(game_id: str, pdf_path: Path, doc_name: str) -> Path:
    dest_dir = DATA_DIR / "games" / game_id / "pdfs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{doc_name}.pdf"
    if dest != pdf_path:
        shutil.copy2(pdf_path, dest)
    return dest


def _index_single_pdf(game_id: str, pdf_path: Path, doc_name: str) -> None:
    stored_path = _copy_pdf_to_store(game_id, pdf_path, doc_name)
    pages = get_or_extract(stored_path, game_id, doc_name)
    chunks = chunk_by_sections(pages)
    build_index(chunks)
    cache_path = DATA_DIR / "games" / game_id / "extracted" / f"{doc_name}.json"
    register_document(game_id, doc_name, stored_path, cache_path)


def _index_uploaded_pdfs(game_id: str, uploaded_files) -> None:
    import tempfile, os

    progress = st.progress(0, text="Indexing…")
    for i, uf in enumerate(uploaded_files):
        doc_name = Path(uf.name).stem
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uf.read())
            tmp_path = Path(tmp.name)
        try:
            with st.spinner(f"Processing {uf.name}…"):
                _index_single_pdf(game_id, tmp_path, doc_name)
        finally:
            os.unlink(tmp_path)
        progress.progress((i + 1) / len(uploaded_files), text=f"Indexed {uf.name}")
    progress.empty()
    st.success(f"Indexed {len(uploaded_files)} document(s).")


def _index_folder(game_id: str, folder: Path) -> None:
    if not folder.is_dir():
        st.error(f"Not a directory: {folder}")
        return
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        st.warning("No PDF files found in that folder.")
        return
    progress = st.progress(0, text="Indexing folder…")
    for i, pdf_path in enumerate(pdfs):
        doc_name = pdf_path.stem
        with st.spinner(f"Processing {pdf_path.name}…"):
            _index_single_pdf(game_id, pdf_path, doc_name)
        progress.progress((i + 1) / len(pdfs), text=f"Indexed {pdf_path.name}")
    progress.empty()
    st.success(f"Indexed {len(pdfs)} document(s) from folder.")


def _remove_document(game_id: str, doc_name: str) -> None:
    remove_doc_from_index(doc_name, game_id)
    delete_document(game_id, doc_name)
    # Remove cached extraction
    cache_path = DATA_DIR / "games" / game_id / "extracted" / f"{doc_name}.json"
    if cache_path.exists():
        cache_path.unlink()
    st.toast(f"Removed {doc_name}")
