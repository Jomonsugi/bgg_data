"""Boardgame Rules Agent — Streamlit app.

Layout
------
  Sidebar  : game selector, document management, web search domains
  Left col : chat interface
  Right col: PDF viewer (scrollable) with highlighted citation overlay
"""

from __future__ import annotations

import streamlit as st

from bgg_data.boardgame_agent.db.games import init_db, save_qa, set_qa_status
from bgg_data.boardgame_agent.agent.graph import build_agent, run_query
from bgg_data.boardgame_agent.agent.schemas import Citation, QAWithCitations
from bgg_data.boardgame_agent.ui.sidebar import render_sidebar
from bgg_data.boardgame_agent.ui.pdf_panel import render_highlighted_page, show_pdf_viewer

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Board Game Rules",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def get_agent(game_id: str, game_name: str, model_name: str, top_k: int):
    """Build and cache the LangGraph agent keyed by game + model + top_k."""
    return build_agent(game_id, game_name, model_name=model_name, top_k=top_k)


# ── Session state defaults ────────────────────────────────────────────────────

_LAYOUT_PRESETS = {
    "Chat":  [3, 2],
    "Equal": [1, 1],
    "PDF":   [2, 3],
}

def _init_session() -> None:
    defaults = {
        "messages": [],          # list of {"role", "content", "citations"}
        "active_citation": None, # Citation | None
        "active_doc": None,      # doc_name of the PDF currently in the viewer
        "layout": "Equal",       # one of the _LAYOUT_PRESETS keys
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Citation rendering ────────────────────────────────────────────────────────

def _render_citation_chips(citations: list[dict], game_id: str) -> None:
    """Render each citation as a clickable button that updates the PDF panel."""
    if not citations:
        return
    st.markdown("**Citations:**")
    cols = st.columns(min(len(citations), 4))
    for i, c in enumerate(citations):
        doc = c.get("doc_name", "")
        page = c.get("page_num", "?")
        label = f"📄 {doc} · p.{page}"
        with cols[i % 4]:
            if st.button(label, key=f"cite_{id(c)}_{i}", use_container_width=True):
                st.session_state.active_citation = c
                st.session_state.active_doc = doc
                st.rerun()


def _render_accept_buttons(msg: dict) -> None:
    """Render accept / reject buttons for an assistant message.

    Buttons update the DB immediately and toggle — clicking the active status
    again resets it to unreviewed (NULL), allowing corrections at any time.
    """
    qa_id = msg.get("qa_id")
    if qa_id is None:
        return

    status = st.session_state.get(f"qa_status_{qa_id}")  # True / False / None

    col_a, col_b, _ = st.columns([1, 1, 6])
    accept_label = "✓ Accepted" if status is True else "✓ Accept"
    reject_label = "✗ Rejected" if status is False else "✗ Reject"

    if col_a.button(accept_label, key=f"accept_{qa_id}", type="primary" if status is True else "secondary"):
        new_status = None if status is True else True  # toggle off if already accepted
        set_qa_status(qa_id, new_status)
        st.session_state[f"qa_status_{qa_id}"] = new_status
        st.rerun()

    if col_b.button(reject_label, key=f"reject_{qa_id}", type="primary" if status is False else "secondary"):
        new_status = None if status is False else False  # toggle off if already rejected
        set_qa_status(qa_id, new_status)
        st.session_state[f"qa_status_{qa_id}"] = new_status
        st.rerun()


def _render_web_sources(web_sources: list[str]) -> None:
    """Render clickable web source links when search_web was used."""
    if not web_sources:
        return
    st.markdown("**Web sources:**")
    for url in web_sources:
        st.markdown(f"- [{url}]({url})")


def _render_message(msg: dict, game_id: str) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("citations"):
                _render_citation_chips(msg["citations"], game_id)
            if msg.get("web_sources"):
                _render_web_sources(msg["web_sources"])
            _render_accept_buttons(msg)


# ── PDF panel ─────────────────────────────────────────────────────────────────

def _render_pdf_panel(game_id: str) -> None:
    citation = st.session_state.active_citation
    active_doc = st.session_state.active_doc

    if citation:
        doc_name = citation.get("doc_name", active_doc)
        page_num = citation.get("page_num", 1)
        bbox_indices = citation.get("bbox_indices", [])

        st.markdown(f"#### {doc_name} · Page {page_num}")

        # Render highlighted page image at the top
        img = render_highlighted_page(game_id, doc_name, page_num, bbox_indices)
        if img:
            st.image(img, use_container_width=True)
        else:
            st.warning("Could not render page — ensure the PDF is indexed.")

        if st.button("Clear citation", key="clear_citation"):
            st.session_state.active_citation = None
            st.rerun()

        st.divider()
        st.markdown("**Full document:**")
        show_pdf_viewer(game_id, doc_name, scroll_to_page=page_num)

    elif active_doc:
        st.markdown(f"#### {active_doc}")
        show_pdf_viewer(game_id, active_doc)
    else:
        st.markdown("#### PDF Viewer")
        st.info("Click a citation in the chat to view the source page with highlights.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    init_db()
    _init_session()

    game_id, game_name, selected_model, top_k = render_sidebar()

    if game_id is None:
        st.markdown("## Welcome to the Board Game Rules Agent")
        st.markdown(
            "Use the sidebar to **create a game** and **add your rulebook PDFs**. "
            "Once indexed, ask any rules question and get cited answers instantly."
        )
        return

    # Clear chat history only when the active game changes.
    # Model and top-k changes are fluid — history stays intact.
    if st.session_state.get("current_game_id") != game_id:
        st.session_state.messages = []
        st.session_state.active_citation = None
        st.session_state.active_doc = None
        st.session_state.current_game_id = game_id

    # ── Header row: title + layout presets ───────────────────────────────────
    title_col, layout_col = st.columns([3, 1])
    title_col.markdown(f"## {game_name} — Rules Assistant")
    with layout_col:
        chosen = st.radio(
            "Layout",
            options=list(_LAYOUT_PRESETS.keys()),
            index=list(_LAYOUT_PRESETS.keys()).index(st.session_state.layout),
            horizontal=True,
            label_visibility="collapsed",
        )
        if chosen != st.session_state.layout:
            st.session_state.layout = chosen
            st.rerun()

    chat_col, pdf_col = st.columns(_LAYOUT_PRESETS[st.session_state.layout], gap="large")

    # ── Chat column ───────────────────────────────────────────────────────────
    with chat_col:
        # Render conversation history
        for msg in st.session_state.messages:
            _render_message(msg, game_id)

        # Input
        if query := st.chat_input("Ask a rules question…"):
            # Show user message immediately
            st.session_state.messages.append(
                {"role": "user", "content": query, "citations": []}
            )
            with st.chat_message("user"):
                st.markdown(query)

            # Run agent
            compiled, llm, qdrant_client, text_model = get_agent(
                game_id, game_name, selected_model, top_k
            )

            with st.chat_message("assistant"):
                with st.spinner("Consulting the rulebook…"):
                    qa: QAWithCitations = run_query(compiled, game_id, query)

                st.markdown(qa.answer)

                citations_dicts = [c.model_dump() for c in qa.citations]
                _render_citation_chips(citations_dicts, game_id)

                if qa.web_sources:
                    _render_web_sources(qa.web_sources)

                # Set the first citation's doc as the active PDF
                if qa.citations:
                    st.session_state.active_doc = qa.citations[0].doc_name

            # Save to Q&A history (with embedding for future get_past_answers lookups)
            qa_id: int | None = None
            try:
                query_emb = list(text_model.embed([query]))[0]
                qa_id = save_qa(
                    game_id,
                    query,
                    qa.answer,
                    citations_dicts,
                    embedding=query_emb,
                    model_name=selected_model,
                    top_k=top_k,
                )
                # Seed session status as unreviewed
                st.session_state[f"qa_status_{qa_id}"] = None
            except Exception as e:
                st.warning(f"Could not save Q&A to history: {e}")

            # Persist message with qa_id so accept/reject buttons can reference it
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": qa.answer,
                    "citations": citations_dicts,
                    "web_sources": qa.web_sources,
                    "qa_id": qa_id,
                }
            )

            st.rerun()

    # ── PDF column ────────────────────────────────────────────────────────────
    with pdf_col:
        _render_pdf_panel(game_id)


if __name__ == "__main__":
    main()
