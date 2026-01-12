from __future__ import annotations

"""
Optional Streamlit UI for ad-hoc runs.

Install separately:
  pip install streamlit

Run:
  streamlit run -m bgg_data.rule_book_agent.ui_streamlit

This UI supports HITL pause/resume *within the same Python process*.
"""

def main():
    try:
        import streamlit as st  # type: ignore
    except Exception as e:
        raise RuntimeError("Streamlit is not installed. Install `streamlit` to use this UI.") from e

    from .runner import FindBatchParams, FindOneParams, find_batch, find_one, resume

    st.set_page_config(page_title="Rule Book Agent", layout="wide")
    st.title("Rule Book Agent (ad-hoc)")

    tabs = st.tabs(["Find One", "Find Batch", "Resume"])

    with tabs[0]:
        st.subheader("Find One")
        game_name = st.text_input("Game name (must exist in DB)", value="")
        bgg_id = st.text_input("BGG id (optional)", value="")
        db_path = st.text_input("DB path", value="")
        recursion_limit = st.number_input("recursion_limit", min_value=5, max_value=200, value=30, step=5)
        if st.button("Run find_one"):
            params = FindOneParams(
                game_name=game_name or None,
                bgg_id=int(bgg_id) if bgg_id.strip() else None,
                db_path=db_path or "",
                recursion_limit=int(recursion_limit),
            )
            out = find_one(params)
            st.session_state["last_run_id"] = out.get("run_id")
            st.json(out)

            if out.get("run_paused"):
                st.warning("Run paused (human-in-the-loop). Solve the blocking step in the live browser session, then resume.")

    with tabs[1]:
        st.subheader("Find Batch (skips existing rulebooks)")
        rank_from = st.number_input("rank_from", min_value=1, max_value=100000, value=1, step=1)
        rank_to = st.number_input("rank_to", min_value=1, max_value=100000, value=50, step=1)
        limit = st.text_input("limit (optional)", value="")
        db_path_b = st.text_input("DB path (batch)", value="")
        recursion_limit_b = st.number_input("recursion_limit (batch)", min_value=5, max_value=200, value=30, step=5)
        if st.button("Run find_batch"):
            params = FindBatchParams(
                rank_from=int(rank_from),
                rank_to=int(rank_to),
                limit=int(limit) if limit.strip() else None,
                db_path=db_path_b or "",
                recursion_limit=int(recursion_limit_b),
            )
            out = find_batch(params)
            st.json(out)

    with tabs[2]:
        st.subheader("Resume (HITL)")
        default_run = st.session_state.get("last_run_id", "")
        run_id = st.text_input("run_id", value=default_run)
        recursion_limit_r = st.number_input("recursion_limit (resume)", min_value=5, max_value=200, value=30, step=5)
        if st.button("Resume run"):
            out = resume(run_id, recursion_limit=int(recursion_limit_r))
            st.json(out)

    st.divider()
    st.caption("Artifacts live under `bgg_data/bgg_data/rule_book_agent/runs/<run_id>/` and rulebooks under `.../rulebooks/`.")


if __name__ == "__main__":
    main()


