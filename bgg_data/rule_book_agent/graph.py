from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .config import default_db_path
from .prompts import SYSTEM_PROMPT
from .types import AgentState


def _default_budgets() -> dict:
    return {
        "max_downloads": 8,
        "max_sites_opened": 15,
    }


def create_graph(chat_model, tools: list[Any]):
    """
    Create the LangGraph tool loop.

    The model decides tool order. This graph only enforces loop mechanics
    and provides a finalization step.

    Note on budgets:
    - Use LangGraph's built-in `recursion_limit` (passed via invoke/stream config)
      for step/iteration limits.
    - App-specific limits (downloads, sites opened) should be enforced by tools
      and/or the caller, since LangGraph does not have semantic knowledge of them.
    """

    sys_msg = SystemMessage(content=SYSTEM_PROMPT)
    chat_with_tools = chat_model.bind_tools(tools)

    def init_state(state: AgentState) -> dict:
        budgets = state.get("budgets") or _default_budgets()
        counters = state.get("counters") or {"downloads": 0, "sites_opened": 0}
        return {"budgets": budgets, "counters": counters}

    def ensure_game_in_db(state: AgentState) -> dict:
        """
        Deterministic step: Check if game is in database, and if not, find and add it.
        
        This runs BEFORE the agentic loop. If this step fails, the graph stops.
        The agentic loop should NEVER run without a valid game record in the DB.
        
        Handles two cases:
        1. BGG ID provided but not in DB -> fetch directly from BGG XML API
        2. Game name only -> search BGG via Tavily, then fetch details
        
        Returns:
            - On success: dict with updated "game" info
            - On failure: dict with "run_paused": True and error message (stops graph)
        """
        from ..database.tools import find_and_add_game_to_db
        import sqlite3
        
        game = state.get("game", {})
        game_name = game.get("name", "")
        game_id = game.get("id", "")
        
        if not game_name and not game_id:
            return {
                "run_paused": True,
                "pause_reason": "No game name or BGG ID provided. Cannot proceed without game information.",
            }
        
        # Get db_path from state metadata or use default
        db_path = state.get("db_path") or default_db_path()
        
        # Check if game exists in DB
        from ..database.operations import BGGDatabase
        from pathlib import Path
        
        db = BGGDatabase(Path(db_path))
        
        # If we have a BGG ID, check by ID first
        if game_id and game_id != "0":
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute(
                "SELECT bgg_id, name, rank, url, publisher, year_published FROM games WHERE bgg_id = ?",
                (int(game_id),),
            )
            row = cur.fetchone()
            conn.close()
            
            if row:
                # Game exists in DB - proceed to agentic loop
                return {
                    "game": {
                        "id": str(row[0]),
                        "name": row[1],
                        "rank": row[2] or 0,
                        "url": row[3] or "",
                        "publisher": row[4] or "",
                        "year_published": row[5] or 0,
                    }
                }
            
            # Game ID provided but not in DB - fetch directly using the ID
            result = find_and_add_game_to_db(
                game_name=game_name or "Unknown",  # Use name if available, otherwise placeholder
                db_path=db_path,
                bgg_id=game_id  # Pass the BGG ID directly
            )
            
            if result.get('ok') and result.get('game'):
                game_data = result['game']
                return {
                    "game": {
                        "id": game_data['id'],
                        "name": game_data['name'],
                        "rank": game_data.get('rank', 0),
                        "url": game_data.get('url', ''),
                        "publisher": game_data.get('publisher', ''),
                        "year_published": game_data.get('year_published', 0),
                    }
                }
            else:
                # FAIL HARD - stop the graph, don't proceed to agentic loop
                error_msg = result.get('error', 'Unknown error')
                return {
                    "run_paused": True,
                    "pause_reason": f"Could not add game with BGG ID {game_id} to database: {error_msg}. The game must be successfully added to the database before the rulebook search can begin.",
                }
        
        # No BGG ID or BGG ID is "0" - check by name, then search if needed
        if game_name:
            existing = db.get_game_by_name(game_name)
            if existing and existing.id and existing.id != "0":
                # Game is in DB - proceed to agentic loop
                return {
                    "game": {
                        "id": existing.id,
                        "name": existing.name,
                        "rank": existing.rank or 0,
                        "url": existing.url or "",
                        "publisher": existing.publisher or "",
                        "year_published": existing.year_published or 0,
                    }
                }
            
            # Game not in DB - search and add it
            result = find_and_add_game_to_db(game_name, db_path=db_path)
            
            if result.get('ok') and result.get('game'):
                game_data = result['game']
                return {
                    "game": {
                        "id": game_data['id'],
                        "name": game_data['name'],
                        "rank": game_data.get('rank', 0),
                        "url": game_data.get('url', ''),
                        "publisher": game_data.get('publisher', ''),
                        "year_published": game_data.get('year_published', 0),
                    }
                }
            else:
                # FAIL HARD - stop the graph, don't proceed to agentic loop
                error_msg = result.get('error', 'Unknown error')
                return {
                    "run_paused": True,
                    "pause_reason": f"Could not add game '{game_name}' to database: {error_msg}. The game must be successfully added to the database before the rulebook search can begin.",
                }
        
        # Fallback: no name or ID
        return {
            "run_paused": True,
            "pause_reason": "Cannot proceed without a game name or BGG ID.",
        }

    def apply_tool_effects(state: AgentState) -> dict:
        """
        Apply state updates based on the latest tool outputs.

        This is NOT a fallback ladder. It only interprets explicit state-setting tools
        (e.g. set_validated_rulebook, request_human_help) so the agent can end/pause runs.
        """
        from langchain_core.messages import ToolMessage
        import ast
        import json

        msgs = state.get("messages", [])
        if not msgs or not isinstance(msgs[-1], ToolMessage):
            return {}

        last: ToolMessage = msgs[-1]
        tool_name = getattr(last, "name", None) or ""
        content = last.content

        def parse_content(val):
            if isinstance(val, dict):
                return val
            if not isinstance(val, str):
                return {"raw": str(val)}
            s = val.strip()
            if not s:
                return {}
            try:
                return json.loads(s)
            except Exception:
                try:
                    return ast.literal_eval(s)
                except Exception:
                    return {"raw": s}

        data = parse_content(content)

        if tool_name == "set_validated_rulebook":
            vr = data.get("validated_rulebook")
            if vr:
                return {"validated_rulebook": vr}

        if tool_name == "request_human_help":
            reason = (data.get("instructions") or data.get("reason") or "human help requested").strip()
            return {"run_paused": True, "pause_reason": reason}

        return {}

    def assistant(state: AgentState, config: RunnableConfig):
        # Ensure messages list is always message objects
        msgs = state.get("messages", [])
        if msgs and isinstance(msgs[0], str):
            msgs = [HumanMessage(content=msgs[0])]

        out = chat_with_tools.invoke([sys_msg] + msgs, config=config)
        return {"messages": [out]}

    def should_continue(state: AgentState):
        # Stop if we already have a validated rulebook
        if state.get("validated_rulebook"):
            return END
        if state.get("run_paused"):
            return END

        # Otherwise continue based on tool calling condition
        return tools_condition(state)

    def check_game_in_db_result(state: AgentState):
        """
        Check if ensure_game_in_db succeeded or failed.
        If failed (run_paused=True), stop the graph.
        If succeeded, proceed to agentic loop.
        """
        if state.get("run_paused"):
            return END
        return "assistant"

    def finalize(state: AgentState):
        # Clean up browser sessions for this run
        from .tools.browser_primitives import cleanup_browser_sessions_for_run
        run_id = state.get("run_id")
        if run_id:
            try:
                cleanup_browser_sessions_for_run(run_id)
            except Exception:
                pass  # Don't fail finalization if cleanup fails
        
        # Replace last AI message with a concise final status if desired.
        # Keep minimal: if the model already produced a final answer, do nothing.
        if state.get("validated_rulebook"):
            vr = state["validated_rulebook"]
            msg = AIMessage(
                content=(
                    "Validated rulebook downloaded.\n"
                    f"file_path: {vr.get('file_path')}\n"
                    f"url: {vr.get('url')}\n"
                )
            )
            messages = state.get("messages", [])
            return {"messages": messages + [msg]}
        return {}

    builder = StateGraph(AgentState)
    builder.add_node("init_state", init_state)
    builder.add_node("ensure_game_in_db", ensure_game_in_db)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("apply_tool_effects", apply_tool_effects)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "init_state")
    builder.add_edge("init_state", "ensure_game_in_db")
    builder.add_conditional_edges("ensure_game_in_db", check_game_in_db_result, {"assistant": "assistant", END: "finalize"})
    builder.add_conditional_edges("assistant", should_continue, {"tools": "tools", END: "finalize"})
    builder.add_edge("tools", "apply_tool_effects")
    builder.add_edge("apply_tool_effects", "assistant")
    builder.add_edge("finalize", END)

    return builder.compile()


