from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Suppress ML framework warnings (we use API-based models, not local ones)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from bgg_data.database.operations import BGGDatabase
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.errors import GraphRecursionError

from .graph import create_graph
from .runs import get_rulebooks_dir, init_run_dirs, make_run_id, write_json
from .tools.browser_helpers import build_browser_helper_tools
from .tools.browser_primitives import build_browser_primitive_tools
from .tools.download import build_download_tools
from .tools.extract import build_extract_tools
from .tools.hitl import build_hitl_tools
from .tools.search import build_search_tools
from .tools.state import build_state_tools
from .tools.validate import build_validate_tools
from .types import AgentState


@dataclass
class FindOneParams:
    game_name: Optional[str] = None
    bgg_id: Optional[int] = None
    db_path: str = ""
    recursion_limit: int = 50


@dataclass
class FindBatchParams:
    rank_from: int = 1
    rank_to: int = 50
    limit: Optional[int] = None
    db_path: str = ""
    recursion_limit: int = 50


def _invoke_graph_capture_last_state(graph, state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Invoke the graph but, if a recursion limit is hit, return the last emitted state
    instead of raising. This keeps batch runs going and ensures we can write a
    useful `final_state.json` for debugging.
    """
    last_state: AgentState | None = None
    try:
        for chunk in graph.stream(state, config=config, stream_mode="values"):
            # stream_mode="values" yields the full state dict over time
            last_state = chunk
        return last_state or state
    except GraphRecursionError as e:
        s = dict(last_state or state)
        s["error"] = {
            "type": "recursion_limit",
            "message": str(e),
            "recursion_limit": int(config.get("recursion_limit", 0)) if config else None,
        }
        return s  # type: ignore[return-value]
    except Exception as e:
        s = dict(last_state or state)
        s["error"] = {"type": "exception", "message": str(e)}
        return s  # type: ignore[return-value]


def _default_db_path() -> str:
    # bgg_data/bgg_data/rule_book_agent/runner.py -> project root is parents[2]
    return str(Path(__file__).resolve().parents[2] / "bgg_games.db")


def _load_chat_model():
    """
    Load a chat model for LangGraph.
    
    Supports:
    - Together API (via langchain_community)
    - OpenAI (via langchain_openai)
    
    Environment variables:
    - RULEBOOK_AGENT_PROVIDER: "together" or "openai" (default: "together")
    - TOGETHER_API_KEY: Together API key (required if provider=together)
    - OPENAI_API_KEY: OpenAI API key (required if provider=openai)
    - RULEBOOK_AGENT_MODEL: Model name (defaults vary by provider)
    """
    provider = os.getenv("RULEBOOK_AGENT_PROVIDER", "together").lower()
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (required for provider=openai)")
        
        model = os.getenv("RULEBOOK_AGENT_MODEL", "gpt-4o")
        return ChatOpenAI(model=model, api_key=api_key, temperature=0, verbose=True)
    
    elif provider == "together":
        from langchain_together import ChatTogether
        
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY is not set (required for provider=together)")
        
        model = os.getenv("RULEBOOK_AGENT_MODEL", "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8")
        # ChatTogether reads TOGETHER_API_KEY from env, but we can also pass it explicitly
        return ChatTogether(model=model, together_api_key=api_key, temperature=0, verbose=True)
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'together' or 'openai'")


def _build_tools():
    return (
        build_search_tools()
        + build_browser_primitive_tools()
        + build_browser_helper_tools()
        + build_extract_tools()
        + build_download_tools()
        + build_validate_tools()
        + build_hitl_tools()
        + build_state_tools()
    )


def _resolve_game(game_name: Optional[str], bgg_id: Optional[int], db_path: str):
    if not db_path:
        db_path = _default_db_path()

    db = BGGDatabase(Path(db_path))
    if bgg_id is not None:
        # Query directly without changing shared database module.
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(
            "SELECT bgg_id, name, rank, url, publisher, year_published FROM games WHERE bgg_id = ?",
            (int(bgg_id),),
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            raise ValueError("bgg_id not found in DB")
        from bgg_data.database.models import Game

        return Game(
            id=str(row[0]),
            name=row[1],
            rank=row[2],
            url=row[3],
            publisher=row[4],
            year_published=row[5],
        )

    if not game_name:
        raise ValueError("Provide game_name or bgg_id")
    game = db.get_game_by_name(game_name)
    if not game:
        raise ValueError("game_name not found in DB")
    return game


def _existing_rulebook_paths(game_name: str, bgg_id: Optional[str]) -> list[str]:
    rulebooks_dir = get_rulebooks_dir()
    base = game_name.replace(" ", "-").replace(":", "").replace("'", "")
    candidates = []
    if bgg_id:
        candidates.append(rulebooks_dir / f"{base}_{bgg_id}_rules.pdf")
        candidates.append(rulebooks_dir / f"{base}_{bgg_id}_rules.html")
    candidates.append(rulebooks_dir / f"{base}_rules.pdf")
    candidates.append(rulebooks_dir / f"{base}_rules.html")
    return [str(p) for p in candidates if Path(p).exists()]


def _ensure_langsmith_project_default() -> None:
    # Only set a default if the user hasn't already configured it.
    os.environ.setdefault("LANGCHAIN_PROJECT", "boardgame-rulebook-finder")


def find_one(params: FindOneParams, parent_config: RunnableConfig | None = None) -> dict:
    """
    Run the agent for a single game.

    If parent_config is provided (e.g., from a batch run), we reuse its callbacks
    so the run becomes a child trace in LangSmith.
    """
    _ensure_langsmith_project_default()
    db_path = params.db_path or _default_db_path()
    game = _resolve_game(params.game_name, params.bgg_id, db_path)

    existing = _existing_rulebook_paths(game.name, str(game.id))
    if existing:
        return {"skipped": True, "reason": "rulebook already exists", "files": existing, "game": {"id": game.id, "name": game.name}}

    run_id = make_run_id()
    run_dir = init_run_dirs(run_id)

    chat = _load_chat_model()
    tools = _build_tools()
    graph = create_graph(chat, tools)

    state: AgentState = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "game": {
            "id": game.id,
            "name": game.name,
            "rank": game.rank or 0,
            "url": game.url or "",
            "publisher": game.publisher or "",
            "year_published": game.year_published or 0,
        },
        "messages": [
            {
                "type": "human",
                "content": (
                    f"Find and download the correct rulebook for the board game '{game.name}'. "
                    "Use your tools to search, browse, download, and validate. "
                    "When validated, call set_validated_rulebook."
                ),
            }
        ],
    }

    invoke_config: RunnableConfig = {
        "recursion_limit": int(params.recursion_limit),
        "run_name": f"rulebook:{game.name}",
        "tags": ["rule_book_agent", "rulebook", "bgg"],
        "metadata": {
            "run_id": run_id,
            "bgg_id": str(game.id),
            "game_name": game.name,
            "rank": game.rank,
            "db_path": db_path,
        },
    }
    # Reuse parent callbacks for nested LangSmith traces
    if parent_config and parent_config.get("callbacks") is not None:
        invoke_config["callbacks"] = parent_config.get("callbacks")

    from .tools.browser_primitives import cleanup_browser_sessions_for_run

    try:
        result = _invoke_graph_capture_last_state(graph, state, invoke_config)
        write_json(run_dir / "final_state.json", result)
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "validated_rulebook": result.get("validated_rulebook"),
            "run_paused": bool(result.get("run_paused")),
            "pause_reason": result.get("pause_reason"),
            "error": result.get("error"),
        }
    finally:
        # Important: if the graph throws (e.g. recursion limit), finalize() won't run.
        # Always close browser sessions so Playwright's Node driver doesn't die with EPIPE.
        try:
            cleanup_browser_sessions_for_run(run_id)
        except Exception:
            pass


def find_batch(params: FindBatchParams) -> dict:
    """
    Batch runner with LangSmith-friendly nesting:
    - one parent run for the batch
    - one child run per game (reusing callback manager)
    """
    _ensure_langsmith_project_default()
    db_path = params.db_path or _default_db_path()

    def _batch(_: dict, config: RunnableConfig) -> dict:
        db = BGGDatabase(Path(db_path))
        games = db.get_games(limit=params.limit, rank_from=params.rank_from, rank_to=params.rank_to)
        missing = [g for g in games if not _existing_rulebook_paths(g.name, str(g.id))]

        results = []
        for g in missing:
            one = FindOneParams(game_name=g.name, db_path=db_path, recursion_limit=params.recursion_limit)
            result = find_one(one, parent_config=config)
            # Add game info for better batch output
            result["game_name"] = g.name
            result["game_rank"] = g.rank
            result["game_id"] = g.id
            results.append(result)

        return {"total": len(games), "missing": len(missing), "results": results}

    batch_runnable = RunnableLambda(_batch).with_config(
        run_name=f"batch:rank_{params.rank_from}_{params.rank_to}",
        tags=["rule_book_agent", "batch", "bgg"],
        metadata={"rank_from": params.rank_from, "rank_to": params.rank_to, "limit": params.limit, "db_path": db_path},
    )
    return batch_runnable.invoke({}, config={})


def resume(run_id: str, recursion_limit: int = 30) -> dict:
    """
    Resume a paused run by loading saved state and re-invoking the graph.

    Important: browser sessions are held in-memory; resume expects the same Python
    process to still be running.
    """
    run_dir = init_run_dirs(run_id)
    state_path = run_dir / "final_state.json"
    if not state_path.exists():
        raise ValueError("No saved state found for run_id")
    loaded = json.loads(state_path.read_text(encoding="utf-8"))

    chat = _load_chat_model()
    tools = _build_tools()
    graph = create_graph(chat, tools)
    from .tools.browser_primitives import cleanup_browser_sessions_for_run

    try:
        result = _invoke_graph_capture_last_state(graph, loaded, {"recursion_limit": int(recursion_limit)})
        write_json(state_path, result)
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "validated_rulebook": result.get("validated_rulebook"),
            "run_paused": bool(result.get("run_paused")),
            "pause_reason": result.get("pause_reason"),
            "error": result.get("error"),
        }
    finally:
        try:
            cleanup_browser_sessions_for_run(run_id)
        except Exception:
            pass


