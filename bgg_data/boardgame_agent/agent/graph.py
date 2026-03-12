"""LangGraph ReAct agent for the boardgame rules assistant.

Architecture
------------
1. call_agent  — LLM with bound tools (ReAct loop)
2. call_tools  — ToolNode executes requested tool calls
3. format_answer — final LLM call with structured output → QAWithCitations

The graph loops between call_agent and call_tools until the LLM stops
requesting tools, then routes to format_answer which produces the structured
response stored in state["final_answer"].
"""

from __future__ import annotations

import sqlite3
import uuid
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_together import ChatTogether
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from fastembed import TextEmbedding

from bgg_data.boardgame_agent.agent.prompts import FORMAT_PROMPT, SYSTEM_PROMPT_TEMPLATE
from bgg_data.boardgame_agent.agent.schemas import QAWithCitations
from bgg_data.boardgame_agent.agent.state import AgentState
from bgg_data.boardgame_agent.agent.tools import make_all_tools
from bgg_data.boardgame_agent.config import (
    ANTHROPIC_API_KEY,
    CHECKPOINTS_DB_PATH,
    DEFAULT_MODEL,
    EMBED_MODEL_NAME,
    GAMES_DB_PATH,
    MODEL_OPTIONS,
    OPENAI_API_KEY,
    TOGETHER_API_KEY,
)
from bgg_data.boardgame_agent.rag.indexer import get_qdrant_client


def _build_llm(model_name: str):
    """Instantiate the correct LangChain chat class based on MODEL_OPTIONS."""
    provider = MODEL_OPTIONS.get(model_name, "together")
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, api_key=ANTHROPIC_API_KEY, temperature=0)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, temperature=0)
    else:
        return ChatTogether(model=model_name, together_api_key=TOGETHER_API_KEY, temperature=0)


def build_agent(
    game_id: str,
    game_name: str,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 5,
) -> tuple[Any, Any, QdrantClient, TextEmbedding]:
    """Compile the LangGraph agent for *game_id*.

    Returns (compiled_graph, llm, qdrant_client, text_model).
    The caller should cache the result keyed by (game_id, model_name, top_k).
    """
    qdrant_client = get_qdrant_client()
    text_model = TextEmbedding(model_name=EMBED_MODEL_NAME)
    tools = make_all_tools(
        game_id, game_name, qdrant_client, text_model, GAMES_DB_PATH, top_k=top_k
    )

    llm = _build_llm(model_name)
    llm_with_tools = llm.bind_tools(tools)
    system_message = SystemMessage(
        content=SYSTEM_PROMPT_TEMPLATE.format(game_name=game_name)
    )

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def call_agent(state: AgentState) -> dict:
        all_messages = list(state["messages"])

        # Find the last AIMessage so we know which tool outputs have been processed.
        last_ai_idx = max(
            (i for i, m in enumerate(all_messages) if isinstance(m, AIMessage)),
            default=-1,
        )

        # Compress ToolMessages that the LLM has already seen (before last AI turn)
        # to free context space, while preserving tool_call_id pairing.
        compressed: list = []
        for i, m in enumerate(all_messages):
            if isinstance(m, ToolMessage) and i < last_ai_idx:
                compressed.append(
                    ToolMessage(
                        content=f"[retrieved {len(m.content)} chars — already processed]",
                        tool_call_id=m.tool_call_id,
                        name=getattr(m, "name", "tool"),
                    )
                )
            else:
                compressed.append(m)

        response = llm_with_tools.invoke([system_message] + compressed)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    def format_answer(state: AgentState) -> dict:
        """Produce structured QAWithCitations from the completed message history."""
        structured_llm = llm.with_structured_output(QAWithCitations)

        # Collect tool outputs so the formatter has full citation context.
        tool_outputs = "\n\n".join(
            f"[Tool: {m.name}]\n{m.content}"
            for m in state["messages"]
            if isinstance(m, ToolMessage)
        )
        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            None,
        )
        agent_answer = last_ai.content if last_ai else ""

        format_input = (
            f"Agent answer:\n{agent_answer}\n\n"
            f"Tool outputs used:\n{tool_outputs}"
        )
        qa: QAWithCitations = structured_llm.invoke(
            [SystemMessage(content=FORMAT_PROMPT), HumanMessage(content=format_input)]
        )
        return {"final_answer": qa.model_dump()}

    # ── Routing ───────────────────────────────────────────────────────────────

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return "format_answer"

    # ── Graph ─────────────────────────────────────────────────────────────────

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_agent)
    graph.add_node("tools", tool_node)
    graph.add_node("format_answer", format_answer)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "format_answer": "format_answer"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("format_answer", END)

    conn = sqlite3.connect(str(CHECKPOINTS_DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    compiled = graph.compile(checkpointer=checkpointer)
    return compiled, llm, qdrant_client, text_model


def run_query(
    compiled_graph: Any,
    game_id: str,
    query: str,
    thread_id: str | None = None,
) -> QAWithCitations:
    """Invoke the agent for *query* and return structured QAWithCitations.

    Pass a stable *thread_id* to share conversation context across queries in a
    session (enables follow-up questions). Old RAG tool outputs are compressed
    in call_agent so only Q&A text accumulates, not raw retrieval dumps.
    """
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}, "recursion_limit": 12}
    result = compiled_graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "game_id": game_id,
            "game_name": "",
            "final_answer": None,
        },
        config=config,
    )
    raw = result.get("final_answer") or {}
    return QAWithCitations(**raw) if raw else QAWithCitations(
        reasoning="", answer="No answer produced.", citations=[]
    )
