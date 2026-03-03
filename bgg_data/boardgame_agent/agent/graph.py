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
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_together import ChatTogether
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from bgg_data.boardgame_agent.agent.prompts import FORMAT_PROMPT, SYSTEM_PROMPT_TEMPLATE
from bgg_data.boardgame_agent.agent.schemas import QAWithCitations
from bgg_data.boardgame_agent.agent.state import AgentState
from bgg_data.boardgame_agent.agent.tools import make_all_tools
from bgg_data.boardgame_agent.config import (
    CHECKPOINTS_DB_PATH,
    EMBED_MODEL_NAME,
    GAMES_DB_PATH,
    QDRANT_PATH,
    TOGETHER_API_KEY,
    TOGETHER_MODEL_NAME,
)


def build_agent(
    game_id: str,
    game_name: str,
    model_name: str = TOGETHER_MODEL_NAME,
    top_k: int = 5,
) -> tuple[Any, ChatTogether, QdrantClient, TextEmbedding]:
    """Compile the LangGraph agent for *game_id*.

    Returns (compiled_graph, llm, qdrant_client, text_model).
    The caller should cache the result keyed by (game_id, model_name, top_k).
    """
    qdrant_client = QdrantClient(path=str(QDRANT_PATH))
    text_model = TextEmbedding(model_name=EMBED_MODEL_NAME)
    tools = make_all_tools(
        game_id, game_name, qdrant_client, text_model, GAMES_DB_PATH, top_k=top_k
    )

    llm = ChatTogether(
        model=model_name,
        together_api_key=TOGETHER_API_KEY,
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)
    system_message = SystemMessage(
        content=SYSTEM_PROMPT_TEMPLATE.format(game_name=game_name)
    )

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def call_agent(state: AgentState) -> dict:
        messages = [system_message] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
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
) -> QAWithCitations:
    """Invoke the agent for *query* and return structured QAWithCitations."""
    config = {"configurable": {"thread_id": game_id}}
    result = compiled_graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "game_id": game_id,
            "game_name": "",  # already in system prompt; placeholder here
            "final_answer": None,
        },
        config=config,
    )
    raw = result.get("final_answer") or {}
    return QAWithCitations(**raw) if raw else QAWithCitations(
        reasoning="", answer="No answer produced.", citations=[]
    )
