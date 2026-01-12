from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

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
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("apply_tool_effects", apply_tool_effects)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "init_state")
    builder.add_edge("init_state", "assistant")
    builder.add_conditional_edges("assistant", should_continue, {"tools": "tools", END: "finalize"})
    builder.add_edge("tools", "apply_tool_effects")
    builder.add_edge("apply_tool_effects", "assistant")
    builder.add_edge("finalize", END)

    return builder.compile()


