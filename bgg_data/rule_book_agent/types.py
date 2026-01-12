from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class GameInfo(TypedDict, total=False):
    id: str
    name: str
    rank: int
    url: str
    publisher: str
    year_published: int


class Budgets(TypedDict, total=False):
    max_steps: int
    max_downloads: int
    max_sites_opened: int
    max_runtime_s: int


class Counters(TypedDict, total=False):
    steps: int
    downloads: int
    sites_opened: int


class ValidatedRulebook(TypedDict, total=False):
    url: str
    file_path: str
    sha256: str
    notes: str
    confidence: float


class AgentState(TypedDict, total=False):
    # LangGraph messages
    messages: Annotated[list[AnyMessage], add_messages]

    # Deterministic inputs
    game: GameInfo
    run_id: str
    run_dir: str

    # Budgets / counters
    budgets: Budgets
    counters: Counters
    started_at_ts: float

    # Browser session (Playwright)
    session_id: Optional[str]
    current_url: Optional[str]

    # Results
    validated_rulebook: Optional[ValidatedRulebook]

    # HITL
    run_paused: bool
    pause_reason: Optional[str]


ToolStatus = Literal["ok", "error"]


class ToolEnvelope(TypedDict, total=False):
    """
    Standard tool return format. Tools should return JSON (string) of this shape.
    We keep it explicit so the agent can reason over failures and artifacts.
    """

    tool: str
    status: ToolStatus
    result: dict
    error: str


