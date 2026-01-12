from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class RequestHumanHelpIn(BaseModel):
    reason: str = Field(..., description="Why human help is needed (e.g., CAPTCHA/login)")
    session_id: str = Field(..., description="Browser session id to continue")
    screenshot_path: str | None = Field(default=None, description="Optional screenshot path for context")


def request_human_help(reason: str, session_id: str, screenshot_path: str | None = None) -> dict:
    """
    Pause the run and request human assistance (CAPTCHA/login).

    The surrounding API/runner should treat this as a pause signal and allow resume.
    """
    instructions = (
        "Human assistance requested.\n"
        f"reason: {reason}\n"
        f"session_id: {session_id}\n"
        + (f"screenshot_path: {screenshot_path}\n" if screenshot_path else "")
        + "Next: open the page in a headed browser session and solve the blocking step, then resume the run."
    )
    return {"run_paused": True, "instructions": instructions}


def build_hitl_tools():
    return [
        StructuredTool.from_function(
            func=request_human_help,
            name="request_human_help",
            description="Request human assistance when blocked by CAPTCHA/login. Causes the run to pause.",
            args_schema=RequestHumanHelpIn,
        )
    ]


