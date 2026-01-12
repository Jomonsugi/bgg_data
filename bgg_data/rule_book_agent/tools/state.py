from __future__ import annotations

import hashlib
import shutil
import re
from pathlib import Path
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..runs import get_rulebooks_dir


def _safe_base(name: str) -> str:
    s = (name or "").strip()
    s = s.replace(" ", "-").replace(":", "").replace("'", "")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", s)
    return (s[:160] or "game").strip("-")


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class SetValidatedRulebookIn(BaseModel):
    downloaded_file_path: str = Field(..., description="Path to the downloaded candidate file")
    source_url: str = Field(..., description="Source URL where the file came from")
    game_name: str = Field(..., description="Board game name")
    bgg_id: Optional[str] = Field(default=None, description="Optional BGG id for stable naming")
    notes: str = Field(default="", description="Short notes about why this is the correct rulebook")
    confidence: float = Field(default=0.8, description="Confidence 0-1")


def set_validated_rulebook(
    downloaded_file_path: str,
    source_url: str,
    game_name: str,
    bgg_id: Optional[str] = None,
    notes: str = "",
    confidence: float = 0.8,
) -> dict:
    """
    Finalize a validated rulebook by copying it into the rulebook library folder
    with a stable filename, and returning a `validated_rulebook` object.

    The agent should call this ONLY after it is satisfied validation passed.
    """
    src = Path(downloaded_file_path)
    if not src.exists():
        return {"ok": False, "fail_reason": "downloaded_file_path not found"}

    lib = get_rulebooks_dir()
    base = _safe_base(game_name)
    suffix = f"_{bgg_id}" if bgg_id else ""
    ext = src.suffix.lower() if src.suffix else ".pdf"
    if ext not in [".pdf", ".html", ".bin"]:
        ext = ".pdf"
    dst = lib / f"{base}{suffix}_rules{ext}"

    try:
        shutil.copyfile(src, dst)
        sha256 = _sha256_path(dst)
        return {
            "ok": True,
            "validated_rulebook": {
                "url": source_url,
                "file_path": str(dst),
                "sha256": sha256,
                "notes": notes,
                "confidence": float(confidence),
            },
        }
    except Exception as e:
        return {"ok": False, "fail_reason": f"{e}"}


def build_state_tools():
    return [
        StructuredTool.from_function(
            func=set_validated_rulebook,
            name="set_validated_rulebook",
            description="Finalize a validated rulebook: copy downloaded file into rulebooks folder with stable filename and return validated_rulebook object.",
            args_schema=SetValidatedRulebookIn,
        )
    ]


