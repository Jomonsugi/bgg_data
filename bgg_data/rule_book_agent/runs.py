from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional


def make_run_id() -> str:
    return uuid.uuid4().hex


def get_package_dir() -> Path:
    return Path(__file__).resolve().parent


def get_rulebooks_dir() -> Path:
    d = get_package_dir() / "rulebooks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_runs_root() -> Path:
    d = get_package_dir() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def init_run_dirs(run_id: str) -> Path:
    run_dir = get_runs_root() / run_id
    (run_dir / "screenshots").mkdir(parents=True, exist_ok=True)
    (run_dir / "downloads").mkdir(parents=True, exist_ok=True)
    (run_dir / "rendered_pages").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


def _serialize_for_json(obj: Any) -> Any:
    """
    Recursively convert LangChain message objects and other non-serializable
    objects to JSON-serializable dicts.
    """
    # Handle LangChain messages
    if hasattr(obj, "dict") and callable(obj.dict):
        try:
            return obj.dict()
        except Exception:
            pass
    
    # Handle dicts
    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    
    # Handle lists/tuples
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item) for item in obj]
    
    # Handle common non-serializable types
    if hasattr(obj, "__dict__"):
        try:
            return {"_type": type(obj).__name__, "_repr": str(obj)}
        except Exception:
            pass
    
    # Try default JSON serialization
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return {"_type": type(obj).__name__, "_repr": str(obj)}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = _serialize_for_json(data)
    path.write_text(json.dumps(serialized, indent=2, ensure_ascii=False))


def append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def log_event(run_id: str, event: str, payload: Optional[dict] = None) -> None:
    run_dir = init_run_dirs(run_id)
    append_jsonl(
        run_dir / "logs" / "events.jsonl",
        {
            "ts": time.time(),
            "event": event,
            "payload": payload or {},
        },
    )


