from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..runs import init_run_dirs
from . import browser_primitives as bp


def _safe_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", (name or "").strip())
    return s[:120] or "file"


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


class DownloadCandidateIn(BaseModel):
    url: str
    run_id: str
    session_id: Optional[str] = Field(default=None, description="Optional browser session to reuse cookies")
    timeout_s: int = Field(default=30)
    max_bytes: int = Field(default=40 * 1024 * 1024, description="Max download size in bytes")


def download_candidate(
    url: str,
    run_id: str,
    session_id: Optional[str] = None,
    timeout_s: int = 30,
    max_bytes: int = 40 * 1024 * 1024,
) -> dict:
    """
    Download a candidate URL with guardrails (timeout, max size) and optional cookie reuse from a browser session.
    Returns file_path + metadata; does not validate rulebook correctness.
    """
    if not url.startswith(("http://", "https://")):
        return {"ok": False, "file_path": None, "fail_reason": "url must start with http(s)"}

    # Dropbox -> direct
    if "dropbox.com" in url and ("?dl=0" in url or "&dl=0" in url):
        url = url.replace("?dl=0", "?dl=1").replace("&dl=0", "&dl=1")

    # Basic session
    sess = requests.Session()
    sess.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
            "Accept": "application/pdf, text/html;q=0.9,*/*;q=0.8",
        }
    )

    # Try cookie reuse if available
    if session_id:
        try:
            out = bp._call_worker("cookies", session_id=session_id)
            cookies = (out.get("cookies") or []) if out.get("ok") else []
            for c in cookies or []:
                try:
                    name = c.get("name")
                    value = c.get("value")
                    domain = c.get("domain")
                    path = c.get("path") or "/"
                    if name and value:
                        sess.cookies.set(name, value, domain=domain, path=path)
                except Exception:
                    continue
        except Exception:
            pass

    try:
        resp = sess.get(url, timeout=max(5, int(timeout_s)), stream=True, allow_redirects=True)
        resp.raise_for_status()

        content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()

        # Stream with max_bytes cap
        chunks = []
        size = 0
        for chunk in resp.iter_content(chunk_size=1024 * 128):
            if not chunk:
                continue
            chunks.append(chunk)
            size += len(chunk)
            if size > int(max_bytes):
                return {"ok": False, "file_path": None, "fail_reason": f"download exceeded max_bytes={max_bytes}"}

        data = b"".join(chunks)
        if not data:
            return {"ok": False, "file_path": None, "fail_reason": "empty response body"}

        sha256 = _sha256_bytes(data)
        run_dir = init_run_dirs(run_id)

        # Pick extension (PDF if signature, else guess from content type)
        ext = ".bin"
        if data[:4] == b"%PDF":
            ext = ".pdf"
        elif "html" in content_type:
            ext = ".html"

        ts = int(time.time())
        file_path = run_dir / "downloads" / f"{_safe_name(urlparse(url).netloc)}_{ts}_{sha256[:10]}{ext}"
        file_path.write_bytes(data)

        return {
            "ok": True,
            "file_path": str(file_path),
            "content_type": content_type or "",
            "size_bytes": len(data),
            "sha256": sha256,
            "final_url": str(resp.url),
            "fail_reason": "",
        }
    except Exception as e:
        return {"ok": False, "file_path": None, "fail_reason": f"{e}"}
    finally:
        try:
            sess.close()
        except Exception:
            pass


def build_download_tools():
    return [
        StructuredTool.from_function(
            func=download_candidate,
            name="download_candidate",
            description="Download a URL with guardrails; optionally reuse cookies from a browser session_id. Returns file_path + sha256 + size.",
            args_schema=DownloadCandidateIn,
        )
    ]


