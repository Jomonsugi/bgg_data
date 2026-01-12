from __future__ import annotations

import re
import time
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from ..runs import init_run_dirs


@dataclass
class _BrowserSession:
    browser: Any
    context: Any
    page: Any
    run_id: str


@dataclass
class _Req:
    op: str
    kwargs: dict
    resp_q: "queue.Queue[dict]"


_WORKER_THREAD: threading.Thread | None = None
_WORKER_Q: "queue.Queue[_Req]" | None = None
_WORKER_LOCK = threading.Lock()


def _safe_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s.strip())
    return s[:180] or "screenshot"


def _ensure_worker_started() -> "queue.Queue[_Req]":
    global _WORKER_THREAD, _WORKER_Q
    with _WORKER_LOCK:
        if _WORKER_THREAD and _WORKER_Q:
            return _WORKER_Q

        q: "queue.Queue[_Req]" = queue.Queue()
        _WORKER_Q = q

        def _run():
            # All Playwright sync API calls happen in this single thread.
            # This avoids greenlet/thread issues when LangGraph executes tools.
            p = sync_playwright().start()
            sessions: dict[str, _BrowserSession] = {}
            try:
                while True:
                    req = q.get()
                    if req.op == "__stop__":
                        req.resp_q.put({"ok": True})
                        break
                    try:
                        op = req.op
                        kw = req.kwargs

                        if op == "open":
                            url = kw["url"]
                            run_id = kw["run_id"]
                            headless = bool(kw.get("headless", True))
                            browser = p.chromium.launch(headless=headless)
                            context = browser.new_context()
                            page = context.new_page()
                            try:
                                page.goto(url, wait_until="networkidle", timeout=20000)
                            except PlaywrightTimeoutError:
                                page.goto(url, wait_until="domcontentloaded", timeout=20000)
                            session_id = f"sess_{int(time.time()*1000)}"
                            sessions[session_id] = _BrowserSession(browser=browser, context=context, page=page, run_id=run_id)
                            req.resp_q.put({"ok": True, "session_id": session_id, "current_url": page.url})
                            continue

                        if op == "close":
                            session_id = kw["session_id"]
                            sess = sessions.pop(session_id, None)
                            if not sess:
                                req.resp_q.put({"ok": False, "reason": "unknown session_id"})
                                continue
                            try:
                                try:
                                    sess.context.close()
                                except Exception:
                                    pass
                                try:
                                    sess.browser.close()
                                except Exception:
                                    pass
                                req.resp_q.put({"ok": True})
                            except Exception as e:
                                req.resp_q.put({"ok": False, "reason": f"{e}"})
                            continue

                        if op == "cleanup_run":
                            run_id = kw["run_id"]
                            to_close = [sid for sid, sess in sessions.items() if sess.run_id == run_id]
                            closed = 0
                            for sid in to_close:
                                sess = sessions.pop(sid, None)
                                if not sess:
                                    continue
                                try:
                                    try:
                                        sess.context.close()
                                    except Exception:
                                        pass
                                    try:
                                        sess.browser.close()
                                    except Exception:
                                        pass
                                    closed += 1
                                except Exception:
                                    pass
                            req.resp_q.put({"ok": True, "closed": closed})
                            continue

                        # Ops that require an existing session:
                        session_id = kw.get("session_id")
                        sess = sessions.get(session_id) if session_id else None
                        if not sess:
                            req.resp_q.put({"ok": False, "reason": "unknown session_id"})
                            continue
                        page = sess.page
                        context = sess.context

                        if op == "cookies":
                            # Return cookies for the current browser context (for reuse in requests downloads).
                            try:
                                req.resp_q.put({"ok": True, "cookies": context.cookies()})
                            except Exception as e:
                                req.resp_q.put({"ok": False, "reason": f"{e}", "cookies": []})
                            continue

                        if op == "snapshot":
                            max_text_chars = int(kw.get("max_text_chars", 20000))
                            run_dir = init_run_dirs(sess.run_id)
                            screenshot_path = run_dir / "screenshots" / f"{_safe_filename(page.url)}.png"
                            try:
                                try:
                                    page.wait_for_load_state("domcontentloaded", timeout=5000)
                                except Exception:
                                    pass
                                try:
                                    page.screenshot(path=str(screenshot_path), full_page=True)
                                except Exception:
                                    # If screenshot fails, still return text/links.
                                    pass

                                text = (page.text_content("body") or "").strip()
                                if len(text) > max_text_chars:
                                    text = text[:max_text_chars]

                                links = []
                                try:
                                    anchors = page.eval_on_selector_all(
                                        "a[href]",
                                        """(els) => els.slice(0, 400).map(a => ({text: (a.innerText||'').trim().slice(0,200), url: a.href}))""",
                                    )
                                    for a in anchors or []:
                                        url = (a.get("url") or "").strip()
                                        if not url:
                                            continue
                                        links.append({"text": (a.get("text") or url)[:200], "url": url})
                                except Exception:
                                    pass

                                req.resp_q.put(
                                    {
                                        "ok": True,
                                        "current_url": page.url,
                                        "text": text,
                                        "links": links,
                                        "screenshot_path": str(screenshot_path),
                                    }
                                )
                            except PlaywrightError as e:
                                req.resp_q.put({"ok": False, "current_url": page.url, "reason": f"{e}"})
                            continue

                        if op == "click":
                            target = kw["target"]
                            timeout_ms = int(kw.get("timeout_ms", 5000))
                            before = page.url
                            try:
                                if target.startswith("css:"):
                                    sel = target[len("css:") :]
                                    page.click(sel, timeout=timeout_ms)
                                else:
                                    # Use role=link/button heuristics; fall back to text selector
                                    try:
                                        page.get_by_role("button", name=re.compile(re.escape(target), re.I)).click(timeout=timeout_ms)
                                    except Exception:
                                        try:
                                            page.get_by_role("link", name=re.compile(re.escape(target), re.I)).click(timeout=timeout_ms)
                                        except Exception:
                                            page.get_by_text(target, exact=False).first.click(timeout=timeout_ms)
                                try:
                                    page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
                                except Exception:
                                    pass
                                req.resp_q.put({"ok": True, "clicked": True, "current_url": page.url, "navigated": page.url != before})
                            except Exception as e:
                                req.resp_q.put({"ok": False, "clicked": False, "current_url": page.url, "reason": f"{e}"})
                            continue

                        if op == "type":
                            selector = kw["selector"]
                            text = kw["text"]
                            submit = bool(kw.get("submit", False))
                            try:
                                page.fill(selector, text)
                                if submit:
                                    page.press(selector, "Enter")
                                req.resp_q.put({"ok": True, "typed": True})
                            except Exception as e:
                                req.resp_q.put({"ok": False, "typed": False, "reason": f"{e}"})
                            continue

                        if op == "scroll":
                            amount = int(kw["amount"])
                            try:
                                page.evaluate("({amount}) => window.scrollBy(0, amount)", {"amount": amount})
                                req.resp_q.put({"ok": True, "scrolled": True})
                            except Exception as e:
                                req.resp_q.put({"ok": False, "scrolled": False, "reason": f"{e}"})
                            continue

                        if op == "wait":
                            seconds = float(kw.get("seconds", 1.5))
                            try:
                                page.wait_for_timeout(int(max(0.0, seconds) * 1000))
                                req.resp_q.put({"ok": True, "waited": True})
                            except Exception as e:
                                req.resp_q.put({"ok": False, "waited": False, "reason": f"{e}"})
                            continue

                        if op == "back":
                            try:
                                page.go_back(timeout=10000)
                                try:
                                    page.wait_for_load_state("domcontentloaded", timeout=5000)
                                except Exception:
                                    pass
                                req.resp_q.put({"ok": True, "current_url": page.url})
                            except Exception as e:
                                req.resp_q.put({"ok": False, "current_url": page.url, "reason": f"{e}"})
                            continue

                        if op == "goto":
                            url = kw["url"]
                            try:
                                page.goto(url, wait_until="domcontentloaded", timeout=20000)
                                req.resp_q.put({"ok": True, "current_url": page.url})
                            except Exception as e:
                                req.resp_q.put({"ok": False, "current_url": page.url, "reason": f"{e}"})
                            continue

                        if op == "wait_for_selector":
                            selector = kw["selector"]
                            timeout_ms = int(kw.get("timeout_ms", 6000))
                            try:
                                page.wait_for_selector(selector, timeout=timeout_ms)
                                req.resp_q.put({"ok": True})
                            except Exception as e:
                                req.resp_q.put({"ok": False, "reason": f"{e}"})
                            continue

                        if op == "eval":
                            script = kw["script"]
                            try:
                                val = page.evaluate(script)
                                req.resp_q.put({"ok": True, "value": val, "current_url": page.url})
                            except Exception as e:
                                req.resp_q.put({"ok": False, "reason": f"{e}", "current_url": page.url})
                            continue

                        if op == "close_popups":
                            actions = []
                            closed_any = False
                            patterns = [
                                r"accept(?: all)?",
                                r"agree",
                                r"i\s*accept",
                                r"ok(?:ay)?",
                                r"continue",
                                r"(?:reject|decline|disagree)",
                                r"no thanks",
                                r"got it",
                                r"close",
                            ]
                            for pat in patterns:
                                try:
                                    loc = page.get_by_role("button", name=re.compile(pat, re.IGNORECASE))
                                    if loc.count() > 0:
                                        loc.first.click(timeout=1000)
                                        actions.append({"action": "click_button", "detail": pat})
                                        closed_any = True
                                except Exception:
                                    continue
                            try:
                                page.keyboard.press("Escape")
                                actions.append({"action": "press_key", "detail": "Escape"})
                            except Exception:
                                pass
                            req.resp_q.put({"ok": True, "closed_any": closed_any, "actions": actions})
                            continue

                        if op == "find_links":
                            include_patterns = kw.get("include_patterns") or []
                            exclude_patterns = kw.get("exclude_patterns") or []
                            max_matches = int(kw.get("max_matches", 50))
                            inc = [re.compile(p, re.IGNORECASE) for p in include_patterns if p]
                            exc = [re.compile(p, re.IGNORECASE) for p in exclude_patterns if p]
                            matches = []
                            try:
                                anchors = page.eval_on_selector_all(
                                    "a[href]",
                                    """(els) => els.map(a => ({text: (a.innerText||'').trim().slice(0,200), url: a.href}))""",
                                )
                            except Exception as e:
                                req.resp_q.put({"ok": False, "matches": [], "reason": f"{e}"})
                                continue
                            for a in anchors or []:
                                text = (a.get("text") or "").strip()
                                url = (a.get("url") or "").strip()
                                hay = f"{text}\n{url}"
                                if inc and not any(p.search(hay) for p in inc):
                                    continue
                                if exc and any(p.search(hay) for p in exc):
                                    continue
                                why = []
                                for ptn in inc:
                                    if ptn.search(hay):
                                        why.append(ptn.pattern)
                                matches.append({"text": text or url, "url": url, "why": ",".join(why) if why else ""})
                                if len(matches) >= max(1, max_matches):
                                    break
                            req.resp_q.put({"ok": True, "matches": matches})
                            continue

                        req.resp_q.put({"ok": False, "reason": f"unknown op: {op}"})
                    except Exception as e:
                        req.resp_q.put({"ok": False, "reason": f"{e}"})
            finally:
                # Best-effort cleanup
                for sess in list(sessions.values()):
                    try:
                        sess.context.close()
                    except Exception:
                        pass
                    try:
                        sess.browser.close()
                    except Exception:
                        pass
                try:
                    p.stop()
                except Exception:
                    pass

        t = threading.Thread(target=_run, name="rulebook-playwright", daemon=True)
        _WORKER_THREAD = t
        t.start()
        return q


def _call_worker(op: str, **kwargs) -> dict:
    q = _ensure_worker_started()
    resp_q: "queue.Queue[dict]" = queue.Queue(maxsize=1)
    q.put(_Req(op=op, kwargs=kwargs, resp_q=resp_q))
    return resp_q.get()


class BrowserOpenIn(BaseModel):
    url: str = Field(..., description="Absolute http(s) URL to open")
    run_id: str = Field(..., description="Run id for artifact storage")
    headless: bool = Field(default=True, description="Whether to run browser headless")


def browser_open(url: str, run_id: str, headless: bool = True) -> dict:
    """Open a new Playwright browser session and navigate to a URL."""
    if not url.startswith(("http://", "https://")):
        return {"session_id": None, "current_url": None, "reason": "url must start with http:// or https://"}

    out = _call_worker("open", url=url, run_id=run_id, headless=headless)
    if out.get("ok"):
        return {"session_id": out.get("session_id"), "current_url": out.get("current_url")}
    return {"session_id": None, "current_url": None, "reason": out.get("reason", "failed to open")}


class BrowserCloseIn(BaseModel):
    session_id: str


def browser_close(session_id: str) -> dict:
    """Close a Playwright browser session."""
    out = _call_worker("close", session_id=session_id)
    if out.get("ok"):
        return {"closed": True}
    return {"closed": False, "reason": out.get("reason", "unknown error")}


def cleanup_browser_sessions_for_run(run_id: str) -> int:
    """
    Close all browser sessions associated with a specific run_id.
    Returns the number of sessions closed.
    """
    out = _call_worker("cleanup_run", run_id=run_id)
    if out.get("ok"):
        return int(out.get("closed", 0))
    return 0


class BrowserSnapshotIn(BaseModel):
    session_id: str
    max_text_chars: int = Field(default=20000, description="Max characters of body text to return")


def browser_snapshot(session_id: str, max_text_chars: int = 20000) -> dict:
    """
    Capture the current URL, visible text, outgoing links, and a screenshot.
    """
    out = _call_worker("snapshot", session_id=session_id, max_text_chars=int(max_text_chars))
    if out.get("ok"):
        return {
            "current_url": out.get("current_url"),
            "text": out.get("text", ""),
            "links": out.get("links", []),
            "screenshot_path": out.get("screenshot_path"),
        }
    return {"current_url": out.get("current_url"), "text": "", "links": [], "screenshot_path": None, "reason": out.get("reason", "error")}


class BrowserClickIn(BaseModel):
    session_id: str
    target: str = Field(..., description="Text to click, or CSS selector if prefixed with 'css:'")
    timeout_ms: int = Field(default=5000)


def browser_click(session_id: str, target: str, timeout_ms: int = 5000) -> dict:
    """
    Click a target on the page. Uses text click by default; `css:...` for selector.
    """
    out = _call_worker("click", session_id=session_id, target=target, timeout_ms=int(timeout_ms))
    if out.get("ok"):
        return {"clicked": True, "current_url": out.get("current_url"), "navigated": bool(out.get("navigated"))}
    return {"clicked": False, "current_url": out.get("current_url"), "reason": out.get("reason", "error")}


class BrowserTypeIn(BaseModel):
    session_id: str
    selector: str = Field(..., description="CSS selector for the input element")
    text: str
    submit: bool = Field(default=False, description="Press Enter after typing")


def browser_type(session_id: str, selector: str, text: str, submit: bool = False) -> dict:
    out = _call_worker("type", session_id=session_id, selector=selector, text=text, submit=bool(submit))
    if out.get("ok"):
        return {"typed": True}
    return {"typed": False, "reason": out.get("reason", "error")}


class BrowserScrollIn(BaseModel):
    session_id: str
    amount: int = Field(..., description="Pixels to scroll (positive=down, negative=up)")


def browser_scroll(session_id: str, amount: int) -> dict:
    out = _call_worker("scroll", session_id=session_id, amount=int(amount))
    if out.get("ok"):
        return {"scrolled": True}
    return {"scrolled": False, "reason": out.get("reason", "error")}


class BrowserWaitIn(BaseModel):
    session_id: str
    seconds: float = Field(default=1.5, description="Seconds to wait")


def browser_wait(session_id: str, seconds: float = 1.5) -> dict:
    out = _call_worker("wait", session_id=session_id, seconds=float(seconds))
    if out.get("ok"):
        return {"waited": True}
    return {"waited": False, "reason": out.get("reason", "error")}


class BrowserBackIn(BaseModel):
    session_id: str


def browser_back(session_id: str) -> dict:
    out = _call_worker("back", session_id=session_id)
    if out.get("ok"):
        return {"ok": True, "current_url": out.get("current_url")}
    return {"ok": False, "current_url": out.get("current_url"), "reason": out.get("reason", "error")}


def build_browser_primitive_tools():
    return [
        StructuredTool.from_function(
            func=browser_open,
            name="browser_open",
            description="Open a Playwright browser session and navigate to an absolute URL. Returns session_id and current_url.",
            args_schema=BrowserOpenIn,
        ),
        StructuredTool.from_function(
            func=browser_snapshot,
            name="browser_snapshot",
            description="Capture current_url, visible text, outgoing links, and a screenshot for the current page.",
            args_schema=BrowserSnapshotIn,
        ),
        StructuredTool.from_function(
            func=browser_click,
            name="browser_click",
            description="Click an element by visible text, or by CSS selector using prefix 'css:'. Returns whether clicked and current_url.",
            args_schema=BrowserClickIn,
        ),
        StructuredTool.from_function(
            func=browser_type,
            name="browser_type",
            description="Type into an input selected by CSS selector; optionally press Enter to submit.",
            args_schema=BrowserTypeIn,
        ),
        StructuredTool.from_function(
            func=browser_scroll,
            name="browser_scroll",
            description="Scroll the page by a pixel amount (positive=down, negative=up).",
            args_schema=BrowserScrollIn,
        ),
        StructuredTool.from_function(
            func=browser_wait,
            name="browser_wait",
            description="Wait for a short period to allow dynamic content to load.",
            args_schema=BrowserWaitIn,
        ),
        StructuredTool.from_function(
            func=browser_back,
            name="browser_back",
            description="Navigate back in browser history for the session.",
            args_schema=BrowserBackIn,
        ),
        StructuredTool.from_function(
            func=browser_close,
            name="browser_close",
            description="Close the Playwright browser session and free resources.",
            args_schema=BrowserCloseIn,
        ),
    ]


