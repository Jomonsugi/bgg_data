from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import replicate
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..runs import init_run_dirs


class ValidatePdfBasicIn(BaseModel):
    file_path: str


def validate_pdf_basic(file_path: str) -> dict:
    """Check PDF signature and page count. Cheap pre-check."""
    p = Path(file_path)
    if not p.exists():
        return {"is_pdf": False, "page_count": 0, "fail_reason": "file not found"}
    try:
        b = p.read_bytes()
        if b[:4] != b"%PDF":
            return {"is_pdf": False, "page_count": 0, "fail_reason": "missing %PDF signature"}
        doc = fitz.open(stream=b, filetype="pdf")
        n = doc.page_count
        doc.close()
        return {"is_pdf": True, "page_count": int(n), "fail_reason": ""}
    except Exception as e:
        return {"is_pdf": False, "page_count": 0, "fail_reason": f"{e}"}


class RenderPdfPagesIn(BaseModel):
    file_path: str
    run_id: str
    pages: int = Field(default=3, description="Number of first pages to render to images")


def render_pdf_pages(file_path: str, run_id: str, pages: int = 3) -> dict:
    """Render the first N pages of a PDF to PNG images and return their paths."""
    p = Path(file_path)
    if not p.exists():
        return {"image_paths": [], "fail_reason": "file not found"}
    try:
        run_dir = init_run_dirs(run_id)
        out_dir = run_dir / "rendered_pages"
        out_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(p))
        count = min(int(pages), doc.page_count, 10)
        image_paths = []
        for i in range(count):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=150)
            img_path = out_dir / f"{p.stem}_p{i+1}.png"
            pix.save(str(img_path))
            image_paths.append(str(img_path))
        doc.close()
        return {"image_paths": image_paths, "fail_reason": ""}
    except Exception as e:
        return {"image_paths": [], "fail_reason": f"{e}"}


class VisionQaIn(BaseModel):
    image_path: str
    question: str


def vision_qa(image_path: str, question: str) -> dict:
    """
    Ask a vision model a question about an image.

    Uses Replicate model `google/gemini-2.5-flash`. Requires `REPLICATE_API_TOKEN`.
    """
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        return {"answer": "ERROR: REPLICATE_API_TOKEN not set"}
    p = Path(image_path)
    if not p.exists():
        return {"answer": f"ERROR: Image not found at {image_path}"}
    try:
        with open(p, "rb") as f:
            output = replicate.run(
                "google/gemini-2.5-flash",
                input={"images": [f], "prompt": question},
            )
        if isinstance(output, list):
            text = "".join(str(x) for x in output).strip()
        else:
            text = str(output).strip()
        return {"answer": text or "ERROR: Empty response from vision model"}
    except Exception as e:
        return {"answer": f"ERROR vision_qa: {e}"}


class ValidateRulebookVisionIn(BaseModel):
    file_path: str
    run_id: str
    game_name: str
    pages: int = Field(default=3)


def _try_parse_json_object(text: str) -> dict | None:
    """
    Best-effort JSON object parse.

    We instruct the model to return JSON only. In practice, models sometimes wrap JSON
    in Markdown fences or add a short preamble. We'll attempt to extract the first
    {...} block if needed.
    """
    if not text:
        return None
    s = text.strip()
    # Fast path: pure JSON
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Fallback: extract first JSON object substring
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def validate_rulebook_vision(file_path: str, run_id: str, game_name: str, pages: int = 3) -> dict:
    """
    Validate that a downloaded file looks like the correct rulebook for the given game.

    Renders first pages, then asks Gemini vision targeted questions.
    """
    basic = validate_pdf_basic(file_path)
    if not basic.get("is_pdf"):
        return {"looks_like_rulebook": False, "confidence": 0.0, "notes": "", "fail_reason": basic.get("fail_reason", "")}

    rendered = render_pdf_pages(file_path, run_id, pages=pages)
    imgs = rendered.get("image_paths") or []
    if not imgs:
        return {"looks_like_rulebook": False, "confidence": 0.0, "notes": "", "fail_reason": rendered.get("fail_reason", "failed to render pages")}

    # Ask about page 1 (cover/title). STRICT JSON ONLY.
    # We explicitly distinguish MAIN RULEBOOK vs REFERENCE docs (rules reference / glossary / quick reference).
    q1 = (
        f"Game name: {game_name}\n"
        "Task: Determine whether this PDF is the MAIN rulebook (learn-to-play with setup + full rules) or a REFERENCE/supplement.\n"
        "IMPORTANT: A document titled 'Rules Reference' or 'Rules Reference Guide' is NOT the main rulebook. Reject it.\n"
        "Return ONLY valid JSON (no markdown, no prose) with this schema:\n"
        "{\n"
        '  "doc_type": "main_rulebook" | "reference" | "supplement" | "unknown",\n'
        '  "is_english": true | false | null,\n'
        '  "confidence": 0.0-1.0,\n'
        '  "reason": "short"\n'
        "}\n"
    )
    a1_raw = vision_qa(imgs[0], q1).get("answer", "")
    a1 = _try_parse_json_object(a1_raw)
    notes_parts = [f"page1_raw: {a1_raw}"]
    if not a1:
        return {
            "looks_like_rulebook": False,
            "confidence": 0.0,
            "notes": "\n".join(notes_parts)[:4000],
            "fail_reason": "vision_qa did not return valid JSON for page 1; cannot validate reliably; continue searching",
        }

    notes_parts.append(f"page1: {json.dumps(a1, ensure_ascii=False)}")

    doc_type = (a1.get("doc_type") or "").strip().lower()
    if doc_type in {"reference"}:
        return {
            "looks_like_rulebook": False,
            "confidence": float(a1.get("confidence") or 0.0),
            "notes": "\n".join(notes_parts)[:4000],
            "fail_reason": "This appears to be a rules reference / quick reference / glossary document, not the main rulebook. Continue searching.",
        }
    if doc_type in {"supplement"}:
        return {
            "looks_like_rulebook": False,
            "confidence": float(a1.get("confidence") or 0.0),
            "notes": "\n".join(notes_parts)[:4000],
            "fail_reason": "This appears to be supplementary material (FAQ/expansion/errata/aid), not the main rulebook. Continue searching.",
        }
    if doc_type not in {"main_rulebook"}:
        return {
            "looks_like_rulebook": False,
            "confidence": float(a1.get("confidence") or 0.0),
            "notes": "\n".join(notes_parts)[:4000],
            "fail_reason": "Could not confirm this is the main rulebook from page 1. Continue searching.",
        }

    # Page 2/3: confirm it contains setup/components/full rules (not just a glossary/index).
    # STRICT JSON ONLY.
    confirmations = []
    for idx, img in enumerate(imgs[1:3], start=2):
        q = (
            f"Game name: {game_name}\n"
            "Task: Determine whether this page supports that the document is the MAIN rulebook.\n"
            "Return ONLY valid JSON with this schema:\n"
            "{\n"
            '  "has_setup_or_components": true | false | null,\n'
            '  "looks_like_reference_only": true | false | null,\n'
            '  "confidence": 0.0-1.0,\n'
            '  "reason": "short"\n'
            "}\n"
        )
        raw = vision_qa(img, q).get("answer", "")
        parsed = _try_parse_json_object(raw)
        notes_parts.append(f"page{idx}_raw: {raw}")
        if not parsed:
            return {
                "looks_like_rulebook": False,
                "confidence": 0.0,
                "notes": "\n".join(notes_parts)[:4000],
                "fail_reason": f"vision_qa did not return valid JSON for page {idx}; cannot validate reliably; continue searching",
            }
        notes_parts.append(f"page{idx}: {json.dumps(parsed, ensure_ascii=False)}")
        confirmations.append(parsed)

    # Decide: require at least one of pages 2-3 to indicate setup/components
    has_setup = any(c.get("has_setup_or_components") is True for c in confirmations)
    looks_reference_only = any(c.get("looks_like_reference_only") is True for c in confirmations)
    confs = [float(a1.get("confidence") or 0.0)] + [float(c.get("confidence") or 0.0) for c in confirmations]
    avg_conf = sum(confs) / max(1, len(confs))

    if looks_reference_only and not has_setup:
        return {
            "looks_like_rulebook": False,
            "confidence": float(avg_conf),
            "notes": "\n".join(notes_parts)[:4000],
            "fail_reason": "Pages indicate this is primarily a reference document (glossary/index/lookup) and not the main rulebook. Continue searching.",
        }
    if not has_setup:
        return {
            "looks_like_rulebook": False,
            "confidence": float(avg_conf),
            "notes": "\n".join(notes_parts)[:4000],
            "fail_reason": "Could not confirm setup/components/full rules on early pages; may be a reference or supplement. Continue searching.",
        }

    return {
        "looks_like_rulebook": True,
        "confidence": float(avg_conf),
        "notes": "\n".join(notes_parts)[:4000],
        "fail_reason": "",
    }


def build_validate_tools():
    return [
        StructuredTool.from_function(
            func=validate_pdf_basic,
            name="validate_pdf_basic",
            description="Cheap PDF validation: checks %PDF signature and returns page count.",
            args_schema=ValidatePdfBasicIn,
        ),
        StructuredTool.from_function(
            func=render_pdf_pages,
            name="render_pdf_pages",
            description="Render first N pages of a PDF to PNG images and return image paths (for vision QA).",
            args_schema=RenderPdfPagesIn,
        ),
        StructuredTool.from_function(
            func=vision_qa,
            name="vision_qa",
            description="Ask a vision model a question about an image (Replicate Gemini 2.5 Flash).",
            args_schema=VisionQaIn,
        ),
        StructuredTool.from_function(
            func=validate_rulebook_vision,
            name="validate_rulebook_vision",
            description="Validate if a PDF looks like the correct board game rulebook using Gemini vision on first pages.",
            args_schema=ValidateRulebookVisionIn,
        ),
    ]


