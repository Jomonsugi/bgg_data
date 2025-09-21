from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import fitz  # PyMuPDF

from .tools import (
    render_first_page_image,
    classify_image_with_vlm,
)


def make_number_pdf(out_path: Path, number_text: str = "42") -> Path:
    """Create a one-page PDF with a large number centered."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # US Letter points
    rect = page.rect
    # Draw big number centered
    page.insert_textbox(
        rect,
        number_text,
        fontsize=240,
        fontname="helv",
        align=1,  # center
        color=(0, 0, 0),
    )
    doc.save(out_path.as_posix())
    doc.close()
    return out_path


def run_once(image_path: Path, model_id: str, provider: str) -> str:
    if provider == "local":
        # Simpler prompt for local MLX models
        prompt = "What number is shown in this image? Answer with only the number, nothing else."
    else:
        # More detailed prompt for remote models
        prompt = (
            "What number is shown in this image? Answer with only the number, nothing else."
        )
    return classify_image_with_vlm(image_path, prompt, model_id, provider)


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description="Sanity-check VLM image calls for local and together providers")
    p.add_argument("--number", default="42", help="Number to render in the test PDF")
    p.add_argument("--together", action="store_true", help="Run Together (HF router) test")
    p.add_argument("--local", action="store_true", help="Run local MLX test")
    p.add_argument("--together-model", default="meta-llama/Llama-4-Scout-17B-16E-Instruct", help="Together vision model id")
    p.add_argument("--local-model", default="mlx-community/Llama-3.2-11B-Vision-Instruct-4bit", help="Local MLX vision model id")
    args = p.parse_args()

    artifacts_dir = Path(__file__).resolve().parent / "_vlm_check"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = artifacts_dir / "number_test.pdf"
    img_path: Optional[Path] = None

    # Create PDF and render first page image
    make_number_pdf(pdf_path, args.number)
    img_path = render_first_page_image(pdf_path, dpi=256)
    if not img_path or not img_path.exists():
        print("Failed to render image from test PDF")
        return

    print(f"Image: {img_path}")

    # Decide defaults: if neither flag provided, run both
    run_together = args.together or (not args.together and not args.local)
    run_local = args.local or (not args.together and not args.local)

    # Together path via HF router
    if run_together:
        try:
            print(f"Together model: {args.together_model}")
            out = run_once(img_path, args.together_model, provider="together")
            print("[together] response:", out)
        except Exception as e:
            print("[together] error:", e)

    # Local MLX path
    if run_local:
        try:
            print(f"Local model: {args.local_model}")
            out = run_once(img_path, args.local_model, provider="local")
            print("[local] response:", out)
        except Exception as e:
            print("[local] error:", e)


if __name__ == "__main__":
    main()


