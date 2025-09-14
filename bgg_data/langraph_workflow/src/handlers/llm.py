"""
LLM handler for Together.ai integration in the rulebook fetcher.
"""

import base64
import logging
from typing import Optional, Tuple, Literal, List, Dict
from pathlib import Path
import io
from PIL import Image
import PyPDF2
from pydantic import BaseModel, Field, ValidationError

from ...config import (
    TOGETHER_API_KEY,
    MODEL_NAME,
    RULEBOOK_EXTRACTION_PROMPT,
    VISION_BACKEND,
    MLX_VLM_MODEL,
)

logger = logging.getLogger(__name__)


class PDFAssessment(BaseModel):
    """Pydantic model for structured LLM responses when assessing PDF rulebooks."""
    is_official: bool = Field(description="Whether this is an official rulebook")
    is_english: bool = Field(description="Whether the document is in English")



class LLMHandler:
    """
    Handles communication with Together.ai for rulebook URL extraction.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = MODEL_NAME):
        """
        Initialize the LLM handler.
        
        Args:
            api_key: Together.ai API key (defaults to environment variable)
            model_name: Name of the model to use
        """
        self.api_key = api_key or TOGETHER_API_KEY
        self.model_name = model_name
        self.client = None
        self._mlx_model_id = None
        
        # Initialize backend
        if VISION_BACKEND == "mlx":
            self._initialize_mlx()
        else:
            if not self.api_key:
                logger.warning("TOGETHER_API_KEY not set; skipping Together client init. Set VISION_BACKEND=mlx to suppress this.")
            else:
                self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Together.ai client."""
        try:
            from together import Together
            self.client = Together(api_key=self.api_key)
            logger.info(f"Together.ai client initialized with model: {self.model_name}")
        except ImportError:
            raise ImportError("Together.ai Python client not installed. Run: pip install together")
        except Exception as e:
            logger.error(f"Failed to initialize Together.ai client: {e}")
            raise

    def _initialize_mlx(self) -> None:
        """Initialize local MLX VLM backend."""
        try:
            # Defer heavy imports until use; track model id
            import mlx_vlm  # noqa: F401
            self._mlx_model_id = MLX_VLM_MODEL
            logger.info(f"Using MLX VLM model: {self._mlx_model_id}")
        except ImportError:
            raise ImportError("mlx-vlm not installed. Run: pip install -U mlx-vlm")
        except Exception as e:
            logger.error(f"Failed to initialize MLX VLM: {e}")
            raise
    
    def _encode_image(self, image_bytes: bytes) -> str:
        """
        Encode image bytes to base64 string for API transmission.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Base64 encoded image string
        """
        try:
            # Convert to base64
            encoded_string = base64.b64encode(image_bytes).decode('utf-8')
            return encoded_string
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise
    
    def _optimize_image(self, image_bytes: bytes, max_size_mb: float = 5.0) -> bytes:
        """
        Optimize image size to reduce token costs while maintaining quality.
        
        Args:
            image_bytes: Original image bytes
            max_size_mb: Maximum size in MB
            
        Returns:
            Optimized image bytes
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Check current size
            current_size_mb = len(image_bytes) / (1024 * 1024)
            
            if current_size_mb <= max_size_mb:
                logger.info(f"Image size ({current_size_mb:.2f}MB) is within limits, no optimization needed")
                return image_bytes
            
            logger.info(f"Optimizing image from {current_size_mb:.2f}MB to target {max_size_mb}MB")
            
            # Calculate target dimensions while maintaining aspect ratio
            width, height = image.size
            aspect_ratio = width / height if height else 1.0
            
            # Convert to RGB for JPEG compatibility
            try:
                image = image.convert('RGB')
            except Exception:
                pass
            
            # Start with current dimensions and reduce gradually, using JPEG compression
            target_width = width
            target_height = height
            quality = 85
            
            while True:
                temp_image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                temp_bytes = io.BytesIO()
                temp_image.save(temp_bytes, format='JPEG', quality=quality, optimize=True, progressive=True)
                temp_size_mb = len(temp_bytes.getvalue()) / (1024 * 1024)
                
                if temp_size_mb <= max_size_mb or (target_width < 800 and quality <= 70):
                    optimized_bytes = temp_bytes.getvalue()
                    logger.info(f"Image optimized to {temp_size_mb:.2f}MB at {target_width}x{target_height}, q={quality}")
                    return optimized_bytes
                
                # Prefer reducing quality a bit before resizing further
                if quality > 70:
                    quality -= 5
                else:
                    target_width = int(target_width * 0.9)
                    target_height = max(1, int(target_width / aspect_ratio))
            
        except Exception as e:
            logger.warning(f"Error optimizing image, using original: {e}")
            return image_bytes
    
    def extract_rulebook_url(self, image_bytes: bytes, game_name: str = "", candidates: Optional[List[Dict[str, str]]] = None, extra_images: Optional[List[bytes]] = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Extract rulebook URL from a webpage screenshot using Together.ai.
        
        Args:
            image_bytes: Screenshot image bytes
            game_name: Name of the game (for context in logging)
            
        Returns:
            Tuple of (success, url_or_error, confidence_indicator)
        """
        try:
            backend_name = "MLX" if VISION_BACKEND == "mlx" else "Together.ai"
            logger.info(f"Extracting rulebook URL for '{game_name}' using {backend_name}...")
            
            # Optimize image to reduce costs (JPEG compression)
            optimized_bytes = self._optimize_image(image_bytes)
            
            if VISION_BACKEND == "mlx":
                # Write to temp file and invoke the mlx_vlm CLI to avoid API mismatches
                import tempfile
                import subprocess
                import sys
                cand_text = ""
                if candidates:
                    preview = "\n".join([f"- {c.get('text','')} — {c.get('href','')}" for c in candidates[:8]])
                    cand_text = f"\n\nCandidates (choose one if suitable):\n{preview}\n\n"
                prompt = f"Game: {game_name}\n\n{RULEBOOK_EXTRACTION_PROMPT}{cand_text}"
                with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                    tmp.write(optimized_bytes)
                    tmp.flush()
                    cmd = [
                        sys.executable,
                        "-m",
                        "mlx_vlm.generate",
                        "--model",
                        str(self._mlx_model_id),
                        "--max-tokens",
                        "120",
                        "--temp",
                        "0.0",
                        "--images",
                        tmp.name,
                        "--prompt",
                        prompt,
                    ]
                    try:
                        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
                        content = (proc.stdout or proc.stderr or "").strip()
                    except Exception as e:
                        content = f"ERROR: {e}"
                if content and "http" in content.lower():
                    url = self._clean_url_response(content)
                    if url:
                        return True, url, "medium"
                return False, "No rulebook link found", "low"
            else:
                # Together.ai path
                encoded_image = self._encode_image(optimized_bytes)
                cand_text = ""
                if candidates:
                    preview = "\n".join([f"- {c.get('text','')} — {c.get('href','')}" for c in candidates[:8]])
                    cand_text = f"\n\nCandidates (choose one URL if it matches the official rulebook, otherwise ignore):\n{preview}\n\nReturn only a single URL."
                prompt = f"Game: {game_name}\n\n{RULEBOOK_EXTRACTION_PROMPT}{cand_text}"
                content_items = [{"type": "text", "text": prompt}]
                content_items.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}})
                if extra_images:
                    for img in extra_images[:2]:
                        enc = self._encode_image(self._optimize_image(img))
                        content_items.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{enc}"}})
                messages = [{"role": "user", "content": content_items}]
                # Simple retry for transient 5xx
                response = None
                last_err: Optional[Exception] = None
                for _ in range(2):
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            max_tokens=200,
                            temperature=0.1,
                        )
                        break
                    except Exception as e:
                        last_err = e
                        import time as _t
                        _t.sleep(0.6)
                if response is None:
                    raise last_err or RuntimeError("LLM request failed")
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content.strip()
                    if content and not any(phrase in content for phrase in ["None Found", "none found", "no rulebook download link", "no download"]):
                        url = self._clean_url_response(content)
                        if url:
                            return True, url, "high"
                        return False, "Invalid URL format in response", "low"
                    return False, "No rulebook link found", "high"
                return False, "No response from LLM API", "low"
                
        except Exception as e:
            error_msg = f"Error during LLM extraction: {e}"
            logger.error(error_msg)
            # Optional auto-fallback to MLX if available
            try:
                if VISION_BACKEND == "together":
                    logger.info("Falling back to MLX vision backend after Together.ai error")
                    # emulate MLX branch quickly
                    # Optimize again to ensure reasonable size
                    optimized_bytes = self._optimize_image(image_bytes)
                    import tempfile, subprocess, sys
                    prompt = f"Game: {game_name}\n\n{RULEBOOK_EXTRACTION_PROMPT}"
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
                        tmp.write(optimized_bytes)
                        tmp.flush()
                        cmd = [
                            sys.executable,
                            "-m",
                            "mlx_vlm.generate",
                            "--model",
                            str(self._mlx_model_id or MLX_VLM_MODEL),
                            "--max-tokens",
                            "120",
                            "--temp",
                            "0.0",
                            "--images",
                            tmp.name,
                            "--prompt",
                            prompt,
                        ]
                        try:
                            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
                            content = (proc.stdout or proc.stderr or "").strip()
                            if content and "http" in content.lower():
                                url = self._clean_url_response(content)
                                if url:
                                    return True, url, "medium"
                        except Exception:
                            pass
            except Exception:
                pass
            return False, error_msg, "low"
    
    def _clean_url_response(self, response: str) -> Optional[str]:
        """
        Clean and validate the URL response from the LLM.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Cleaned URL if valid, None otherwise
        """
        try:
            # Remove common prefixes and suffixes
            cleaned = response.strip()
            
            # Remove quotes
            cleaned = cleaned.strip('"\'')
            
            # Remove common prefixes
            prefixes_to_remove = [
                "The rulebook URL is: ",
                "I found the rulebook at: ",
                "Here's the link: ",
                "URL: ",
                "Link: "
            ]
            
            for prefix in prefixes_to_remove:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
            
            # Remove common suffixes
            suffixes_to_remove = [
                ".",
                ",",
                " - ",
                " (PDF)",
                " [PDF]"
            ]
            
            for suffix in suffixes_to_remove:
                if cleaned.endswith(suffix):
                    cleaned = cleaned[:-len(suffix)].strip()
            
            # Basic absolute URL validation
            if cleaned.startswith(('http://', 'https://')):
                return cleaned
            
            # Allow relative URLs; absolute resolution will be done by caller
            if cleaned.startswith(('/', './')) or cleaned.startswith('../'):
                return cleaned
            
            logger.warning(f"Response doesn't appear to be a valid URL: {cleaned}")
            return None
            
        except Exception as e:
            logger.warning(f"Error cleaning URL response: {e}")
            return None
    
    def test_connection(self) -> bool:
        """
        Test the vision backend availability.
        
        Returns:
            True if connection successful, False otherwise
        """
        if VISION_BACKEND == "mlx":
            try:
                import mlx_vlm  # noqa: F401
                return True
            except Exception as e:
                logger.error(f"MLX VLM not available: {e}")
                return False
        try:
            # Use a tiny 1x1 image to validate multimodal path for vision models
            import io as _io
            from PIL import Image as _Image
            tiny = _Image.new('RGB', (1, 1), (255, 255, 255))
            buf = _io.BytesIO()
            tiny.save(buf, format='JPEG')
            encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Ping"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                ]
            }]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=5
            )
            return bool(getattr(response, 'choices', None))
        except Exception as e:
            logger.error(f"Together.ai connection test failed: {e}")
            return False

    def assess_file_official_rulebook(self, file_path: Path, game_name: str = "", save_debug: bool = True) -> Tuple[bool, bool, str]:
        """
        Assess whether a PDF appears to be the official English rulebook.
        Returns (is_official, is_english, rationale_or_error).
        
        Args:
            file_path: Path to the file to assess
            game_name: Name of the game for context
            save_debug: Whether to save debug information when assessment fails
        """
        try:
            logger.info(f"Starting file assessment for {file_path.name} ({file_path.stat().st_size / 1024 / 1024:.2f} MB)")
            
            sample_text = ""
            file_type = ""
            
            if file_path.suffix.lower() == '.pdf':
                file_type = "PDF"
                # Extract text from first 2 pages only for faster, more reliable processing
                reader = PyPDF2.PdfReader(str(file_path))
                total_pages = len(reader.pages)
                num_pages = min(2, total_pages)  # Only first 2 pages for speed and accuracy
                logger.info(f"PDF has {total_pages} pages, extracting text from first {num_pages}")
                
                text_parts = []
                for i in range(num_pages):
                    try:
                        logger.debug(f"Extracting text from page {i+1}")
                        page_text = reader.pages[i].extract_text() or ""
                        # Limit each page to reasonable size to avoid massive text
                        if len(page_text) > 2000:
                            page_text = page_text[:2000] + "..."
                        text_parts.append(page_text)
                    except Exception as e:
                        logger.debug(f"Failed to extract text from page {i+1}: {e}")
                        pass
                
                sample_text = "\n\n".join(text_parts)
                if not sample_text.strip():
                    sample_text = "(No extractable text; may be image-based PDF)"
                    # Optional: basic OCR fallback for image-based PDFs
                    try:
                        import pytesseract
                        from pdf2image import convert_from_path
                        images = convert_from_path(str(file_path), first_page=1, last_page=min(2, total_pages))
                        ocr_text = []
                        for img in images:
                            ocr_text.append(pytesseract.image_to_string(img) or "")
                        sample_text = ("\n".join(ocr_text)).strip() or sample_text
                        logger.info(f"OCR extracted {len(sample_text)} characters of text")
                    except Exception as e:
                        logger.debug(f"OCR fallback unavailable/failed: {e}")
                    
            elif file_path.suffix.lower() in ['.html', '.htm']:
                file_type = "HTML"
                # Extract text and structural cues from HTML file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()

                # Use BeautifulSoup to extract clean text and metadata
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')

                # Title and headers for structural cues
                doc_title = (soup.title.string if soup.title and soup.title.string else "").strip()
                headers = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2'])]
                headers_text = " | ".join(headers[:10])

                # Count images as a proxy for designed documents (vs plain text FAQ)
                num_images = len(soup.find_all('img'))

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Get text content
                text = soup.get_text()

                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                sample_text = ' '.join(chunk for chunk in chunks if chunk)

                # Limit size for LLM
                if len(sample_text) > 6000:
                    sample_text = sample_text[:6000] + "..."
                    
            else:
                return False, False, f"Unsupported file type: {file_path.suffix}"
                
            logger.info(f"Extracted {len(sample_text)} characters of text for LLM analysis")

            # Provide extra structural context for HTML to steer the model
            extra_context = ""
            try:
                if file_type == "HTML":
                    title_txt = locals().get("doc_title", "")
                    headers_txt = locals().get("headers_text", "")
                    num_imgs = locals().get("num_images", 0)
                    extra_context = (
                        f"\nTitle: {title_txt}\n"
                        f"Top headers: {headers_txt}\n"
                        f"Image count: {num_imgs}\n"
                    )
            except Exception:
                pass

            prompt = f"""
Look at this text from a {file_type} file for the board game '{game_name}':

{sample_text[:4000]}
{extra_context}

Answer these 2 questions strictly:
1. Is this text in English? (yes/no)
2. Is this the official game rulebook document? (yes/no)

Important instructions:
- FAQs, errata, rules reference, quick reference, quickstart/learn-to-play guides, player aids, blog posts, news posts, product pages, and forum/discussion pages are NOT the rulebook.
- If the title or headers contain words like "FAQ", "Errata", "Reference", "Quick", "Guide", treat it as NOT a rulebook.
- An official HTML rulebook usually contains structured sections like Components, Setup, Gameplay/Turn Order/Phases, and often includes images or diagrams.
- Only answer "yes" for official if it clearly appears to be the rulebook itself.

Respond ONLY with a valid JSON object with two boolean fields.
Example: {{"is_english": true, "is_official": false}}
"""

            if VISION_BACKEND == "mlx":
                # Lightweight vision check for PDFs: render first page → ask MLX VLM a simple JSON yes/no
                if file_type == "PDF":
                    try:
                        import tempfile
                        import subprocess
                        from pdf2image import convert_from_path
                        images = convert_from_path(str(file_path), first_page=1, last_page=1, fmt="png")
                        if images:
                            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                                images[0].save(tmp.name, format="PNG")
                                prompt_img = (
                                    "You will see the first page of a board game document.\n"
                                    f"Game: {game_name}\n\n"
                                    "Answer strictly in JSON with two booleans: {\"is_english\": <bool>, \"is_official\": <bool>}\n"
                                    "Guidance: An official rulebook usually has sections like Components, Setup, How to Play/Rules,\n"
                                    "publisher branding, and page numbers. Exclude FAQs, errata, product pages, and marketing flyers.\n"
                                )
                                cmd = [
                                    __import__("sys").executable,
                                    "-m",
                                    "mlx_vlm.generate",
                                    "--model",
                                    str(self._mlx_model_id),
                                    "--max-tokens",
                                    "60",
                                    "--temp",
                                    "0.0",
                                    "--images",
                                    tmp.name,
                                    "--prompt",
                                    prompt_img,
                                ]
                                proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
                                content = (proc.stdout or proc.stderr or "").strip()
                                try:
                                    import json as _json
                                    # Extract first JSON object found in output
                                    start = content.find("{")
                                    end = content.find("}", start) + 1 if start != -1 else -1
                                    if start != -1 and end != -1:
                                        data = _json.loads(content[start:end])
                                        ans_off = bool(data.get("is_official", False))
                                        ans_en = bool(data.get("is_english", False))
                                        return ans_off, ans_en, "mlx_vision_pdf"
                                except Exception:
                                    pass
                    except Exception:
                        # Fall through to heuristic if vision path fails
                        pass

                # Heuristic assessment (used for HTML or PDF fallback)
                lower = sample_text.lower()
                english_cues = [" the ", "setup", "components", "players", "round", "phase", "victory", "turn order", "objective"]
                non_english_cues = ["spieler", "regeln", "regle", "reglas", "spiel", "punkte", "objet", "objetivo"]
                is_english = sum(tok in lower for tok in english_cues) >= 2 and sum(tok in lower for tok in non_english_cues) == 0

                is_official = False
                if file_type == "HTML":
                    title_l = (locals().get("doc_title", "") or "").lower()
                    headers_l = (locals().get("headers_text", "") or "").lower()
                    num_imgs = int(locals().get("num_images", 0) or 0)

                    exclusion_terms = ["faq", "errata", "reference", "quickstart", "quick start", "learn to play", "player aid", "guide"]
                    has_exclusion = any(term in title_l or term in headers_l for term in exclusion_terms)

                    section_cues = ["setup", "components", "gameplay", "turn order", "phases", "objective"]
                    has_sections = sum(cue in lower for cue in section_cues) >= 2
                    has_rulebook_word = ("rulebook" in title_l) or ("rulebook" in headers_l) or ("rulebook" in lower) or ("rules" in headers_l) or ("rules" in title_l)
                    has_images = num_imgs >= 2

                    is_official = (not has_exclusion) and has_sections and (has_rulebook_word or has_images)
                else:
                    # Relaxed PDF heuristic
                    cues = ["setup", "components", "rules", "game setup", "gameplay", "round", "turn", "victory", "objective"]
                    is_official = sum(c in lower for c in cues) >= 2

                return is_official, is_english, "heuristic"
            else:
                logger.info("Sending PDF assessment request to LLM...")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                if not response.choices:
                    return False, False, "No response from LLM"
                content = response.choices[0].message.content.strip()
                try:
                    assessment = PDFAssessment.model_validate_json(content)
                    if save_debug and (not assessment.is_official or not assessment.is_english):
                        self._save_debug_info(file_path, game_name, sample_text, prompt, content, assessment)
                    return assessment.is_official, assessment.is_english, "LLM assessment successful"
                except ValidationError as e:
                    lower = content.lower()
                    is_official = ("rulebook" in lower) and ("official" in lower)
                    is_english = any(tok in lower for tok in ["english", "en-us", "en-gb"]) and not any(tok in lower for tok in ["french", "german", "spanish", "italian"])
                    if save_debug:
                        self._save_debug_info(file_path, game_name, sample_text, prompt, content, PDFAssessment(is_official=is_official, is_english=is_english))
                    return is_official, is_english, f"Fallback parsing used due to validation error: {str(e)[:100]}"
                except Exception as e:
                    return False, False, f"Error parsing LLM response: {str(e)[:100]}"
        except Exception as e:
            return False, False, f"Error assessing PDF: {e}"
    
    def _save_debug_info(self, file_path: Path, game_name: str, sample_text: str, prompt: str, 
                         llm_response: str, assessment: PDFAssessment) -> None:
        """
        Save debug information when LLM assessment fails.
        This helps debug why files are being rejected.
        """
        try:
            from pathlib import Path
            import json
            from datetime import datetime
            
            # Create debug directory if it doesn't exist
            debug_dir = Path("bgg_data/debug/llm_assessments")
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a unique filename for this assessment
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_game_name = "".join(c for c in game_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_game_name = safe_game_name.replace(' ', '_')
            debug_filename = f"{timestamp}_{safe_game_name}_{file_path.stem}_assessment.json"
            debug_path = debug_dir / debug_filename
            
            # Prepare debug data
            debug_data = {
                "timestamp": timestamp,
                "game_name": game_name,
                "file_path": str(file_path),
                "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                "file_type": file_path.suffix.lower(),
                "llm_prompt": prompt,
                "llm_response": llm_response,
                "assessment_result": {
                    "is_official": assessment.is_official,
                    "is_english": assessment.is_english
                },
                "sample_text_preview": sample_text[:1000] + "..." if len(sample_text) > 1000 else sample_text,
                "sample_text_length": len(sample_text)
            }
            
            # Save debug data
            with open(debug_path, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved LLM assessment debug info to: {debug_path}")
            
        except Exception as e:
            logger.error(f"Failed to save debug info: {e}")
