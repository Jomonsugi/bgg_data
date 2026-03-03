
**Task:** Build a Jupyter Notebook for a "Visual-Aware & Grounded RAG Pipeline" for board game rulebooks on Apple Silicon (MPS).

**Primary Goal:** Implement a pipeline that not only retrieves rules but also preserves the exact page coordinates () for every rule, enabling "Visual Citations" (highlighting the PDF).

**The Stack:**

* **Parsing:** `docling` (PDF to Markdown + Coordinate Extraction).
* **Vector DB:** `qdrant-client` (Local storage).
* **Reranking:** `fastembed` (Model: `BAAI/bge-reranker-v2-m3`).
* **Visuals:** `colpali-engine` (following the logic in `L2.ipynb`).
* **Hardware:** Use `device="mps"` for Docling and ColPali.

**Notebook Requirements:**

1. **Stage 1: Enhanced Docling Ingestion**
* Load a sample rulebook PDF.
* Use `DocumentConverter` to parse the PDF.
* **CRITICAL:** Instead of just calling `export_to_markdown()`, iterate through the `result.document.iterate_items()`.
* For each text element, extract:
* The **Markdown text**.
* The **Page Number**.
* The **Bounding Box ()** coordinates from the `item.prov` (provenance) attribute.




2. **Stage 2: Qdrant Indexing with "Addresses"**
* Upsert chunks into Qdrant.
* **Metadata Payload:** Store the `text`, `page_number`, and a list/dict of the `bbox` coordinates. This is the "address" we will use later for highlighting.


3. **Stage 3: Hybrid Retrieval & Local Rerank**
* Implement a function that retrieves 20 chunks via vector search and reranks them using `FastEmbed`.
* Display the top 3 results, showing the text alongside their "Page + Bounding Box" metadata.


4. **Stage 4: PDF Visual Highlighter (The "Answer Proof")**
* Use `PyMuPDF` (import as `fitz`) to create a function `highlight_rule(pdf_path, page_num, bbox)`.
* This function should:
1. Open the original PDF.
2. Go to the specific page.
3. Draw a semi-transparent yellow rectangle over the coordinates provided by Docling.
4. Render the page as a PNG and display it in the notebook using `IPython.display`.




5. **Stage 5: ColPali Integration**
* Use the provided `helper.py` and `L2.ipynb` to run a ColPali search if the user's query is visual or if the reranked text chunk contains an image tag.


6. **Stage 6: Final Demonstration**
* Run a query.
* Output the text answer.
* Display the "Evidence": The original PDF page with a yellow highlight over the rule.

---

### **Why this works for a future Agent**

By doing this now, a "map" of your rulebooks is built.

* **The LLM Agent** reads the clean Markdown to understand the rules.
* **The UI** uses the `bbox` metadata to show the user exactly where to look.
* **ColPali** acts as the "Optical Zoom" for diagrams that text can't describe.