# Visual-Aware Rulebook RAG: Retrieval Evaluation Notebook

**Purpose:** Demonstrate and compare retrieval strategies for board game rulebooks to inform an agentic RAG system. Focus on visualizing what each retrieval method returns, enabling informed decisions about when text-retrieval suffices vs. when visual/multimodal approaches are needed.

---

## Stack

- **PDF Processing:** `docling` (Advanced PDF understanding + bbox/provenance extraction)
- **Vector Database:** `qdrant-client` (Local mode, multi-vector support)
- **Text Embeddings:** `fastembed` (lightweight, fast embeddings)
- **Reranking:** `fastembed` (`BAAI/bge-reranker-v2-m3`)
- **Visual Retrieval:** `colpali-engine` (multi-vector image embeddings)
- **Visualization:** `PyMuPDF` (fitz) for highlighting, `pdf2image` for rendering
- **Hardware:** MPS acceleration on Apple Silicon

---

## Notebook Structure

### **Stage 0: Setup & Imports**
- Load all dependencies
- Set MPS device configs
- Define helper functions for visualization
- Load sample rulebook (e.g., `Azul_230802_rules.pdf` or `Spirit-Island_rules.pdf`)

### **Stage 1: Advanced Docling Ingestion**

**Goal:** Parse the rulebook and extract structured page-level documents with full provenance tracking.

**Implementation:**
```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Configure Docling for maximum information extraction
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

result = converter.convert("rulebooks/Azul_230802_rules.pdf")
```

**Data Extraction Per Page:**
For each page, extract:
1. **Full page text** (markdown format)
2. **Page number** (from document structure)
3. **Bounding boxes** with provenance:
   - Iterate through `result.document.iterate_items()`
   - For each item with `item.prov`, extract bbox coordinates
   - Store relationship: text chunk → bbox coordinates
4. **Page image** (for ColPali and visualization)

**Output Structure:**
```python
pages = [
    {
        "page_num": 0,
        "text": "Full page text...",
        "bboxes": [
            {"x0": 100, "y0": 50, "x1": 500, "y1": 100, "text": "Game Overview"},
            # ... more bboxes
        ],
        "page_image_path": "pages/azul_page_00.png",
        "has_tables": True,
        "has_figures": True
    },
    # ... more pages
]
```

**Visualization:**
- Display sample page with its extracted text
- Show bbox overlay on the page image to verify coordinate accuracy
- Print statistics: number of pages, text elements per page, table count, etc.

---

### **Stage 2: Dual-Vector Qdrant Indexing**

**Goal:** Create a Qdrant collection that supports both text-based and visual retrieval.

**Implementation:**
```python
from qdrant_client import QdrantClient, models
import uuid

client = QdrantClient(path="./qdrant_rulebook_db")

collection_name = "rulebook_pages"

# Create collection with multiple named vectors
client.create_collection(
    collection_name,
    vectors_config={
        # Dense text vector for semantic search
        "text": models.VectorParams(
            size=768,  # fastembed dimension
            distance=models.Distance.COSINE
        ),
        # ColPali multi-vector for visual similarity
        "colpali": models.VectorParams(
            size=128,  # ColPali token dimension
            distance=models.Distance.DOT,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            hnsw_config=models.HnswConfigDiff(m=0)  # Disable HNSW for multivector
        )
    }
)
```

**Embedding Generation:**
1. **Text embeddings:** Use FastEmbed to encode page text
2. **Visual embeddings:** Use ColPali to encode page images (reuse L2.ipynb approach)

**Upsert Strategy:**
```python
from fastembed import TextEmbedding
from colpali_engine.models import ColPali, ColPaliProcessor
import torch

text_model = TextEmbedding("BAAI/bge-small-en-v1.5")
colpali_model = ColPali.from_pretrained("vidore/colpali-v1.3").to("mps")
colpali_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3")

for page_data in pages:
    # Generate embeddings
    text_embedding = list(text_model.embed([page_data["text"]]))[0]
    
    page_image = Image.open(page_data["page_image_path"])
    colpali_embedding = encode_page_image(page_image, colpali_model, colpali_processor)
    
    # Upsert to Qdrant
    client.upsert(
        collection_name,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "text": text_embedding,
                    "colpali": colpali_embedding
                },
                payload={
                    "page_num": page_data["page_num"],
                    "text": page_data["text"],
                    "bboxes": page_data["bboxes"],
                    "page_image_path": page_data["page_image_path"],
                    "rulebook_name": "Azul",
                    "has_tables": page_data["has_tables"],
                    "has_figures": page_data["has_figures"]
                }
            )
        ]
    )
```

**Visualization:**
- Print collection info (vector count, dimensions)
- Show sample point with full payload structure

---

### **Stage 3: Text-Only Retrieval Baseline**

**Goal:** Establish baseline retrieval performance using only text embeddings.

**Test Queries:**
```python
test_queries = [
    "How is scoring calculated?",
    "What happens when tiles fall on the floor?",
    "How do you set up the game?",
    "What does the first player marker look like?",  # Visual query
    "Can you show an example of a complete row?"     # Diagram query
]
```

**Retrieval Function:**
```python
def retrieve_text_only(query: str, limit: int = 5):
    query_embedding = list(text_model.embed([query]))[0]
    
    results = client.search(
        collection_name=collection_name,
        query_vector=("text", query_embedding),
        limit=limit,
        with_payload=True
    )
    return results
```

**Visualization for Each Query:**
1. Display query text
2. Show top-3 retrieved pages with:
   - Page number
   - Similarity score
   - Text snippet (first 200 chars)
   - **Full page image** rendered from PDF
3. Create a grid view showing all retrieved pages side-by-side

**Analysis:**
- Which queries work well with text-only?
- Which queries return wrong pages? (especially visual/diagram queries)

---

### **Stage 4: ColPali Visual Retrieval**

**Goal:** Demonstrate pure visual retrieval for comparison with text-based approach.

**Retrieval Function:**
```python
def retrieve_colpali_only(query: str, limit: int = 5):
    # Encode query with ColPali processor
    batch_query = colpali_processor.process_queries([query]).to("mps")
    with torch.no_grad():
        query_embedding = colpali_model(**batch_query)[0].cpu().numpy()
    
    results = client.search(
        collection_name=collection_name,
        query_vector=("colpali", query_embedding),
        limit=limit,
        with_payload=True
    )
    return results
```

**Run Same Test Queries:**
- Use identical queries from Stage 3
- Display results in same format for comparison

**Visualization:**
1. Side-by-side comparison: Text retrieval vs. ColPali retrieval
2. Highlight where ColPali succeeds (diagram queries) vs. where it fails
3. Show similarity map heatmaps for visual queries (using ColPali interpretability tools)

**Analysis:**
- Does ColPali better retrieve pages with visual elements?
- How does it handle pure text queries?
- Validate your notes.md hypothesis about ColPali focusing on text

---

### **Stage 5: Hybrid Retrieval Strategy**

**Goal:** Combine text and visual signals for optimal retrieval.

**Implementation Options:**

**Option A: Reciprocal Rank Fusion (RRF)**
```python
def retrieve_hybrid_rrf(query: str, limit: int = 5):
    # Get both result sets
    text_results = retrieve_text_only(query, limit=20)
    colpali_results = retrieve_colpali_only(query, limit=20)
    
    # Apply RRF
    fused_scores = compute_rrf(text_results, colpali_results, k=60)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
```

**Option B: Query Classification**
```python
def retrieve_adaptive(query: str, limit: int = 5):
    # Classify query type
    visual_keywords = ["show", "look like", "example", "diagram", "picture"]
    is_visual = any(kw in query.lower() for kw in visual_keywords)
    
    if is_visual:
        return retrieve_colpali_only(query, limit)
    else:
        return retrieve_text_only(query, limit)
```

**Visualization:**
- Show results for hybrid approach
- Compare with Stage 3 & 4 results
- Highlight improvements and trade-offs

---

### **Stage 6: Reranking with FastEmbed**

**Goal:** Apply cross-encoder reranking to improve top-k precision.

**Implementation:**
```python
from fastembed import LateInteractionTextEmbedding

reranker = LateInteractionTextEmbedding("BAAI/bge-reranker-v2-m3")

def retrieve_with_reranking(query: str, initial_k: int = 20, final_k: int = 3):
    # Initial retrieval (text or hybrid)
    initial_results = retrieve_text_only(query, limit=initial_k)
    
    # Extract texts for reranking
    documents = [r.payload["text"] for r in initial_results]
    
    # Rerank
    rerank_scores = reranker.query_embed(query, documents)
    
    # Sort by reranked scores
    reranked = sorted(
        zip(initial_results, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    return [r[0] for r in reranked[:final_k]]
```

**Visualization:**
1. Show "Before Reranking" top-5 with scores
2. Show "After Reranking" top-5 with new scores
3. Highlight rank changes (page moved up/down)
4. Display the final top-3 pages with full images

**Analysis:**
- Does reranking improve relevance?
- Which queries benefit most from reranking?

---

### **Stage 7: Visual Citation & Highlighting**

**Goal:** Demonstrate bbox-based highlighting for "grounded" retrieval results.

**Implementation:**
```python
import fitz
from PIL import Image

def highlight_bboxes_on_page(pdf_path: str, page_num: int, bboxes: list) -> Image:
    """
    Open PDF, highlight specific bboxes, return rendered page image.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    for bbox in bboxes:
        # Note: May need coordinate transformation depending on Docling's bbox format
        rect = fitz.Rect(bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"])
        highlight = page.add_highlight_annot(rect)
        highlight.set_colors(stroke=[1, 1, 0])  # Yellow
        highlight.update()
    
    # Render to image
    pix = page.get_pixmap(dpi=150)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img
```

**Demonstration:**
1. Take a query: "How is scoring calculated?"
2. Retrieve top result
3. Show:
   - Retrieved text snippet
   - Full page image (unhighlighted)
   - Page with relevant bboxes highlighted (yellow overlay)
   
**Analysis:**
- How accurately do bboxes align with relevant text?
- Are there coordinate system mismatches to fix?

---

### **Stage 8: Comparative Analysis & Insights**

**Goal:** Summarize findings to inform the agentic system design.

**Create Comparison Table:**
| Query Type | Text-Only | ColPali-Only | Hybrid | Reranked | Best Approach |
|------------|-----------|--------------|--------|----------|---------------|
| Text rule  | ✅ Good   | ⚠️ OK        | ✅ Good | ✅ Great | Text + Rerank |
| Visual element | ❌ Poor | ✅ Good    | ✅ Good | ⚠️ OK   | ColPali/Hybrid |
| Scoring calculation | ✅ Good | ⚠️ OK | ✅ Good | ✅ Great | Text + Rerank |

**Visualizations:**
1. Precision@K curves for each approach
2. Query time comparisons (text vs. ColPali vs. hybrid)
3. Heatmap: Query type × Retrieval method → Relevance score

**Key Insights for Agent:**
- When should the agent use text-only retrieval? (fast, good for factual rules)
- When should it invoke ColPali? (diagrams, visual examples, "show me" queries)
- When is reranking worth the compute? (ambiguous queries, low initial confidence)
- What confidence threshold should trigger VLM calls? (based on retrieval scores)

---

### **Stage 9: Mock Agent Decision Flow**

**Goal:** Simulate how the future agent would use these retrieval tools.

**Pseudo-Agent:**
```python
def agent_retrieve_and_decide(query: str):
    """
    Simulates agent logic for deciding retrieval strategy and VLM invocation.
    """
    # Step 1: Classify query
    is_visual_query = classify_query_type(query)
    
    # Step 2: Choose retrieval strategy
    if is_visual_query:
        results = retrieve_colpali_only(query, limit=3)
        confidence = results[0].score
    else:
        results = retrieve_with_reranking(query, initial_k=20, final_k=3)
        confidence = results[0].score
    
    # Step 3: Decision point
    if confidence > 0.85:
        decision = "SUFFICIENT - Return text from top page"
    elif confidence > 0.70:
        decision = "MEDIUM - Consider VLM call with top page image"
    else:
        decision = "LOW CONFIDENCE - VLM required with top 3 pages"
    
    # Visualize decision
    print(f"Query: {query}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Decision: {decision}")
    display_retrieved_pages(results)
    
    return decision, results
```

**Test Cases:**
Run through 5-10 diverse queries and show:
- The retrieval results
- The agent's decision (text-only vs. VLM call)
- Why that decision makes sense

---

## Deliverables

By the end of this notebook, you will have:

1. ✅ **Parsed rulebook** with full page-level structure + bbox metadata
2. ✅ **Multi-vector Qdrant collection** with text + ColPali embeddings
3. ✅ **Comparative evaluation** of 4 retrieval strategies
4. ✅ **Visual demonstrations** showing actual PDF pages retrieved
5. ✅ **Bbox-based highlighting** for grounded citations
6. ✅ **Quantitative insights** on when each approach works best
7. ✅ **Decision framework** for the future agent system

---

## Notes on Code Quality

- **Readable:** Each cell should be self-contained with clear comments
- **Educational:** Explain why choices are made (e.g., "Using RRF because...")
- **Reproducible:** Include version info, seed setting for consistency
- **Modular:** Define reusable functions, avoid copy-paste code
- **Visual:** Every retrieval experiment should show the actual pages, not just scores

---

## Success Criteria

The notebook successfully demonstrates retrieval capabilities if:
1. You can visually verify what pages are retrieved for different query types
2. You can quantify performance differences between retrieval strategies
3. You have clear guidelines on when to use text vs. visual vs. hybrid retrieval
4. The bbox highlighting accurately pinpoints relevant content on pages
5. The insights inform concrete design decisions for the agentic RAG system

---

## Extensions (Optional)

- **Multi-rulebook indexing:** Index 3-5 different games, test cross-game queries
- **Table-aware retrieval:** Special handling for pages with tables
- **Query expansion:** Use LLM to reformulate ambiguous queries before retrieval
- **A/B testing:** Formal evaluation on a labeled test set of query-page pairs
