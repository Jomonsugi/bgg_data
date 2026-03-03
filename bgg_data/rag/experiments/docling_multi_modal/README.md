# Visual-Aware Rulebook RAG: Retrieval Evaluation

An experimental notebook for evaluating different retrieval strategies on board game rulebooks using Docling, Qdrant, and ColPali.

## Overview

This notebook demonstrates:
- **Page-level PDF parsing** with bounding box extraction using Docling
- **Dual-vector indexing** (text + visual) in Qdrant
- **Four retrieval strategies**: Text-only, ColPali visual, Hybrid RRF, and Reranked
- **Visual citations** with bbox-based highlighting
- **Agent decision logic** for smart VLM invocation

## Setup

### Prerequisites

- Python 3.10 or higher (3.12 recommended)
- Apple Silicon Mac (for MPS acceleration) or CUDA-capable GPU
- 8GB+ RAM recommended

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Additional Requirements

- **poppler-utils** (for pdf2image):
  ```bash
  # macOS
  brew install poppler
  
  # Ubuntu/Debian
  sudo apt-get install poppler-utils
  ```

## Usage

1. **Start Jupyter**:
   ```bash
   jupyter notebook rulebook_rag_retrieval.ipynb
   ```

2. **Configure paths** in the notebook:
   - Set `RULEBOOK_PATH` to your PDF
   - Adjust `OUTPUT_DIR` as needed

3. **Run all cells** sequentially

The notebook is designed to be run cell-by-cell so you can observe and learn from each stage.

## Notebook Structure

- **Stage 0**: Setup & imports
- **Stage 1**: Docling ingestion (PDF → pages with bboxes)
- **Stage 2**: Qdrant dual-vector indexing
- **Stage 3**: Text-only retrieval baseline
- **Stage 4**: ColPali visual retrieval
- **Stage 5**: Hybrid retrieval with RRF
- **Stage 6**: Reranking with FastEmbed
- **Stage 7**: Visual citations with bbox highlighting
- **Stage 8**: Comparative analysis
- **Stage 9**: Mock agent decision flow

## Expected Output

- **Parsed pages** with extracted text and bboxes
- **Qdrant collection** with ~6 pages (for Azul)
- **Comparison visualizations** showing retrieval method differences
- **Highlighted PDFs** showing grounded citations
- **Agent decision summary** for different query types

## Performance Notes

- **First run**: Expect 5-10 minutes (model downloads + embedding generation)
- **Subsequent runs**: ~2-3 minutes if using cached embeddings
- **Memory usage**: ~4-6GB during ColPali encoding

## Outputs

Generated files in `./output/`:
- `pages/` - PNG images of each PDF page
- `qdrant_rulebook_db/` - Local Qdrant database
- `highlighted_page_*.png` - Pages with bbox highlights
- `experiment_summary.json` - Results summary

## Tips for Learning

1. **Read cell explanations** before running
2. **Examine retrieved pages visually** - this is key!
3. **Try your own queries** after Stage 3
4. **Adjust confidence thresholds** in Stage 9
5. **Compare scores across methods** in Stage 8

## Extending the Notebook

Ideas for further experiments:
- Index multiple rulebooks
- Test cross-game queries
- Fine-tune confidence thresholds
- Implement query expansion
- Add actual VLM integration
- Test on scanned (OCR) rulebooks

## Troubleshooting

**ColPali fails to load:**
- Check torch/MPS compatibility
- Try `device="cpu"` if MPS issues occur

**Coordinate misalignment:**
- Docling and PyMuPDF may use different coordinate systems
- Adjust the transformation in `highlight_bboxes_on_page()`

**Out of memory:**
- Process fewer pages at once
- Reduce batch sizes in embedding generation
- Use CPU instead of MPS for ColPali

## References

- [Docling Documentation](https://docling-project.github.io/docling/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [ColPali Paper](https://arxiv.org/abs/2407.01449)
- [M3DocRAG Paper](https://huggingface.co/papers/2411.04952)

## License

Experimental notebook for educational purposes.
