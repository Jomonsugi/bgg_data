# Board Game Rules Agent

A local AI assistant that answers board game rules questions with **cited, highlighted** references to the official rulebook — built for fast lookups during actual gameplay.

## Quick start

Install prerequisites:
- [uv](https://docs.astral.sh/uv/getting-started/installation/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [Ollama](https://ollama.com/download): download the macOS app, then `ollama pull qwen3-embedding`
- A [Together API](https://www.together.ai/) key (free tier works — this is the default LLM provider)

Then:

```bash
cd bgg_data
uv sync
cp bgg_data/boardgame_agent/.env.example bgg_data/boardgame_agent/.env
# Edit .env and add your TOGETHER_API_KEY
uv run streamlit run bgg_data/boardgame_agent/app.py
```

Create a game in the sidebar, upload a rulebook PDF, and ask a question. That's it.

---

## Using the app

**Create a game and add documents.** Click **Add new game** — the new game is auto-selected. Upload rulebook PDFs or point to a folder. Docling parses each PDF once (can take a few minutes for large rulebooks). The first query also downloads the SPLADE++ sparse model (~530 MB, one-time).

**Ask questions.** Type a rules question in the chat. The agent searches the indexed rulebook, retrieves relevant pages, and returns a cited answer. Click any **citation chip** to jump to that page in the PDF viewer with the passage highlighted.

**Rate answers.** Each response has ✅ and ❌ buttons. Accepted answers feed into the `get_past_answers` tool so the agent stays consistent with prior verified rulings. Click again to undo.

**Top-k slider.** Adjusts how many rulebook pages are retrieved per query. Takes effect immediately — no session reset.

**Web search (optional).** Requires a `TAVILY_API_KEY` in `.env`. When set, a checkbox appears in the sidebar to enable/disable web search. Add trusted domains (e.g., `boardgamegeek.com`) to restrict where the agent searches.

**Switching LLM models.** Use the dropdown in the sidebar. Changing the model resets the current conversation (you'll be warned first).

**Rebuild index.** After changing the embedding model in `config.py`, click **Rebuild index** in the sidebar. This re-embeds all cached documents — Docling does not re-run.

## LLM providers

The default models use Together API, but you can use Anthropic, OpenAI, or any combination. Models and their providers are configured in `config.py` under `MODEL_OPTIONS` — map each model ID to `"together"`, `"anthropic"`, or `"openai"`. Only add API keys for the providers you use. If a key is missing when you select a model, you'll get a clear error telling you which key to set.

## Embeddings

Dense vectors via Ollama (default `qwen3-embedding`, 4096-d). Sparse vectors via FastEmbed SPLADE++. Results are fused with Qdrant-native RRF hybrid search. Any Ollama embedding model can be used — change `OLLAMA_EMBED_MODEL` in `config.py` and click **Rebuild index**.

Ollama launches automatically if the app is installed but not running.

## Project structure

```
boardgame_agent/
├── app.py              # Streamlit entry point
├── config.py           # All tunable settings
├── agent/
│   ├── graph.py        # LangGraph ReAct agent
│   ├── prompts.py      # System and format prompts
│   ├── schemas.py      # QAWithCitations, Citation
│   ├── state.py        # AgentState
│   └── tools/
│       ├── __init__.py # Tool registry
│       ├── rag.py      # search_rulebook (hybrid dense + sparse)
│       ├── web_search.py # search_web (Tavily, optional)
│       └── history.py  # get_past_answers
├── rag/
│   ├── extractor.py    # Docling PDF parsing + JSON cache
│   ├── indexer.py      # Qdrant hybrid indexing (Ollama + SPLADE++)
│   └── retriever.py    # Hybrid retrieval with RRF fusion
├── db/
│   └── games.py        # SQLite: games, documents, domains, Q&A history
├── ui/
│   ├── pdf_panel.py    # PyMuPDF highlights + PDF viewer
│   └── sidebar.py      # Game & document management UI
└── data/               # Runtime data (gitignored)
    ├── qdrant/
    ├── games.db
    └── games/{game_id}/
        ├── pdfs/
        └── extracted/  # Cached Docling JSON
```
