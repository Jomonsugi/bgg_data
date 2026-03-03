# Board Game Rules Agent

A local AI assistant that answers board game rules questions with **cited, highlighted** references to the official rulebook — built for fast lookups during actual gameplay.

## How it works

1. You provide PDFs (rulebook + any supplemental docs) for a game once. Docling parses them into text + bounding boxes and caches the output.
2. You ask a rules question in the chat.
3. The agent retrieves the most relevant rulebook pages (RAG), optionally searches BoardGameGeek for community clarifications, and returns a structured answer with citations.
4. Click any citation chip to jump to that exact page in the PDF with the relevant passage highlighted.

---

## Setup

### 1. Install dependencies

```bash
cd /path/to/bgg_data
uv sync   # or: pip install -e ".[dev]"
```

### 2. Configure API keys

Either export from your shell:

```bash
export TOGETHER_API_KEY="..."
export TAVILY_API_KEY="..."
```

Or create a `.env` file in `bgg_data/boardgame_agent/`:

```bash
cp bgg_data/boardgame_agent/.env.example bgg_data/boardgame_agent/.env
# edit .env and fill in your keys
```

### 3. Run the app

From the project root (where `pyproject.toml` lives):

```bash
streamlit run -m bgg_data.boardgame_agent.app
```

This runs the app as an installed module so `bgg_data` imports resolve correctly.

---

## First use

1. In the sidebar, click **Add new game** → enter a game name → **Create game**.
2. Upload your rulebook PDF(s) via the **Add PDF(s)** uploader, or paste a folder path and click **Index folder**.
3. Docling will parse and cache the PDFs (one-time, takes ~30–60s per document).
4. Ask a question in the chat.

---

## Switching models

### LLM (Together API)
Change `TOGETHER_MODEL_NAME` in `.env` or `config.py`. Takes effect immediately — no reindexing needed.

### Embedding model
Change `EMBED_MODEL_NAME` in `.env` or `config.py`, then click **Rebuild index** in the sidebar. This re-embeds all cached Docling output (Docling does **not** re-run). Only the vector representations are rebuilt.

---

## Adding more documents to a game

Upload additional PDFs in the sidebar at any time. New documents are added to the existing index without affecting anything already indexed.

---

## Web search domains

By default the agent restricts web search to `boardgamegeek.com`. You can add or remove domains in the sidebar per game, or clear all to allow unrestricted search.

---

## Adding new agent tools (for developers)

1. Create `bgg_data/boardgame_agent/agent/tools/your_tool.py` with a `make_your_tool()` factory that returns a `@tool`-decorated function.
2. Import it in `agent/tools/__init__.py` and add it to the `make_all_tools()` return list.

That's the only change needed — `graph.py` picks up whatever `make_all_tools()` returns.

---

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
│       ├── __init__.py # Tool registry (add new tools here)
│       ├── rag.py      # search_rulebook
│       ├── web_search.py # search_web
│       └── history.py  # get_past_answers
├── rag/
│   ├── extractor.py    # Docling PDF parsing + JSON cache
│   ├── indexer.py      # Qdrant indexing + reindex_all()
│   └── retriever.py    # Vector search
├── db/
│   └── games.py        # SQLite: games, documents, domains, Q&A history
├── ui/
│   ├── pdf_panel.py    # PyMuPDF highlights + PDF viewer
│   └── sidebar.py      # Game & document management UI
└── data/               # Runtime data (gitignored)
    ├── qdrant/
    ├── games.db
    ├── agent_checkpoints.db
    └── games/
        └── {game_id}/
            ├── pdfs/
            └── extracted/   # Cached Docling JSON
```
