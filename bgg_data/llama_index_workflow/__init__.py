"""Simple LlamaIndex Workflow for fetching board game rulebooks.

This package provides a minimal workflows v2 setup with tools to:
- query the local SQLite DB for games by rank
- search for rulebook PDF links via DuckDuckGo
- download PDFs to a local folder

Designed to be easy to read and extend.
"""

__all__ = ["tools", "workflow", "cli"]


