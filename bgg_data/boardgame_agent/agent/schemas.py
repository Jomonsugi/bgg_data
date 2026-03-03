"""Shared Pydantic schemas used by the agent and the RAG pipeline."""

from typing import List
from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_name: str = Field(description="Stem of the source PDF (e.g. 'Ark-Nova_342942_rules')")
    page_num: int = Field(description="1-indexed page number in that document")
    bbox_indices: List[int] = Field(
        description="Indices into that page's bbox array that contain the cited text"
    )


class QAWithCitations(BaseModel):
    reasoning: str = Field(description="Brief chain-of-thought explaining the answer")
    answer: str = Field(description="Clear, concise answer to the rules question")
    citations: List[Citation] = Field(
        description="Rulebook citations grounding the answer (always required)"
    )
    web_sources: List[str] = Field(
        default=[],
        description=(
            "URLs from search_web results that were used to confirm or clarify the answer. "
            "Empty list if search_web was not called."
        ),
    )
