"""Shared data contracts. Changes must be agreed by both members."""
from typing import Literal, Optional, Protocol
from pydantic import BaseModel, Field

Lang = Literal["fr", "en", "ar", "darija"]


class Chunk(BaseModel):
    id: str                      # ex: "wiki:Diabète_de_type_2#3"
    text: str
    source: str                  # "wikipedia_fr", "who", "kdigo_2024"...
    doc_title: str
    section: Optional[str] = None
    page: Optional[int] = None
    lang: Lang = "fr"
    version: Optional[str] = None  # dump date / guideline version
    url: Optional[str] = None
    module: Optional[str] = None   # résidanat module (cardiologie...)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    rank: int


class Question(BaseModel):
    id: str
    text: str
    lang: Lang
    type: Literal["open", "mcq"]
    options: dict[str, str] = Field(default_factory=dict)   # {"A": "...", ...}
    gold_answer: Optional[str] = None
    gold_options: list[str] = Field(default_factory=list)   # ["A", "C"]
    gold_chunk_ids: list[str] = Field(default_factory=list)
    module: Optional[str] = None
    answerable: bool = True
    split: Literal["dev", "test"] = "dev"


class Answer(BaseModel):
    question_id: str
    text: str
    selected_options: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)       # chunk ids
    refused: bool = False
    model: str
    latency_s: float = 0.0


class Retriever(Protocol):
    """Every retriever (BM25, dense, hybrid, reranker) implements this."""
    def search(self, query: str, k: int = 10) -> list[RetrievedChunk]: ...
