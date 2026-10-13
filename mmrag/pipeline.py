"""Retrieve the most relevant page, then let the VLM answer from that page's image and text."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .index import ChunkIndex


class Reader(Protocol):
    def answer(self, question: str, image_path: str, context: list[str]) -> str: ...


@dataclass
class Answer:
    question: str
    answer: str
    doc_id: str
    page: int
    image_path: str
    sources: list[dict] = field(default_factory=list)


class RAGPipeline:
    def __init__(self, index: ChunkIndex, reader: Reader, k: int = 5, mode: str = "bm25",
                 max_context_chunks: int = 6):
        self.index, self.reader = index, reader
        self.k, self.mode, self.max_context_chunks = k, mode, max_context_chunks

    def retrieve(self, question: str):
        hits = self.index.search(question, k=self.k, mode=self.mode)
        if not hits:
            raise LookupError("retrieval returned no chunks for this question")
        top = hits[0].chunk
        # Context = the chosen page's own chunks in reading order, so the text matches the image.
        page_chunks = [c for c in self.index.chunks if (c.doc_id, c.page) == (top.doc_id, top.page)]
        return hits, top, [c.text for c in page_chunks[: self.max_context_chunks]]

    def ask(self, question: str) -> Answer:
        hits, top, context = self.retrieve(question)
        text = self.reader.answer(question, top.image_path, context)
        sources = [{"doc_id": h.chunk.doc_id, "page": h.chunk.page, "chunk_id": h.chunk.chunk_id,
                    "score": round(h.score, 4)} for h in hits]
        return Answer(question, text, top.doc_id, top.page, top.image_path, sources)
