"""Chunk index: dense (sentence-transformers + FAISS), lexical (BM25) and hybrid (RRF) retrieval."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from .ingest import Chunk

DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
Encoder = Callable[[list[str]], np.ndarray]
MODES = ("dense", "bm25", "hybrid")


def load_encoder(name: str = DEFAULT_EMBEDDER, device: str | None = None) -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name, device=device)
    return lambda texts: model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Okapi BM25 over pre-tokenised documents (k1=1.5, b=0.75)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.lens.mean()) if len(docs) else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tfs), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.lens / max(self.avgdl, 1e-9))
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            tf = np.array([c.get(t, 0) for c in self.tfs], dtype=np.float32)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out


@dataclass
class Hit:
    chunk: Chunk
    score: float


class ChunkIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, encoder: Encoder | None = None):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self.chunks = chunks
        self.encoder = encoder
        self.faiss = faiss.IndexFlatIP(embeddings.shape[1])  # inner product on unit vectors = cosine
        self.faiss.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        self.bm25 = BM25([tokenize(c.text) for c in chunks])

    @classmethod
    def build(cls, chunks: list[Chunk], encoder: Encoder) -> ChunkIndex:
        if not chunks:
            raise ValueError("no chunks to index (did OCR/text extraction return nothing?)")
        return cls(chunks, encoder([c.text for c in chunks]), encoder)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss, str(directory / "chunks.faiss"))
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict()) + "\n")

    @classmethod
    def load(cls, directory: Path, encoder: Encoder | None = None) -> ChunkIndex:
        index = faiss.read_index(str(directory / "chunks.faiss"))
        with (directory / "chunks.jsonl").open(encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        embeddings = index.reconstruct_n(0, index.ntotal)
        return cls(chunks, embeddings, encoder)

    def _dense_ranking(self, query: str) -> list[tuple[int, float]]:
        if self.encoder is None:
            raise RuntimeError("dense search needs an encoder; pass one to ChunkIndex.load()")
        q = np.ascontiguousarray(self.encoder([query]), dtype=np.float32)
        scores, ids = self.faiss.search(q, len(self.chunks))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=False) if i >= 0]

    def _bm25_ranking(self, query: str) -> list[tuple[int, float]]:
        s = self.bm25.scores(tokenize(query))
        order = np.argsort(-s, kind="stable")
        return [(int(i), float(s[i])) for i in order if s[i] > 0]  # no shared terms = not a match

    def search(self, query: str, k: int = 5, mode: str = "bm25", rrf_k: int = 60) -> list[Hit]:
        if mode == "dense":
            ranking = self._dense_ranking(query)
        elif mode == "bm25":
            ranking = self._bm25_ranking(query)
        elif mode == "hybrid":
            # Reciprocal Rank Fusion: score = sum over rankers of 1 / (rrf_k + rank)
            fused: dict[int, float] = {}
            for rk in (self._dense_ranking(query), self._bm25_ranking(query)):
                for rank, (i, _) in enumerate(rk, start=1):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank)
            ranking = sorted(fused.items(), key=lambda x: -x[1])
        else:
            raise ValueError(f"mode must be one of {MODES}")
        return [Hit(self.chunks[i], s) for i, s in ranking[:k]]

    def search_pages(self, query: str, k: int = 3, mode: str = "bm25") -> list[tuple[str, int, float]]:
        """Rank pages by their best chunk. Returns (doc_id, page, score) tuples."""
        best: dict[tuple[str, int], float] = {}
        for h in self.search(query, k=len(self.chunks), mode=mode):
            key = (h.chunk.doc_id, h.chunk.page)
            best.setdefault(key, h.score)  # hits are sorted, so the first one is the page's best
        return [(d, p, s) for (d, p), s in list(best.items())[:k]]
