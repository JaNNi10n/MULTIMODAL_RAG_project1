"""Minimal dense-retrieval example: MiniLM embeddings + FAISS exact search.

The original notebook cell used placeholder chunks and a `!pip` line; this version runs as a plain
script on a few real chunks. Install deps with `pip install -r requirements.txt`.
Usage: python examples/faiss_retrieval.py "your question"
"""

import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chunks = [
    "Tax invoice from Lotus Print Works for brochure printing and laser toner cartridges.",
    "Electricity bill from Valley Energy: 888 kWh consumed in August, amount payable INR 5,700.",
    "Bank account statement for June with grocery, fuel and broadband payments.",
    "Qwen2.5-VL is a vision-language model that can read text in document images.",
]

embeddings = embedder.encode(chunks, normalize_embeddings=True)
index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product on unit vectors = cosine similarity
index.add(np.asarray(embeddings, dtype=np.float32))

query = sys.argv[1] if len(sys.argv) > 1 else "How much electricity was used?"
query_vec = embedder.encode([query], normalize_embeddings=True)
scores, indices = index.search(np.asarray(query_vec, dtype=np.float32), k=2)
for score, i in zip(scores[0], indices[0], strict=False):
    print(f"{score:.3f}  {chunks[i]}")
