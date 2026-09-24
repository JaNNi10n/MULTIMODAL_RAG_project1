!pip install sentence-transformers faiss-cpu -q

from sentence_transformers import SentenceTransformer
import faiss, numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")

chunks = [
    "Paste a real paragraph.",
    "Paste another real chunk here.",
]

embeddings = embedder.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

query = "your question about the text-only content"
query_vec = embedder.encode([query])
_, indices = index.search(np.array(query_vec), k=2)
print([chunks[i] for i in indices[0]])


