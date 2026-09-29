"""Phase 4: FAISS-backed similarity search over chunk embeddings."""

import faiss
import numpy as np


class VectorStore:
    """Wraps a FAISS IndexFlatIP over normalized embeddings (cosine similarity)."""

    def __init__(self, chunks: list[dict], embeddings: np.ndarray):
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings must be the same length")
        self.chunks = chunks
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        """Return the top_k most similar chunks as {text, page, score}, best first.

        Any extra keys on a chunk (e.g. "doc" for multi-document sessions) are
        carried through to the result."""
        top_k = min(top_k, len(self.chunks))
        if top_k == 0:
            return []
        query = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({**chunk, "score": float(score)})
        return results
