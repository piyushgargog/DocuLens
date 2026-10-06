"""Phase 4: chunk embeddings plus the structures that search them.

Holds the chunks, their embeddings (kept for exact scoring and for storing the
document), the BM25 term statistics, and a `VectorIndex` (exact or
approximate -- see vector_index.py) for similarity search."""

import numpy as np

from retriever import TermIndex
from vector_index import VectorIndex, make_index


class VectorStore:
    """Chunks + embeddings + a similarity index over them (cosine, via inner
    product on normalised vectors)."""

    def __init__(
        self,
        chunks: list[dict],
        embeddings: np.ndarray,
        index: VectorIndex | None = None,
        index_kind: str | None = None,
    ):
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings must be the same length")
        self.chunks = chunks
        self.embeddings = embeddings  # kept for exact scoring and for storing the document
        self.terms = TermIndex([c["text"] for c in chunks])  # BM25 statistics
        if index is not None and index.size() != len(chunks):
            raise ValueError("index and chunks must be the same length")
        self.index = index if index is not None else make_index(embeddings, index_kind)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        """Return the top_k most similar chunks as {text, page, score}, best first.

        Any extra keys on a chunk (e.g. "doc" for multi-document sessions) are
        carried through to the result."""
        top_k = min(top_k, len(self.chunks))
        if top_k == 0:
            return []
        scores, ids = self.index.search(query_embedding, top_k)
        return [{**self.chunks[int(i)], "score": float(s)} for s, i in zip(scores, ids, strict=True)]

    def without(self, drop: set[int]) -> "VectorStore":
        """A new store with the chunks at these positions removed. Indexes
        cannot delete in place, so this rebuilds over the chunks that remain."""
        keep = [i for i in range(len(self.chunks)) if i not in drop]
        return VectorStore(
            [self.chunks[i] for i in keep],
            np.ascontiguousarray(self.embeddings[keep]),
            index=self.index.rebuild(self.embeddings[keep]),
        )
