"""Vector search behind one small interface, so the rest of the app never touches FAISS.

Two implementations, both cosine similarity over L2-normalised vectors (inner
product == cosine, exactly as before):

* `FlatIndex`  -- exact brute force (`IndexFlatIP`). The default and the
  reference: results are exact, building is free, nothing to tune.
* `HNSWIndex`  -- approximate nearest neighbours (`IndexHNSWFlat`, inner
  product). Sub-linear search for large indexes at the price of a build step,
  extra memory for the graph and a small recall loss.

`make_index()` picks one: VECTOR_INDEX=flat|hnsw|auto (default auto), where
auto switches to HNSW at ANN_THRESHOLD vectors. The threshold comes from
`bench_vector_index.py` (see DECISIONS.md), not from a guess; at the app's
per-document limit (MAX_CHUNKS_PER_DOC) exact search is used.

An index only stores vectors, in insertion order; ids are positions. Removing
vectors means rebuilding over the ones kept (`rebuild`): HNSW has no deletion.
"""

import os
from abc import ABC, abstractmethod

import faiss
import numpy as np

DEFAULT_ANN_THRESHOLD = 20_000
HNSW_M = 32  # graph degree: more = better recall, more memory
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128


class IndexFormatError(ValueError):
    """Serialized index bytes are not a valid index of the expected kind."""


class VectorIndex(ABC):
    """Insertion-ordered vectors with top-k similarity search."""

    kind: str

    def __init__(self, dim: int) -> None:
        self.dim = dim

    @abstractmethod
    def add(self, vectors: np.ndarray) -> None:
        """Append float32 vectors of shape (n, dim)."""

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """The k most similar stored vectors to one query vector, best first:
        (scores, ids). Fewer than k are returned if fewer exist."""

    @abstractmethod
    def size(self) -> int:
        """Number of stored vectors."""

    @abstractmethod
    def save(self) -> bytes:
        """Serialise the whole index."""

    @classmethod
    @abstractmethod
    def load(cls, data: bytes) -> "VectorIndex":
        """Inverse of save(). Raises IndexFormatError for anything else."""

    def rebuild(self, vectors: np.ndarray) -> "VectorIndex":
        """A fresh index of the same kind over exactly these vectors (the way to
        'remove' entries: pass the ones to keep)."""
        index = type(self)(self.dim)
        if len(vectors):
            index.add(vectors)
        return index

    def _check(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of shape (n, {self.dim})")
        return vectors


def _deserialize(data: bytes) -> "faiss.Index":
    try:
        return faiss.deserialize_index(np.frombuffer(data, dtype="uint8"))
    except Exception as e:  # faiss raises RuntimeError on bad bytes
        raise IndexFormatError("not a valid serialized index") from e


class FlatIndex(VectorIndex):
    kind = "flat"

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self._index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray) -> None:
        self._index.add(self._check(vectors))

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, self.size())
        if k <= 0:
            return np.empty(0, dtype="float32"), np.empty(0, dtype="int64")
        scores, ids = self._index.search(self._check(query.reshape(1, -1)), k)
        keep = ids[0] >= 0
        return scores[0][keep], ids[0][keep]

    def size(self) -> int:
        return int(self._index.ntotal)

    def save(self) -> bytes:
        return faiss.serialize_index(self._index).tobytes()

    @classmethod
    def load(cls, data: bytes) -> "FlatIndex":
        raw = _deserialize(data)
        if not isinstance(raw, faiss.IndexFlatIP):
            raise IndexFormatError("serialized index is not a flat inner-product index")
        index = cls(raw.d)
        index._index = raw
        return index


class HNSWIndex(VectorIndex):
    kind = "hnsw"

    def __init__(self, dim: int, m: int = HNSW_M, ef_search: int = HNSW_EF_SEARCH) -> None:
        super().__init__(dim)
        self._index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        self._index.hnsw.efSearch = ef_search

    def add(self, vectors: np.ndarray) -> None:
        self._index.add(self._check(vectors))

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, self.size())
        if k <= 0:
            return np.empty(0, dtype="float32"), np.empty(0, dtype="int64")
        self._index.hnsw.efSearch = max(HNSW_EF_SEARCH, k)  # ef below k loses results
        scores, ids = self._index.search(self._check(query.reshape(1, -1)), k)
        keep = ids[0] >= 0
        return scores[0][keep], ids[0][keep]

    def size(self) -> int:
        return int(self._index.ntotal)

    def save(self) -> bytes:
        return faiss.serialize_index(self._index).tobytes()

    @classmethod
    def load(cls, data: bytes) -> "HNSWIndex":
        raw = _deserialize(data)
        if not isinstance(raw, faiss.IndexHNSWFlat):
            raise IndexFormatError("serialized index is not an HNSW index")
        index = cls(raw.d)
        index._index = raw
        index._index.hnsw.efSearch = HNSW_EF_SEARCH
        return index


KINDS: dict[str, type[VectorIndex]] = {"flat": FlatIndex, "hnsw": HNSWIndex}


def ann_threshold() -> int:
    try:
        return max(1, int(os.environ.get("ANN_THRESHOLD", DEFAULT_ANN_THRESHOLD)))
    except ValueError:
        return DEFAULT_ANN_THRESHOLD


def choose_kind(count: int) -> str:
    """'flat' or 'hnsw' for an index that will hold `count` vectors."""
    setting = os.environ.get("VECTOR_INDEX", "auto").strip().lower()
    if setting in KINDS:
        return setting
    return "hnsw" if count >= ann_threshold() else "flat"


def make_index(vectors: np.ndarray, kind: str | None = None) -> VectorIndex:
    """Build an index over `vectors`, of the given kind or the one choose_kind() picks."""
    kind = kind or choose_kind(len(vectors))
    index = KINDS[kind](int(vectors.shape[1]))
    index.add(vectors)
    return index


def load_index(kind: str, data: bytes) -> VectorIndex:
    """Deserialise an index saved by `save()`; `kind` is the stored kind."""
    if kind not in KINDS:
        raise IndexFormatError(f"unknown index kind {kind!r}")
    return KINDS[kind].load(data)
