"""Phase 3: sentence-transformers embedding wrapper."""

import hashlib
import logging
import os
import threading
from collections import OrderedDict

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None
_log = logging.getLogger("doculens.embedder")


def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    global _model
    if _model is None:
        _log.info("Loading embedding model %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        _log.info("Embedding model ready")
    return _model


# Vectors already computed in this process, by text. A repeated question, a
# document uploaded twice, or text repeated across documents costs a dict lookup
# instead of ~20 ms of CPU per chunk (measured: bench_latency.py). In-process
# only, so nothing is shared between users or instances; ~1.5 KB per entry.
CACHE_SIZE = 4096
_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
_cache_lock = threading.Lock()


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts, L2-normalized (so dot product == cosine similarity).
    Texts seen before in this process are served from a bounded cache; the rest
    are encoded together in one batch."""
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    keys = [_key(t) for t in texts]
    found: dict[str, np.ndarray] = {}
    with _cache_lock:
        for key in keys:
            if key in _cache:
                _cache.move_to_end(key)
                found[key] = _cache[key]
    missing = list(dict.fromkeys(k for k in keys if k not in found))
    if missing:
        first_text = {k: t for k, t in zip(keys, texts)}
        model = get_model()
        encoded = model.encode([first_text[k] for k in missing], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        with _cache_lock:
            for key, vector in zip(missing, encoded):
                _cache[key] = vector
                found[key] = vector
            while len(_cache) > CACHE_SIZE:
                _cache.popitem(last=False)
    return np.stack([found[k] for k in keys])


def prefetch_enabled() -> bool:
    """True when PREFETCH_MODEL=1: load the model at app startup instead of on
    the first upload (the app calls prefetch() from its lifespan)."""
    return os.environ.get("PREFETCH_MODEL", "").strip() == "1"


def prefetch() -> None:
    """Load the model now if PREFETCH_MODEL=1; otherwise do nothing."""
    if prefetch_enabled():
        get_model()
