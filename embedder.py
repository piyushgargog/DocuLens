"""Phase 3: sentence-transformers embedding wrapper."""

import logging
import os

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


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts, L2-normalized (so dot product == cosine similarity)."""
    model = get_model()
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.astype("float32")


def prefetch_enabled() -> bool:
    """True when PREFETCH_MODEL=1: load the model at app startup instead of on
    the first upload (the app calls prefetch() from its lifespan)."""
    return os.environ.get("PREFETCH_MODEL", "").strip() == "1"


def prefetch() -> None:
    """Load the model now if PREFETCH_MODEL=1; otherwise do nothing."""
    if prefetch_enabled():
        get_model()
