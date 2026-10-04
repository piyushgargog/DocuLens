"""Downloads/loads the real sentence-transformers model on first run, same as
the app itself does — no network mocking, since the point is to verify the
actual embedding behavior, not a stand-in for it."""

import numpy as np

from embedder import embed


def test_embed_returns_normalized_384_dim_vectors():
    vectors = embed(["hello world", "another sentence"])
    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_related_sentences_score_higher_than_unrelated():
    vectors = embed(
        [
            "Jupiter is the largest planet in the Solar System.",
            "Saturn has a famous ring system.",
            "Bananas are a good source of potassium.",
        ]
    )
    sim_related = float(np.dot(vectors[0], vectors[1]))
    sim_unrelated = float(np.dot(vectors[0], vectors[2]))
    assert sim_related > sim_unrelated


def test_prefetch_loads_the_model_only_when_enabled(monkeypatch):
    import embedder

    loads = []
    monkeypatch.setattr(embedder, "get_model", lambda: loads.append(1))

    monkeypatch.delenv("PREFETCH_MODEL", raising=False)
    embedder.prefetch()
    assert loads == []  # off by default: the model still loads lazily

    monkeypatch.setenv("PREFETCH_MODEL", "1")
    embedder.prefetch()
    assert loads == [1]


def test_app_startup_prefetches_the_model_when_enabled(monkeypatch):
    from fastapi.testclient import TestClient

    import embedder
    import main

    calls = []
    monkeypatch.setattr(embedder, "get_model", lambda: calls.append(1))

    monkeypatch.delenv("PREFETCH_MODEL", raising=False)
    with TestClient(main.app):
        pass
    assert calls == []

    monkeypatch.setenv("PREFETCH_MODEL", "1")
    with TestClient(main.app):
        assert calls == [1]  # loaded during startup, before any request
