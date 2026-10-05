"""The vector-index abstraction: exact and approximate indexes behind one
interface, persistence, rebuilds, and the app-level guarantees (same answers
from either kind on real documents, derived data never fatal)."""

import asyncio

import numpy as np
import pytest

import docstore
import pipeline
import store
import vector_index
from embedder import embed
from vector_index import FlatIndex, HNSWIndex, IndexFormatError, make_index
from vector_store import VectorStore


def unit(n, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


KINDS = [FlatIndex, HNSWIndex]


@pytest.fixture(params=KINDS, ids=["flat", "hnsw"])
def kind(request):
    return request.param


# ---------- interface ----------


def test_results_are_best_first_and_match_brute_force(kind):
    vectors = unit(300)
    index = make_index(vectors, kind.kind)
    query = unit(1, seed=9)[0]
    scores, ids = index.search(query, 10)
    truth = np.argsort(-(vectors @ query))[:10]
    assert list(scores) == sorted(scores, reverse=True)
    if kind is FlatIndex:
        assert list(ids) == list(truth)
        assert np.allclose(scores, (vectors @ query)[truth], atol=1e-5)
    else:
        assert len(set(ids) & set(truth)) >= 8  # approximate, but close
        assert np.allclose(scores, vectors[ids] @ query, atol=1e-4)  # scores are true cosines


def test_k_is_capped_and_empty_is_safe(kind):
    index = kind(8)
    scores, ids = index.search(np.ones(8, dtype="float32"), 5)
    assert len(scores) == len(ids) == 0 and index.size() == 0
    index.add(unit(3, dim=8))
    scores, ids = index.search(np.ones(8, dtype="float32") / np.sqrt(8), 50)
    assert len(ids) == 3 and index.size() == 3


def test_wrong_dimension_is_rejected(kind):
    with pytest.raises(ValueError):
        kind(8).add(unit(2, dim=4))


def test_add_appends_in_order(kind):
    index = kind(16)
    first, second = unit(5, 16, seed=1), unit(5, 16, seed=2)
    index.add(first)
    index.add(second)
    _, ids = index.search(second[3], 1)
    assert int(ids[0]) == 5 + 3


def test_save_load_roundtrip_gives_identical_results(kind):
    vectors = unit(200)
    index = make_index(vectors, kind.kind)
    restored = vector_index.load_index(kind.kind, index.save())
    assert restored.kind == kind.kind and restored.size() == 200 and restored.dim == 32
    for seed in range(5):
        q = unit(1, seed=100 + seed)[0]
        a, b = index.search(q, 10), restored.search(q, 10)
        assert list(a[1]) == list(b[1]) and np.allclose(a[0], b[0])


def test_loading_the_wrong_kind_or_garbage_fails_cleanly():
    flat, hnsw = make_index(unit(50), "flat"), make_index(unit(50), "hnsw")
    with pytest.raises(IndexFormatError):
        HNSWIndex.load(flat.save())
    with pytest.raises(IndexFormatError):
        FlatIndex.load(hnsw.save())
    for bad in (b"", b"garbage" * 10):
        with pytest.raises(IndexFormatError):
            vector_index.load_index("flat", bad)
    with pytest.raises(IndexFormatError):
        vector_index.load_index("mystery", b"x")


def test_rebuild_is_how_vectors_are_removed(kind):
    vectors = unit(40)
    index = make_index(vectors, kind.kind)
    kept = np.delete(vectors, [3, 7], axis=0)
    smaller = index.rebuild(kept)
    assert smaller.kind == kind.kind and smaller.size() == 38 and index.size() == 40
    assert int(smaller.search(kept[10], 1)[1][0]) == 10
    assert smaller.rebuild(kept[:0]).size() == 0


# ---------- selection ----------


def test_auto_picks_exact_below_the_threshold_and_hnsw_at_it(monkeypatch):
    monkeypatch.delenv("VECTOR_INDEX", raising=False)
    monkeypatch.setenv("ANN_THRESHOLD", "100")
    assert vector_index.choose_kind(99) == "flat"
    assert vector_index.choose_kind(100) == "hnsw"


def test_configuration_can_force_either_kind_and_survives_garbage(monkeypatch):
    monkeypatch.setenv("VECTOR_INDEX", "hnsw")
    assert vector_index.choose_kind(1) == "hnsw"
    monkeypatch.setenv("VECTOR_INDEX", "flat")
    assert vector_index.choose_kind(10**9) == "flat"
    monkeypatch.setenv("VECTOR_INDEX", "nonsense")
    monkeypatch.setenv("ANN_THRESHOLD", "not-a-number")
    assert vector_index.choose_kind(10) == "flat"
    assert vector_index.ann_threshold() == vector_index.DEFAULT_ANN_THRESHOLD


def test_the_default_threshold_leaves_every_per_document_size_exact(monkeypatch):
    import main

    monkeypatch.delenv("VECTOR_INDEX", raising=False)
    monkeypatch.delenv("ANN_THRESHOLD", raising=False)
    assert vector_index.choose_kind(main.MAX_CHUNKS_PER_DOC) == "flat"


# ---------- VectorStore ----------


def test_store_search_works_with_either_index(kind):
    chunks = [{"text": t, "page": i + 1} for i, t in enumerate(
        ["Jupiter is the largest planet.", "Bananas have potassium.", "Paris is in France.", "Cats purr softly."]
    )]
    vectors = embed([c["text"] for c in chunks])
    s = VectorStore(chunks, vectors, index_kind=kind.kind)
    assert s.index.kind == kind.kind
    assert s.search(embed(["biggest planet"])[0], 1)[0]["page"] == 1


def test_store_without_rebuilds_over_the_remaining_chunks(kind):
    chunks = [{"text": f"chunk {i}", "page": i} for i in range(6)]
    s = VectorStore(chunks, unit(6, 16), index_kind=kind.kind)
    smaller = s.without({1, 2})
    assert [c["page"] for c in smaller.chunks] == [0, 3, 4, 5] and smaller.index.size() == 4
    assert smaller.index.kind == kind.kind and len(s.chunks) == 6  # original untouched
    assert s.without(set(range(6))).index.size() == 0


def test_store_rejects_a_mismatched_index():
    with pytest.raises(ValueError):
        VectorStore([{"text": "a", "page": 1}], unit(1, 8), index=make_index(unit(3, 8), "flat"))


# ---------- the app's retrieval, either index ----------

DOC_TEXT = (
    "Mercury is the smallest planet in the Solar System.\n\n"
    "Venus is the hottest planet, with a surface near 465 degrees Celsius.\n\n"
    "Earth is the only planet known to support life.\n\n"
    "Mars has the tallest volcano, Olympus Mons.\n\n"
    "Jupiter is the largest planet and has a Great Red Spot.\n\n"
    "Saturn is known for its spectacular ring system.\n\n"
    "Neptune has the strongest winds of any planet."
)


def _state(kind_name, monkeypatch):
    monkeypatch.setenv("VECTOR_INDEX", kind_name)
    state = pipeline.ingest(DOC_TEXT.encode(), name="planets.txt", chunk_size=120, chunk_overlap=20)
    assert state.store.index.kind == kind_name
    return state


def test_retrieval_returns_the_same_passages_from_either_index(monkeypatch):
    flat, hnsw = _state("flat", monkeypatch), _state("hnsw", monkeypatch)
    for question in ("Which planet is the hottest?", "What has the biggest volcano?", "ring system"):
        a = pipeline.retrieve([question], [flat])
        b = pipeline.retrieve([question], [hnsw])
        assert [p["text"] for p in a] == [p["text"] for p in b]
        assert np.allclose([p["score"] for p in a], [p["score"] for p in b], atol=1e-4)


def test_mixed_exact_and_approximate_documents_search_together(monkeypatch):
    a, b = _state("flat", monkeypatch), _state("hnsw", monkeypatch)
    results = pipeline.retrieve(["Which planet is the hottest?"], [a, b], top_k=4)
    assert any("Venus" in r["text"] for r in results) and all(-1.0 <= r["score"] <= 1.0001 for r in results)


# ---------- persisted ----------


def _roundtrip(state, **load_kwargs):
    backend = store.MemoryStore()
    repo = docstore.DocumentRepository(backend, docstore.StoreArtifacts(backend))

    async def go():
        packed = docstore.pack("d1", state.name, state.num_pages, state.chunk_size, state.chunk_overlap,
                               state.store.chunks, state.store.embeddings, index=state.store.index)
        await repo.save("o", packed)
        meta = (await repo.list_docs("o"))["d1"]
        return packed, meta, backend, repo

    return asyncio.run(go())


def test_an_approximate_index_is_saved_and_reloaded_not_rebuilt(monkeypatch):
    state = _state("hnsw", monkeypatch)
    packed, meta, backend, repo = _roundtrip(state)
    assert packed.index_blob and meta.index_kind == "hnsw"
    loaded = asyncio.run(repo.load("o", meta))
    assert loaded.index is not None and loaded.index.kind == "hnsw" and loaded.index.size() == meta.num_chunks


def test_exact_indexes_are_not_stored_because_rebuilding_them_is_free(monkeypatch):
    state = _state("flat", monkeypatch)
    packed, meta, backend, repo = _roundtrip(state)
    assert packed.index_blob is None and meta.index_kind == "flat"
    assert asyncio.run(repo.load("o", meta)).index is None


@pytest.mark.parametrize("damage", ["missing", "tampered", "wrong-size"])
def test_a_damaged_saved_index_is_rebuilt_never_fatal(monkeypatch, damage):
    state = _state("hnsw", monkeypatch)
    packed, meta, backend, repo = _roundtrip(state)
    key = repo._blob_key("o", "d1", "index")
    if damage == "missing":
        asyncio.run(repo._artifacts.delete(key))
    elif damage == "tampered":
        asyncio.run(repo._artifacts.put(key, packed.index_blob[:-5] + b"xxxxx", 60))
    else:
        small = make_index(unit(2, state.store.embeddings.shape[1]), "hnsw").save()
        asyncio.run(repo._artifacts.put(key, small, 60))
        meta = docstore.DocMeta(**{**meta.to_dict(), "index_digest": __import__("hashlib").sha256(small).hexdigest()})
    loaded = asyncio.run(repo.load("o", meta))
    assert loaded.index is None and len(loaded.chunks) == meta.num_chunks  # the document itself is intact
    restored = pipeline.restore(loaded.chunks, loaded.embeddings, "planets.txt", 1, 120, 20, loaded.index)
    assert restored.store.index.size() == meta.num_chunks


def test_deleting_a_document_deletes_its_index_blob_too(monkeypatch):
    state = _state("hnsw", monkeypatch)
    packed, meta, backend, repo = _roundtrip(state)
    assert any(k.startswith("a:o:d1:index") for k in backend._blobs)
    asyncio.run(repo.delete("o", "d1"))
    assert not any(k.startswith("a:o:d1") for k in backend._blobs)
