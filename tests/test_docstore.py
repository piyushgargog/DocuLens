"""Persistent document storage: packing, integrity, ownership, quota, expiry,
and recovery from missing or corrupt artifacts. Runs on every backend."""

import asyncio
import json
import time
import zlib

import fakeredis
import numpy as np
import pytest

import docstore
import store

CHUNKS = [
    {"text": "Mercury is the smallest planet.", "page": 1, "doc": "space.txt"},
    {"text": "Venus is the hottest planet: 465 °C.", "page": 2, "doc": "space.txt"},
    {"text": "Ünïcode — ok ✓", "page": 2, "doc": "space.txt"},
]


def vectors(n=3, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def run(coro):
    return asyncio.run(coro)


def packed(doc_id="d1", chunks=CHUNKS, emb=None, **kw):
    emb = vectors(len(chunks)) if emb is None else emb
    return docstore.pack(doc_id, "space.txt", 2, 800, 150, chunks, emb, **kw)


@pytest.fixture(params=["memory-store", "redis-store", "disk"])
def repo(request, tmp_path):
    if request.param == "memory-store":
        backend = store.MemoryStore()
        return docstore.DocumentRepository(backend, docstore.StoreArtifacts(backend))
    if request.param == "redis-store":
        backend = store.RedisStore("redis://unused", client=fakeredis.FakeAsyncRedis())
        return docstore.DocumentRepository(backend, docstore.StoreArtifacts(backend))
    backend = store.MemoryStore()
    return docstore.DocumentRepository(backend, docstore.DiskArtifacts(tmp_path / "artifacts"))


# ---------- pack / unpack ----------


def test_pack_unpack_roundtrip_keeps_text_pages_and_ranking():
    emb = vectors()
    p = packed(emb=emb)
    chunks, restored = docstore.unpack(p.meta, p.chunks_blob, p.emb_blob)
    assert chunks == CHUNKS
    assert restored.dtype == np.float32 and restored.shape == emb.shape
    assert np.allclose(np.linalg.norm(restored, axis=1), 1.0, atol=1e-5)
    # float16 storage must not change which chunk is nearest to any chunk
    assert (np.argmax(restored @ emb.T, axis=1) == np.argmax(emb @ emb.T, axis=1)).all()


def test_meta_roundtrips_through_json_and_rejects_garbage():
    meta = packed().meta
    assert docstore.DocMeta.from_dict(json.loads(json.dumps(meta.to_dict()))) == meta
    for bad in (None, [], {}, {**meta.to_dict(), "num_chunks": -1}, {**meta.to_dict(), "version": 99}, {**meta.to_dict(), "name": 5}):
        with pytest.raises(ValueError):
            docstore.DocMeta.from_dict(bad)


@pytest.mark.parametrize("damage", ["chunks", "emb", "swap"])
def test_corrupt_artifacts_are_detected(damage):
    p = packed()
    chunks_blob, emb_blob = p.chunks_blob, p.emb_blob
    if damage == "chunks":
        chunks_blob = chunks_blob[:-3] + b"xyz"
    elif damage == "emb":
        emb_blob = bytes([emb_blob[0] ^ 1]) + emb_blob[1:]
    else:
        chunks_blob, emb_blob = emb_blob, chunks_blob
    with pytest.raises(docstore.DocumentCorruptError):
        docstore.unpack(p.meta, chunks_blob, emb_blob)


def test_decompression_bomb_is_refused(monkeypatch):
    bomb = zlib.compress(b"[" + b" " * 5_000_000 + b"]")
    p = packed()
    meta = docstore.DocMeta(**{**p.meta.to_dict(), "digest": __import__("hashlib").sha256(bomb + p.emb_blob).hexdigest()})
    monkeypatch.setattr(docstore, "MAX_UNPACKED_BYTES", 1_000_000)
    with pytest.raises(docstore.DocumentCorruptError):
        docstore.unpack(meta, bomb, p.emb_blob)


def test_valid_digest_but_malformed_chunks_is_still_corrupt():
    bad_chunks = [{"text": 1, "page": "x"}] * 3
    blob = zlib.compress(json.dumps(bad_chunks).encode())
    p = packed()
    emb = p.emb_blob
    meta = docstore.DocMeta(**{**p.meta.to_dict(), "digest": __import__("hashlib").sha256(blob + emb).hexdigest()})
    with pytest.raises(docstore.DocumentCorruptError):
        docstore.unpack(meta, blob, emb)


# ---------- repository ----------


def test_save_list_load_roundtrip(repo):
    async def go():
        owner = docstore.owner_key("uid-1")
        p = packed()
        await repo.save(owner, p)
        listed = await repo.list_docs(owner)
        assert list(listed) == ["d1"] and listed["d1"].num_chunks == 3
        chunks, emb, _ = await repo.load(owner, listed["d1"])
        assert chunks == CHUNKS and emb.shape == (3, 8)

    run(go())


def test_large_blobs_are_split_and_reassembled(repo, monkeypatch):
    monkeypatch.setattr(docstore, "BLOB_PART_BYTES", 1000)

    async def go():
        owner = docstore.owner_key("uid-1")
        big = [{"text": f"passage {i} " + "lorem ipsum " * 60, "page": 1} for i in range(40)]
        await repo.save(owner, packed(chunks=big))
        meta = (await repo.list_docs(owner))["d1"]
        chunks, _, _ = await repo.load(owner, meta)
        assert chunks == big

    run(go())


def test_users_cannot_see_or_load_each_others_documents(repo):
    async def go():
        a, b = docstore.owner_key("alice"), docstore.owner_key("bob")
        assert a != b
        await repo.save(a, packed("a-doc"))
        assert await repo.list_docs(b) == {}
        # even holding Alice's metadata, Bob's namespace has no artifacts for it
        alice_meta = (await repo.list_docs(a))["a-doc"]
        with pytest.raises(docstore.DocumentMissingError):
            await repo.load(b, alice_meta)
        await repo.delete(b, "a-doc")  # Bob deleting that id touches nothing of Alice's
        assert list(await repo.list_docs(a)) == ["a-doc"]
        await repo.delete_all(b)
        assert list(await repo.list_docs(a)) == ["a-doc"]

    run(go())


def test_owner_key_is_not_the_raw_uid():
    assert "alice" not in docstore.owner_key("alice") and len(docstore.owner_key("alice")) == 40


def test_delete_removes_metadata_and_artifacts(repo):
    async def go():
        owner = docstore.owner_key("uid-1")
        await repo.save(owner, packed("d1"))
        meta = (await repo.list_docs(owner))["d1"]
        await repo.delete(owner, "d1")
        assert await repo.list_docs(owner) == {}
        with pytest.raises(docstore.DocumentMissingError):
            await repo.load(owner, meta)

    run(go())


def test_delete_all_removes_documents_but_keeps_saved_chats(repo):
    async def go():
        owner = docstore.owner_key("uid-1")
        await repo.save(owner, packed("d1"))
        await repo.save(owner, packed("d2"))
        chat = await repo.create_chat(owner, "About planets")
        await repo.save_chat_turns(owner, chat, [{"question": "q", "answer": "a"}])
        await repo.delete_all(owner)
        assert await repo.list_docs(owner) == {}
        assert [c["title"] for c in await repo.list_chats(owner)] == ["About planets"]  # chats are the user's to delete
        assert await repo.get_chat_history(owner, chat["id"]) == [{"question": "q", "answer": "a"}]

    run(go())


def test_storage_quota_blocks_before_writing(repo, monkeypatch):
    async def go():
        owner = docstore.owner_key("uid-1")
        p = packed("d1")
        monkeypatch.setattr(docstore, "max_user_bytes", lambda: p.meta.size_bytes * 2 - 1)
        await repo.save(owner, p)
        with pytest.raises(docstore.StorageQuotaError):
            await repo.save(owner, packed("d2"))
        assert list(await repo.list_docs(owner)) == ["d1"]

    run(go())


def test_expired_documents_are_dropped_and_cleaned(repo):
    async def go():
        owner = docstore.owner_key("uid-1")
        await repo.save(owner, packed("old", ttl=-5))
        await repo.save(owner, packed("fresh"))
        assert list(await repo.list_docs(owner)) == ["fresh"]
        assert list(await repo.list_docs(owner)) == ["fresh"]  # and stays gone

    run(go())


def test_documents_list_in_upload_order(repo):
    async def go():
        owner = docstore.owner_key("uid-1")
        for name in ("c", "a", "b"):
            await repo.save(owner, packed(name))
            await asyncio.sleep(0.01)
        assert list(await repo.list_docs(owner)) == ["c", "a", "b"]

    run(go())


def test_stale_metadata_whose_artifacts_vanished_is_reported_missing(repo):
    async def go():
        owner = docstore.owner_key("uid-1")
        await repo.save(owner, packed("d1"))
        meta = (await repo.list_docs(owner))["d1"]
        await repo._artifacts.delete(repo._blob_key(owner, "d1", "emb"))  # artifact evicted, metadata remains
        with pytest.raises(docstore.DocumentMissingError):
            await repo.load(owner, meta)

    run(go())


def test_tampered_artifact_is_reported_corrupt(repo):
    async def go():
        owner = docstore.owner_key("uid-1")
        await repo.save(owner, packed("d1"))
        meta = (await repo.list_docs(owner))["d1"]
        key = repo._blob_key(owner, "d1", "chunks")
        original = await repo._artifacts.get(key)
        await repo._artifacts.put(key, original[:-1] + bytes([original[-1] ^ 0xFF]), 60)
        with pytest.raises(docstore.DocumentCorruptError):
            await repo.load(owner, meta)

    run(go())


def test_malformed_metadata_entry_is_dropped_not_fatal():
    async def go():
        backend = store.MemoryStore()
        repo = docstore.DocumentRepository(backend, docstore.StoreArtifacts(backend))
        owner = docstore.owner_key("uid-1")
        await repo.save(owner, packed("good"))
        await backend.hset_json(f"u:{owner}:docs", "bad", {"id": "bad", "name": 3}, 60)
        await backend.hset_json(f"u:{owner}:docs", "liar", {**packed("other").meta.to_dict()}, 60)  # id != field
        assert list(await repo.list_docs(owner)) == ["good"]
        assert set((await backend.hgetall_json(f"u:{owner}:docs")).keys()) == {"good"}

    run(go())


def test_two_instances_share_state_through_the_store():
    """Instance A saves, instance B (sharing only the Redis server) sees and loads it."""
    async def go():
        server = fakeredis.FakeServer()
        a = docstore.DocumentRepository(store.RedisStore("redis://x", client=fakeredis.FakeAsyncRedis(server=server)))
        b = docstore.DocumentRepository(store.RedisStore("redis://x", client=fakeredis.FakeAsyncRedis(server=server)))
        owner = docstore.owner_key("uid-1")
        await a.save(owner, packed("from-a"))
        listed = await b.list_docs(owner)
        assert list(listed) == ["from-a"]
        chunks, _, _ = await b.load(owner, listed["from-a"])
        assert chunks == CHUNKS
        await b.delete(owner, "from-a")
        assert await a.list_docs(owner) == {}

    run(go())


def test_restart_keeps_documents_when_the_backend_survives():
    """A new repository object over the same Redis (a restarted API) still has the user's documents."""
    async def go():
        server = fakeredis.FakeServer()
        owner = docstore.owner_key("uid-1")
        first = docstore.DocumentRepository(store.RedisStore("redis://x", client=fakeredis.FakeAsyncRedis(server=server)))
        await first.save(owner, packed("keep"))
        await first.set_history(owner, [{"question": "q", "answer": "a"}])
        second = docstore.DocumentRepository(store.RedisStore("redis://x", client=fakeredis.FakeAsyncRedis(server=server)))
        assert list(await second.list_docs(owner)) == ["keep"]
        assert await second.get_history(owner) == [{"question": "q", "answer": "a"}]

    run(go())


def test_history_is_bounded_and_filtered(repo):
    async def go():
        owner = docstore.owner_key("uid-1")
        turns = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(25)]
        await repo.set_history(owner, turns)
        stored = await repo.get_history(owner)
        assert len(stored) == docstore.MAX_HISTORY_TURNS_STORED and stored[-1]["question"] == "q24"
        await repo._store.set_json(f"u:{owner}:hist", [{"question": "ok", "answer": "ok"}, {"question": 1}, "junk"], 60)
        assert await repo.get_history(owner) == [{"question": "ok", "answer": "ok"}]

    run(go())


def test_disk_artifacts_expire(tmp_path):
    async def go():
        disk = docstore.DiskArtifacts(tmp_path)
        await disk.put("k", b"data", 1)
        assert await disk.get("k") == b"data"
        await asyncio.sleep(1.2)
        assert await disk.get("k") is None

    run(go())


def test_make_artifacts_picks_disk_when_configured(monkeypatch, tmp_path):
    backend = store.MemoryStore()
    assert isinstance(docstore.make_artifacts(backend), docstore.StoreArtifacts)
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "blobs"))
    assert isinstance(docstore.make_artifacts(backend), docstore.DiskArtifacts)


def test_retention_and_quota_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("DOC_RETENTION_DAYS", "2")
    monkeypatch.setenv("USER_STORAGE_MB", "3")
    assert docstore.retention_seconds() == 2 * 24 * 3600
    assert docstore.max_user_bytes() == 3 * 1024 * 1024
    monkeypatch.setenv("DOC_RETENTION_DAYS", "garbage")
    assert docstore.retention_seconds() == 30 * 24 * 3600
    assert time.time() > 0
