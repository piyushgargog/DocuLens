"""A signed-in user's documents are stored durably, owned by their verified
Firebase uid, and survive restarts, other instances, and damaged storage.

"Restart" here means dropping the process's in-memory workspaces
(`main._user_sessions`) while the store survives -- exactly what a new
FastAPI process over the same Upstash database would see. A second "instance"
is the same thing: no shared memory, only the store."""

import json

import pytest
from fastapi.testclient import TestClient

import docstore
import main
from firebase_helpers import TXT, make_token

pytestmark = pytest.mark.usefixtures("signin_on")

PLANETS = {"file": ("planets.txt", b"Mercury is the smallest planet. Venus is the hottest planet.", "text/plain")}
MOONS = {"file": ("moons.txt", b"Europa is a moon of Jupiter covered in ice.", "text/plain")}


def new_client(uid="uid-123", **claims):
    client = TestClient(main.app)
    client.__enter__()
    response = client.post("/api/login", json={"id_token": make_token(sub=uid, **claims)})
    assert response.status_code == 200, response.text
    return client


@pytest.fixture
def alice():
    client = new_client("alice", email="alice@example.com", name="Alice")
    yield client
    client.__exit__(None, None, None)


@pytest.fixture
def bob():
    client = new_client("bob", email="bob@example.com", name="Bob")
    yield client
    client.__exit__(None, None, None)


@pytest.fixture
def seen_states(monkeypatch):
    """Capture what the pipeline is given instead of calling an LLM."""
    seen = []

    def fake_answer(question, states, history=None):
        seen.append({"question": question, "docs": [s.name for s in states], "chunks": [c["text"] for s in states for c in s.store.chunks], "history": history})
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(main.pipeline, "answer", fake_answer)
    return seen


def restart():
    main._user_sessions.clear()


def stored_blob_keys():
    return [k for k in main._store._blobs if k.startswith("a:")]


def stored_meta(uid):
    owner = docstore.owner_key(uid)
    return {f: json.loads(v) for f, v in main._store._hashes.get(f"u:{owner}:docs", ({}, 0))[0].items()}


# ---------- persistence ----------


def test_signed_in_document_is_stored_durably(alice):
    response = alice.post("/api/ingest", files=PLANETS)
    assert response.status_code == 200
    assert len(stored_meta("alice")) == 1 and len(stored_blob_keys()) >= 4  # 2 manifests + 2 parts


def test_documents_survive_a_restart(alice, seen_states):
    doc = alice.post("/api/ingest", files=PLANETS).json()
    restart()
    listed = alice.get("/api/session").json()["documents"]
    assert [(d["id"], d["filename"]) for d in listed] == [(doc["id"], "planets.txt")]
    assert alice.post("/api/ask", json={"question": "hottest planet?"}).status_code == 200
    assert seen_states[0]["docs"] == ["planets.txt"]
    assert any("Venus" in text for text in seen_states[0]["chunks"])


def test_restored_document_ranks_like_the_original(alice, monkeypatch):
    """float16 storage must not change what retrieval returns."""
    from pipeline import retrieve

    alice.post("/api/ingest", files=PLANETS)
    original = next(iter(main._user_sessions.values())).docs
    before = [c["text"] for c in retrieve(["which planet is hottest"], list(original.values()))]
    restart()
    captured = {}

    def grab(question, states, history=None):
        captured["states"] = states
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(main.pipeline, "answer", grab)
    alice.post("/api/ask", json={"question": "x?"})
    after = [c["text"] for c in retrieve(["which planet is hottest"], captured["states"])]
    assert before == after


def test_a_chat_survives_a_restart_and_can_be_continued(alice, seen_states):
    alice.post("/api/ingest", files=PLANETS)
    chat = alice.post("/api/ask", json={"question": "first?"}).json()["chat"]
    restart()
    assert [c["title"] for c in alice.get("/api/chats").json()["chats"]] == ["first?"]
    assert [t["question"] for t in alice.get(f"/api/chat?id={chat['id']}").json()["history"]] == ["first?"]
    alice.post("/api/ask", json={"question": "second?", "chat_id": chat["id"]})
    assert [t["question"] for t in seen_states[-1]["history"]] == ["first?"]


def test_signing_out_and_back_in_finds_the_documents(alice):
    alice.post("/api/ingest", files=PLANETS)
    assert alice.post("/api/logout").json() == {"ok": True}
    assert alice.get("/api/session").json()["documents"] == []  # signed out: a guest sees nothing of Alice's
    assert alice.post("/api/login", json={"id_token": make_token(sub="alice", email="alice@example.com", name="Alice")}).status_code == 200
    assert [d["filename"] for d in alice.get("/api/session").json()["documents"]] == ["planets.txt"]


def test_guest_documents_stay_ephemeral(signin_on):
    with TestClient(main.app) as guest:
        assert guest.post("/api/ingest", files=PLANETS).status_code == 200
        assert stored_meta("anyone") == {} and stored_blob_keys() == []
        restart()  # a guest session is not a user workspace: unaffected
        assert [d["filename"] for d in guest.get("/api/session").json()["documents"]] == ["planets.txt"]


def test_guest_document_is_adopted_on_sign_in(signin_on, seen_states):
    with TestClient(main.app) as client:
        client.post("/api/ingest", files=PLANETS)
        client.post("/api/ask", json={"question": "as a guest?"})
        assert client.post("/api/login", json={"id_token": make_token(sub="carol", email="c@example.com", name="Carol")}).status_code == 200
        assert [d["filename"] for d in client.get("/api/session").json()["documents"]] == ["planets.txt"]
        assert len(stored_meta("carol")) == 1
        chats = client.get("/api/chats").json()["chats"]
        assert [c["title"] for c in chats] == ["Chat from before you signed in"]
        assert [t["question"] for t in client.get(f"/api/chat?id={chats[0]['id']}").json()["history"]] == ["as a guest?"]
        restart()
        assert [d["filename"] for d in client.get("/api/session").json()["documents"]] == ["planets.txt"]
        assert len(main._sessions) == 0  # the guest session was retired


# ---------- ownership ----------


def test_users_cannot_see_each_others_documents(alice, bob, seen_states):
    alice.post("/api/ingest", files=PLANETS)
    assert bob.get("/api/session").json() == {"documents": [], "history": []}
    assert bob.post("/api/ask", json={"question": "anything?"}).status_code == 400


def test_a_document_id_from_another_user_is_refused_everywhere(alice, bob, seen_states):
    alice_doc = alice.post("/api/ingest", files=PLANETS).json()["id"]
    bob.post("/api/ingest", files=MOONS)
    assert bob.post("/api/ask", json={"question": "q?", "doc_ids": [alice_doc]}).status_code == 400
    assert bob.post("/api/summary", json={"id": alice_doc}).status_code == 404
    assert bob.post("/api/suggestions", json={"id": alice_doc}).status_code == 404
    assert bob.post("/api/remove", json={"id": alice_doc}).json()["documents"][0]["filename"] == "moons.txt"
    # Alice's document is untouched by Bob's attempts
    assert [d["id"] for d in alice.get("/api/session").json()["documents"]] == [alice_doc]
    assert alice.post("/api/ask", json={"question": "q?"}).status_code == 200


def test_questions_only_ever_see_the_callers_documents(alice, bob, seen_states):
    alice.post("/api/ingest", files=PLANETS)
    bob.post("/api/ingest", files=MOONS)
    alice.post("/api/ask", json={"question": "q?"})
    bob.post("/api/ask", json={"question": "q?"})
    assert seen_states[0]["docs"] == ["planets.txt"] and seen_states[1]["docs"] == ["moons.txt"]
    assert not any("Europa" in t for t in seen_states[0]["chunks"])
    assert not any("Venus" in t for t in seen_states[1]["chunks"])


def test_the_browser_cannot_choose_its_identity(signin_on):
    """A forged or missing login cookie gives a guest, never someone's workspace."""
    with TestClient(main.app) as victim:
        victim.post("/api/login", json={"id_token": make_token(sub="victim", email="v@example.com", name="V")})
        victim.post("/api/ingest", files=PLANETS)
    with TestClient(main.app) as attacker:
        attacker.cookies.set("auth_id", "A" * 43)
        attacker.cookies.set("user_id", "victim")
        attacker.headers["X-User-Id"] = "victim"
        assert attacker.get("/api/session").json() == {"documents": [], "history": []}
        assert attacker.get("/api/me").json()["user"] is None


def test_storage_keys_never_contain_the_raw_uid(alice):
    alice.post("/api/ingest", files=PLANETS)
    keys = [*main._store._hashes, *main._store._blobs, *main._store._values]
    assert not any("alice" in k for k in keys)


# ---------- other instances ----------


def test_a_second_instance_sees_documents_added_elsewhere(alice, seen_states):
    alice.post("/api/ingest", files=PLANETS)
    restart()  # instance B starts cold
    alice.get("/api/session")  # B resolves the user from shared state only
    meta_before = dict(stored_meta("alice"))
    # instance A (simulated by writing straight to the shared store) adds another document
    owner = docstore.owner_key("alice")
    import numpy as np

    packed = docstore.pack("from-a", "moons.txt", 1, 800, 150, [{"text": "Europa is icy.", "page": 1, "doc": "moons.txt"}], np.ones((1, 4), dtype="float32") / 2)
    import asyncio

    asyncio.run(docstore.DocumentRepository(main._store).save(owner, packed))
    names = [d["filename"] for d in alice.get("/api/session").json()["documents"]]
    assert sorted(names) == ["moons.txt", "planets.txt"] and len(stored_meta("alice")) == len(meta_before) + 1


def test_a_document_deleted_elsewhere_disappears_from_a_warm_instance(alice, seen_states):
    doc_id = alice.post("/api/ingest", files=PLANETS).json()["id"]
    alice.post("/api/ask", json={"question": "warm the cache?"})  # indexes now loaded in this process
    owner = docstore.owner_key("alice")
    import asyncio

    asyncio.run(docstore.DocumentRepository(main._store).delete(owner, doc_id))  # another instance removes it
    assert alice.get("/api/session").json()["documents"] == []
    assert alice.post("/api/ask", json={"question": "still there?"}).status_code == 400


# ---------- deletion ----------


def test_removing_a_document_deletes_all_of_its_state(alice):
    first = alice.post("/api/ingest", files=PLANETS).json()["id"]
    alice.post("/api/ingest", files=MOONS)
    result = alice.post("/api/remove", json={"id": first}).json()
    assert [d["filename"] for d in result["documents"]] == ["moons.txt"]
    assert first not in stored_meta("alice")
    assert not any(first in k for k in stored_blob_keys())
    restart()
    assert [d["filename"] for d in alice.get("/api/session").json()["documents"]] == ["moons.txt"]


def test_removing_everything_clears_documents_and_artifacts_but_keeps_chats(alice, seen_states):
    alice.post("/api/ingest", files=PLANETS)
    alice.post("/api/ask", json={"question": "q?"})
    assert alice.post("/api/remove", json={}).json() == {"ok": True, "documents": []}
    assert stored_meta("alice") == {} and stored_blob_keys() == []
    restart()
    assert alice.get("/api/session").json()["documents"] == []
    assert [c["title"] for c in alice.get("/api/chats").json()["chats"]] == ["q?"]


def test_removing_an_unknown_id_changes_nothing(alice):
    alice.post("/api/ingest", files=PLANETS)
    assert [d["filename"] for d in alice.post("/api/remove", json={"id": "nope"}).json()["documents"]] == ["planets.txt"]
    assert len(stored_meta("alice")) == 1


# ---------- damaged or stale storage ----------


def test_stale_metadata_with_missing_artifacts_is_removed_and_reported(alice, seen_states):
    alice.post("/api/ingest", files=PLANETS)
    for key in stored_blob_keys():  # artifacts evicted/expired, metadata remains
        del main._store._blobs[key]
    restart()
    response = alice.post("/api/ask", json={"question": "q?"})
    assert response.status_code == 409
    assert "planets.txt" in response.json()["error"] and "upload them again" in response.json()["error"]
    assert seen_states == []
    assert alice.get("/api/session").json()["documents"] == []  # cleaned up, not left broken
    assert stored_meta("alice") == {}


def test_corrupt_artifact_is_detected_removed_and_other_documents_still_work(alice, seen_states):
    alice.post("/api/ingest", files=PLANETS)
    alice.post("/api/ingest", files=MOONS)
    owner = docstore.owner_key("alice")
    planets_id = next(i for i, m in stored_meta("alice").items() if m["name"] == "planets.txt")
    key = f"a:{owner}:{planets_id}:emb:0"
    blob, expires = main._store._blobs[key]
    main._store._blobs[key] = (bytes([blob[0] ^ 0xFF]) + blob[1:], expires)
    restart()
    assert alice.post("/api/ask", json={"question": "q?"}).status_code == 409
    assert [d["filename"] for d in alice.get("/api/session").json()["documents"]] == ["moons.txt"]
    assert alice.post("/api/ask", json={"question": "q?"}).status_code == 200
    assert seen_states[-1]["docs"] == ["moons.txt"]


def test_storage_trouble_is_not_mistaken_for_a_missing_document(alice, seen_states):
    alice.post("/api/ingest", files=PLANETS)
    for key in stored_blob_keys():
        del main._store._blobs[key]
    restart()
    main._store.degraded = True  # what ResilientStore reports while Redis is being skipped
    try:
        response = alice.post("/api/ask", json={"question": "q?"})
    finally:
        main._store.degraded = False
    assert response.status_code == 503
    assert len(stored_meta("alice")) == 1  # nothing was deleted


def test_malformed_metadata_does_not_break_the_workspace(alice):
    alice.post("/api/ingest", files=PLANETS)
    owner = docstore.owner_key("alice")
    fields, expires = main._store._hashes[f"u:{owner}:docs"]
    fields["junk"] = '{"id": "junk", "name": 7}'
    restart()
    assert [d["filename"] for d in alice.get("/api/session").json()["documents"]] == ["planets.txt"]


# ---------- limits and memory ----------


def test_storage_allowance_is_enforced(alice, monkeypatch):
    monkeypatch.setattr(docstore, "max_user_bytes", lambda: 1)
    response = alice.post("/api/ingest", files=PLANETS)
    assert response.status_code == 413 and "storage is full" in response.json()["error"]
    assert stored_meta("alice") == {} and stored_blob_keys() == []
    assert alice.get("/api/session").json()["documents"] == []


def test_signed_in_user_gets_the_larger_document_cap(alice):
    for _ in range(main.MAX_DOCS_PER_SESSION):
        assert alice.post("/api/ingest", files=PLANETS).status_code == 200
    assert alice.post("/api/ingest", files=PLANETS).status_code == 400
    assert len(stored_meta("alice")) == main.MAX_DOCS_PER_SESSION


def test_memory_pressure_evicts_another_users_cache_and_it_reloads(alice, bob, seen_states, monkeypatch):
    alice.post("/api/ingest", files=PLANETS)
    alice_chunks = next(iter(main._user_sessions.values())).docs
    needed = sum(s.num_chunks for s in alice_chunks.values())
    bob.post("/api/ingest", files=MOONS)
    monkeypatch.setattr(main, "MAX_TOTAL_CHUNKS", needed + 1)  # room for roughly one workspace at a time
    assert bob.post("/api/ask", json={"question": "q?"}).status_code == 200
    assert alice.post("/api/ask", json={"question": "q?"}).status_code == 200  # reloaded from storage
    assert seen_states[-1]["docs"] == ["planets.txt"]
    assert main._total_chunks() <= main.MAX_TOTAL_CHUNKS


def test_idle_user_workspaces_are_dropped_from_memory_but_not_from_storage(alice):
    alice.post("/api/ingest", files=PLANETS)
    for session in main._user_sessions.values():
        session.last_used -= main.SESSION_TTL_SECONDS + 1
    main._prune_sessions()
    assert main._user_sessions == {}
    assert [d["filename"] for d in alice.get("/api/session").json()["documents"]] == ["planets.txt"]


def test_everything_still_works_without_sign_in_configured(monkeypatch):
    """No Firebase config: no tiers, no persistence, exactly the earlier behaviour."""
    monkeypatch.delenv("FIREBASE_PROJECT_ID", raising=False)
    with TestClient(main.app) as client:
        assert client.post("/api/ingest", files=TXT).status_code == 200
        assert client.post("/api/ingest", files=TXT).status_code == 200
        assert main._store._hashes == {} and main._store._blobs == {}
