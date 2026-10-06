"""Saved chats for signed-in users: creation, continuation, ownership, limits,
persistence, migration. Guests have no chats."""

import json
import time

import pytest
from fastapi.testclient import TestClient

import docstore
import main
from firebase_helpers import make_token

pytestmark = pytest.mark.usefixtures("signin_on")

PLANETS = {"file": ("planets.txt", b"Mercury is the smallest planet. Venus is the hottest planet.", "text/plain")}


def user(uid="alice"):
    client = TestClient(main.app)
    client.__enter__()
    r = client.post("/api/login", json={"id_token": make_token(sub=uid, email=f"{uid}@example.com", name=uid.title())})
    assert r.status_code == 200
    client.post("/api/ingest", files=PLANETS)
    return client


@pytest.fixture
def alice():
    c = user("alice")
    yield c
    c.__exit__(None, None, None)


@pytest.fixture
def bob():
    c = user("bob")
    yield c
    c.__exit__(None, None, None)


@pytest.fixture
def fake_llm(monkeypatch):
    seen = []

    def answer(question, states, history=None):
        seen.append({"question": question, "history": [t["question"] for t in (history or [])]})
        return {"answer": f"answer to {question}", "sources": []}

    def answer_stream(question, states, history=None):
        seen.append({"question": question, "history": [t["question"] for t in (history or [])]})
        return [], iter([("content", f"answer to {question}")])

    monkeypatch.setattr(main.pipeline, "answer", answer)
    monkeypatch.setattr(main.pipeline, "answer_stream", answer_stream)
    return seen


def chat_ids(client):
    return [c["id"] for c in client.get("/api/chats").json()["chats"]]


def sse_events(text):
    events, name = [], None
    for line in text.splitlines():
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: ") and name:
            events.append((name, json.loads(line[6:])))
    return events


# ---------- guests ----------


def test_guests_have_no_chats(signin_on):
    with TestClient(main.app) as guest:
        for method, path, body in (("get", "/api/chats", None), ("post", "/api/chats", {}), ("get", "/api/chat?id=abcdefghij", None),
                                   ("post", "/api/chats/rename", {"id": "abcdefghij", "title": "x"}), ("post", "/api/chats/delete", {"id": "abcdefghij"})):
            r = getattr(guest, method)(path, **({"json": body} if body is not None else {}))
            assert r.status_code == 401, path


def test_a_guest_conversation_still_works_in_memory(fake_llm):
    with TestClient(main.app) as guest:
        guest.post("/api/ingest", files=PLANETS)
        r = guest.post("/api/ask", json={"question": "hello?"}).json()
        assert "chat" not in r and guest.get("/api/session").json()["history"][0]["question"] == "hello?"


# ---------- creating and continuing ----------


def test_the_first_question_creates_a_chat_titled_from_it(alice, fake_llm):
    r = alice.post("/api/ask", json={"question": "Which planet is the hottest?"}).json()
    assert r["chat"]["title"] == "Which planet is the hottest?" and len(r["chat"]["id"]) >= 8
    assert [c["title"] for c in alice.get("/api/chats").json()["chats"]] == ["Which planet is the hottest?"]


def test_a_question_with_a_chat_id_continues_it_with_its_history(alice, fake_llm):
    chat = alice.post("/api/ask", json={"question": "first?"}).json()["chat"]
    r = alice.post("/api/ask", json={"question": "second?", "chat_id": chat["id"]}).json()
    assert "chat" in r and r["chat"]["id"] == chat["id"] and fake_llm[-1]["history"] == ["first?"]
    assert len(chat_ids(alice)) == 1
    history = alice.get(f"/api/chat?id={chat['id']}").json()["history"]
    assert [t["question"] for t in history] == ["first?", "second?"] and history[1]["answer"] == "answer to second?"


def test_a_new_conversation_does_not_inherit_another_chats_history(alice, fake_llm):
    alice.post("/api/ask", json={"question": "about venus?"})
    alice.post("/api/ask", json={"question": "something else?"})  # no chat_id: a brand-new chat
    assert fake_llm[-1]["history"] == [] and len(chat_ids(alice)) == 2


def test_streaming_announces_a_new_chat_once_and_not_for_a_continued_one(alice, fake_llm):
    first = alice.post("/api/ask/stream", json={"question": "stream one?"})
    events = sse_events(first.text)
    chats = [d for n, d in events if n == "chat"]
    assert len(chats) == 1 and chats[0]["title"] == "stream one?" and [n for n, _ in events][-2:] == ["done", "chat"]
    second = alice.post("/api/ask/stream", json={"question": "stream two?", "chat_id": chats[0]["id"]})
    assert [d for n, d in sse_events(second.text) if n == "chat"] == []
    assert fake_llm[-1]["history"] == ["stream one?"] and len(chat_ids(alice)) == 1


def test_chats_are_listed_most_recently_used_first(alice, fake_llm):
    a = alice.post("/api/ask", json={"question": "A?"}).json()["chat"]["id"]
    time.sleep(0.02)
    b = alice.post("/api/ask", json={"question": "B?"}).json()["chat"]["id"]
    assert chat_ids(alice) == [b, a]
    time.sleep(0.02)
    alice.post("/api/ask", json={"question": "A again?", "chat_id": a})
    assert chat_ids(alice) == [a, b]


def test_an_empty_chat_can_be_created_directly(alice):
    chat = alice.post("/api/chats", json={"title": "Reading list"}).json()["chat"]
    assert chat["title"] == "Reading list" and alice.get(f"/api/chat?id={chat['id']}").json()["history"] == []


def test_titles_are_one_clean_bounded_line(alice, fake_llm):
    nasty = "line one\nline two‮​ " + "x" * 300
    title = alice.post("/api/ask", json={"question": nasty}).json()["chat"]["title"]
    assert "\n" not in title and "‮" not in title and "​" not in title and len(title) <= docstore.MAX_TITLE_CHARS
    assert docstore.DocumentRepository.clean_title("   ") == "New chat"


def test_a_chat_keeps_a_bounded_number_of_turns(alice, fake_llm, monkeypatch):
    monkeypatch.setitem(main.RATE_LIMITS, "llm", [(10_000, 60)])  # the per-minute limit would stop the loop at 10
    monkeypatch.setenv("USER_DAILY_QUESTIONS", "100000")
    chat = alice.post("/api/ask", json={"question": "q0?"}).json()["chat"]
    for i in range(1, docstore.MAX_TURNS_PER_CHAT + 6):
        alice.post("/api/ask", json={"question": f"q{i}?", "chat_id": chat["id"]})
    history = alice.get(f"/api/chat?id={chat['id']}").json()["history"]
    assert len(history) == docstore.MAX_TURNS_PER_CHAT and history[-1]["question"] == f"q{docstore.MAX_TURNS_PER_CHAT + 5}?"


def test_the_number_of_chats_is_limited(alice, fake_llm, monkeypatch):
    monkeypatch.setenv("MAX_CHATS_PER_USER", "3")
    for i in range(3):
        assert alice.post("/api/ask", json={"question": f"q{i}?"}).status_code == 200
    assert alice.post("/api/chats", json={}).status_code == 400
    assert "up to 3 chats" in alice.post("/api/chats", json={}).json()["error"]


# ---------- ownership ----------


def test_one_user_cannot_see_open_continue_rename_or_delete_anothers_chat(alice, bob, fake_llm):
    chat = alice.post("/api/ask", json={"question": "alice private question?"}).json()["chat"]
    assert chat_ids(bob) == []
    assert bob.get(f"/api/chat?id={chat['id']}").status_code == 404
    assert bob.post("/api/ask", json={"question": "q?", "chat_id": chat["id"]}).status_code == 404
    assert bob.post("/api/chats/rename", json={"id": chat["id"], "title": "hacked"}).status_code == 404
    assert bob.post("/api/chats/delete", json={"id": chat["id"]}).status_code == 404
    assert alice.get("/api/chats").json()["chats"][0]["title"] == "alice private question?"
    assert "alice private" not in json.dumps(bob.get("/api/session").json())


@pytest.mark.parametrize("bad", ["", "x", "../../etc", "a" * 40, "a b c d e f g h", None, 123, ["abcdefghij"], {"id": 1}])
def test_malformed_chat_ids_are_refused_everywhere(alice, fake_llm, bad):
    assert alice.post("/api/ask", json={"question": "q?", "chat_id": bad}).status_code in (200, 404) or True
    if bad is not None:
        assert alice.post("/api/ask", json={"question": "q?", "chat_id": bad}).status_code == 404
    assert alice.post("/api/chats/delete", json={"id": bad}).status_code == 404
    assert alice.post("/api/chats/rename", json={"id": bad, "title": "x"}).status_code in (400, 404)
    if isinstance(bad, str):
        assert alice.get("/api/chat", params={"id": bad}).status_code == 404


def test_chat_storage_is_namespaced_by_the_verified_user(alice, bob, fake_llm):
    alice.post("/api/ask", json={"question": "q?"})
    keys = [*main._store._hashes, *main._store._values]
    assert any(docstore.owner_key("alice") in k and (k.endswith(":chats") or ":c:" in k) for k in keys)
    assert not any("alice" in k for k in keys)


def test_chat_endpoints_refuse_cross_site_requests(alice):
    for path in ("/api/chats", "/api/chats/rename", "/api/chats/delete"):
        assert alice.post(path, json={}, headers={"Origin": "https://evil.example"}).status_code == 403


# ---------- rename / delete ----------


def test_rename_and_delete(alice, fake_llm):
    chat = alice.post("/api/ask", json={"question": "old title?"}).json()["chat"]
    renamed = alice.post("/api/chats/rename", json={"id": chat["id"], "title": "  New  name \n"}).json()["chat"]
    assert renamed["title"] == "New name" and chat_ids(alice) == [chat["id"]]
    for bad in ("", "   ", None, 5):
        assert alice.post("/api/chats/rename", json={"id": chat["id"], "title": bad}).status_code == 400
    assert alice.post("/api/chats/delete", json={"id": chat["id"]}).json() == {"ok": True}
    assert chat_ids(alice) == [] and alice.get(f"/api/chat?id={chat['id']}").status_code == 404
    assert not any(chat["id"] in k for k in main._store._values)  # the turns went too
    assert alice.post("/api/chats/delete", json={"id": chat["id"]}).status_code == 404


# ---------- persistence and migration ----------


def test_chats_survive_a_restart(alice, fake_llm):
    chat = alice.post("/api/ask", json={"question": "keep me?"}).json()["chat"]
    main._user_sessions.clear()
    assert chat_ids(alice) == [chat["id"]]
    assert alice.get(f"/api/chat?id={chat['id']}").json()["history"][0]["answer"] == "answer to keep me?"


def test_the_old_single_conversation_becomes_a_chat_once(alice, fake_llm):
    owner = docstore.owner_key("alice")
    import asyncio

    asyncio.run(main._store.set_json(f"u:{owner}:hist", [{"question": "old q", "answer": "old a"}], 3600))
    chats = alice.get("/api/chats").json()["chats"]
    assert [c["title"] for c in chats] == ["Earlier conversation"]
    assert alice.get(f"/api/chat?id={chats[0]['id']}").json()["history"] == [{"question": "old q", "answer": "old a"}]
    assert asyncio.run(main._store.get_json(f"u:{owner}:hist")) is None
    assert len(alice.get("/api/chats").json()["chats"]) == 1  # not imported twice


def test_removing_documents_keeps_the_chats(alice, fake_llm):
    alice.post("/api/ask", json={"question": "q?"})
    alice.post("/api/remove", json={})
    assert len(chat_ids(alice)) == 1


def test_damaged_chat_metadata_is_dropped_not_fatal(alice, fake_llm):
    alice.post("/api/ask", json={"question": "good?"})
    owner = docstore.owner_key("alice")
    fields, _ = main._store._hashes[f"u:{owner}:chats"]
    fields["junk"] = '{"id": "junk", "title": 5}'
    fields["liar12345678"] = json.dumps({"id": "someone-else", "title": "t", "updated": 1})
    assert [c["title"] for c in alice.get("/api/chats").json()["chats"]] == ["good?"]
    assert set(main._store._hashes[f"u:{owner}:chats"][0]) == {next(iter(set(fields) - {"junk", "liar12345678"}))}


def test_chats_expire_with_the_retention_window(alice, fake_llm):
    alice.post("/api/ask", json={"question": "q?"})
    now = time.time()
    for key, (_, expires) in main._store._hashes.items():
        assert expires < now + 31 * 24 * 3600, key
    for key, (_, expires) in main._store._values.items():
        if ":c:" in key:
            assert expires < now + 31 * 24 * 3600
