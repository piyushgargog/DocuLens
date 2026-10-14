"""v2 API behaviour: streamed answers, document scope, load control."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def fake_stream(monkeypatch):
    """Replace the streaming LLM call; records the passages it was given."""
    import llm_client

    seen = {}

    def fake_ask_stream(question, passages, timeout=30, history=None):
        seen["passages"], seen["history"] = passages, history
        yield from [("content", "Jupiter "), ("content", "is the "), ("content", "largest planet.")]

    monkeypatch.setattr(llm_client, "ask_stream", fake_ask_stream)
    return seen


def _upload(client, data, name="a.pdf"):
    return client.post("/api/ingest", files={"file": (name, data, "application/pdf")}).json()


def _events(response):
    events = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_stream_sends_sources_then_tokens_then_done(client, sample_pdf_bytes, fake_stream):
    _upload(client, sample_pdf_bytes)
    response = client.post("/api/ask/stream", json={"question": "What is Jupiter known for?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
    events = _events(response)
    assert events[0][0] == "sources" and events[0][1]
    assert "".join(e[1]["text"] for e in events if e[0] == "token") == "Jupiter is the largest planet."
    assert events[-1][0] == "done"
    history = client.get("/api/session").json()["history"]
    assert history[-1]["answer"] == "Jupiter is the largest planet."


def test_stream_errors_before_the_first_token_are_proper_http_errors(client, sample_pdf_bytes, monkeypatch):
    import llm_client

    def failing(*args, **kwargs):
        raise llm_client.LLMConfigError("no key")
        yield  # pragma: no cover

    monkeypatch.setattr(llm_client, "ask_stream", failing)
    _upload(client, sample_pdf_bytes)
    response = client.post("/api/ask/stream", json={"question": "anything?"})
    assert response.status_code == 503
    assert "reference" in response.json()["error"]


def test_a_failure_mid_stream_is_reported_and_not_saved(client, sample_pdf_bytes, monkeypatch):
    import llm_client

    def breaks(*args, **kwargs):
        yield ("content", "Partial ")
        raise llm_client.LLMRequestError("connection dropped")

    monkeypatch.setattr(llm_client, "ask_stream", breaks)
    _upload(client, sample_pdf_bytes)
    events = _events(client.post("/api/ask/stream", json={"question": "anything?"}))
    assert events[-1][0] == "error" and "reference" in events[-1][1]["error"]
    assert client.get("/api/session").json()["history"] == []


def test_doc_ids_limit_which_documents_are_searched(client, sample_pdf_bytes, fake_stream):
    a = _upload(client, sample_pdf_bytes, "a.pdf")
    _upload(client, sample_pdf_bytes, "b.pdf")
    client.post("/api/ask/stream", json={"question": "What is Jupiter known for?", "doc_ids": [a["id"]]})
    assert {p["doc"] for p in fake_stream["passages"]} == {"a.pdf"}

    bad = client.post("/api/ask/stream", json={"question": "x?", "doc_ids": ["not-mine"]})
    assert bad.status_code == 400


def test_busy_server_answers_503_instead_of_queueing_forever(client, sample_pdf_bytes, fake_stream, monkeypatch):
    _upload(client, sample_pdf_bytes)
    monkeypatch.setattr(main, "QUEUE_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(main, "_llm_slots", asyncio.Semaphore(0))  # every slot taken
    response = client.post("/api/ask/stream", json={"question": "anything?"})
    assert response.status_code == 503
    assert response.headers["retry-after"] == "10"


# --- Request guard (CSRF, size) and cookie ----------------------------------


def test_cross_site_posts_are_refused(client, sample_pdf_bytes):
    _upload(client, sample_pdf_bytes)
    evil = client.post("/api/remove", json={}, headers={"Origin": "https://evil.example"})
    assert evil.status_code == 403
    fetch_meta = client.post("/api/remove", json={}, headers={"Sec-Fetch-Site": "cross-site"})
    assert fetch_meta.status_code == 403
    same = client.post("/api/remove", json={}, headers={"Origin": "http://testserver"})
    assert same.status_code == 200


def test_oversized_json_bodies_are_refused(client):
    response = client.post("/api/ask", content=b'{"question": "' + b"x" * 20000 + b'"}', headers={"Content-Type": "application/json"})
    assert response.status_code == 413


def test_https_sessions_use_a_host_prefixed_cookie(sample_pdf_bytes):
    secure = TestClient(main.app, base_url="https://testserver")
    response = _upload(secure, sample_pdf_bytes)
    assert "__Host-session" in secure.cookies
    assert "session_id" not in secure.cookies
    assert [d["id"] for d in secure.get("/api/session").json()["documents"]] == [response["id"]]


# --- v2.2: which provider answered -------------------------------------------


def test_stream_names_the_provider_that_answers(client, sample_pdf_bytes, monkeypatch):
    import llm_client
    from providers import Route

    route = Route("openrouter", "OpenRouter", "https://example.invalid", "openai/gpt-oss-120b:free", "secret-key")

    def fake_ask_stream(question, passages, timeout=30, history=None):
        yield ("reasoning", "The passage says Jupiter is largest.")
        yield ("content", llm_client._tagged("Jupiter ", route))
        yield ("content", "is largest.")

    monkeypatch.setattr(llm_client, "ask_stream", fake_ask_stream)
    _upload(client, sample_pdf_bytes)
    response = client.post("/api/ask/stream", json={"question": "What is Jupiter known for?"})
    events = _events(response)
    kinds = [name for name, _ in events]
    assert kinds[0] == "sources"
    # route is announced once, right before the first answer token
    assert "route" in kinds and kinds.index("route") == kinds.index("token") - 1
    route = next(d for n, d in events if n == "route")
    assert route == {"provider": "OpenRouter", "model": "openai/gpt-oss-120b:free"}
    assert "secret-key" not in response.text
    # Kept with the turn, so a reloaded page still says who answered.
    assert client.get("/api/session").json()["history"][-1]["answered_by"]["provider"] == "OpenRouter"


def test_status_lists_providers_without_secrets(client, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "groq,nvidia")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "gsk-secret")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-secret")
    monkeypatch.delenv("LLM_FALLBACK_MODEL", raising=False)
    response = client.get("/api/status")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == main.APP_VERSION
    assert [p["provider"] for p in body["providers"]] == ["Groq", "NVIDIA", "NVIDIA"]
    assert "secret" not in response.text and "https://" not in response.text


def test_health_check(client):
    assert client.get("/api/health").json() == {"ok": True}


def test_reasoning_is_streamed_as_its_own_event_before_the_answer(client, sample_pdf_bytes, monkeypatch):
    import llm_client

    def fake_ask_stream(question, passages, timeout=30, history=None):
        yield ("reasoning", "Let me check the passage. ")
        yield ("reasoning", "It names Jupiter. ")
        yield ("content", "Jupiter is largest.")

    monkeypatch.setattr(llm_client, "ask_stream", fake_ask_stream)
    _upload(client, sample_pdf_bytes)
    events = _events(client.post("/api/ask/stream", json={"question": "which is largest?"}))
    kinds = [name for name, _ in events]
    assert kinds[0] == "sources"
    assert "reasoning" in kinds and "token" in kinds
    # every reasoning event comes before the first answer token
    assert kinds.index("reasoning") < kinds.index("token")
    reasoning = "".join(d["text"] for n, d in events if n == "reasoning")
    assert reasoning == "Let me check the passage. It names Jupiter. "
    # reasoning is not saved as the answer
    assert client.get("/api/session").json()["history"][-1]["answer"] == "Jupiter is largest."
