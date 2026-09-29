"""Tests for the FastAPI layer (main.py) itself -- separate from the
pipeline-level tests, which already cover retrieval/grounding correctness.
These just verify the HTTP contract: upload, session handling, and error
shapes.

Each test gets its own TestClient (via the `client` fixture) so that one
test's session cookie can never leak into another -- TestClient behaves like
a real browser session and persists cookies across requests made on the same
instance."""

import os

import pytest
from fastapi.testclient import TestClient

from main import app

requires_api_key = pytest.mark.skipif(
    not os.environ.get("LLM_API_KEY"),
    reason="LLM_API_KEY not set; skipping tests that call a real LLM API",
)


@pytest.fixture
def client():
    return TestClient(app)


def test_index_page_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "AI Document Assistant" in response.text


def test_ingest_rejects_non_pdf(client):
    response = client.post("/api/ingest", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert "PDF" in response.json()["error"]


def test_ingest_valid_pdf_returns_session_cookie(client, sample_pdf_bytes):
    response = client.post("/api/ingest", files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")})
    assert response.status_code == 200
    body = response.json()
    assert body["num_pages"] == 4
    assert body["num_chunks"] > 0
    assert "session_id" in response.cookies


def test_ask_without_a_session_returns_400(client):
    response = client.post("/api/ask", json={"question": "anything?"})
    assert response.status_code == 400
    assert "No document is loaded" in response.json()["error"]


def test_ask_empty_question_rejected(client, sample_pdf_bytes):
    client.post("/api/ingest", files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")})
    response = client.post("/api/ask", json={"question": "   "})
    assert response.status_code == 400


@requires_api_key
def test_ask_end_to_end_after_ingest(client, sample_pdf_bytes):
    client.post("/api/ingest", files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")})
    response = client.post("/api/ask", json={"question": "What is Jupiter known for?"})
    assert response.status_code == 200
    body = response.json()
    assert "jupiter" in body["answer"].lower()
    assert body["sources"][0]["page"] == 3


def test_remove_clears_the_session(client, sample_pdf_bytes):
    client.post("/api/ingest", files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")})
    client.post("/api/remove")
    response = client.post("/api/ask", json={"question": "anything?"})
    assert response.status_code == 400


# --- Error handling ---------------------------------------------------------


def test_ingest_unextractable_pdf_returns_422(client):
    response = client.post("/api/ingest", files={"file": ("broken.pdf", b"not a pdf", "application/pdf")})
    assert response.status_code == 422
    assert "extract" in response.json()["error"]


def test_ask_with_invalid_json_body_is_a_400_not_a_crash(client, sample_pdf_bytes):
    client.post("/api/ingest", files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")})
    for body in (b"{not json", b"[1, 2]", b'{"question": 42}'):
        response = client.post("/api/ask", content=body, headers={"Content-Type": "application/json"})
        assert response.status_code == 400


def test_llm_failure_returns_502(client, sample_pdf_bytes, monkeypatch):
    import llm_client

    def failing_ask(*args, **kwargs):
        raise llm_client.LLMRequestError("boom")

    monkeypatch.setattr(llm_client, "ask", failing_ask)
    client.post("/api/ingest", files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")})
    response = client.post("/api/ask", json={"question": "What is Jupiter known for?"})
    assert response.status_code == 502
    assert "boom" not in response.json()["error"]  # exception text is never echoed


# --- Multiple documents, history, summary (LLM replaced by a fake) ----------


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the LLM calls and record what they were given."""
    import llm_client

    calls = []

    def fake_ask(question, passages, timeout=30, history=None):
        calls.append({"question": question, "passages": passages, "history": history})
        return f"answer to: {question}"

    monkeypatch.setattr(llm_client, "ask", fake_ask)
    monkeypatch.setattr(llm_client, "summarize", lambda passages, timeout=30: "a summary")
    return calls


def test_second_upload_adds_a_document_to_the_same_session(client, sample_pdf_bytes):
    first = client.post("/api/ingest", files={"file": ("a.pdf", sample_pdf_bytes, "application/pdf")})
    second = client.post("/api/ingest", files={"file": ("a.pdf", sample_pdf_bytes, "application/pdf")})
    assert second.status_code == 200
    assert "session_id" not in second.cookies  # existing session reused, not replaced
    names = [d["filename"] for d in second.json()["documents"]]
    assert names == ["a.pdf", "a.pdf (2)"]
    assert first.json()["id"] != second.json()["id"]


def test_document_limit_per_session(client, sample_pdf_bytes):
    import main

    for _ in range(main.MAX_DOCS_PER_SESSION):
        client.post("/api/ingest", files={"file": ("a.pdf", sample_pdf_bytes, "application/pdf")})
    response = client.post("/api/ingest", files={"file": ("a.pdf", sample_pdf_bytes, "application/pdf")})
    assert response.status_code == 400


def test_sources_name_their_document_and_history_reaches_the_llm(client, sample_pdf_bytes, fake_llm):
    client.post("/api/ingest", files={"file": ("planets.pdf", sample_pdf_bytes, "application/pdf")})
    first = client.post("/api/ask", json={"question": "What is Jupiter known for?"}).json()
    assert first["sources"][0]["doc"] == "planets.pdf"
    assert fake_llm[0]["history"] == []

    client.post("/api/ask", json={"question": "How many moons does it have?"})
    assert fake_llm[1]["history"] == [
        {"question": "What is Jupiter known for?", "answer": "answer to: What is Jupiter known for?"}
    ]

    session = client.get("/api/session").json()
    assert [d["filename"] for d in session["documents"]] == ["planets.pdf"]
    assert len(session["history"]) == 2


def test_removing_one_document_keeps_the_others(client, sample_pdf_bytes):
    a = client.post("/api/ingest", files={"file": ("a.pdf", sample_pdf_bytes, "application/pdf")}).json()
    client.post("/api/ingest", files={"file": ("b.pdf", sample_pdf_bytes, "application/pdf")})
    remaining = client.post("/api/remove", json={"id": a["id"]}).json()["documents"]
    assert [d["filename"] for d in remaining] == ["b.pdf"]


def test_summary_of_a_loaded_document(client, sample_pdf_bytes, fake_llm):
    doc = client.post("/api/ingest", files={"file": ("a.pdf", sample_pdf_bytes, "application/pdf")}).json()
    response = client.post("/api/summary", json={"id": doc["id"]})
    assert response.status_code == 200
    assert response.json()["summary"] == "a summary"
    assert response.json()["sources"]

    assert client.post("/api/summary", json={"id": "nope"}).status_code == 404


def test_session_endpoint_without_a_session(client):
    assert client.get("/api/session").json() == {"documents": [], "history": []}
