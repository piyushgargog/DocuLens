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
    with TestClient(app) as c:
        yield c


def test_index_page_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "DocuLens" in response.text


def test_ingest_accepts_a_plain_text_file(client):
    # Since v2.5.0 text/Markdown/docx are accepted, not only PDFs.
    response = client.post("/api/ingest", files={"file": ("notes.txt", b"Mercury is the smallest planet.", "text/plain")})
    assert response.status_code == 200
    assert response.json()["num_chunks"] > 0


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
    # Starts like a PDF (passes the signature check) but can't be parsed.
    response = client.post("/api/ingest", files={"file": ("broken.pdf", b"%PDF-1.4 not really", "application/pdf")})
    assert response.status_code == 422
    assert "read any text" in response.json()["error"]


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
    assert second.cookies.get("session_id") == first.cookies.get("session_id")  # existing session reused, not replaced
    names = [d["filename"] for d in second.json()["documents"]]
    assert names == ["a.pdf", "a (2).pdf"]
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


def test_frontend_files_are_revalidated_but_api_is_untouched(client):
    # Heuristic browser caching once served a stale app.js with a new index.html.
    for path in ("/", "/static/app.js", "/static/style.css"):
        assert client.get(path).headers["cache-control"] == "no-cache"
    assert "cache-control" not in client.get("/api/session").headers


# --- Uploaded documents don't outlive their session ------------------------


def _expired_session(main):
    session = main.Session()
    session.last_used -= main.SESSION_TTL_SECONDS + 1
    return session


def test_expired_sessions_are_pruned():
    import main

    main._sessions["stale"] = _expired_session(main)
    main._sessions["fresh"] = main.Session()
    main._prune_sessions()
    assert "stale" not in main._sessions
    assert "fresh" in main._sessions
    del main._sessions["fresh"]


def test_background_sweeper_deletes_expired_sessions_without_any_request(monkeypatch):
    import time

    import main

    monkeypatch.setattr(main, "CLEANUP_INTERVAL_SECONDS", 0.05)
    with TestClient(main.app):  # runs the lifespan, which starts the sweeper
        main._sessions["stale"] = _expired_session(main)
        deadline = time.time() + 3
        while "stale" in main._sessions and time.time() < deadline:
            time.sleep(0.05)
        assert "stale" not in main._sessions


def test_upload_temp_file_is_closed_right_after_reading(client, sample_pdf_bytes, monkeypatch):
    from starlette.datastructures import UploadFile

    closed = []
    original_close = UploadFile.close

    async def tracking_close(self):
        closed.append(self.filename)
        await original_close(self)

    monkeypatch.setattr(UploadFile, "close", tracking_close)
    client.post("/api/ingest", files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")})
    assert "sample.pdf" in closed


def test_page_shows_the_same_version_as_the_app():
    """static/index.html repeats the version (asset ?v=, footer, release link);
    this catches a release where one of them wasn't bumped."""
    import re

    import main

    html = (main.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    found = (
        re.findall(r"\?v=([\d.]+)", html)
        + re.findall(r"releases/tag/v([\d.]+)", html)
        + re.findall(r">v([\d.]+)<", html)
    )
    # Every version string in the page (asset ?v=, release links, badge labels)
    # must equal APP_VERSION; the exact count grows as more badges are added.
    assert len(found) >= 5
    assert set(found) == {main.APP_VERSION}
    assert main.app.version == main.APP_VERSION


def test_footer_credits_the_author_and_links_the_repository(client):
    html = client.get("/").text
    assert "Made with" in html and "Piyush Garg" in html
    assert 'href="https://github.com/piyushgargog/DocuLens"' in html


def test_suggestions_for_a_loaded_document(client, sample_pdf_bytes, monkeypatch):
    import llm_client

    monkeypatch.setattr(llm_client, "suggest_questions", lambda passages, timeout=30: ["What is Jupiter known for?"])
    doc = client.post("/api/ingest", files={"file": ("a.pdf", sample_pdf_bytes, "application/pdf")}).json()
    response = client.post("/api/suggestions", json={"id": doc["id"]})
    assert response.status_code == 200
    assert response.json() == {"questions": ["What is Jupiter known for?"]}
    assert client.post("/api/suggestions", json={"id": "nope"}).status_code == 404


def test_ingest_accepts_text_and_docx(client):
    txt = ("Mercury is the smallest planet. " * 30).encode()
    r = client.post("/api/ingest", files={"file": ("notes.txt", txt, "text/plain")})
    assert r.status_code == 200 and r.json()["num_chunks"] > 0

    import io as _io

    from docx import Document

    d = Document()
    for _ in range(20):
        d.add_paragraph("Saturn has many confirmed moons in the Solar System.")
    buf = _io.BytesIO(); d.save(buf)
    r = client.post("/api/ingest", files={"file": ("report.docx", buf.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})
    assert r.status_code == 200 and r.json()["num_chunks"] > 0


def test_ingest_rejects_unsupported_extension(client):
    r = client.post("/api/ingest", files={"file": ("malware.exe", b"MZ...", "application/octet-stream")})
    assert r.status_code == 400


def test_responses_carry_a_request_id(client):
    r = client.get("/api/status")
    assert r.headers.get("X-Request-ID")
    # A client-supplied id is echoed back (for log correlation).
    r2 = client.get("/api/status", headers={"X-Request-ID": "abc123trace"})
    assert r2.headers["X-Request-ID"] == "abc123trace"

def test_session_cookie_max_age_is_refreshed_on_active_requests(client, sample_pdf_bytes):
    # The initial ingest sets the cookie
    first = client.post("/api/ingest", files={"file": ("a.pdf", sample_pdf_bytes, "application/pdf")})
    assert "session_id" in first.cookies
    # A subsequent API request should refresh the cookie max-age
    session_resp = client.get("/api/session")
    # The Set-Cookie header should be present with the same session id
    set_cookie = session_resp.headers.get("set-cookie", "")
    assert "session_id=" in set_cookie
    assert "Max-Age=" in set_cookie
    import main
    assert str(main.SESSION_TTL_SECONDS) in set_cookie
