"""FastAPI backend for the AI Document Assistant.

Wraps the RAG pipeline (pdf_loader -> chunker -> embedder -> vector_store ->
llm_client -> pipeline) with a minimal HTTP API and serves the static
frontend from the same origin (so no CORS setup is needed). Nothing in this
file touches retrieval, chunking, embedding, or prompt logic -- it only adapts
pipeline.ingest()/answer()/summarize() to HTTP and holds per-session state.
"""

import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import pipeline
from llm_client import LLMConfigError, LLMRequestError

load_dotenv()

STATIC_DIR = Path(__file__).parent / "static"

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25MB -- unchanged from the previous UI's limit
MAX_QUESTION_CHARS = 1000  # unchanged from the previous UI's limit
MAX_DOCS_PER_SESSION = 5
MAX_SESSIONS = 50  # oldest-idle session is evicted beyond this, bounding server memory
MAX_STORED_TURNS = 10  # conversation turns kept per session (only the last few reach the LLM)
SESSION_TTL_SECONDS = 2 * 60 * 60  # 2 hours of inactivity

app = FastAPI(title="AI Document Assistant")


@dataclass
class Session:
    docs: dict[str, pipeline.IndexState] = field(default_factory=dict)  # doc_id -> index
    history: list[dict] = field(default_factory=list)  # [{"question", "answer"}], oldest first
    last_used: float = field(default_factory=time.time)


# In-memory sessions keyed by a random cookie value. A single-process
# in-memory store is the simplest sensible choice at this project's scale (a
# personal/portfolio tool, not a multi-instance service). Sessions are lost on
# restart and are not shared across processes -- see README "Known limitations".
_sessions: dict[str, Session] = {}


def _prune_sessions() -> None:
    now = time.time()
    for sid in [sid for sid, s in _sessions.items() if now - s.last_used > SESSION_TTL_SECONDS]:
        del _sessions[sid]
    while len(_sessions) > MAX_SESSIONS:
        oldest = min(_sessions, key=lambda sid: _sessions[sid].last_used)
        del _sessions[oldest]


def _get_session(request: Request) -> Session | None:
    session = _sessions.get(request.cookies.get("session_id", ""))
    if session is not None:
        session.last_used = time.time()
    return session


def _is_https(request: Request) -> bool:
    """True if this request reached us over HTTPS, directly or via Nginx.

    Nginx (see DECISIONS.md) forwards X-Forwarded-Proto based on its own
    $scheme, so the session cookie is marked Secure exactly when the browser
    is on HTTPS, with no code change needed between local and deployed runs.
    """
    if request.url.scheme == "https":
        return True
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    return forwarded_proto.split(",")[0].strip().lower() == "https"


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


async def _json_body(request: Request) -> dict:
    """Parsed JSON object body, or {} for a missing/invalid/non-object body."""
    try:
        body = await request.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _documents(session: Session) -> list[dict]:
    return [
        {"id": doc_id, "filename": s.name, "num_pages": s.num_pages, "num_chunks": s.num_chunks}
        for doc_id, s in session.docs.items()
    ]


def _unique_name(session: Session, filename: str) -> str:
    """Filenames label sources in answers, so two uploads of 'notes.pdf'
    become 'notes.pdf' and 'notes.pdf (2)'."""
    taken = {s.name for s in session.docs.values()}
    name, n = filename, 2
    while name in taken:
        name, n = f"{filename} ({n})", n + 1
    return name


def _llm_error_response(e: Exception, action: str) -> JSONResponse:
    # Logged server-side only -- never echo exception text back to the
    # client (see DECISIONS.md: CodeQL py/stack-trace-exposure).
    print(f"Error while {action}: {e!r}")
    if isinstance(e, LLMConfigError):
        return _error("The assistant is not configured correctly. Please contact the site administrator.", 503)
    if isinstance(e, LLMRequestError):
        return _error("The LLM API request failed. Please try again in a moment.", 502)
    return _error(f"Something went wrong while {action}. Please try again.", 500)


@app.post("/api/ingest")
async def ingest(request: Request, file: UploadFile = File(...)):
    _prune_sessions()

    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        return _error("Please upload a PDF file.", 400)

    session = _get_session(request)
    if session is not None and len(session.docs) >= MAX_DOCS_PER_SESSION:
        return _error(f"You can load up to {MAX_DOCS_PER_SESSION} documents at once. Remove one first.", 400)

    # Read at most one byte past the limit, so an oversized upload is never
    # held in memory in full.
    pdf_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(pdf_bytes) > MAX_UPLOAD_BYTES:
        return _error("File is too large. The limit is 25MB.", 413)

    name = _unique_name(session, filename) if session else filename
    try:
        # Extraction + embedding is CPU-bound; running it in a worker thread
        # keeps the event loop free to serve other users meanwhile.
        index_state = await run_in_threadpool(pipeline.ingest, pdf_bytes, name=name)
    except Exception as e:
        print(f"Unexpected error during ingestion: {e!r}")
        index_state = None

    if index_state is None:
        return _error(
            "Couldn't extract any text from this PDF. It may be empty, image-only "
            "(scanned without OCR), password-protected, or corrupted.",
            422,
        )

    # Look the session up again: it may have expired or been evicted while
    # this upload was being indexed.
    session = _get_session(request)
    is_new_session = session is None
    if is_new_session:
        session_id = secrets.token_urlsafe(32)
        session = Session()
        _sessions[session_id] = session
        _prune_sessions()

    doc_id = secrets.token_urlsafe(8)
    session.docs[doc_id] = index_state

    response = JSONResponse(
        {
            "id": doc_id,
            "filename": index_state.name,
            "num_pages": index_state.num_pages,
            "num_chunks": index_state.num_chunks,
            "documents": _documents(session),
        }
    )
    if is_new_session:
        response.set_cookie(
            "session_id",
            session_id,
            httponly=True,
            samesite="lax",
            secure=_is_https(request),
            max_age=SESSION_TTL_SECONDS,
        )
    return response


@app.get("/api/session")
async def get_session(request: Request):
    """Current documents and conversation, so a page reload can restore the UI."""
    session = _get_session(request)
    if session is None:
        return {"documents": [], "history": []}
    return {"documents": _documents(session), "history": session.history}


@app.post("/api/ask")
async def ask(request: Request):
    _prune_sessions()

    session = _get_session(request)
    if session is None or not session.docs:
        return _error("No document is loaded. Please upload a PDF first.", 400)

    body = await _json_body(request)
    question = body.get("question")
    question = question.strip() if isinstance(question, str) else ""
    if not question:
        return _error("Please enter a question.", 400)
    if len(question) > MAX_QUESTION_CHARS:
        return _error(f"Question is too long (max {MAX_QUESTION_CHARS} characters).", 400)

    try:
        result = await run_in_threadpool(
            pipeline.answer, question, list(session.docs.values()), history=list(session.history)
        )
    except Exception as e:
        return _llm_error_response(e, "answering that question")

    session.history.append({"question": question, "answer": result["answer"]})
    del session.history[:-MAX_STORED_TURNS]
    return {"answer": result["answer"], "sources": result["sources"]}


@app.post("/api/summary")
async def summary(request: Request):
    session = _get_session(request)
    if session is None or not session.docs:
        return _error("No document is loaded. Please upload a PDF first.", 400)

    body = await _json_body(request)
    index_state = session.docs.get(body.get("id"))
    if index_state is None:
        return _error("That document is not loaded.", 404)

    try:
        result = await run_in_threadpool(pipeline.summarize, index_state)
    except Exception as e:
        return _llm_error_response(e, "summarizing the document")
    return {"filename": index_state.name, "summary": result["summary"], "sources": result["sources"]}


@app.post("/api/remove")
async def remove(request: Request):
    """Remove one document (body {"id": ...}) or, with no id, everything."""
    session_id = request.cookies.get("session_id", "")
    session = _sessions.get(session_id)
    doc_id = (await _json_body(request)).get("id")

    if session is not None and doc_id is not None:
        session.docs.pop(doc_id, None)
        if session.docs:
            return {"ok": True, "documents": _documents(session)}

    _sessions.pop(session_id, None)
    response = JSONResponse({"ok": True, "documents": []})
    response.delete_cookie("session_id")
    return response


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")
