"""FastAPI backend for DocuLens — An AI Powered Document Assistant.

Wraps the RAG pipeline (pdf_loader -> chunker -> embedder -> vector_store ->
llm_client -> pipeline) with a minimal HTTP API and serves the static
frontend from the same origin (so no CORS setup is needed). Nothing in this
file touches retrieval, chunking, embedding, or prompt logic -- it only adapts
pipeline.ingest()/answer()/summarize() to HTTP and holds per-session state.
"""

import asyncio
import ipaddress
import json
import os
import re
import secrets
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.concurrency import iterate_in_threadpool, run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import document_loader
import embedder
import observability
import pipeline
import providers
from observability import log, new_request_id, request_id_var

observability.setup_logging()
from llm_client import LLMConfigError, LLMRequestError, answered_by

load_dotenv()

STATIC_DIR = Path(__file__).parent / "static"

# The release version. static/index.html repeats it (asset ?v= query, footer,
# release link) and tests/test_api.py fails if the two ever disagree.
APP_VERSION = "3.6.0"

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25MB -- unchanged from the previous UI's limit
MAX_QUESTION_CHARS = 1000  # unchanged from the previous UI's limit
MAX_DOCS_PER_SESSION = 5
MAX_SESSIONS = 50  # oldest-idle session is evicted beyond this, bounding server memory
MAX_STORED_TURNS = 10  # conversation turns kept per session (only the last few reach the LLM)
SESSION_TTL_SECONDS = 2 * 60 * 60  # 2 hours of inactivity
CLEANUP_INTERVAL_SECONDS = 5 * 60
# ~360 pages of text at the default chunking. Bounds the memory and embedding
# time one upload can take: a 25MB text-heavy PDF could otherwise hold
# gigabytes across sessions on a 2GB server.
MAX_CHUNKS_PER_DOC = 1500
MAX_FILENAME_CHARS = 120

# Total in-memory chunk budget across all sessions.  At ~800 chars per chunk
# plus a 384-dim float32 embedding vector (~1.5 KB each) and BM25 term index,
# each chunk costs roughly 3 KB.  The default 75 000 chunks ≈ 225 MB of chunk
# data, leaving headroom on a 2 GB server for the embedding model (~200 MB),
# the Python runtime, and request buffers.  Adjust via MAX_TOTAL_CHUNKS.
# This is a best-effort soft limit, not horizontal scalability.
try:
    MAX_TOTAL_CHUNKS = max(1, int(os.environ.get("MAX_TOTAL_CHUNKS", "75000")))
except ValueError:
    log.warning("Invalid MAX_TOTAL_CHUNKS; using 75000")
    MAX_TOTAL_CHUNKS = 75000


# Per-client request limits: (max requests, window in seconds). Every
# LLM-backed call spends the shared provider quota (on Groq's free tier, a
# daily token budget), so one script could otherwise take the assistant down
# for everyone; ingestion is CPU-heavy (embedding).
RATE_LIMITS = {
    "llm": [(10, 60), (100, 60 * 60)],  # /api/ask, /api/summary, /api/suggestions
    "ingest": [(10, 10 * 60)],  # /api/ingest
}

# CIDR networks whose X-Real-IP header is trusted for rate limiting.
# Default: loopback only (safe for Nginx on the same host). Behind a
# Docker bridge or external reverse proxy, set TRUSTED_PROXIES to a
# comma-separated list of CIDRs, e.g. "172.17.0.0/16,10.0.0.0/8".
# Never include 0.0.0.0/0: any client could then spoof their IP.
_TRUSTED_PROXY_NETS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []

def _parse_trusted_proxies() -> list:
    raw = os.environ.get("TRUSTED_PROXIES", "127.0.0.0/8,::1/128").strip()
    nets = []
    for cidr in raw.split(","):
        cidr = cidr.strip()
        if not cidr:
            continue
        try:
            nets.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            log.warning("Ignoring invalid TRUSTED_PROXIES entry: %s", cidr)
    return nets

_TRUSTED_PROXY_NETS = _parse_trusted_proxies()

# Sent with every response. The page loads nothing from other origins (fonts
# are self-hosted), so the policy can be 'self' throughout; the favicon is an
# inline SVG data: URL.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; "
        "form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}

# JSON request bodies are a question and a few ids; anything bigger is abuse.
MAX_JSON_BYTES = 16 * 1024

# How many LLM calls and ingestions may run at once. The server has 2 CPUs and
# 2GB of RAM, and the LLM provider has per-minute limits: beyond these, extra
# requests wait briefly and then get a clear "busy" answer instead of piling up
# threads and memory until everything slows down together.
LLM_CONCURRENCY = 4
INGEST_CONCURRENCY = 2
QUEUE_WAIT_SECONDS = 20
_llm_slots = asyncio.Semaphore(LLM_CONCURRENCY)
_ingest_slots = asyncio.Semaphore(INGEST_CONCURRENCY)


async def _sweep_expired_sessions() -> None:
    """Delete expired sessions on a timer, not only when a request arrives.

    Uploaded PDFs are never written to disk; their extracted text and
    embeddings live only in `_sessions`. Pruning used to run only at the start
    of a request, so with no traffic an expired session's document text could
    stay in memory indefinitely. This makes the 2-hour limit a real deadline.
    """
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        _prune_sessions()


@asynccontextmanager
async def lifespan(app: FastAPI):
    observability.init_sentry(release=f"doculens@{APP_VERSION}")
    providers_configured = len(providers.routes())
    log.info("DocuLens %s starting; %d LLM provider(s) configured", APP_VERSION, providers_configured)
    if providers_configured == 0:
        # Not fatal: uploads and retrieval still work, and every LLM-backed
        # endpoint answers 503 with a clear message. Refusing to start would
        # also break the credential-free test suite and CI.
        log.warning("No LLM provider is configured -- questions, summaries and suggestions will fail.")
    if embedder.prefetch_enabled():
        # Moves the ~2s model load from the first upload to startup.
        await run_in_threadpool(embedder.prefetch)
    sweeper = asyncio.create_task(_sweep_expired_sessions())
    try:
        yield
    finally:
        sweeper.cancel()


# The interactive API docs (/docs, /redoc, /openapi.json) are off: they would
# publish a map of every endpoint, and nothing here needs them in production.
app = FastAPI(
    title="DocuLens",
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.middleware("http")
async def api_request_guard(request: Request, call_next):
    """Reject cross-site and oversized API requests before any handler runs.

    CSRF: the session cookie is already SameSite=Lax, which stops browsers
    sending it on cross-site POSTs; this is the second layer. A browser always
    sends Origin on a POST fetch and Sec-Fetch-Site on modern engines, so a
    request that another site triggers is refused outright. Requests with no
    Origin at all (curl, scripts, the test client) aren't from a browser page
    and can't carry a victim's cookie, so they pass.
    """
    if request.url.path.startswith("/api/") and request.method not in ("GET", "HEAD", "OPTIONS"):
        if request.headers.get("sec-fetch-site") == "cross-site" or not _same_origin(request):
            return _error("Cross-site requests are not allowed.", 403)
        if request.url.path != "/api/ingest":
            length = request.headers.get("content-length")
            if length and length.isdigit() and int(length) > MAX_JSON_BYTES:
                return _error("Request is too large.", 413)
    return await call_next(request)


@app.middleware("http")
async def refresh_session_cookie(request: Request, call_next):
    """Refresh the session cookie max-age on active cookie-authenticated
    responses so the browser expiry tracks the server-side TTL.  The
    session id itself is unchanged -- only the expiry slides forward."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        return response
    cookie_name = _cookie_name(request)
    session_id = request.cookies.get(cookie_name, "")
    if session_id and session_id in _sessions:
        response.set_cookie(
            cookie_name,
            session_id,
            httponly=True,
            samesite="lax",
            secure=_is_https(request),
            max_age=SESSION_TTL_SECONDS,
        )
    return response


@app.middleware("http")
async def response_headers(request: Request, call_next):
    """Security headers on every response, plus revalidation of the frontend.

    Without a Cache-Control header, browsers cache static files heuristically
    and may skip asking the server at all -- after the v1.2.0 deploy a browser
    kept running the old app.js against the new index.html and hung on
    "Reading and indexing document". With no-cache, an unchanged file still
    costs only a 304.
    """
    response = await call_next(request)
    response.headers.update(SECURITY_HEADERS)
    if _is_https(request):
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Outermost middleware: give every request a short id (echoed in the
    X-Request-ID header and attached to all of its log lines), and log each
    API call with its status and latency. Health checks are skipped so the
    30-second Docker probe doesn't flood the log."""
    rid = request.headers.get("x-request-id", "")[:32] or new_request_id()
    token = request_id_var.set(rid)
    started = time.perf_counter()
    path = request.url.path
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        # Log while the id is still set so the line carries it (health checks
        # excluded so the 30s Docker probe doesn't flood the log).
        if path.startswith("/api/") and path != "/api/health":
            ms = (time.perf_counter() - started) * 1000
            log.info("%s %s -> %d (%.0fms)", request.method, path, response.status_code, ms)
        return response
    except Exception:
        log.exception("Unhandled error on %s %s", request.method, path)
        raise
    finally:
        request_id_var.reset(token)


@dataclass
class Session:
    docs: dict[str, pipeline.IndexState] = field(default_factory=dict)  # doc_id -> index
    history: list[dict] = field(default_factory=list)  # [{"question", "answer"}], oldest first
    last_used: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


# In-memory sessions keyed by a random cookie value. A single-process
# in-memory store is the simplest sensible choice at this project's scale (a
# personal/portfolio tool, not a multi-instance service). Sessions are lost on
# restart and are not shared across processes -- see README "Known limitations".
_sessions: dict[str, Session] = {}


def _prune_sessions() -> None:
    now = time.time()
    expired = []
    for sid, session in list(_sessions.items()):
        with session.lock:
            if now - session.last_used > SESSION_TTL_SECONDS:
                expired.append(sid)
    for sid in expired:
        _sessions.pop(sid, None)
    while len(_sessions) > MAX_SESSIONS:
        oldest = min(
            _sessions,
            key=lambda sid: _sessions[sid].last_used,
        )
        _sessions.pop(oldest, None)
    # Rate-limit history older than the longest window is no longer needed.
    longest = max(window for limits in RATE_LIMITS.values() for _, window in limits)
    for key in [k for k, hits in _rate_log.items() if not hits or now - hits[-1] > longest]:
        del _rate_log[key]


def _total_chunks() -> int:
    """Current total chunks across all sessions."""
    total = 0
    for session in list(_sessions.values()):
        with session.lock:
            total += sum(state.num_chunks for state in session.docs.values())
    return total


# (client, bucket) -> timestamps of recent allowed requests, oldest first.
_rate_log: dict[tuple[str, str], deque] = {}


def _client_ip(request: Request) -> str:
    """The caller's address for rate limiting.

    When the peer IP belongs to a configured trusted-proxy network
    (default: loopback only), we read X-Real-IP as set by the reverse
    proxy.  The header is ignored from any other peer, so it can't be
    spoofed to dodge the rate limit.  Set TRUSTED_PROXIES to a
    comma-separated list of CIDRs for Docker bridge or similar setups.
    """
    peer = request.client.host if request.client else "unknown"
    try:
        peer_addr = ipaddress.ip_address(peer)
    except ValueError:
        return peer
    if any(peer_addr in net for net in _TRUSTED_PROXY_NETS):
        return request.headers.get("x-real-ip", peer)
    return peer


def _rate_limited(request: Request, bucket: str) -> JSONResponse | None:
    """A 429 response if this client is over any of the bucket's limits,
    otherwise None (and the request is counted)."""
    now = time.time()
    hits = _rate_log.setdefault((_client_ip(request), bucket), deque())
    longest = max(window for _, window in RATE_LIMITS[bucket])
    while hits and now - hits[0] > longest:
        hits.popleft()
    for limit, window in RATE_LIMITS[bucket]:
        recent = [t for t in hits if now - t <= window]
        if len(recent) >= limit:
            retry_after = int(window - (now - recent[0])) + 1
            response = _error(
                f"Too many requests. Please wait about {max(1, round(retry_after / 60))} "
                f"minute(s) and try again.",
                429,
            )
            response.headers["Retry-After"] = str(retry_after)
            return response
    hits.append(now)
    return None


def _cookie_name(request: Request) -> str:
    """Over HTTPS the session cookie uses the __Host- prefix: browsers then
    only accept it if it is Secure, has Path=/ and no Domain, so no subdomain
    or plain-HTTP response can set or overwrite it. (Plain-HTTP local runs
    can't use the prefix.)"""
    return "__Host-session" if _is_https(request) else "session_id"


def _get_session(request: Request) -> Session | None:
    session = _sessions.get(request.cookies.get(_cookie_name(request), ""))
    if session is not None:
        with session.lock:
            session.last_used = time.time()
    return session


def _same_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    if not origin:
        return True
    return urlsplit(origin).netloc == request.headers.get("host", "")


async def _take_slot(slots: asyncio.Semaphore) -> bool:
    try:
        await asyncio.wait_for(slots.acquire(), timeout=QUEUE_WAIT_SECONDS)
        return True
    except asyncio.TimeoutError:
        return False


def _busy() -> JSONResponse:
    response = _error("The assistant is busy right now. Please try again in a few seconds.", 503)
    response.headers["Retry-After"] = "10"
    return response


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
    """Parsed JSON object body, or {} for a missing/invalid/non-object/oversized
    body (the size check also covers bodies sent without Content-Length)."""
    raw = await request.body()
    if len(raw) > MAX_JSON_BYTES:
        return {}
    try:
        body = json.loads(raw)
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


async def _question_request(request: Request) -> tuple[Session, str, list] | JSONResponse:
    """Validate an ask request: a session with documents, a question, and an
    optional `doc_ids` list choosing which of the session's documents to search."""
    session = _get_session(request)
    if session is None:
        return _error("No document is loaded. Please upload a PDF first.", 400)

    body = await _json_body(request)
    question = body.get("question")
    question = question.strip() if isinstance(question, str) else ""
    if not question:
        return _error("Please enter a question.", 400)
    if len(question) > MAX_QUESTION_CHARS:
        return _error(f"Question is too long (max {MAX_QUESTION_CHARS} characters).", 400)

    doc_ids = body.get("doc_ids")
    with session.lock:
        if not session.docs:
            return _error("No document is loaded. Please upload a PDF first.", 400)
        if doc_ids is None:
            states = list(session.docs.values())
        else:
            if not isinstance(doc_ids, list):
                return _error("doc_ids must be a list.", 400)
            # Only ids from this session count: another session's ids simply don't match.
            states = [session.docs[d] for d in doc_ids if isinstance(d, str) and d in session.docs]
            if not states:
                return _error("Choose at least one of your documents to search.", 400)
    return session, question, states


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _documents(session: Session) -> list[dict]:
    with session.lock:
        return [
            {"id": doc_id, "filename": state.name, "num_pages": state.num_pages, "num_chunks": state.num_chunks}
            for doc_id, state in session.docs.items()
        ]


def _unique_name(session: Session, filename: str) -> str:
    """Filenames label sources in answers *and* select the document loader by
    extension, so a duplicate inserts its counter before the extension: two
    uploads of 'notes.pdf' become 'notes.pdf' and 'notes (2).pdf' (not
    'notes.pdf (2)', which would no longer end in .pdf)."""
    taken = {s.name for s in session.docs.values()}
    if filename not in taken:
        return filename
    stem, dot, ext = filename.rpartition(".")
    base = stem if dot else filename
    suffix = f".{ext}" if dot else ""
    n = 2
    while f"{base} ({n}){suffix}" in taken:
        n += 1
    return f"{base} ({n}){suffix}"


def _clean_filename(raw: str | None) -> str:
    """Basename only, no control characters, at most MAX_FILENAME_CHARS
    (keeping the extension). Filenames are shown in the UI and label passages
    in the prompt, so an attacker-chosen name must stay short and inert."""
    name = re.sub(r"[\x00-\x1f\x7f]", "", Path(raw or "").name).strip()
    if len(name) > MAX_FILENAME_CHARS:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: MAX_FILENAME_CHARS - len(ext) - 2] + "…" + dot + ext) if dot else name[:MAX_FILENAME_CHARS]
    return name


def _server_error(e: Exception, action: str, message: str, status_code: int) -> JSONResponse:
    """Log the real error with a short reference and return a generic message
    carrying the same reference, so a user's report can be matched to the log
    without exposing internals (CodeQL py/stack-trace-exposure)."""
    reference = secrets.token_hex(4)
    log.error("[%s] Error while %s: %r", reference, action, e)
    return _error(f"{message} (reference {reference})", status_code)


def _llm_error_response(e: Exception, action: str) -> JSONResponse:
    if isinstance(e, LLMConfigError):
        return _server_error(
            e, action, "The assistant is not configured correctly. Please contact the site administrator.", 503
        )
    if isinstance(e, LLMRequestError):
        return _server_error(e, action, "The LLM API request failed. Please try again in a moment.", 502)
    return _server_error(e, action, f"Something went wrong while {action}. Please try again.", 500)


@app.post("/api/ingest")
async def ingest(request: Request, file: UploadFile = File(...)):
    _prune_sessions()
    if limited := _rate_limited(request, "ingest"):
        return limited

    filename = _clean_filename(file.filename)
    if not document_loader.is_supported(filename):
        return _error("Please upload a PDF, Word (.docx), text or Markdown file.", 400)

    session = _get_session(request)
    if session is not None:
        with session.lock:
            if len(session.docs) >= MAX_DOCS_PER_SESSION:
                return _error(f"You can load up to {MAX_DOCS_PER_SESSION} documents at once. Remove one first.", 400)

    # Read at most one byte past the limit, so an oversized upload is never
    # held in memory in full.
    pdf_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    # Uploads over 1MB are spooled by Starlette to an anonymous temporary
    # file; close it now rather than at the end of the request, so no copy
    # of the file outlives this read. From here on only the extracted text is
    # kept, in memory, until the document is removed or the session expires.
    await file.close()
    if len(pdf_bytes) > MAX_UPLOAD_BYTES:
        return _error("File is too large. The limit is 25MB.", 413)
    # The extension is only a claim; verify a PDF really starts with "%PDF-"
    # within its first 1024 bytes (the spec allows leading junk). Text, Markdown
    # and docx have no single reliable magic byte, so they are validated by
    # whether any text can actually be extracted (below).
    if filename.lower().endswith(".pdf") and b"%PDF-" not in pdf_bytes[:1024]:
        return _error("That file isn't a PDF. Please upload a valid PDF file.", 400)

    if session is not None:
        with session.lock:
            name = _unique_name(session, filename)
    else:
        name = filename
    available_chunks = MAX_TOTAL_CHUNKS - _total_chunks()
    if available_chunks <= 0:
        return _error(
            "The server's document memory is full. Try again later or ask "
            "another user to free space by removing documents.",
            503,
        )
    # Pass the remaining aggregate budget into the pipeline so an upload that
    # cannot fit is rejected before embedding, not after allocating its index.
    max_chunks_for_upload = min(MAX_CHUNKS_PER_DOC, available_chunks)
    if not await _take_slot(_ingest_slots):
        return _busy()
    try:
        # Extraction + embedding is CPU-bound; running it in a worker thread
        # keeps the event loop free to serve other users meanwhile.
        index_state = await run_in_threadpool(
            pipeline.ingest, pdf_bytes, name=name, max_chunks=max_chunks_for_upload
        )
    except pipeline.DocumentTooLargeError:
        if max_chunks_for_upload < MAX_CHUNKS_PER_DOC:
            return _error(
                "The server's document memory is full for this upload. Remove "
                "another document or try a shorter file.",
                503,
            )
        return _error(
            "This document has too much text to process here (roughly 360 pages is the limit). "
            "Try a shorter document or split it.",
            413,
        )
    except Exception as e:
        # Previously this fell through to the "couldn't extract any text"
        # message, which misreported server faults as a bad PDF.
        return _server_error(e, "reading the document", "Something went wrong while reading this PDF.", 500)
    finally:
        _ingest_slots.release()

    if index_state is None:
        return _error(
            "Couldn't read any text from this file. It may be empty, a scan OCR "
            "couldn't make out, password-protected, or corrupted.",
            422,
        )

    # Look the session up again: it may have expired or been evicted while
    # this upload was being indexed.
    session = _get_session(request)
    is_new_session = session is None
    if is_new_session:
        session_id = secrets.token_urlsafe(32)
        session = Session()

    # Re-check the aggregate budget now that this document's real size is
    # known (other uploads may have landed while it was being indexed). A new
    # session is registered only after the document is added, so a refused
    # upload never leaves an empty orphan session behind.
    if _total_chunks() + index_state.num_chunks > MAX_TOTAL_CHUNKS:
        return _error(
            "The server's document memory is full. Try again later or ask "
            "another user to free space by removing documents.",
            503,
        )

    doc_id = secrets.token_urlsafe(8)
    with session.lock:
        if len(session.docs) >= MAX_DOCS_PER_SESSION:
            return _error(f"You can load up to {MAX_DOCS_PER_SESSION} documents at once. Remove one first.", 400)
        session.docs[doc_id] = index_state
    if is_new_session:
        _sessions[session_id] = session
        _prune_sessions()

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
            _cookie_name(request),
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
    with session.lock:
        history = list(session.history)
    return {"documents": _documents(session), "history": history}


def _turn(question: str, answer: str, by: dict | None) -> dict:
    """One conversation turn as stored in the session (and restored on reload)."""
    turn = {"question": question, "answer": answer}
    if by:
        turn["answered_by"] = by
    return turn


@app.get("/api/status")
async def status():
    """The AI providers this server can use and whether each is usable right
    now, for the footer. Names and models only -- never keys or URLs."""
    return {"version": APP_VERSION, "providers": providers.status()}


@app.get("/api/health")
async def health():
    """Liveness check for the container (no LLM call)."""
    return {"ok": True}


@app.post("/api/ask")
async def ask(request: Request):
    """Answer as one JSON response (the UI uses /api/ask/stream)."""
    _prune_sessions()
    if limited := _rate_limited(request, "llm"):
        return limited
    checked = await _question_request(request)
    if isinstance(checked, JSONResponse):
        return checked
    session, question, states = checked

    if not await _take_slot(_llm_slots):
        return _busy()
    try:
        with session.lock:
            history = list(session.history)
        result = await run_in_threadpool(pipeline.answer, question, states, history=history)
    except Exception as e:
        return _llm_error_response(e, "answering that question")
    finally:
        _llm_slots.release()

    by = answered_by(result["answer"])
    with session.lock:
        session.history.append(_turn(question, result["answer"], by))
        del session.history[:-MAX_STORED_TURNS]
    return {"answer": result["answer"], "sources": result["sources"], "answered_by": by}


@app.post("/api/ask/stream")
async def ask_stream(request: Request):
    """Answer as server-sent events: `sources` first, then `reasoning` events
    (the model's thinking, if it exposes any), a `route` event naming the
    provider and model, `token` events as the answer is written, then `done`
    (or `error`).

    Retrieval and the first piece of the answer are fetched *before* the
    response starts, so a missing API key, a rate limit or a provider failure
    still produces a proper HTTP status and message instead of a stream that
    breaks halfway. If the client disconnects (the Stop button), the turn is
    not saved to the conversation history.
    """
    _prune_sessions()
    if limited := _rate_limited(request, "llm"):
        return limited
    checked = await _question_request(request)
    if isinstance(checked, JSONResponse):
        return checked
    session, question, states = checked

    if not await _take_slot(_llm_slots):
        return _busy()
    pieces = None
    try:
        with session.lock:
            history = list(session.history)
        sources, pieces = await run_in_threadpool(
            pipeline.answer_stream, question, states, history=history
        )
        # (kind, text) or None; pulling the first piece surfaces early errors.
        first = await run_in_threadpool(next, pieces, None)
    except Exception as e:
        _llm_slots.release()
        if pieces is not None:
            try:
                pieces.close()
            except (ValueError, StopIteration):
                pass
        return _llm_error_response(e, "answering that question")

    async def events():
        parts = []  # answer content only; reasoning is shown but never stored
        route = {}
        completed = False
        try:
            yield _sse("sources", sources)

            async def stream():
                if first is not None:
                    yield first
                async for piece in iterate_in_threadpool(pieces):
                    yield piece

            async for kind, text in stream():
                if kind == "reasoning":
                    yield _sse("reasoning", {"text": text})
                    continue
                if not route:
                    # The route rides on the first content piece (see llm_client).
                    route.update(answered_by(text) or {})
                    if route:
                        yield _sse("route", route)
                parts.append(text)
                yield _sse("token", {"text": text})
            completed = True
            answer = "".join(parts)
            with session.lock:
                session.history.append(_turn(question, answer, route or None))
                del session.history[:-MAX_STORED_TURNS]
            yield _sse("done", {})
        except Exception as e:
            reference = secrets.token_hex(4)
            log.error("[%s] Error while streaming an answer: %r", reference, e)
            yield _sse("error", {"error": f"The answer was interrupted. Please try again. (reference {reference})"})
        finally:
            if not completed:
                try:
                    pieces.close()  # stop reading from the provider
                except ValueError:
                    pass  # still running in its worker thread; it ends on its own
            _llm_slots.release()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # X-Accel-Buffering stops Nginx from holding the stream back until it ends.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/summary")
async def summary(request: Request):
    if limited := _rate_limited(request, "llm"):
        return limited
    session = _get_session(request)
    if session is None:
        return _error("No document is loaded. Please upload a PDF first.", 400)

    body = await _json_body(request)
    with session.lock:
        if not session.docs:
            return _error("No document is loaded. Please upload a PDF first.", 400)
        index_state = session.docs.get(body.get("id"))
    if index_state is None:
        return _error("That document is not loaded.", 404)

    if not await _take_slot(_llm_slots):
        return _busy()
    try:
        result = await run_in_threadpool(pipeline.summarize, index_state)
    except Exception as e:
        return _llm_error_response(e, "summarizing the document")
    finally:
        _llm_slots.release()
    return {
        "filename": index_state.name,
        "summary": result["summary"],
        "sources": result["sources"],
        "answered_by": answered_by(result["summary"]),
    }


@app.post("/api/suggestions")
async def suggestions(request: Request):
    """Starter questions for a document, written by the LLM from a sample of it."""
    if limited := _rate_limited(request, "llm"):
        return limited
    session = _get_session(request)
    if session is None:
        return _error("No document is loaded. Please upload a PDF first.", 400)

    body = await _json_body(request)
    with session.lock:
        if not session.docs:
            return _error("No document is loaded. Please upload a PDF first.", 400)
        index_state = session.docs.get(body.get("id"))
    if index_state is None:
        return _error("That document is not loaded.", 404)

    if not await _take_slot(_llm_slots):
        return _busy()
    try:
        questions = await run_in_threadpool(pipeline.suggest_questions, index_state)
    except Exception as e:
        return _llm_error_response(e, "suggesting questions")
    finally:
        _llm_slots.release()
    return {"questions": questions}


@app.post("/api/remove")
async def remove(request: Request):
    """Remove one document (body {"id": ...}) or, with no id, everything."""
    session_id = request.cookies.get(_cookie_name(request), "")
    session = _sessions.get(session_id)
    doc_id = (await _json_body(request)).get("id")

    if session is not None and doc_id is not None:
        with session.lock:
            session.docs.pop(doc_id, None)
            has_documents = bool(session.docs)
        if has_documents:
            return {"ok": True, "documents": _documents(session)}

    _sessions.pop(session_id, None)
    response = JSONResponse({"ok": True, "documents": []})
    response.delete_cookie(_cookie_name(request), secure=_is_https(request), httponly=True, samesite="lax")
    return response


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")
