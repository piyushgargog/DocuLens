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
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.concurrency import iterate_in_threadpool, run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import auth
import docstore
import document_loader
import embedder
import observability
import pipeline
import providers
import store
from observability import log, new_request_id, request_id_var

observability.setup_logging()
from llm_client import LLMConfigError, LLMRequestError, answered_by

load_dotenv()

STATIC_DIR = Path(__file__).parent / "static"

# The release version. static/index.html repeats it (asset ?v= query, footer,
# release link) and tests/test_api.py fails if the two ever disagree.
APP_VERSION = "3.8.0"

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
    "login": [(20, 10 * 60)],  # /api/login (each hit may fetch Google's signing certificates)
}

# Login sessions and rate-limit counters live here: Redis when
# REDIS_URL is set, in-process otherwise (see store.py). Documents do not.
_store = store.make_store()

# CIDR networks whose X-Real-IP header is trusted for rate limiting.
# Default: loopback only (safe for Nginx on the same host). Behind a
# Docker bridge or external reverse proxy, set TRUSTED_PROXIES to a
# comma-separated list of CIDRs, e.g. "172.17.0.0/16,10.0.0.0/8".
# Never include 0.0.0.0/0: any client could then spoof their IP.
_TRUSTED_PROXY_NETS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []

def _parse_trusted_proxies() -> list:
    raw = os.environ.get("TRUSTED_PROXIES", "127.0.0.0/8,::1/128").strip()
    nets = []
    for position, cidr in enumerate(raw.split(","), start=1):
        cidr = cidr.strip()
        if not cidr:
            continue
        try:
            nets.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            # Log the position only, never the configured text.
            log.warning("Ignoring invalid TRUSTED_PROXIES entry #%d", position)
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

def _security_headers() -> dict:
    """SECURITY_HEADERS, widened only as far as Firebase sign-in needs when it is
    configured: the SDK is served from this origin, but it talks to Google's
    identity APIs, loads Google's gapi script and opens the sign-in popup (which
    COOP `same-origin` would sever, hence `same-origin-allow-popups`)."""
    if not auth.enabled():
        return SECURITY_HEADERS
    domain = auth.web_config()["authDomain"]
    csp = SECURITY_HEADERS["Content-Security-Policy"]
    csp = csp.replace("script-src 'self'", "script-src 'self' https://apis.google.com")
    csp = csp.replace(
        "connect-src 'self'",
        "connect-src 'self' https://identitytoolkit.googleapis.com https://securetoken.googleapis.com",
    )
    csp += f"; frame-src https://{domain} https://accounts.google.com"
    return {**SECURITY_HEADERS, "Content-Security-Policy": csp, "Cross-Origin-Opener-Policy": "same-origin-allow-popups"}


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
        await _store.sweep()


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
    log.info(
        "State store: %s; Firebase sign-in %s",
        "Redis" if isinstance(_store, store.ResilientStore) else "in-process memory",
        "on (guest limits apply)" if auth.enabled() else "off (no tiers)",
    )
    sweeper = asyncio.create_task(_sweep_expired_sessions())
    try:
        yield
    finally:
        sweeper.cancel()
        await _store.close()


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
        _set_cookie(response, cookie_name, session_id, request, SESSION_TTL_SECONDS)
    if await _current_user(request):
        login_cookie = _auth_cookie_name(request)
        _set_cookie(response, login_cookie, request.cookies[login_cookie], request, auth.LOGIN_TTL)
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
    response.headers.update(_security_headers())
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
    """One workspace: a guest's browser session, or a signed-in user's
    persisted documents (`owner` set).

    `docs` holds the *loaded* indexes. A guest's documents live only here. A
    signed-in user's are stored durably (docstore.py) and described by `meta`;
    `docs` is then just a cache that is filled lazily and may be emptied under
    memory pressure or after a restart."""

    docs: dict[str, pipeline.IndexState] = field(default_factory=dict)  # doc_id -> index
    history: list[dict] = field(default_factory=list)  # guests only; users' history is in the repository
    last_used: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    owner: str | None = None  # docstore namespace for a signed-in user; None for a guest
    meta: dict[str, docstore.DocMeta] = field(default_factory=dict)
    load_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


# Guest sessions, in process memory keyed by a random cookie value: a guest's
# single document is deliberately ephemeral (lost on restart, not shared).
_sessions: dict[str, Session] = {}
# Signed-in users' workspaces, keyed by their docstore owner namespace. Only a
# cache: the documents themselves are persisted and re-synced on every request.
_user_sessions: dict[str, Session] = {}


def _prune_sessions() -> None:
    now = time.time()
    for table in (_sessions, _user_sessions):
        expired = []
        for sid, session in list(table.items()):
            with session.lock:
                if now - session.last_used > SESSION_TTL_SECONDS:
                    expired.append(sid)
        for sid in expired:
            table.pop(sid, None)
        while len(table) > MAX_SESSIONS:
            oldest = min(table, key=lambda sid: table[sid].last_used)
            table.pop(oldest, None)


def _total_chunks() -> int:
    """Chunks currently held in memory across all sessions."""
    total = 0
    for session in [*_sessions.values(), *_user_sessions.values()]:
        with session.lock:
            total += sum(state.num_chunks for state in session.docs.values())
    return total


def _make_room(needed: int, keep: Session) -> bool:
    """Free memory for `needed` chunks by dropping the loaded indexes of other
    signed-in users, least recently used first. Safe because theirs are
    persisted and reload on demand; a guest's are never touched."""
    free = MAX_TOTAL_CHUNKS - _total_chunks()
    for other in sorted((u for u in _user_sessions.values() if u is not keep), key=lambda u: u.last_used):
        if free >= needed:
            break
        with other.lock:
            free += sum(state.num_chunks for state in other.docs.values())
            other.docs.clear()
    return free >= needed


def _repo() -> docstore.DocumentRepository:
    return docstore.DocumentRepository(_store)


async def _user_session(uid: str) -> Session:
    """The signed-in user's workspace, synced with what is stored right now:
    documents added or removed through another instance appear or disappear."""
    owner = docstore.owner_key(uid)
    metas = await _repo().list_docs(owner)
    session = _user_sessions.get(owner)
    if session is None:
        session = _user_sessions[owner] = Session(owner=owner)
        _prune_sessions()
    with session.lock:
        session.last_used = time.time()
        session.meta = metas
        for doc_id in [d for d in session.docs if d not in metas]:
            del session.docs[doc_id]
    return session


def _doc_count(session: Session) -> int:
    with session.lock:
        return len(session.meta) if session.owner else len(session.docs)


def _doc_ids(session: Session) -> list[str]:
    with session.lock:
        return list(session.meta) if session.owner else list(session.docs)


class StorageUnavailable(Exception):
    """Saved documents can't be read right now (the store is degraded)."""


class MemoryFull(Exception):
    """No room to load another document into memory."""


async def _load_states(session: Session, doc_ids: list[str]) -> tuple[list[pipeline.IndexState], list[str]]:
    """The indexes for `doc_ids` (ids this workspace does not own are ignored)
    and the names of any saved documents that could not be restored.

    A guest's are already in memory. A signed-in user's are loaded from storage
    on first use; a document whose artifacts are missing or corrupt is removed
    (it cannot be rebuilt: the upload itself is never kept) and reported."""
    if not session.owner:
        with session.lock:
            return [session.docs[d] for d in doc_ids if d in session.docs], []
    states, lost = [], []
    async with session.load_lock:
        for doc_id in doc_ids:
            with session.lock:
                state, meta = session.docs.get(doc_id), session.meta.get(doc_id)
            if meta is None:
                continue
            if state is None:
                if not _make_room(meta.num_chunks, keep=session):
                    raise MemoryFull()
                try:
                    loaded = await _repo().load(session.owner, meta)
                except (docstore.DocumentMissingError, docstore.DocumentCorruptError) as e:
                    if getattr(_store, "degraded", False):
                        raise StorageUnavailable() from e  # not proof the document is gone
                    log.warning("Removing unrecoverable document (%s)", type(e).__name__)
                    await _repo().delete(session.owner, doc_id)
                    with session.lock:
                        session.meta.pop(doc_id, None)
                        session.docs.pop(doc_id, None)
                    lost.append(meta.name)
                    continue
                state = await run_in_threadpool(
                    pipeline.restore,
                    loaded.chunks,
                    loaded.embeddings,
                    meta.name,
                    meta.num_pages,
                    meta.chunk_size,
                    meta.chunk_overlap,
                    loaded.index,
                )
                with session.lock:
                    session.docs[doc_id] = state
            states.append(state)
    return states, lost


def _restore_problem(e: Exception | None, lost: list[str]) -> JSONResponse | None:
    """The HTTP error for a failed or partial restore, or None if all is well."""
    if isinstance(e, MemoryFull):
        return _error(
            "The server's document memory is full. Try again in a moment.",
            503,
        )
    if isinstance(e, StorageUnavailable):
        return _error("Your saved documents are temporarily unavailable. Please try again shortly.", 503)
    if lost:
        names = ", ".join(lost)[:200]
        return _error(
            f"These saved documents could no longer be restored and were removed: {names}. Please upload them again.",
            409,
        )
    return None


async def _history(session: Session) -> list[dict]:
    if session.owner:
        return await _repo().get_history(session.owner)
    with session.lock:
        return list(session.history)


async def _record_turn(session: Session, turn: dict, history: list[dict] | None = None) -> None:
    """Add a turn to the conversation. `history` is the list already read for this
    request, so a signed-in user's write is one round trip, not read-then-write."""
    if session.owner:
        turns = list(history) if history is not None else await _repo().get_history(session.owner)
        turns.append(turn)
        await _repo().set_history(session.owner, turns[-MAX_STORED_TURNS:])
        return
    with session.lock:
        session.history.append(turn)
        del session.history[:-MAX_STORED_TURNS]


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


async def _rate_limited(
    request: Request, bucket: str, limits: list[tuple[int, int]] | None = None, message: str | None = None
) -> JSONResponse | None:
    """A 429 response if this client is over any of the bucket's limits,
    otherwise None (and the request is counted). Signed-in users are counted
    per account, everyone else per IP address."""
    who = await _current_user(request)
    identity = f"u:{who['sub']}" if who else f"ip:{_client_ip(request)}"
    retry_after = await _store.rate_hit(f"rl:{identity}:{bucket}", limits or RATE_LIMITS[bucket])
    if not retry_after:
        return None
    wait = f"about {max(1, round(retry_after / 60))} minute(s)" if retry_after < 3600 else f"about {round(retry_after / 3600)} hour(s)"
    response = _error(message or f"Too many requests. Please wait {wait} and try again.", 429)
    response.headers["Retry-After"] = str(retry_after)
    return response


def _cookie_name(request: Request) -> str:
    """Over HTTPS the session cookie uses the __Host- prefix: browsers then
    only accept it if it is Secure, has Path=/ and no Domain, so no subdomain
    or plain-HTTP response can set or overwrite it. (Plain-HTTP local runs
    can't use the prefix.)"""
    return "__Host-session" if _is_https(request) else "session_id"


async def _get_session(request: Request) -> Session | None:
    """The caller's workspace: a signed-in user's persisted one, otherwise the
    guest session named by the cookie (None if there isn't one yet)."""
    user = await _current_user(request)
    if user:
        return await _user_session(user["sub"])
    session = _sessions.get(request.cookies.get(_cookie_name(request), ""))
    if session is not None:
        with session.lock:
            session.last_used = time.time()
    return session


_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{16,64}")


def _auth_cookie_name(request: Request) -> str:
    return "__Host-auth" if _is_https(request) else "auth_id"


def _set_cookie(response, name: str, value: str, request: Request, max_age: int) -> None:
    response.set_cookie(name, value, httponly=True, samesite="lax", secure=_is_https(request), max_age=max_age)


def _drop_cookie(response, name: str, request: Request) -> None:
    response.delete_cookie(name, secure=_is_https(request), httponly=True, samesite="lax")


async def _current_user(request: Request) -> dict | None:
    """The signed-in user ({"sub", "email", "name", "t"}) or None. Looked up once
    per request; an in-use login is renewed at most once an hour."""
    if not auth.enabled():
        return None
    if hasattr(request.state, "user"):
        return request.state.user
    sid = request.cookies.get(_auth_cookie_name(request), "")
    user = None
    if _TOKEN_RE.fullmatch(sid):
        user = await _store.get_json(f"login:{sid}")
        if user and time.time() - user.get("t", 0) > auth.LOGIN_REFRESH:
            user["t"] = time.time()
            await _store.set_json(f"login:{sid}", user, auth.LOGIN_TTL)
    request.state.user = user
    return user


async def _tier(request: Request) -> auth.Tier | None:
    """The caller's limits, or None when sign-in is not configured (no tiers)."""
    if not auth.enabled():
        return None
    return auth.user_tier(MAX_DOCS_PER_SESSION) if await _current_user(request) else auth.guest_tier()


async def _quota_limited(request: Request, tier: auth.Tier | None, kind: str) -> JSONResponse | None:
    """Daily cap on questions ("ask") or uploads ("upload") for the caller's tier."""
    if tier is None:
        return None
    per_day = tier.ask_per_day if kind == "ask" else tier.uploads_per_day
    noun = "questions" if kind == "ask" else "uploads"
    if tier.name == "guest":
        message = f"Guest limit reached ({per_day} {noun} per day). Sign in with Google to keep going."
    else:
        message = f"Daily limit reached ({per_day} {noun}). Please try again tomorrow."
    return await _rate_limited(request, f"{kind}_daily", [(per_day, auth.DAY)], message)


def _doc_limit_message(tier: auth.Tier | None, max_docs: int) -> str:
    if tier is not None and tier.name == "guest":
        return (
            f"Guests can load {max_docs} document{'s' if max_docs != 1 else ''} at a time. "
            f"Remove it, or sign in with Google to load up to {MAX_DOCS_PER_SESSION}."
        )
    return f"You can load up to {max_docs} documents at once. Remove one first."


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


async def _question_request(request: Request, session: Session | None) -> tuple[Session, str, list] | JSONResponse:
    """Validate an ask request: a session with documents, a question, and an
    optional `doc_ids` list choosing which of the session's documents to search."""
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
    known = _doc_ids(session)
    if not known:
        return _error("No document is loaded. Please upload a PDF first.", 400)
    if doc_ids is None:
        wanted = known
    else:
        if not isinstance(doc_ids, list):
            return _error("doc_ids must be a list.", 400)
        # Only ids from this workspace count: another user's ids simply don't match.
        wanted = [d for d in doc_ids if isinstance(d, str) and d in known]
        if not wanted:
            return _error("Choose at least one of your documents to search.", 400)
    try:
        states, lost = await _load_states(session, wanted)
    except (MemoryFull, StorageUnavailable) as e:
        return _restore_problem(e, [])
    if problem := _restore_problem(None, lost):
        return problem
    return session, question, states


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _documents(session: Session) -> list[dict]:
    with session.lock:
        if session.owner:
            return [
                {"id": doc_id, "filename": m.name, "num_pages": m.num_pages, "num_chunks": m.num_chunks}
                for doc_id, m in session.meta.items()
            ]
        return [
            {"id": doc_id, "filename": state.name, "num_pages": state.num_pages, "num_chunks": state.num_chunks}
            for doc_id, state in session.docs.items()
        ]


def _unique_name(session: Session, filename: str) -> str:
    """Filenames label sources in answers *and* select the document loader by
    extension, so a duplicate inserts its counter before the extension: two
    uploads of 'notes.pdf' become 'notes.pdf' and 'notes (2).pdf' (not
    'notes.pdf (2)', which would no longer end in .pdf)."""
    taken = {d["filename"] for d in _documents(session)}
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
    if limited := await _rate_limited(request, "ingest"):
        return limited
    tier = await _tier(request)
    max_docs = tier.max_docs if tier else MAX_DOCS_PER_SESSION

    filename = _clean_filename(file.filename)
    if not document_loader.is_supported(filename):
        return _error("Please upload a PDF, Word (.docx), text or Markdown file.", 400)

    session = await _get_session(request)
    if session is not None and _doc_count(session) >= max_docs:
        return _error(_doc_limit_message(tier, max_docs), 400)
    if limited := await _quota_limited(request, tier, "upload"):
        return limited

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
    session = await _get_session(request)
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
    if _doc_count(session) >= max_docs:
        return _error(_doc_limit_message(tier, max_docs), 400)
    saved = None
    if session.owner:
        # A signed-in user's document is stored durably before it counts as
        # uploaded; packing (compression) is CPU work, so it runs in a thread.
        try:
            packed = await run_in_threadpool(
                docstore.pack,
                doc_id,
                index_state.name,
                index_state.num_pages,
                index_state.chunk_size,
                index_state.chunk_overlap,
                index_state.store.chunks,
                index_state.store.embeddings,
                None,
                index_state.store.index,
            )
            saved = await _repo().save(session.owner, packed, existing=session.meta)
        except docstore.StorageQuotaError:
            return _error(
                f"Your saved-documents storage is full ({docstore.max_user_bytes() // (1024 * 1024)} MB). "
                "Remove a document first.",
                413,
            )
        except Exception as e:
            return _server_error(e, "saving the document", "Something went wrong while saving this document.", 500)
    with session.lock:
        session.docs[doc_id] = index_state
        if saved is not None:
            session.meta[doc_id] = saved
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
            "warnings": index_state.warnings[:5],
        }
    )
    if is_new_session:
        _set_cookie(response, _cookie_name(request), session_id, request, SESSION_TTL_SECONDS)
    return response


@app.get("/api/session")
async def get_session(request: Request):
    """Current documents and conversation, so a page reload can restore the UI."""
    session = await _get_session(request)
    if session is None:
        return {"documents": [], "history": []}
    return {"documents": _documents(session), "history": await _history(session)}


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
    await _current_user(request)  # one lookup, cached for the calls below
    limited, session = await asyncio.gather(_rate_limited(request, "llm"), _get_session(request))
    if limited:
        return limited
    checked = await _question_request(request, session)
    if isinstance(checked, JSONResponse):
        return checked
    session, question, states = checked
    quota, history = await asyncio.gather(_quota_limited(request, await _tier(request), "ask"), _history(session))
    if quota:
        return quota

    if not await _take_slot(_llm_slots):
        return _busy()
    try:
        result = await run_in_threadpool(pipeline.answer, question, states, history=history)
    except Exception as e:
        return _llm_error_response(e, "answering that question")
    finally:
        _llm_slots.release()

    by = answered_by(result["answer"])
    await _record_turn(session, _turn(question, result["answer"], by), history)
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
    await _current_user(request)  # one lookup, cached for the calls below
    limited, session = await asyncio.gather(_rate_limited(request, "llm"), _get_session(request))
    if limited:
        return limited
    checked = await _question_request(request, session)
    if isinstance(checked, JSONResponse):
        return checked
    session, question, states = checked
    quota, history = await asyncio.gather(_quota_limited(request, await _tier(request), "ask"), _history(session))
    if quota:
        return quota

    if not await _take_slot(_llm_slots):
        return _busy()
    pieces = None
    try:
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
            yield _sse("done", {})  # the answer is complete: tell the browser first,
            await _record_turn(session, _turn(question, answer, route or None), history)  # then save the turn
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


async def _one_document(session: Session, doc_id) -> pipeline.IndexState | JSONResponse:
    """The loaded index for one document id, or the error response to send."""
    if not isinstance(doc_id, str) or doc_id not in _doc_ids(session):
        return _error("That document is not loaded.", 404)
    try:
        states, lost = await _load_states(session, [doc_id])
    except (MemoryFull, StorageUnavailable) as e:
        return _restore_problem(e, [])
    if problem := _restore_problem(None, lost):
        return problem
    return states[0]


@app.post("/api/summary")
async def summary(request: Request):
    if limited := await _rate_limited(request, "llm"):
        return limited
    session = await _get_session(request)
    if session is None or not _doc_count(session):
        return _error("No document is loaded. Please upload a PDF first.", 400)

    doc_id = (await _json_body(request)).get("id")
    index_state = await _one_document(session, doc_id)
    if isinstance(index_state, JSONResponse):
        return index_state

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
        "coverage": result.get("coverage"),
        "answered_by": answered_by(result["summary"]),
    }


@app.post("/api/suggestions")
async def suggestions(request: Request):
    """Starter questions for a document, written by the LLM from a sample of it."""
    if limited := await _rate_limited(request, "llm"):
        return limited
    session = await _get_session(request)
    if session is None or not _doc_count(session):
        return _error("No document is loaded. Please upload a PDF first.", 400)

    doc_id = (await _json_body(request)).get("id")
    index_state = await _one_document(session, doc_id)
    if isinstance(index_state, JSONResponse):
        return index_state

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
    """Remove one document (body {"id": ...}) or, with no id, everything.
    For a signed-in user this deletes the stored copy too: metadata, both
    artifacts and (once nothing is left) the conversation."""
    doc_id = (await _json_body(request)).get("id")

    user = await _current_user(request)
    if user:
        session = await _user_session(user["sub"])
        if doc_id is None:
            await _repo().delete_all(session.owner)
            with session.lock:
                session.meta.clear()
                session.docs.clear()
        elif isinstance(doc_id, str) and doc_id in _doc_ids(session):
            await _repo().delete(session.owner, doc_id)
            with session.lock:
                session.meta.pop(doc_id, None)
                session.docs.pop(doc_id, None)
            if not _doc_count(session):
                await _repo().clear_history(session.owner)
        return {"ok": True, "documents": _documents(session)}

    session_id = request.cookies.get(_cookie_name(request), "")
    session = _sessions.get(session_id)

    if session is not None and doc_id is not None:
        with session.lock:
            session.docs.pop(doc_id, None)
            has_documents = bool(session.docs)
        if has_documents:
            return {"ok": True, "documents": _documents(session)}

    _sessions.pop(session_id, None)
    response = JSONResponse({"ok": True, "documents": []})
    _drop_cookie(response, _cookie_name(request), request)
    return response


async def _adopt_guest_documents(request: Request, uid: str) -> bool:
    """On sign-in, move the browser's guest documents (and conversation) into
    the user's saved workspace so nothing uploaded before signing in is lost.
    Returns True if there was a guest session to retire."""
    guest = _sessions.get(request.cookies.get(_cookie_name(request), ""))
    if guest is None:
        return False
    session = await _user_session(uid)
    with guest.lock:
        items = list(guest.docs.items())
        turns = list(guest.history)
    for doc_id, state in items:
        if _doc_count(session) >= MAX_DOCS_PER_SESSION:
            break
        try:
            packed = await run_in_threadpool(
                docstore.pack,
                doc_id,
                state.name,
                state.num_pages,
                state.chunk_size,
                state.chunk_overlap,
                state.store.chunks,
                state.store.embeddings,
                None,
                state.store.index,
            )
            saved = await _repo().save(session.owner, packed, existing=session.meta)
        except Exception as e:  # over quota, storage trouble: keep going with the rest
            log.warning("Could not keep a guest document on sign-in (%s)", type(e).__name__)
            continue
        with session.lock:
            session.docs[doc_id] = state
            session.meta[doc_id] = saved
    if turns and _doc_count(session):
        await _repo().set_history(session.owner, turns)
    _sessions.pop(request.cookies.get(_cookie_name(request), ""), None)
    return True


@app.post("/api/login")
async def login(request: Request):
    """Exchange a Firebase ID token (from the browser's Google sign-in) for our
    own login session. The token is verified, used once, and not kept."""
    if not auth.enabled():
        return _error("Sign-in is not configured on this server.", 503)
    if limited := await _rate_limited(request, "login"):
        return limited
    token = (await _json_body(request)).get("id_token")
    try:
        profile = await run_in_threadpool(auth.verify_id_token, token)
    except auth.AuthError as e:
        log.warning("Sign-in rejected: %s", e)
        return _error("Couldn't sign you in. Please try again.", 401)
    sid = secrets.token_urlsafe(32)
    await _store.set_json(f"login:{sid}", {**profile, "t": time.time()}, auth.LOGIN_TTL)
    log.info("User signed in")
    adopted = await _adopt_guest_documents(request, profile["sub"])
    response = JSONResponse({"ok": True, "user": {"name": profile["name"], "email": profile["email"]}})
    _set_cookie(response, _auth_cookie_name(request), sid, request, auth.LOGIN_TTL)
    if adopted:
        _drop_cookie(response, _cookie_name(request), request)
    return response


@app.get("/api/me")
async def me(request: Request):
    """Who is signed in and what limits apply, for the header and guest notice."""
    user = await _current_user(request)
    tier = await _tier(request)
    return {
        "auth_enabled": auth.enabled(),
        "firebase": auth.web_config() if auth.enabled() else None,
        "user": {"name": user["name"], "email": user["email"]} if user else None,
        "limits": (
            {"tier": tier.name, "max_docs": tier.max_docs, "questions_per_day": tier.ask_per_day}
            if tier
            else None
        ),
    }


@app.post("/api/logout")
async def logout(request: Request):
    """Sign out: end the login and drop everything this process holds for the
    browser (the user's cached indexes and any guest session). The user's saved
    documents stay in storage, reachable only by signing in again."""
    user = await _current_user(request)
    if user:
        _user_sessions.pop(docstore.owner_key(user["sub"]), None)
    sid = request.cookies.get(_auth_cookie_name(request), "")
    if _TOKEN_RE.fullmatch(sid):
        await _store.delete(f"login:{sid}")
    request.state.user = None
    _sessions.pop(request.cookies.get(_cookie_name(request), ""), None)
    response = JSONResponse({"ok": True})
    _drop_cookie(response, _auth_cookie_name(request), request)
    _drop_cookie(response, _cookie_name(request), request)
    return response


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")
