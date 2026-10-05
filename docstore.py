"""Persistent, per-user document storage.

What is kept for a signed-in user's document, and where:

* **Metadata** (name, pages, chunks, size, digest, expiry) -- one JSON field in
  the Redis hash `u:<owner>:docs`, one field per document, so two instances
  adding documents at once never overwrite each other.
* **Artifacts** -- the extracted chunks (JSON, zlib) and the embeddings
  (float16) as two blobs behind an `ArtifactStore`: `StoreArtifacts` (Redis,
  split into small parts) or `DiskArtifacts` (a directory). The vector index
  itself is *rebuilt* from the embeddings when a document is loaded; FAISS
  objects are never pushed into Redis.
* **History** -- the last few conversation turns, `u:<owner>:hist`.

`<owner>` is a SHA-256 of the verified Firebase uid, never a value the browser
sends, so every key is namespaced by the authenticated user and no lookup can
cross users. Everything expires after DOC_RETENTION_DAYS (default 30).

This module knows nothing about FastAPI, FAISS or the pipeline: it moves
(chunks, embeddings) in and out. A document is written artifacts-first and its
metadata last, so a half-written upload is never listed; metadata that outlives
its artifacts (expiry, eviction, corruption) is detected on load and cleaned up.
"""

import asyncio
import hashlib
import hmac
import json
import os
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np

import vector_index
from observability import log

FORMAT_VERSION = 1
BLOB_PART_BYTES = 400_000  # Upstash caps a request at 1 MB; stay well under
MAX_UNPACKED_BYTES = 64 * 1024 * 1024  # decompression-bomb guard for the chunk blob
MAX_HISTORY_TURNS_STORED = 10


class DocumentMissingError(Exception):
    """The document's metadata exists but its artifacts are gone."""


class DocumentCorruptError(Exception):
    """The stored artifacts do not match their metadata."""


class StorageQuotaError(Exception):
    """Saving this document would exceed the user's storage allowance."""


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


def retention_seconds() -> int:
    return _env_int("DOC_RETENTION_DAYS", 30) * 24 * 3600


def max_user_bytes() -> int:
    return _env_int("USER_STORAGE_MB", 20) * 1024 * 1024


def state_key() -> bytes:
    """Secret used to sign saved index blobs (DOCULENS_STATE_KEY); empty = off."""
    return os.environ.get("DOCULENS_STATE_KEY", "").strip().encode("utf-8")


def _index_signature(blob: bytes) -> str:
    return hmac.new(state_key(), blob, hashlib.sha256).hexdigest()


def owner_key(uid: str) -> str:
    """The namespace for one user's keys, derived from the verified uid."""
    return hashlib.sha256(uid.encode("utf-8")).hexdigest()[:40]


# ---------- metadata ----------


@dataclass
class DocMeta:
    id: str
    name: str
    num_pages: int
    num_chunks: int
    chunk_size: int
    chunk_overlap: int
    dim: int
    size_bytes: int
    digest: str
    created: float
    expires: float
    version: int = FORMAT_VERSION
    index_kind: str = "flat"  # a non-flat index (HNSW) is also saved, as its own blob
    index_digest: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw) -> "DocMeta":
        """Strict: anything malformed raises ValueError, so a damaged metadata
        entry is dropped instead of crashing a request."""
        if not isinstance(raw, dict):
            raise ValueError("metadata is not an object")
        try:
            meta = cls(**raw)
        except TypeError as e:
            raise ValueError(str(e)) from e
        ints = (meta.num_pages, meta.num_chunks, meta.chunk_size, meta.chunk_overlap, meta.dim, meta.size_bytes)
        if not (isinstance(meta.id, str) and isinstance(meta.name, str) and isinstance(meta.digest, str)):
            raise ValueError("bad metadata strings")
        if not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in ints):
            raise ValueError("bad metadata numbers")
        if meta.version != FORMAT_VERSION:
            raise ValueError(f"unsupported format version {meta.version}")
        if meta.index_kind not in vector_index.KINDS or not isinstance(meta.index_digest, str):
            raise ValueError("bad index metadata")
        return meta


@dataclass
class Packed:
    chunks_blob: bytes
    emb_blob: bytes
    meta: DocMeta
    index_blob: bytes | None = None  # only for approximate indexes; a flat one is rebuilt for free


class Loaded(NamedTuple):
    chunks: list[dict]
    embeddings: np.ndarray
    saved_index: "vector_index.VectorIndex | None"  # None: build one from the embeddings


def pack(
    doc_id: str,
    name: str,
    num_pages: int,
    chunk_size: int,
    chunk_overlap: int,
    chunks: list[dict],
    embeddings: np.ndarray,
    ttl: int | None = None,
    index: "vector_index.VectorIndex | None" = None,
) -> Packed:
    """CPU-bound (call in a worker thread): serialise a document's chunks and
    embeddings. Embeddings are stored as float16 (half the space; cosine
    ranking is unaffected at that precision) and renormalised on load."""
    if len(chunks) != embeddings.shape[0]:
        raise ValueError("chunks and embeddings must be the same length")
    chunks_blob = zlib.compress(json.dumps(chunks, ensure_ascii=False).encode("utf-8"), 6)
    emb_blob = embeddings.astype("float16").tobytes()
    # A saved HNSW graph is deserialised by FAISS, which is not hardened against
    # hostile bytes, so it is stored only when a signing key is configured and is
    # loaded only if its signature verifies (see state_key()); otherwise it is
    # simply rebuilt from the embeddings, which are always stored.
    index_blob = index.save() if index is not None and index.kind != "flat" and state_key() else None
    now = time.time()
    meta = DocMeta(
        id=doc_id,
        name=name,
        num_pages=num_pages,
        num_chunks=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        dim=int(embeddings.shape[1]),
        size_bytes=len(chunks_blob) + len(emb_blob) + len(index_blob or b""),
        digest=hashlib.sha256(chunks_blob + emb_blob).hexdigest(),
        created=now,
        expires=now + (retention_seconds() if ttl is None else ttl),
        index_kind=index.kind if (index is not None and index_blob is not None) else "flat",
        index_digest=_index_signature(index_blob) if index_blob is not None else "",
    )
    return Packed(chunks_blob, emb_blob, meta, index_blob)


def unpack(meta: DocMeta, chunks_blob: bytes, emb_blob: bytes) -> tuple[list[dict], np.ndarray]:
    """CPU-bound: verify and decode a stored document. Raises
    DocumentCorruptError if anything does not match its metadata."""
    if hashlib.sha256(chunks_blob + emb_blob).hexdigest() != meta.digest:
        raise DocumentCorruptError("digest mismatch")
    if len(emb_blob) != meta.num_chunks * meta.dim * 2:
        raise DocumentCorruptError("embedding size mismatch")
    try:
        inflater = zlib.decompressobj()
        raw = inflater.decompress(chunks_blob, MAX_UNPACKED_BYTES)
        if inflater.unconsumed_tail:
            raise DocumentCorruptError("chunk data too large")
        chunks = json.loads(raw)
    except (zlib.error, ValueError) as e:
        raise DocumentCorruptError("unreadable chunk data") from e
    if not isinstance(chunks, list) or len(chunks) != meta.num_chunks:
        raise DocumentCorruptError("chunk count mismatch")
    for chunk in chunks:
        if not (isinstance(chunk, dict) and isinstance(chunk.get("text"), str) and isinstance(chunk.get("page"), int)):
            raise DocumentCorruptError("malformed chunk")
    vectors = np.frombuffer(emb_blob, dtype="float16").astype("float32").reshape(meta.num_chunks, meta.dim)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1.0, norms)
    return chunks, np.ascontiguousarray(vectors, dtype="float32")


# ---------- artifact backends ----------


class ArtifactStore(Protocol):
    async def put(self, key: str, data: bytes, ttl: int) -> None: ...
    async def get(self, key: str) -> bytes | None: ...
    async def delete(self, key: str) -> None: ...


class StoreArtifacts:
    """Blobs in the shared store (Redis), split into BLOB_PART_BYTES parts. The
    manifest at `key` is written last, so a partial write is invisible."""

    def __init__(self, store) -> None:
        self._store = store

    async def put(self, key: str, data: bytes, ttl: int) -> None:
        parts = [data[i : i + BLOB_PART_BYTES] for i in range(0, len(data), BLOB_PART_BYTES)] or [b""]
        await asyncio.gather(*(self._store.set_bytes(f"{key}:{i}", part, ttl) for i, part in enumerate(parts)))
        manifest = json.dumps({"parts": len(parts), "bytes": len(data)}).encode()
        await self._store.set_bytes(key, manifest, ttl)

    async def get(self, key: str) -> bytes | None:
        raw = await self._store.get_bytes(key)
        if raw is None:
            return None
        try:
            manifest = json.loads(raw)
            count, size = int(manifest["parts"]), int(manifest["bytes"])
        except (ValueError, KeyError, TypeError):
            return None
        pieces = await asyncio.gather(*(self._store.get_bytes(f"{key}:{i}") for i in range(count)))
        if any(part is None for part in pieces):
            return None
        data = b"".join(pieces)
        return data if len(data) == size else None

    async def delete(self, key: str) -> None:
        raw = await self._store.get_bytes(key)
        count = 0
        if raw is not None:
            try:
                count = int(json.loads(raw)["parts"])
            except (ValueError, KeyError, TypeError):
                count = 0
        await self._store.delete_bytes([key, *[f"{key}:{i}" for i in range(count)]])


class DiskArtifacts:
    """Blobs as files in a directory (for a host with a real disk or a mounted
    volume shared by instances). Writes are atomic (temp file + rename)."""

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / hashlib.sha256(key.encode()).hexdigest()

    async def put(self, key: str, data: bytes, ttl: int) -> None:
        path = self._path(key)
        temp = path.with_suffix(f".{os.getpid()}.tmp")
        temp.write_bytes(data)
        os.replace(temp, path)
        os.utime(path, (time.time() + ttl, time.time() + ttl))  # mtime carries the expiry

    async def get(self, key: str) -> bytes | None:
        path = self._path(key)
        try:
            if path.stat().st_mtime <= time.time():
                path.unlink(missing_ok=True)
                return None
            return path.read_bytes()
        except OSError:
            return None

    async def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


def make_artifacts(store) -> ArtifactStore:
    """DiskArtifacts if ARTIFACT_DIR is set, otherwise blobs in the shared store."""
    directory = os.environ.get("ARTIFACT_DIR", "").strip()
    return DiskArtifacts(directory) if directory else StoreArtifacts(store)


# ---------- repository ----------


class DocumentRepository:
    """A user's persisted documents. Every method takes the `owner` namespace
    from `owner_key(verified uid)`; there is no way to name another user's key."""

    def __init__(self, store, artifacts: ArtifactStore | None = None) -> None:
        self._store = store
        self._artifacts = artifacts or make_artifacts(store)

    @staticmethod
    def _docs_key(owner: str) -> str:
        return f"u:{owner}:docs"

    @staticmethod
    def _hist_key(owner: str) -> str:
        return f"u:{owner}:hist"

    @staticmethod
    def _blob_key(owner: str, doc_id: str, kind: str) -> str:
        return f"a:{owner}:{doc_id}:{kind}"

    async def list_docs(self, owner: str) -> dict[str, DocMeta]:
        """The user's live documents, oldest first. Expired or malformed
        entries are removed as they are found."""
        raw = await self._store.hgetall_json(self._docs_key(owner))
        now = time.time()
        docs: dict[str, DocMeta] = {}
        for doc_id, entry in raw.items():
            try:
                meta = DocMeta.from_dict(entry)
                if meta.id != doc_id:
                    raise ValueError("id mismatch")
            except ValueError as e:
                log.warning("Dropping malformed document metadata: %s", e)
                await self._store.hdel(self._docs_key(owner), doc_id)
                continue
            if meta.expires <= now:
                await self.delete(owner, doc_id)
                continue
            docs[doc_id] = meta
        return dict(sorted(docs.items(), key=lambda item: item[1].created))

    async def save(self, owner: str, packed: Packed, existing: dict[str, DocMeta] | None = None) -> DocMeta:
        """Write the artifacts, then the metadata. Raises StorageQuotaError
        before writing anything if the user's allowance would be exceeded."""
        meta = packed.meta
        used = sum(m.size_bytes for m in (existing if existing is not None else await self.list_docs(owner)).values())
        if used + meta.size_bytes > max_user_bytes():
            raise StorageQuotaError("storage allowance exceeded")
        ttl = max(1, int(meta.expires - time.time()))
        writes = [
            self._artifacts.put(self._blob_key(owner, meta.id, "chunks"), packed.chunks_blob, ttl),
            self._artifacts.put(self._blob_key(owner, meta.id, "emb"), packed.emb_blob, ttl),
        ]
        if packed.index_blob is not None:
            writes.append(self._artifacts.put(self._blob_key(owner, meta.id, "index"), packed.index_blob, ttl))
        await asyncio.gather(*writes)
        await self._store.hset_json(self._docs_key(owner), meta.id, meta.to_dict(), retention_seconds())
        return meta

    async def load(self, owner: str, meta: DocMeta) -> Loaded:
        """The stored chunks, embeddings and (for an approximate index) the saved
        index. Raises DocumentMissingError or DocumentCorruptError; the caller
        decides how to recover. A missing or damaged *index* is not an error: it
        is derived data, so `index` is None and the caller rebuilds it."""
        chunks_blob, emb_blob = await asyncio.gather(
            self._artifacts.get(self._blob_key(owner, meta.id, "chunks")),
            self._artifacts.get(self._blob_key(owner, meta.id, "emb")),
        )
        if chunks_blob is None or emb_blob is None:
            raise DocumentMissingError(meta.id)
        chunks, embeddings = unpack(meta, chunks_blob, emb_blob)
        index = None
        if meta.index_kind != "flat" and state_key():
            blob = await self._artifacts.get(self._blob_key(owner, meta.id, "index"))
            if blob is not None and hmac.compare_digest(_index_signature(blob), meta.index_digest):
                try:
                    index = vector_index.load_index(meta.index_kind, blob)
                    if index.size() != meta.num_chunks:
                        index = None
                except vector_index.IndexFormatError:
                    index = None
            if index is None:
                log.warning("Saved %s index unusable; rebuilding it", meta.index_kind)
        return Loaded(chunks, embeddings, index)

    async def delete(self, owner: str, doc_id: str) -> None:
        """Remove a document completely: metadata first (so it stops being
        listed), then both artifacts."""
        await self._store.hdel(self._docs_key(owner), doc_id)
        await asyncio.gather(*(self._artifacts.delete(self._blob_key(owner, doc_id, kind)) for kind in ("chunks", "emb", "index")))

    async def delete_all(self, owner: str) -> None:
        """Remove every document and the conversation."""
        for doc_id in list((await self._store.hgetall_json(self._docs_key(owner))).keys()):
            await self.delete(owner, doc_id)
        await self._store.delete(self._hist_key(owner))

    async def get_history(self, owner: str) -> list[dict]:
        turns = await self._store.get_json(self._hist_key(owner))
        if not isinstance(turns, list):
            return []
        return [t for t in turns if isinstance(t, dict) and isinstance(t.get("question"), str) and isinstance(t.get("answer"), str)]

    async def set_history(self, owner: str, turns: list[dict]) -> None:
        await self._store.set_json(self._hist_key(owner), turns[-MAX_HISTORY_TURNS_STORED:], retention_seconds())

    async def clear_history(self, owner: str) -> None:
        await self._store.delete(self._hist_key(owner))
