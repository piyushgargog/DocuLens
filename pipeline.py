"""Phase 6: orchestration — wires loader, chunker, embedder, vector store, LLM."""

from dataclasses import dataclass

import embedder
import llm_client
from chunker import chunk_pages
from pdf_loader import load_pdf_pages
from vector_store import VectorStore

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
MAX_HISTORY_TURNS = 3  # earlier Q/A turns sent to the LLM for follow-up questions
SUMMARY_SAMPLE_CHUNKS = 10  # evenly spaced chunks fed to the summary prompt


@dataclass
class IndexState:
    store: VectorStore
    num_pages: int
    num_chunks: int
    chunk_size: int
    chunk_overlap: int
    name: str | None = None


def ingest(
    pdf_bytes: bytes,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    name: str | None = None,
) -> IndexState | None:
    """Run extraction -> chunking -> embedding -> indexing. Returns None if the
    PDF has no extractable text (invalid/empty PDF).

    `name` (e.g. the uploaded filename) is tagged onto every chunk so answers
    across several documents can say which document a passage came from."""
    pages = load_pdf_pages(pdf_bytes)
    if not pages:
        return None

    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return None
    if name:
        for chunk in chunks:
            chunk["doc"] = name

    vectors = embedder.embed([c["text"] for c in chunks])
    store = VectorStore(chunks, vectors)
    return IndexState(
        store=store,
        num_pages=len(pages),
        num_chunks=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        name=name,
    )


def retrieve(queries: list[str], states: list[IndexState], top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Search every document with every query and return the overall top_k
    passages, best first. A passage found by more than one query keeps its
    highest score. With one query and one document this is exactly
    `store.search(query, top_k)`."""
    query_vectors = embedder.embed(queries)
    best: dict[tuple, dict] = {}
    for state in states:
        for vector in query_vectors:
            for hit in state.store.search(vector, top_k=top_k):
                key = (hit.get("doc"), hit["page"], hit["text"])
                if key not in best or hit["score"] > best[key]["score"]:
                    best[key] = hit
    return sorted(best.values(), key=lambda h: h["score"], reverse=True)[:top_k]


def answer(
    question: str,
    index_state: IndexState | list[IndexState],
    top_k: int = DEFAULT_TOP_K,
    history: list[dict] | None = None,
) -> dict:
    """Retrieve relevant chunks and generate a grounded answer.

    `index_state` may be one document or a list of documents. `history` is a
    list of earlier {"question", "answer"} turns, oldest first. For a
    follow-up, retrieval also runs on the previous question + this one, so
    "what about its moons?" can still find the passages about "it".

    Returns {"answer": str, "sources": [{"page", "text", "score", ["doc"]}, ...]}.
    """
    states = index_state if isinstance(index_state, list) else [index_state]
    recent = (history or [])[-MAX_HISTORY_TURNS:]

    queries = [question]
    if recent:
        queries.append(f"{recent[-1]['question']} {question}")

    sources = retrieve(queries, states, top_k=top_k)
    answer_text = llm_client.ask(question, sources, history=recent)
    return {"answer": answer_text, "sources": sources}


def summarize(index_state: IndexState) -> dict:
    """Summarize one document from evenly spaced chunks across it (a full
    map-reduce over every chunk would cost one LLM call per chunk, which
    free-tier rate limits don't allow for larger PDFs).

    Returns {"summary": str, "sources": [...]} in the same shape as answer()."""
    chunks = index_state.store.chunks
    count = min(SUMMARY_SAMPLE_CHUNKS, len(chunks))
    step = len(chunks) / count
    sample = [chunks[int(i * step)] for i in range(count)]
    summary_text = llm_client.summarize(sample)
    return {"summary": summary_text, "sources": [{**c, "score": None} for c in sample]}
