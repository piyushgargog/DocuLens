"""Phase 6: orchestration — wires loader, chunker, embedder, vector store, LLM."""

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

import embedder
import llm_client
from chunker import chunk_pages
from document_loader import load_document
from retriever import bm25_scores, ranking, reciprocal_rank_fusion
from vector_store import VectorStore

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
MAX_HISTORY_TURNS = 3  # earlier Q/A turns sent to the LLM for follow-up questions
SUMMARY_SAMPLE_CHUNKS = 10  # evenly spaced chunks fed to the summary prompt
OVERVIEW_CHUNKS = 8  # passages given to the LLM for a whole-document question
OVERVIEW_LEAD_CHUNKS = 2  # always include the opening (abstract/introduction)

# Minimum cosine similarity for the best retrieved passage. When even the best
# of the top-k hits is below it, a non-overview question gets no passages and
# the model answers with its grounded refusal instead of improvising from noise.
# Calibrated with score_floor_eval.py (reports/score_floor_eval.md): off-topic
# and cross-document questions top out at 0.28 (median ~0.1), answerable ones
# have a median of 0.5-0.65. 0.25 refuses ~all off-topic questions while
# wrongly refusing only ~3% of answerable ones; 0.30 would refuse 8%. It is a
# safety net, not the main control (the prompt's refusal rule is). Re-measure
# on a new document set; adjust via the RETRIEVAL_SCORE_FLOOR env var.
try:
    RETRIEVAL_SCORE_FLOOR = float(os.environ.get("RETRIEVAL_SCORE_FLOOR", "0.25"))
except ValueError:
    RETRIEVAL_SCORE_FLOOR = 0.25
RETRIEVAL_SCORE_FLOOR = max(-1.0, min(1.0, RETRIEVAL_SCORE_FLOOR))

# Questions about the document as a whole ("what is this about?", "main
# contribution", "summarize the key findings"). Similarity search is the wrong
# tool for these: no single passage resembles the question, so it returns
# whatever shares a word with it -- in testing, the reference list.
# Sectional qualifiers — if any of these follow a summarize/overview verb,
# the question is about a specific part, not the whole document.
_SECTION_QUALIFIER = re.compile(
    r"\b(section|chapter|part|paragraph|page|table|figure|appendix|slide)\s*\d",
    re.IGNORECASE,
)

OVERVIEW_PATTERN = re.compile(
    r"\b(summari[sz](e|ing)"
    r"|summary(?= of| please|\s*\?|\s*$)|(give|write|need|want)( me)? (a |the )?(short |quick |brief )?summary"
    r"|overview|gist|tl;?dr|in a nutshell"
    r"|(main|key|central|overall|primary|core|biggest) (idea|point|message|contribution|finding|takeaway|argument|goal|topic|theme)s?"
    r"|what('?s| is| are)? (this|the) (document|paper|pdf|file|report|article|book|text)s? (about|for)"
    r"|what does (this|the) (document|paper|pdf|file|report|article|book|text) (do|say|propose|cover))\b",
    re.IGNORECASE,
)


class DocumentTooLargeError(ValueError):
    """The document would produce more chunks than the caller allows."""


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
    max_chunks: int | None = None,
) -> IndexState | None:
    """Run extraction -> chunking -> embedding -> indexing. Returns None if the
    document has no extractable text (invalid/empty/unsupported file).

    The document type is chosen from `name`'s extension (PDF, txt, Markdown,
    docx); with no name the bytes are treated as a PDF, so older callers keep
    working. With `max_chunks`, raises DocumentTooLargeError *before* embedding,
    so an oversized document never costs the memory or CPU time of indexing.

    `name` (e.g. the uploaded filename) is tagged onto every chunk so answers
    across several documents can say which document a passage came from."""
    pages = load_document(name or "document.pdf", pdf_bytes)
    if not pages:
        return None

    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return None
    if max_chunks is not None and len(chunks) > max_chunks:
        raise DocumentTooLargeError(f"{len(chunks)} chunks > {max_chunks}")
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
    """Hybrid retrieval over every given document at once.

    For each query, all chunks of all documents are ranked twice -- by BM25
    (keyword statistics taken over the documents together, so scores are
    comparable) and by embedding cosine similarity -- and every ranking is
    fused with reciprocal rank fusion. With one document and one query this is
    exactly the "Hybrid" retriever measured in `retrieval_eval.py`.

    Each returned passage carries its cosine similarity to the closest query
    as `score`, so the UI's "similarity" stays meaningful.
    """
    chunks = [c for state in states for c in state.store.chunks]
    if not chunks:
        return []
    query_vectors = embedder.embed(queries)
    term_indexes = [state.store.terms for state in states]
    rankings, similarities = [], []
    for query, vector in zip(queries, query_vectors):
        similarity = np.concatenate([state.store.embeddings @ vector for state in states])
        similarities.append(similarity)
        rankings.append(ranking(bm25_scores(query, term_indexes)))
        rankings.append(ranking(similarity))
    best_similarity = np.max(similarities, axis=0)
    order = reciprocal_rank_fusion(rankings)[:top_k]
    return [{**chunks[i], "score": float(best_similarity[i])} for i in order]


def is_overview_question(question: str) -> bool:
    """True for questions about a whole document rather than a specific fact.
    Sectional requests like 'summarize section 4' stay retrieval-based."""
    if _SECTION_QUALIFIER.search(question):
        return False
    return bool(OVERVIEW_PATTERN.search(question))


def overview_sample(index_state: IndexState, count: int = OVERVIEW_CHUNKS) -> list[dict]:
    """The opening chunks (abstract/introduction) plus evenly spaced chunks
    from the rest, in document order -- a cheap stand-in for reading it all."""
    chunks = index_state.store.chunks
    if len(chunks) <= count:
        return list(chunks)
    lead = chunks[:OVERVIEW_LEAD_CHUNKS]
    rest = chunks[OVERVIEW_LEAD_CHUNKS:]
    picks = count - len(lead)
    step = len(rest) / picks
    return lead + [rest[int(i * step)] for i in range(picks)]


def gather_sources(
    question: str,
    index_state: IndexState | list[IndexState],
    top_k: int = DEFAULT_TOP_K,
    history: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """The passages the LLM will be given for this question, and the recent
    turns to send with it. Shared by answer() and answer_stream().

    A whole-document question ("what is this paper about?") is routed to an
    overview sample of each document; anything else goes through hybrid
    retrieval, which for a follow-up also searches "previous question + this
    one", so "what about its moons?" still finds the passages about "it".
    """
    states = index_state if isinstance(index_state, list) else [index_state]
    recent = (history or [])[-MAX_HISTORY_TURNS:]

    if is_overview_question(question):
        per_doc = max(3, OVERVIEW_CHUNKS // len(states))
        sources = [{**c, "score": None} for state in states for c in overview_sample(state, per_doc)]
        return sources, recent

    queries = [question]
    if recent:
        queries.append(f"{recent[-1]['question']} {question}")
    sources = retrieve(queries, states, top_k=top_k)
    # Abstention: if every retrieved passage is below the score floor,
    # return no sources so the LLM sees "(no passages retrieved)" and
    # gives its grounded refusal instead of hallucinating from noise.
    if sources and all(s["score"] < RETRIEVAL_SCORE_FLOOR for s in sources):
        return [], recent
    return sources, recent


def answer(
    question: str,
    index_state: IndexState | list[IndexState],
    top_k: int = DEFAULT_TOP_K,
    history: list[dict] | None = None,
) -> dict:
    """Retrieve relevant chunks and generate a grounded answer.

    `index_state` may be one document or a list of documents. `history` is a
    list of earlier {"question", "answer"} turns, oldest first.

    Returns {"answer": str, "sources": [{"page", "text", "score", ["doc"]}, ...]}.
    """
    sources, recent = gather_sources(question, index_state, top_k, history)
    answer_text = llm_client.ask(question, sources, history=recent)
    return {"answer": answer_text, "sources": sources}


def answer_stream(
    question: str,
    index_state: IndexState | list[IndexState],
    top_k: int = DEFAULT_TOP_K,
    history: list[dict] | None = None,
) -> tuple[list[dict], Iterator[tuple[str, str]]]:
    """Like answer(), but returns the sources at once and the answer as an
    iterator of (kind, text) pieces ("content"/"reasoning"), so the UI can show
    passages, the model's thinking, and the answer text as they come."""
    sources, recent = gather_sources(question, index_state, top_k, history)
    return sources, llm_client.ask_stream(question, sources, history=recent)


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


def suggest_questions(index_state: IndexState) -> list[str]:
    """Up to 4 starter questions for a document, written by the LLM from an
    overview sample of it."""
    return llm_client.suggest_questions(overview_sample(index_state))
