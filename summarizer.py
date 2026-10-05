"""Hierarchical (map-reduce) document summaries with a bounded LLM budget.

The old summary read 10 evenly spaced chunks, so a long document's middle was
never seen. This reads *all* of a document's text when the budget allows:

    chunks -> de-overlap -> batches (by section / page range)
           -> one summary per batch        (map, in parallel, each batch <= SUMMARY_BATCH_CHARS)
           -> merged in groups of REDUCE_FANIN, repeatedly      (reduce)
           -> final summary

* A document that fits in SUMMARY_SINGLE_CALL_CHARS is summarised in one call.
* The number of model calls never exceeds SUMMARY_MAX_LLM_CALLS (default 10).
  If the document is too large for that budget at SUMMARY_BATCH_HARD_CHARS per
  batch, evenly spaced batches are *skipped* and the result says how much of the
  document it covers -- it does not claim to have read what it did not.
* Page ranges ride through every level: each batch summary cites its pages, and
  the reducers are told to keep citations.
* A batch that fails (after llm_client's provider failover) is reported as "not
  summarised" in the final summary instead of failing the whole request; if
  nothing succeeds the error is raised.
* Batch summaries are cached (bounded LRU) so asking again, or re-summarising
  after a failure, does not pay for finished batches twice.

Everything the model sees is fenced untrusted document content (see llm_client).
"""

import hashlib
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import llm_client
from observability import log

PROMPT_VERSION = "1"
CACHE_SIZE = 128
OVERLAP_SEARCH_CHARS = 400
MIN_OVERLAP_CHARS = 20


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except ValueError:
        return default


def single_call_chars() -> int:
    return _env_int("SUMMARY_SINGLE_CALL_CHARS", 12_000, 1000)


def batch_chars() -> int:
    return _env_int("SUMMARY_BATCH_CHARS", 6_000, 1000)


def batch_hard_chars() -> int:
    return max(batch_chars(), _env_int("SUMMARY_BATCH_HARD_CHARS", 12_000, 1000))


def max_llm_calls() -> int:
    return _env_int("SUMMARY_MAX_LLM_CALLS", 10, 2)


def concurrency() -> int:
    return _env_int("SUMMARY_CONCURRENCY", 3, 1)


def reduce_fanin() -> int:
    return _env_int("SUMMARY_REDUCE_FANIN", 6, 2)


@dataclass
class Batch:
    chunks: list[dict]
    text: str = ""
    pages: tuple[int, int] = (0, 0)

    @property
    def label(self) -> str:
        a, b = self.pages
        return f"Page {a}" if a == b else f"Pages {a}-{b}"


@dataclass
class SummaryResult:
    summary: str
    sources: list[dict]
    coverage: dict = field(default_factory=dict)


# ---------- planning ----------


def dedupe_overlap(chunks: list[dict]) -> list[dict]:
    """Drop the text a chunk repeats from the end of the previous chunk of the
    same page (the character chunker overlaps neighbours), so no sentence is
    summarised twice or counted twice against the budget. Chunks are copied."""
    out: list[dict] = []
    for chunk in chunks:
        text = chunk["text"]
        if out and out[-1]["page"] == chunk["page"]:
            prev = out[-1]["_full"]
            for k in range(min(len(prev), len(text), OVERLAP_SEARCH_CHARS), MIN_OVERLAP_CHARS - 1, -1):
                if prev.endswith(text[:k]):
                    text = text[k:]
                    break
        out.append({**chunk, "text": text.strip(), "_full": chunk["text"]})
    return [{k: v for k, v in c.items() if k != "_full"} for c in out if c["text"].strip()]


def _total(chunks: list[dict]) -> int:
    return sum(len(c["text"]) + 1 for c in chunks)


def make_batches(chunks: list[dict], limit: int) -> list[Batch]:
    """Contiguous batches of at most `limit` characters. A batch closes early at
    a section boundary once it is at least 40% full, so sections stay together
    where they can."""
    batches: list[Batch] = []
    current: list[dict] = []
    size = 0
    for chunk in chunks:
        n = len(chunk["text"]) + 1
        new_section = bool(current) and chunk.get("section") and chunk.get("section") != current[-1].get("section")
        if current and (size + n > limit or (new_section and size >= 0.4 * limit)):
            batches.append(Batch(current))
            current, size = [], 0
        current.append(chunk)
        size += n
    if current:
        batches.append(Batch(current))
    for batch in batches:
        batch.text = "\n".join(c["text"] for c in batch.chunks)
        batch.pages = (min(c["page"] for c in batch.chunks), max(c["page"] for c in batch.chunks))
    return batches


def reduce_calls(summaries: int, fanin: int) -> int:
    """Model calls needed to merge `summaries` into one, including the last."""
    calls = 0
    while summaries > 1:
        groups = -(-summaries // fanin)
        calls += groups
        summaries = groups
    return max(calls, 1)


def plan(chunks: list[dict]) -> tuple[list[Batch], dict[str, Any]]:
    """Choose batches that respect the call budget. Returns (batches to summarise,
    coverage info). If everything cannot fit, whole batches are skipped evenly."""
    total = _total(chunks)
    budget, fanin = max_llm_calls(), reduce_fanin()
    size = batch_chars()
    batches = make_batches(chunks, size)
    # grow batches (up to the hard cap) until map + reduce fit in the budget
    while len(batches) + reduce_calls(len(batches), fanin) > budget and size < batch_hard_chars():
        size = min(batch_hard_chars(), int(size * 1.35) + 1)
        batches = make_batches(chunks, size)
    skipped: list[Batch] = []
    if len(batches) + reduce_calls(len(batches), fanin) > budget:
        keep = len(batches)
        while keep > 1 and keep + reduce_calls(keep, fanin) > budget:
            keep -= 1
        step = len(batches) / keep
        chosen = {int(i * step) for i in range(keep)}
        skipped = [b for i, b in enumerate(batches) if i not in chosen]
        batches = [b for i, b in enumerate(batches) if i in chosen]
    used_chars = sum(len(b.text) for b in batches)
    coverage: dict[str, Any] = {
        "chunks_total": len(chunks),
        "chars_total": total,
        "chars_summarised": used_chars,
        "batches": len(batches),
        "skipped_ranges": [b.label for b in skipped],
        "complete": not skipped,
    }
    return batches, coverage


# ---------- map / reduce ----------

_cache: "OrderedDict[str, str]" = OrderedDict()
_cache_lock = threading.Lock()


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _cached(kind: str, text: str, compute):
    key = hashlib.sha256(f"{PROMPT_VERSION}\0{kind}\0{text}".encode()).hexdigest()
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    value = compute()
    with _cache_lock:
        _cache[key] = value
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return value


def _summarise_batch(batch: Batch) -> str:
    # No extra retry here: llm_client already fails over across providers, and a
    # second attempt per batch would break the SUMMARY_MAX_LLM_CALLS guarantee.
    return _cached("batch", batch.label + "\n" + batch.text, lambda: llm_client.summarize_batch(batch.chunks, batch.label))


def _reduce(parts: list[tuple[str, str]], note: str | None, final: bool) -> str:
    key = "\n".join(f"{label}\n{text}" for label, text in parts) + (note or "") + str(final)

    def compute() -> str:
        return llm_client.summarize_parts(parts, note, final=final)

    return _cached("reduce", key, compute)


def summarize_document(chunks: list[dict]) -> SummaryResult:
    """Summarise a whole document from its chunks (see the module docstring).
    Raises llm_client.LLMConfigError / LLMRequestError if no summary can be made."""
    chunks = dedupe_overlap(chunks)
    if not chunks:
        raise ValueError("no text to summarise")
    total = _total(chunks)

    if total <= single_call_chars():
        text = llm_client.summarize(chunks, complete=True)
        pages = sorted({c["page"] for c in chunks})
        coverage: dict[str, Any] = {"chunks_total": len(chunks), "chars_total": total, "chars_summarised": total, "batches": 1,
                    "skipped_ranges": [], "complete": True, "llm_calls": 1, "failed_ranges": [], "pages": [pages[0], pages[-1]]}
        return SummaryResult(text, [{**c, "score": None} for c in chunks[:12]], coverage)

    batches, coverage = plan(chunks)
    results: dict[int, str] = {}
    failed: list[str] = []
    last_error: Exception | None = None
    llm_calls = 0

    def run(i: int) -> tuple[int, str | Exception]:
        try:
            return i, _summarise_batch(batches[i])
        except llm_client.LLMConfigError:
            raise
        except Exception as e:  # a provider failure on one batch must not sink the others
            return i, e

    with ThreadPoolExecutor(max_workers=min(concurrency(), len(batches))) as pool:
        for i, outcome in pool.map(run, range(len(batches))):
            llm_calls += 1
            if isinstance(outcome, Exception):
                failed.append(batches[i].label)
                last_error = outcome
                log.warning("Summary batch %s failed (%s)", batches[i].label, type(outcome).__name__)
            else:
                results[i] = outcome
    if not results:
        raise last_error or llm_client.LLMRequestError("The summary could not be produced.")

    parts = [(batches[i].label, results[i]) for i in sorted(results)]
    notes = []
    if coverage["skipped_ranges"]:
        notes.append(
            "Because of the document's length only part of it was read; these ranges were not summarised: "
            + ", ".join(coverage["skipped_ranges"][:12])
        )
    if failed:
        notes.append("These ranges could not be summarised (provider errors): " + ", ".join(failed))
    note = " ".join(notes) or None

    fanin = reduce_fanin()
    while len(parts) > fanin:
        grouped = [parts[i : i + fanin] for i in range(0, len(parts), fanin)]
        parts = []
        for group in grouped:
            llm_calls += 1
            label = group[0][0] if len(group) == 1 else f"{group[0][0]} to {group[-1][0]}"
            parts.append((label, _reduce(group, None, final=False)))
    llm_calls += 1
    summary = _reduce(parts, note, final=True)

    sources = [{**b.chunks[0], "score": None} for i, b in enumerate(batches) if i in results][:12]
    coverage.update({"llm_calls": llm_calls, "failed_ranges": failed, "complete": coverage["complete"] and not failed})
    return SummaryResult(summary, sources, coverage)
