"""Where does the time go? Latency of every stage of the pipeline, measured.

Stages (each median of several runs, on this machine, warm model):
  extract   load_pdf on the real paper
  chunk     char vs structured chunker
  embed     MiniLM over all chunks (the dominant ingestion cost)
  index     VectorStore build (BM25 statistics + vector index)
  pack      docstore.pack (compress) and unpack + pipeline.restore (cold load after a restart)
  retrieve  hybrid retrieval, 1 and 5 documents, p50 / p95 per query
  redis     round trips of the per-request state calls, in-process vs the live Redis
            named by REDIS_URL, sequential vs gathered
No LLM calls: the model's own latency is the provider's, not ours.

Usage:
    python bench_latency.py [--output reports/latency_bench.md] [--repeat 5] [--synthetic-chunks 1500]
"""

import argparse
import asyncio
import os
import statistics
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

import docstore
import embedder
import pipeline
import store
from chunker import chunk_pages, chunk_pages_structured
from document_loader import load_document_ex

load_dotenv()
ROOT = Path(__file__).parent
PDF = ROOT / "sample_docs" / "dev_real_world_document.pdf"
QUESTIONS = [
    "How many attention heads are used?",
    "What dropout rate did the base model use?",
    "Why is dot-product attention scaled?",
    "What hardware were the models trained on?",
    "How is word order information added to the model?",
    "What BLEU score did the big model reach?",
]


def median_ms(fn, repeat: int) -> float:
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def pct(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def synthetic_pages(chunks_wanted: int, base_pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """The real paper repeated (with a varying prefix so chunks differ) until it
    yields about `chunks_wanted` chunks at 800 characters."""
    pages, n = [], 0
    while sum(len(t) for _, t in pages) < chunks_wanted * 700:
        for p, t in base_pages:
            n += 1
            pages.append((n, f"[copy {n // len(base_pages)}] " + t))
    return pages


async def redis_round_trips(url: str, repeat: int) -> dict:
    backend = store.RedisStore(url)
    key = f"doculens-bench:{int(time.time())}"
    await backend.set_json(key, {"sub": "x", "t": 1}, 60)
    await backend.hset_json(key + ":h", "d1", {"id": "d1"}, 60)

    async def one_get():
        await backend.get_json(key)

    async def one_limit():
        await backend.rate_hit(key + ":rl", [(10_000, 60)])

    async def one_hash():
        await backend.hgetall_json(key + ":h")

    async def timed(coro_fn):
        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            await coro_fn()
            times.append((time.perf_counter() - start) * 1000)
        return statistics.median(times)

    out = {"get": await timed(one_get), "rate_hit (Lua)": await timed(one_limit), "hgetall": await timed(one_hash)}

    async def sequential():
        await one_get()
        await one_limit()
        await one_limit()
        await one_hash()
        await one_get()

    async def gathered():
        await asyncio.gather(one_get(), one_limit(), one_limit(), one_hash(), one_get())

    out["5 calls sequential"] = await timed(sequential)
    out["5 calls gathered"] = await timed(gathered)
    await backend.delete(key)
    await backend.delete(key + ":h")
    await backend.delete(key + ":rl")
    await backend.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", default=None)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--synthetic-chunks", type=int, default=1500)
    args = parser.parse_args()
    rows: list[tuple[str, str, float]] = []

    def record(stage: str, detail: str, ms: float) -> None:
        rows.append((stage, detail, ms))
        print(f"{stage:9} {detail:44} {ms:9.1f} ms", flush=True)

    data = PDF.read_bytes()
    embedder.get_model()
    embedder.embed(["warm up"])

    record("extract", "load_pdf, real paper (15 pages)", median_ms(lambda: load_document_ex("p.pdf", data), args.repeat))
    pages = load_document_ex("p.pdf", data).pages
    big = synthetic_pages(args.synthetic_chunks, pages)

    for label, fn in (("char", chunk_pages), ("structured", chunk_pages_structured)):
        record("chunk", f"{label}, real paper", median_ms(lambda: fn(pages, 800, 150), args.repeat))
        record("chunk", f"{label}, ~{args.synthetic_chunks} chunks", median_ms(lambda: fn(big, 800, 150), max(1, args.repeat // 2)))

    chunks = chunk_pages_structured(pages, 800, 150)
    texts = [c["text"] for c in chunks]
    def cold(batch):
        embedder.clear_cache()  # measure real encoding, not the in-process cache
        return embedder.embed(batch)

    record("embed", f"MiniLM, {len(texts)} chunks (real paper), cold", median_ms(lambda: cold(texts), args.repeat))
    embedder.embed(texts)
    record("embed", f"same {len(texts)} chunks again (cache hit: a re-upload)", median_ms(lambda: embedder.embed(texts), args.repeat))
    big_chunks = chunk_pages_structured(big, 800, 150)
    big_texts = [c["text"] for c in big_chunks]
    record("embed", f"MiniLM, {len(big_texts)} chunks (synthetic), cold", median_ms(lambda: cold(big_texts), 1))
    record("embed", "one query, cold (what every new question pays)", median_ms(lambda: cold([QUESTIONS[0]]), args.repeat * 4))

    vectors = embedder.embed(texts)
    big_vectors = embedder.embed(big_texts)
    from vector_store import VectorStore

    record("index", f"VectorStore build, {len(texts)} chunks", median_ms(lambda: VectorStore(chunks, vectors), args.repeat))
    record("index", f"VectorStore build, {len(big_texts)} chunks", median_ms(lambda: VectorStore(big_chunks, big_vectors), 2))

    state = pipeline.restore(chunks, vectors, "p.pdf", len(pages), 800, 150)
    big_state = pipeline.restore(big_chunks, big_vectors, "big.pdf", len(big), 800, 150)
    packed = docstore.pack("d", "p.pdf", len(pages), 800, 150, chunks, vectors)
    big_packed = docstore.pack("d", "big.pdf", len(big), 800, 150, big_chunks, big_vectors)
    record("pack", f"pack (compress), {len(texts)} chunks, {packed.meta.size_bytes / 1e3:.0f} KB", median_ms(lambda: docstore.pack("d", "p.pdf", len(pages), 800, 150, chunks, vectors), args.repeat))
    record("pack", f"pack (compress), {len(big_texts)} chunks, {big_packed.meta.size_bytes / 1e6:.2f} MB", median_ms(lambda: docstore.pack("d", "b.pdf", len(big), 800, 150, big_chunks, big_vectors), 2))
    record("pack", f"unpack, {len(big_texts)} chunks", median_ms(lambda: docstore.unpack(big_packed.meta, big_packed.chunks_blob, big_packed.emb_blob), 3))
    record(
        "pack",
        f"cold restore (unpack + index build), {len(big_texts)} chunks",
        median_ms(
            lambda: pipeline.restore(*docstore.unpack(big_packed.meta, big_packed.chunks_blob, big_packed.emb_blob), "b.pdf", len(big), 800, 150),
            2,
        ),
    )

    def run_queries(states, queries):
        times = []
        for q in queries:
            embedder.clear_cache()  # a question asked for the first time
            start = time.perf_counter()
            pipeline.retrieve([q], states)
            times.append((time.perf_counter() - start) * 1000)
        return times

    for label, states in (("1 doc (real paper)", [state]), ("5 docs (real paper)", [state] * 5), (f"1 doc ({len(big_texts)} chunks)", [big_state])):
        run_queries(states, QUESTIONS[:2])  # warm
        times = run_queries(states, QUESTIONS * 5)
        record("retrieve", f"{label} p50", pct(times, 50))
        record("retrieve", f"{label} p95", pct(times, 95))

    url = os.environ.get("REDIS_URL", "").strip()
    if url:
        for k, v in asyncio.run(redis_round_trips(url, max(5, args.repeat * 3))).items():
            record("redis", f"live Redis: {k}", v)
    mem = store.MemoryStore()

    async def mem_calls():
        await mem.set_json("k", {"a": 1}, 60)
        await mem.get_json("k")
        await mem.rate_hit("r", [(10_000, 60)])

    record("redis", "in-process store: set+get+rate_hit", median_ms(lambda: asyncio.run(mem_calls()), args.repeat * 4))

    lines = ["# Latency benchmark", "", "Median of repeated runs on the machine that ran it (no LLM calls); compare rows, not absolute numbers.", "", "| stage | what | ms |", "|---|---|---:|"]
    lines += [f"| {s} | {d} | {ms:.1f} |" for s, d, ms in rows]
    text = "\n".join(lines) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
