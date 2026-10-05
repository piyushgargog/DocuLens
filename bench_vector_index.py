"""Exact vs approximate vector search: is HNSW worth it, and from what size?

Measures, for several index sizes, on L2-normalised 384-dimensional vectors (the
app's MiniLM size):
  - build time
  - mean search latency (single query, top-10) and the app's *whole* dense step
    (what `pipeline._dense_similarity` actually costs: a full matrix product for
    an exact index, an index query for HNSW)
  - recall@10 of HNSW against the exact results
  - serialized index size (a proxy for memory)

Two corpora are used because a benchmark on uniform random vectors flatters
neither method honestly: real embeddings cluster. `clustered` draws vectors
around random centres with noise, which is closer to document embeddings;
`random` is the hard case for graph indexes. Neither is a real document corpus;
the retrieval-quality side (Hit@k on real documents) is measured by
`retrieval_eval.py`.

Usage:
    python bench_vector_index.py
    python bench_vector_index.py --sizes 1000 20000 100000 --output reports/vector_index_bench.md
"""

import argparse
import time
from pathlib import Path

import numpy as np

from vector_index import FlatIndex, HNSWIndex

DIM = 384
QUERIES = 200
K = 10


def corpus(kind: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if kind == "random":
        data = rng.normal(size=(n, DIM)).astype("float32")
        queries = rng.normal(size=(QUERIES, DIM)).astype("float32")
    else:
        centres = rng.normal(size=(max(8, n // 200), DIM)).astype("float32")
        data = centres[rng.integers(0, len(centres), n)] + 0.6 * rng.normal(size=(n, DIM)).astype("float32")
        # queries are noisy copies of real vectors, like a question about a passage
        queries = data[rng.integers(0, n, QUERIES)] + 0.5 * rng.normal(size=(QUERIES, DIM)).astype("float32")
    norm = lambda a: (a / np.linalg.norm(a, axis=1, keepdims=True)).astype("float32")  # noqa: E731
    return norm(data), norm(queries)


def timed(fn, repeat: int = 1) -> float:
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - start) / repeat


def run(sizes: list[int]) -> list[dict]:
    rows = []
    for kind in ("clustered", "random"):
        for n in sizes:
            data, queries = corpus(kind, n, seed=n)
            flat, hnsw = FlatIndex(DIM), HNSWIndex(DIM)
            t_flat = timed(lambda: flat.add(data))
            t_hnsw = timed(lambda: hnsw.add(data))

            truth = [set(flat.search(q, K)[1]) for q in queries]
            recall = float(np.mean([len(truth[i] & set(hnsw.search(q, K)[1])) / K for i, q in enumerate(queries)]))

            lat_flat = timed(lambda: [flat.search(q, K) for q in queries]) / QUERIES
            lat_hnsw = timed(lambda: [hnsw.search(q, K) for q in queries]) / QUERIES
            # The app's dense step for an exact index is a full matrix product:
            lat_matrix = timed(lambda: [data @ q for q in queries]) / QUERIES
            lat_hnsw200 = timed(lambda: [hnsw.search(q, 200) for q in queries]) / QUERIES  # pipeline.DENSE_CANDIDATES

            rows.append(
                {
                    "corpus": kind,
                    "n": n,
                    "build_flat_s": t_flat,
                    "build_hnsw_s": t_hnsw,
                    "flat_ms": lat_flat * 1000,
                    "matrix_ms": lat_matrix * 1000,
                    "hnsw_ms": lat_hnsw * 1000,
                    "hnsw200_ms": lat_hnsw200 * 1000,
                    "recall": recall,
                    "flat_mb": len(flat.save()) / 1e6,
                    "hnsw_mb": len(hnsw.save()) / 1e6,
                }
            )
            print(f"{kind:9} n={n:>7}  recall@{K}={recall:.3f}  flat={lat_flat*1000:.3f}ms  hnsw={lat_hnsw*1000:.3f}ms", flush=True)
    return rows


def report(rows: list[dict]) -> str:
    lines = [
        "# Vector index benchmark: exact vs HNSW",
        "",
        f"{DIM}-dimensional normalised vectors, {QUERIES} queries, top-{K}. HNSW: M=32, efConstruction=200, efSearch=128.",
        "Single-threaded timings on the machine that ran the benchmark; compare the columns, not the absolute numbers.",
        "",
        "| corpus | vectors | recall@10 | exact search ms | exact dense step ms (matrix) | HNSW search ms | HNSW top-200 ms | build exact s | build HNSW s | exact MB | HNSW MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['corpus']} | {r['n']:,} | {r['recall']:.3f} | {r['flat_ms']:.3f} | {r['matrix_ms']:.3f} | {r['hnsw_ms']:.3f} | "
            f"{r['hnsw200_ms']:.3f} | {r['build_flat_s']:.2f} | {r['build_hnsw_s']:.2f} | {r['flat_mb']:.1f} | {r['hnsw_mb']:.1f} |"
        )
    lines += [
        "",
        "Corpora: `clustered` = noisy copies of random centres (closer to real embeddings); `random` = isotropic Gaussian",
        "(the hard case for graph indexes). Neither is a real document collection.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 5_000, 20_000, 50_000, 100_000])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    text = report(run(args.sizes))
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
