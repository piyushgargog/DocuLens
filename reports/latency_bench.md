# Latency benchmark

Median of repeated runs on the machine that ran it (no LLM calls); compare rows, not absolute numbers.

| stage | what | ms |
|---|---|---:|
| extract | load_pdf, real paper (15 pages) | 73.2 |
| chunk | char, real paper | 0.0 |
| chunk | char, ~1500 chunks | 1.2 |
| chunk | structured, real paper | 5.4 |
| chunk | structured, ~1500 chunks | 154.6 |
| embed | MiniLM, 66 chunks (real paper), cold | 1857.5 |
| embed | same 66 chunks again (cache hit: a re-upload) | 0.2 |
| embed | MiniLM, 1863 chunks (synthetic), cold | 10085.9 |
| embed | one query, cold (what every new question pays) | 7.5 |
| index | VectorStore build, 66 chunks | 2.4 |
| index | VectorStore build, 1863 chunks | 71.3 |
| pack | pack (compress), 66 chunks, 66 KB | 1.0 |
| pack | pack (compress), 1863 chunks, 1.82 MB | 28.9 |
| pack | unpack, 1863 chunks | 11.0 |
| pack | cold restore (unpack + index build), 1863 chunks | 79.1 |
| retrieve | 1 doc (real paper) p50 | 8.1 |
| retrieve | 1 doc (real paper) p95 | 10.0 |
| retrieve | 5 docs (real paper) p50 | 8.8 |
| retrieve | 5 docs (real paper) p95 | 11.3 |
| retrieve | 1 doc (1863 chunks) p50 | 14.9 |
| retrieve | 1 doc (1863 chunks) p95 | 17.8 |
| redis | live Redis: get | 26.0 |
| redis | live Redis: rate_hit (Lua) | 26.3 |
| redis | live Redis: hgetall | 25.7 |
| redis | live Redis: 5 calls sequential | 125.4 |
| redis | live Redis: 5 calls gathered | 27.2 |
| redis | in-process store: set+get+rate_hit | 0.5 |
