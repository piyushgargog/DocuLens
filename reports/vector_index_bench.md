# Vector index benchmark: exact vs HNSW

384-dimensional normalised vectors, 200 queries, top-10. HNSW: M=32, efConstruction=200, efSearch=128.
Single-threaded timings on the machine that ran the benchmark; compare the columns, not the absolute numbers.

| corpus | vectors | recall@10 | exact search ms | exact dense step ms (matrix) | HNSW search ms | HNSW top-200 ms | build exact s | build HNSW s | exact MB | HNSW MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clustered | 1,000 | 1.000 | 0.052 | 0.032 | 0.086 | 0.152 | 0.00 | 0.11 | 1.5 | 1.8 |
| clustered | 5,000 | 1.000 | 0.290 | 0.132 | 0.156 | 0.304 | 0.00 | 0.93 | 7.7 | 9.0 |
| clustered | 20,000 | 1.000 | 2.628 | 1.235 | 0.352 | 0.530 | 0.01 | 1.90 | 30.7 | 36.2 |
| clustered | 50,000 | 1.000 | 6.225 | 3.609 | 0.497 | 0.631 | 0.01 | 10.00 | 76.8 | 90.4 |
| clustered | 100,000 | 1.000 | 13.291 | 7.683 | 0.652 | 1.272 | 0.03 | 30.80 | 153.6 | 180.8 |
| random | 1,000 | 1.000 | 0.074 | 0.041 | 0.226 | 0.424 | 0.00 | 0.24 | 1.5 | 1.8 |
| random | 5,000 | 0.967 | 0.541 | 0.197 | 0.729 | 0.829 | 0.01 | 3.65 | 7.7 | 9.0 |
| random | 20,000 | 0.721 | 2.740 | 2.542 | 1.725 | 4.153 | 0.01 | 6.26 | 30.7 | 36.2 |
| random | 50,000 | 0.454 | 6.132 | 3.395 | 1.327 | 1.920 | 0.02 | 33.26 | 76.8 | 90.4 |
| random | 100,000 | 0.314 | 11.936 | 7.232 | 1.361 | 4.303 | 0.02 | 78.77 | 153.6 | 180.8 |

Corpora: `clustered` = noisy copies of random centres (closer to real embeddings); `random` = isotropic Gaussian
(the hard case for graph indexes). Neither is a real document collection.
