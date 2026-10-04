# Retrieval score floor

Current floor: 0.25. Cells under `<x` are the share of questions whose best top-4 hit scores below x, i.e. the share that would be refused without calling the LLM. In-scope rows should stay near 0%; off-topic and cross-doc rows should be high.

| Group | n | min | median | max | <0.15 | <0.20 | <0.25 | <0.30 | <0.35 | <0.40 |
|---|---|---|---|---|---|---|---|---|---|---|
| in-scope: Attention paper (labelled set) | 39 | 0.14 | 0.51 | 0.73 | 3% | 3% | 3% | 8% | 10% | 26% |
| in-scope: Attention paper (dev questions) | 6 | 0.40 | 0.61 | 0.74 | 0% | 0% | 0% | 0% | 0% | 17% |
| in-scope: solar-system sample | 5 | 0.47 | 0.65 | 0.81 | 0% | 0% | 0% | 0% | 0% | 0% |
| off-topic vs Attention paper | 12 | 0.04 | 0.14 | 0.24 | 67% | 92% | 100% | 100% | 100% | 100% |
| off-topic vs solar-system sample | 12 | 0.00 | 0.09 | 0.28 | 92% | 92% | 92% | 100% | 100% | 100% |
| cross-doc: solar questions vs Attention | 6 | 0.06 | 0.11 | 0.16 | 83% | 100% | 100% | 100% | 100% | 100% |
| cross-doc: Attention questions vs solar | 6 | 0.02 | 0.06 | 0.10 | 100% | 100% | 100% | 100% | 100% | 100% |
