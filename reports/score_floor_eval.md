# Retrieval score floor

Current floor: 0.0. Cells under `<x` are the share of questions whose best top-4 hit scores below x, i.e. the share that would be refused without calling the LLM. In-scope rows should stay near 0%; off-topic and cross-doc rows should be high.

**Caveat, learned the hard way (v3.6.1):** these sample documents and hand-written questions made a 0.25 floor look safe, but on a real one-page resume answerable questions scored 0.06-0.27 ("where did he work?" 0.06, "education?" 0.19, "main focus?" 0.21) -- the same range as an off-topic one (0.05) -- so the floor refused real questions. It is therefore off by default (RETRIEVAL_SCORE_FLOOR=0). Add your own short, real documents here before trusting any value.

| Group | n | min | median | max | <0.15 | <0.20 | <0.25 | <0.30 | <0.35 | <0.40 |
|---|---|---|---|---|---|---|---|---|---|---|
| in-scope: Attention paper (labelled set) | 39 | 0.14 | 0.51 | 0.73 | 3% | 3% | 3% | 8% | 10% | 26% |
| in-scope: Attention paper (dev questions) | 6 | 0.40 | 0.61 | 0.74 | 0% | 0% | 0% | 0% | 0% | 17% |
| in-scope: solar-system sample | 5 | 0.47 | 0.65 | 0.81 | 0% | 0% | 0% | 0% | 0% | 0% |
| off-topic vs Attention paper | 12 | 0.04 | 0.14 | 0.24 | 67% | 92% | 100% | 100% | 100% | 100% |
| off-topic vs solar-system sample | 12 | 0.00 | 0.09 | 0.28 | 92% | 92% | 92% | 100% | 100% | 100% |
| cross-doc: solar questions vs Attention | 6 | 0.06 | 0.11 | 0.16 | 83% | 100% | 100% | 100% | 100% | 100% |
| cross-doc: Attention questions vs solar | 6 | 0.02 | 0.06 | 0.10 | 100% | 100% | 100% | 100% | 100% | 100% |
