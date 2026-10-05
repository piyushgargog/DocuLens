"""Hybrid retrieval: BM25 keyword scoring fused with embedding similarity.

`retrieval_eval.py` measured that on the evaluation set, fusing BM25 with
MiniLM (reciprocal rank fusion) beats MiniLM alone -- Hit@4 0.77 -> 0.82 and
MRR 0.56 -> 0.70 at the default chunking -- because keyword matching recovers
exact terms (numbers, names, symbols) that embeddings blur, while embeddings
keep handling paraphrase. The app and the evaluation import these same
functions, so the measured numbers describe what the app actually does.
"""

import math
import re
from collections import Counter

import numpy as np

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class TermIndex:
    """Per-document term statistics, computed once at ingestion so each query
    only has to look terms up."""

    def __init__(self, texts: list[str]):
        self.counts = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.df = Counter(term for c in self.counts for term in c)


def bm25_scores(query: str, indexes: list[TermIndex]) -> np.ndarray:
    """Okapi BM25 over the concatenated chunks of several documents, with
    document frequencies and average length taken over all of them -- so scores
    from different documents are comparable and can be ranked together.

    score(q, d) = sum over query terms t of
        idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    """
    lengths = [n for index in indexes for n in index.lengths]
    total = len(lengths)
    scores = np.zeros(total)
    if total == 0:
        return scores
    avgdl = (sum(lengths) / total) or 1.0
    for term in set(tokenize(query)):
        df = sum(index.df.get(term, 0) for index in indexes)
        if not df:
            continue
        idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
        offset = 0
        for index in indexes:
            for i, counts in enumerate(index.counts):
                tf = counts.get(term, 0)
                if tf:
                    norm = BM25_K1 * (1 - BM25_B + BM25_B * index.lengths[i] / avgdl)
                    scores[offset + i] += idf * tf * (BM25_K1 + 1) / (tf + norm)
            offset += len(index.counts)
    return scores


class BM25:
    """BM25 over one fixed list of texts (used by the evaluation script)."""

    def __init__(self, texts: list[str]):
        self.index = TermIndex(texts)

    def scores(self, query: str) -> np.ndarray:
        return bm25_scores(query, [self.index])


def ranking(scores: np.ndarray) -> list[int]:
    """Indices, best first (stable for ties)."""
    return list(np.argsort(-scores, kind="stable"))


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Combine rankings by summing 1 / (k + rank). It ignores the systems'
    score scales entirely, which is why it's the usual way to fuse BM25 and
    cosine similarity."""
    fused: dict[int, float] = {}
    for rank_list in rankings:
        for rank, idx in enumerate(rank_list, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    # sorted() is stable, so ties keep first-seen order (as Counter.most_common did)
    return sorted(fused, key=lambda idx: fused[idx], reverse=True)
