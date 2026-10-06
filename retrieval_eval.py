"""Measure retrieval quality against a labelled question set -- no LLM calls.

`evaluate.py` compares whole answers from the LLM. This script isolates the
step that decides what the LLM gets to see: retrieval. Every question in the
set has the page(s) that answer it and a short evidence phrase copied from
that page; a retrieved chunk is *relevant* when it contains the phrase.

It compares four retrievers on the same chunks:
  - BM25          keyword baseline (implemented below, no extra dependency)
  - MiniLM        all-MiniLM-L6-v2, the embedding model the app uses
  - BGE-small     BAAI/bge-small-en-v1.5, a second Hugging Face model
  - Hybrid        BM25 + MiniLM fused with reciprocal rank fusion
across three chunking configurations, and reports:
  - Hit@1 / Hit@4   share of questions with a relevant chunk in the top 1 / 4
                    (4 is the app's top_k: what the LLM actually receives)
  - MRR@10          mean reciprocal rank of the first relevant chunk
  - Page-Hit@4      share of questions where a top-4 chunk is from a gold page

It can also compare chunking strategies (`--chunkers char structured`), add an
approximate-index system (`--ann`: dense retrieval through HNSW, the way the app
would for a very large index), skip the second embedding model (`--no-bge`) and
break results down by question category when the set labels them.

Usage:
    python retrieval_eval.py
    python retrieval_eval.py --set <set.json> --output reports/retrieval_eval.md
    python retrieval_eval.py --set sample_docs/retrieval_eval_set_v2.json --chunkers char structured --ann --no-bge
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from chunker import chunk_pages, chunk_pages_structured
from pdf_loader import load_pdf_pages
from retriever import BM25, ranking, reciprocal_rank_fusion, tokenize  # noqa: F401 (tokenize re-exported)
from vector_index import HNSWIndex

CHUNKERS = {"char": chunk_pages, "structured": chunk_pages_structured}
DENSE_CANDIDATES = 200  # pipeline.DENSE_CANDIDATES

TOP_K = 4  # pipeline.DEFAULT_TOP_K -- what the LLM is given
MRR_CUTOFF = 10

CHUNK_CONFIGS = {
    "A 300/50": (300, 50),
    "Default 800/150": (800, 150),
    "B 1000/200": (1000, 200),
}

# name -> (Hugging Face model id, prefix added to queries only). BGE models are
# trained with an instruction prefix on the query side; MiniLM is not.
EMBEDDING_MODELS = {
    "MiniLM": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "BGE-small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: "),
}

SYSTEMS = ["BM25", "MiniLM", "BGE-small", "Hybrid"]


def normalize(text: str) -> str:
    """Case, whitespace and line-wrap-hyphen insensitive form used to match
    evidence phrases to chunk text ("English-\\nto-German" == "English-to-German")."""
    return re.sub(r"-\s+", "-", re.sub(r"\s+", " ", text)).strip().lower()


def phrases_of(item: dict) -> list[str]:
    ev = item["evidence"]
    return ev if isinstance(ev, list) else [ev]


def question_metrics(order: list[int], relevant: set[int], chunk_pages_: list[int], gold_pages: set[int]) -> dict:
    first = next((r for r, idx in enumerate(order[:MRR_CUTOFF], start=1) if idx in relevant), None)
    return {
        "hit1": float(bool(set(order[:1]) & relevant)),
        "hitk": float(bool(set(order[:TOP_K]) & relevant)),
        "mrr": 1.0 / first if first else 0.0,
        "pagehit": float(any(chunk_pages_[i] in gold_pages for i in order[:TOP_K])),
    }


def validate_labels(pages: dict[int, str], items: list[dict]) -> None:
    """Every evidence phrase must literally occur on one of its gold pages."""
    bad = [
        item["id"]
        for item in items
        if not all(any(normalize(ph) in normalize(pages.get(p, "")) for p in item["pages"]) for ph in phrases_of(item))
    ]
    if bad:
        raise SystemExit(f"Evidence phrase not found on its gold page for item(s): {bad}")


def multi_metrics(order: list[int], relevant: list[set[int]], chunk_pages_: list[int], gold_pages: set[int]) -> dict:
    """Metrics for an item with several evidence phrases (a cross-page question):
    a result list *covers* the item only when every phrase has a relevant chunk in
    it, and Page-Hit needs every gold page present."""
    def covered(upto: int) -> bool:
        return all(set(order[:upto]) & rel for rel in relevant)

    first = next((r for r in range(1, MRR_CUTOFF + 1) if covered(r)), None)
    return {
        "hit1": float(covered(1)),
        "hitk": float(covered(TOP_K)),
        "mrr": 1.0 / first if first else 0.0,
        "pagehit": float(gold_pages <= {chunk_pages_[i] for i in order[:TOP_K]}),
    }


def item_metrics(order: list[int], item: dict, relevant: list[set[int]], pages_of: list[int]) -> dict:
    if len(relevant) == 1 and item.get("mode") != "all":
        return question_metrics(order, relevant[0], pages_of, set(item["pages"]))
    if item.get("mode") == "all":
        return multi_metrics(order, relevant, pages_of, set(item["pages"]))
    return question_metrics(order, set().union(*relevant), pages_of, set(item["pages"]))


def evaluate(pdf_path: str, items: list[dict], chunkers: tuple[str, ...] = ("char",), ann: bool = False, bge: bool = True) -> dict:
    pages_list = load_pdf_pages(Path(pdf_path).read_bytes())
    validate_labels(dict(pages_list), items)

    wanted = {n: v for n, v in EMBEDDING_MODELS.items() if bge or n == "MiniLM"}
    models = {name: (SentenceTransformer(model_id), prefix) for name, (model_id, prefix) in wanted.items()}
    systems = [s_ for s_ in SYSTEMS if s_ in {"BM25", "Hybrid", *wanted}] + (["Hybrid-HNSW"] if ann else [])
    results, misses, unreachable, timings = {}, {}, {}, {}

    for chunker in chunkers:
        for config_name, (size, overlap) in CHUNK_CONFIGS.items():
            label = config_name if chunkers == ("char",) else f"{chunker}: {config_name}"
            start = time.perf_counter()
            chunks = CHUNKERS[chunker](pages_list, chunk_size=size, chunk_overlap=overlap)
            chunk_seconds = time.perf_counter() - start
            texts = [normalize(c["text"]) for c in chunks]
            pages_of = [c["page"] for c in chunks]
            relevant = {
                item["id"]: [{i for i, t in enumerate(texts) if normalize(ph) in t} for ph in phrases_of(item)]
                for item in items
            }
            unreachable[label] = [i for i, rel in relevant.items() if any(not r for r in rel)]

            bm25 = BM25([c["text"] for c in chunks])
            chunk_vectors, hnsw = {}, None
            for name, (model, _) in models.items():
                start = time.perf_counter()
                chunk_vectors[name] = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
                timings[(label, name)] = time.perf_counter() - start
            if ann:
                hnsw = HNSWIndex(chunk_vectors["MiniLM"].shape[1])
                hnsw.add(chunk_vectors["MiniLM"].astype("float32"))

            per_system = {s_: [] for s_ in systems}
            latency = {s_: [] for s_ in systems}
            categories = {}
            for item in items:
                orders = {}
                t0 = time.perf_counter()
                orders["BM25"] = ranking(bm25.scores(item["question"]))
                latency["BM25"].append(time.perf_counter() - t0)
                for name, (model, prefix) in models.items():
                    t0 = time.perf_counter()
                    q = model.encode([prefix + item["question"]], normalize_embeddings=True)[0]
                    orders[name] = ranking(chunk_vectors[name] @ q)
                    latency[name].append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                orders["Hybrid"] = reciprocal_rank_fusion([orders["BM25"], orders["MiniLM"]])
                latency["Hybrid"].append(latency["BM25"][-1] + latency["MiniLM"][-1] + (time.perf_counter() - t0))
                if ann:
                    t0 = time.perf_counter()
                    qv = models["MiniLM"][0].encode([item["question"]], normalize_embeddings=True)[0].astype("float32")
                    sims = np.full(len(chunks), -1.0, dtype="float32")
                    scores, ids = hnsw.search(qv, DENSE_CANDIDATES)
                    sims[ids] = scores
                    orders["Hybrid-HNSW"] = reciprocal_rank_fusion([orders["BM25"], ranking(sims)])
                    latency["Hybrid-HNSW"].append(time.perf_counter() - t0 + latency["BM25"][-1])
                for system, order in orders.items():
                    m = item_metrics(order, item, relevant[item["id"]], pages_of)
                    per_system[system].append(m)
                    if item.get("category"):
                        categories.setdefault((item["category"], system), []).append(m)
                    if not m["hitk"]:
                        misses.setdefault((label, system), []).append(item["id"])

            mean = lambda ms, k: float(np.mean([m[k] for m in ms]))  # noqa: E731
            results[label] = {
                "num_chunks": len(chunks),
                "avg_chunk_chars": float(np.mean([len(c["text"]) for c in chunks])),
                "max_chunk_chars": max(len(c["text"]) for c in chunks),
                "chunking_seconds": chunk_seconds,
                "systems": {
                    s_: {**{k: mean(ms, k) for k in ("hit1", "hitk", "mrr", "pagehit")}, "latency_ms": 1000 * float(np.mean(latency[s_]))}
                    for s_, ms in per_system.items()
                },
                "categories": {
                    f"{cat}|{system}": {k: mean(ms, k) for k in ("hit1", "hitk", "mrr", "pagehit")} | {"n": len(ms)}
                    for (cat, system), ms in categories.items()
                },
            }
    return {"results": results, "misses": misses, "unreachable": unreachable, "timings": timings}


def format_report(pdf_path: str, items: list[dict], out: dict) -> str:
    lines = [
        "# Retrieval Evaluation",
        "",
        f"Document: `{pdf_path}` — {len(items)} labelled questions.",
        f"A chunk is relevant if it contains the question's evidence phrase (for cross-page items, every phrase must be covered). k = {TOP_K} (the app's top_k).",
        "Produced by `python retrieval_eval.py`; no LLM calls, fully deterministic.",
        "",
    ]
    for config_name, res in out["results"].items():
        lines += [
            f"## Chunking {config_name} ({res['num_chunks']} chunks, mean {res.get('avg_chunk_chars', 0):.0f} chars, max {res.get('max_chunk_chars', 0)})",
            "",
            f"| Retriever | Hit@1 | Hit@{TOP_K} | MRR@{MRR_CUTOFF} | Page-Hit@{TOP_K} | ms/query |",
            "|---|---|---|---|---|---|",
        ]
        for system, m in res["systems"].items():
            n = len(items)
            lines.append(
                f"| {system} | {m['hit1']:.2f} ({round(m['hit1'] * n)}/{n}) | {m['hitk']:.2f} "
                f"({round(m['hitk'] * n)}/{n}) | {m['mrr']:.2f} | {m['pagehit']:.2f} | {m.get('latency_ms', 0):.1f} |"
            )
        if out["unreachable"][config_name]:
            lines.append(f"\nEvidence split across chunks (unreachable): {out['unreachable'][config_name]}")
        cats = sorted({k.split("|")[0] for k in res.get("categories", {})})
        if cats:
            shown = [s_ for s_ in ("BM25", "MiniLM", "Hybrid", "Hybrid-HNSW") if f"{cats[0]}|{s_}" in res["categories"]]
            lines += ["", f"By category, Hit@{TOP_K} (n = questions):", "", "| Category | n | " + " | ".join(shown) + " |", "|---|---|" + "---|" * len(shown)]
            for cat in cats:
                n_cat = res["categories"][f"{cat}|{shown[0]}"]["n"]
                lines.append(f"| {cat} | {n_cat} | " + " | ".join(f"{res['categories'][f'{cat}|{s_}']['hitk']:.2f}" for s_ in shown) + " |")
        lines.append("")

    lines += ["## Questions missed in the top 4", ""]
    by_id = {item["id"]: item["question"] for item in items}
    for (config_name, system), ids in sorted(out["misses"].items()):
        shown = "; ".join(f"#{i} {by_id[i]}" for i in ids)
        lines.append(f"- **{system}, {config_name}** ({len(ids)}): {shown}")
    lines += ["", "## Chunk embedding time (seconds, CPU)", ""]
    for (config_name, name), seconds in out["timings"].items():
        lines.append(f"- {name}, {config_name}: {seconds:.2f}")
    return "\n".join(lines) + "\n"


# Categorical slots 1-4 of the dataviz reference palette, validated for
# colour-vision deficiency on a light surface. Two sit below 3:1 contrast, so
# every bar carries a visible value label (and the report has the table).
SERIES_COLORS = {"BM25": "#2a78d6", "MiniLM": "#eb6834", "BGE-small": "#1baf7a", "Hybrid": "#eda100"}


def write_svg_chart(results: dict, path: str, metric: str = "hitk") -> None:
    """Grouped bar chart of one metric: chunking configs on x, retrievers as bars.
    Plain SVG, so the report needs no plotting library."""
    width, height = 760, 400
    left, right, top, bottom = 56, 16, 84, 52
    plot_w, plot_h = width - left - right, height - top - bottom
    configs = list(results)
    group_w = plot_w / len(configs)
    bar_w, gap = 34, 2
    y = lambda v: top + plot_h * (1 - v)  # noqa: E731 -- 0..1 metric to pixels

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="system-ui, -apple-system, Segoe UI, sans-serif" role="img" '
        f'aria-label="Hit@{TOP_K} by retriever and chunking configuration">',
        f'<rect width="{width}" height="{height}" fill="#fcfcfb"/>',
        f'<text x="{left}" y="26" font-size="16" font-weight="700" fill="#1a1a19">'
        f"Share of questions with a relevant chunk in the top {TOP_K} (Hit@{TOP_K})</text>",
    ]
    # Legend, one row under the title.
    lx = left
    for system, color in SERIES_COLORS.items():
        parts.append(f'<rect x="{lx}" y="42" width="12" height="12" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{lx + 18}" y="52" font-size="13" fill="#3d3d3a">{system}</text>')
        lx += 18 + round(7.6 * len(system)) + 26
    # Recessive grid and y labels.
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" '
                     f'stroke="{"#9a998f" if v == 0 else "#e4e3dc"}" stroke-width="1"/>')
        parts.append(f'<text x="{left - 8}" y="{y(v) + 4:.1f}" font-size="12" fill="#6b6a62" '
                     f'text-anchor="end">{v:.2f}</text>')
    # Bars: 4px rounded data-end, square at the baseline, 2px gaps.
    for gi, config in enumerate(configs):
        systems = results[config]["systems"]
        cluster = len(SERIES_COLORS) * bar_w + (len(SERIES_COLORS) - 1) * gap
        x0 = left + gi * group_w + (group_w - cluster) / 2
        for si, (system, color) in enumerate(SERIES_COLORS.items()):
            if system not in systems:
                continue
            value = systems[system][metric]
            x, yt, yb, r = x0 + si * (bar_w + gap), y(value), y(0), 4
            parts.append(
                f'<path d="M{x:.1f},{yb:.1f} V{yt + r:.1f} Q{x:.1f},{yt:.1f} {x + r:.1f},{yt:.1f} '
                f'H{x + bar_w - r:.1f} Q{x + bar_w:.1f},{yt:.1f} {x + bar_w:.1f},{yt + r:.1f} V{yb:.1f} Z" '
                f'fill="{color}"><title>{system}, {config}: {value:.2f}</title></path>'
            )
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{yt - 6:.1f}" font-size="11" fill="#3d3d3a" '
                         f'text-anchor="middle">{value:.2f}</text>')
        parts.append(f'<text x="{left + gi * group_w + group_w / 2:.1f}" y="{height - bottom + 22}" '
                     f'font-size="13" fill="#3d3d3a" text-anchor="middle">Chunking {config}</text>')
    parts.append("</svg>")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(parts) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="sample_docs/retrieval_eval_set.json", help="Labelled question set (JSON)")
    parser.add_argument("--output", default=None, help="Write the Markdown report here (default: print it)")
    parser.add_argument("--json", default=None, help="Also write raw metrics as JSON here")
    parser.add_argument("--chart", default=None, help="Also write a Hit@4 bar chart (SVG) here")
    parser.add_argument("--chunkers", nargs="+", default=["char"], choices=sorted(CHUNKERS), help="Chunking strategies to compare")
    parser.add_argument("--ann", action="store_true", help="Add a Hybrid-HNSW system (dense retrieval through an HNSW index)")
    parser.add_argument("--no-bge", action="store_true", help="Skip the BGE-small model (faster)")
    args = parser.parse_args()

    spec = json.loads(Path(args.set).read_text(encoding="utf-8"))
    out = evaluate(spec["document"], spec["items"], tuple(args.chunkers), ann=args.ann, bge=not args.no_bge)
    report = format_report(spec["document"], spec["items"], out)

    if args.chart:
        write_svg_chart(out["results"], args.chart)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out["results"], indent=2), encoding="utf-8")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
