"""Does resolving follow-ups help retrieval? Measured, no LLM calls.

Each item in `sample_docs/followup_eval_set.json` is a follow-up that only makes
sense after a `previous` question ("What is the dimension of each of them?").
Four ways of searching for it are compared with the app's own `pipeline.retrieve`:

  alone         the follow-up exactly as typed (what a conversation-blind system does)
  concat (old)  v3 behaviour: also search "<previous question> <follow-up>"
  resolved      v4 `conversation.resolve` with the LLM rewrite off (deterministic)
  standalone    the follow-up written out by hand in the `standalone` field: an
                upper bound that no resolver can beat

Metrics as in retrieval_eval.py: Hit@4, MRR@10, Page-Hit@4; a chunk is relevant
if it contains the item's evidence phrase.

Usage:
    python conversation_eval.py [--output reports/conversation_eval.md] [--chunker structured]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

import conversation
import embedder
import pipeline
from chunker import chunk_document
from pdf_loader import load_pdf_pages

ROOT = Path(__file__).parent
TOP_K = 4
CUTOFF = 10


def normalize(text: str) -> str:
    return re.sub(r"-\s+", "-", re.sub(r"\s+", " ", text)).strip().lower()


def metrics(passages: list[dict], evidence: str, gold_pages: set[int]) -> dict:
    ev = normalize(evidence)
    ranks = [i for i, p in enumerate(passages[:CUTOFF], start=1) if ev in normalize(p["text"])]
    first = ranks[0] if ranks else None
    return {
        "hitk": float(bool(first and first <= TOP_K)),
        "mrr": 1.0 / first if first else 0.0,
        "pagehit": float(any(p["page"] in gold_pages for p in passages[:TOP_K])),
    }


def run(spec_path: str, chunker: str) -> dict:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    pages = load_pdf_pages((ROOT / spec["document"]).read_bytes())
    chunks = chunk_document(pages, 800, 150, strategy=chunker)
    state = pipeline.restore(chunks, embedder.embed([c["text"] for c in chunks]), "doc", len(pages), 800, 150)
    results = {name: [] for name in ("alone", "concat (old)", "resolved", "standalone")}
    by_kind: dict[str, dict] = {}
    rows = []
    for item in spec["items"]:
        history = [{"question": item["previous"], "answer": "ignored"}]
        resolved = conversation.resolve(item["question"], history, use_llm=False)
        queries = {
            "alone": [item["question"]],
            "concat (old)": conversation.legacy_queries(item["question"], history),
            "resolved": resolved.queries,
        }
        if item.get("standalone"):
            queries["standalone"] = [item["standalone"]]
        row = {"id": item["id"], "kind": item.get("kind", "follow_up"), "question": item["question"], "query": resolved.queries[1] if len(resolved.queries) > 1 else item["question"], "method": resolved.method}
        for name, qs in queries.items():
            passages = pipeline.retrieve(qs, [state], top_k=CUTOFF)
            m = metrics(passages, item["evidence"], set(item["pages"]))
            by_kind.setdefault(row["kind"], {}).setdefault(name, []).append(m)
            row[name] = m["hitk"]
        rows.append(row)
    summary = {
        kind: {n: {k: float(np.mean([m[k] for m in ms])) for k in ("hitk", "mrr", "pagehit")} | {"n": len(ms)} for n, ms in strategies.items()}
        for kind, strategies in by_kind.items()
    }
    return {"summary": summary, "rows": rows, "chunks": len(chunks), "chunker": chunker}


def report(out: dict) -> str:
    lines = [
        "# Conversational retrieval evaluation",
        "",
        f"{len(out['rows'])} follow-up questions on one document, {out['chunks']} chunks ({out['chunker']} chunker). "
        "No LLM calls. 'resolved' is the deterministic resolver only; the LLM rewrite is reserved for harder cases and is not scored here.",
        "",
    ]
    titles = {"follow_up": "Follow-ups (pronouns, ellipsis)", "topic_change": "Topic changes (self-contained question after an unrelated one)"}
    for kind, strategies in out["summary"].items():
        lines += ["", f"## {titles.get(kind, kind)}", "", f"| Strategy | Hit@{TOP_K} | MRR@{CUTOFF} | Page-Hit@{TOP_K} | n |", "|---|---|---|---|---|"]
        for name, m in strategies.items():
            lines.append(f"| {name} | {m['hitk']:.2f} | {m['mrr']:.2f} | {m['pagehit']:.2f} | {m['n']} |")
    lines += ["", "## Per question (Hit@4: alone / old concatenation / resolved)", ""]
    for r in out["rows"]:
        lines.append(f"- [{r['kind']}] #{r['id']} {r['question']} -> `{r['query']}` ({r['method']}): {r['alone']:.0f} / {r['concat (old)']:.0f} / {r['resolved']:.0f}")
    lines += [
        "",
        "Limitations: 15 questions, one document, written by the author; the previous question is always the one the follow-up refers to "
        "(no multi-turn drift, no references to earlier answers).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", default="sample_docs/followup_eval_set.json")
    parser.add_argument("--chunker", default="structured", choices=["char", "structured"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    text = report(run(args.set, args.chunker))
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
