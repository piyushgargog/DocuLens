"""Calibrate pipeline.RETRIEVAL_SCORE_FLOOR -- no LLM calls.

The floor withholds passages from the LLM when even the best of the top-4
hybrid hits has a cosine similarity below it. Too high and answerable questions
are refused; too low and it never fires. This script scores the best hit for
three kinds of question and prints how often each would be refused at a range
of floors:

  in-scope     questions the document answers (the labelled eval sets)
  off-topic    questions unrelated to any document ("capital of France?")
  cross-doc    questions written for one sample document, asked of the other

Usage:
    python score_floor_eval.py
    python score_floor_eval.py --output reports/score_floor_eval.md
"""

import argparse
import json
from pathlib import Path

import numpy as np

import pipeline

SAMPLES = Path("sample_docs")
FLOORS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
OFF_TOPIC = [
    "What is the capital of France?", "How do I bake sourdough bread?",
    "Who won the 2018 football world cup?", "What is the price of a flight to Tokyo?",
    "Explain the rules of chess.", "What is the best treatment for a common cold?",
    "Who is the current president of Brazil?", "How do I change a car tyre?",
    "What is the boiling point of mercury?", "Write a poem about the sea.",
    "What is the GDP of India?", "Which programming language is best for games?",
]


def best_score(question: str, state) -> float:
    hits = pipeline.retrieve([question], [state], top_k=pipeline.DEFAULT_TOP_K)
    return max(h["score"] for h in hits) if hits else -1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="write a Markdown report here")
    args = parser.parse_args()

    attention = pipeline.ingest((SAMPLES / "dev_real_world_document.pdf").read_bytes(), name="attention.pdf")
    solar = pipeline.ingest((SAMPLES / "sample.pdf").read_bytes(), name="sample.pdf")
    labelled = [i["question"] for i in json.loads((SAMPLES / "retrieval_eval_set.json").read_text("utf-8"))["items"]]
    attention_qs = json.loads((SAMPLES / "dev_real_world_questions.json").read_text("utf-8"))
    solar_qs = json.loads((SAMPLES / "dev_eval_questions.json").read_text("utf-8"))

    groups = {
        "in-scope: Attention paper (labelled set)": [best_score(q, attention) for q in labelled],
        "in-scope: Attention paper (dev questions)": [best_score(q, attention) for q in attention_qs],
        "in-scope: solar-system sample": [best_score(q, solar) for q in solar_qs if "France" not in q],
        "off-topic vs Attention paper": [best_score(q, attention) for q in OFF_TOPIC],
        "off-topic vs solar-system sample": [best_score(q, solar) for q in OFF_TOPIC],
        "cross-doc: solar questions vs Attention": [best_score(q, attention) for q in solar_qs],
        "cross-doc: Attention questions vs solar": [best_score(q, solar) for q in attention_qs],
    }

    header = "| Group | n | min | median | max | " + " | ".join(f"<{f:.2f}" for f in FLOORS) + " |"
    lines = [header, "|" + "---|" * (5 + len(FLOORS))]
    for name, values in groups.items():
        a = np.array(values)
        refused = " | ".join(f"{100 * np.mean(a < f):.0f}%" for f in FLOORS)
        lines.append(f"| {name} | {len(a)} | {a.min():.2f} | {np.median(a):.2f} | {a.max():.2f} | {refused} |")
    table = "\n".join(lines)
    note = (
        f"Current floor: {pipeline.RETRIEVAL_SCORE_FLOOR}. Cells under `<x` are the share of "
        "questions whose best top-4 hit scores below x, i.e. the share that would be refused "
        "without calling the LLM. In-scope rows should stay near 0%; off-topic and cross-doc rows "
        "should be high."
    )
    print(table, "\n\n" + note)
    if args.output:
        Path(args.output).write_text(f"# Retrieval score floor\n\n{note}\n\n{table}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
