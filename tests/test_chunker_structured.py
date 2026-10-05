"""The structure-aware chunker: invariants first (size, provenance, no loss),
then the behaviours it exists for (headings, tables, sentences, overlap)."""

import re
from pathlib import Path

import pytest

import chunker
from chunker import chunk_document, chunk_pages_structured, split_blocks, split_sentences

ROOT = Path(__file__).resolve().parent.parent

PROSE = (
    "The Transformer relies on attention. It replaces recurrence entirely. "
    "Training is much faster than before. Results improved by 2 BLEU. "
    "The model was trained on eight GPUs. It finished in 3.5 days. "
    "Dr. Smith et al. reported similar findings, e.g. on translation. "
    "Further work will extend this approach to other modalities."
)


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# ---------- invariants ----------


@pytest.mark.parametrize("size,overlap", [(120, 20), (200, 40), (300, 50), (800, 150)])
def test_no_chunk_exceeds_the_size_limit(size, overlap):
    pages = [(1, "Intro\n" + PROSE * 6), (2, "3.1\nMethods\n" + PROSE * 4), (3, "word " * 500)]
    for c in chunk_pages_structured(pages, size, overlap):
        assert 0 < len(c["text"]) <= size, (len(c["text"]), c["text"][:60])


def test_chunks_keep_their_page_and_never_mix_pages():
    pages = [(1, "Alpha one. " * 80), (2, "Beta two. " * 80), (3, "Gamma three. " * 80)]
    for c in chunk_pages_structured(pages, 200, 30):
        marker = {1: "alpha", 2: "beta", 3: "gamma"}[c["page"]]
        assert marker in c["text"].lower()
        assert not any(other in c["text"].lower() for other in {"alpha", "beta", "gamma"} - {marker})


def test_every_word_of_a_page_survives_chunking():
    page = "Overview\n" + PROSE * 5
    text = " ".join(c["text"] for c in chunk_pages_structured([(1, page)], 200, 0))
    assert set(words(page)) <= set(words(text))


def test_a_short_page_stays_whole_like_the_character_chunker():
    chunks = chunk_pages_structured([(1, "short text")], 800, 150)
    assert [(c["text"], c["page"]) for c in chunks] == [("short text", 1)]


def test_empty_and_whitespace_pages_produce_nothing():
    assert chunk_pages_structured([(1, ""), (2, "   \n  ")], 200, 20) == []


def test_chunk_count_is_sane_for_a_very_long_document():
    page = (PROSE + " ") * 400
    chunks = chunk_pages_structured([(1, page)], 800, 150)
    assert len(chunks) <= len(page) // 400  # no pathological explosion of tiny chunks
    assert sum(len(c["text"]) for c in chunks) >= len(page) * 0.9


@pytest.mark.parametrize("bad", [(0, 0), (-5, 0), (100, 100), (100, 250)])
def test_rejects_invalid_sizes(bad):
    with pytest.raises(ValueError):
        chunk_pages_structured([(1, "text")], *bad)


# ---------- headings, sections ----------


def test_a_split_numbered_heading_is_found_and_names_the_section():
    page = "3.1\nEncoder and Decoder Stacks\n" + PROSE * 3
    blocks = split_blocks(page)
    assert blocks[0].kind == "heading" and blocks[0].text == "3.1 Encoder and Decoder Stacks"
    chunks = chunk_pages_structured([(1, page)], 200, 30)
    assert chunks[0]["section"] == "3.1 Encoder and Decoder Stacks"
    assert all(c["section"] == "3.1 Encoder and Decoder Stacks" for c in chunks)


def test_sub_headings_nest_under_the_numbered_section_and_siblings_replace_each_other():
    stack = []
    assert chunker.push_heading(stack, "3 Model") == "3 Model"
    assert chunker.push_heading(stack, "3.1 Encoder and Decoder Stacks") == "3 Model > 3.1 Encoder and Decoder Stacks"
    assert chunker.push_heading(stack, "Encoder:") == "3 Model > 3.1 Encoder and Decoder Stacks > Encoder:"
    assert chunker.push_heading(stack, "Decoder:") == "3 Model > 3.1 Encoder and Decoder Stacks > Decoder:"
    assert chunker.push_heading(stack, "3.2 Attention") == "3 Model > 3.2 Attention"
    assert chunker.push_heading(stack, "4 Why Self-Attention") == "4 Why Self-Attention"
    stack = []
    assert chunker.push_heading(stack, "Abstract") == "Abstract"
    assert chunker.push_heading(stack, "1 Introduction") == "1 Introduction"


def test_later_chunks_carry_their_section_as_a_header_line():
    page = "METHODS\n" + PROSE * 6
    later = chunk_pages_structured([(1, page)], 200, 30)[1:]
    assert later and all(c["text"].startswith("[METHODS]\n") for c in later)


def test_the_section_continues_onto_the_next_page():
    pages = [(1, "2.1 Setup\n" + PROSE * 3), (2, PROSE * 3)]
    page2 = [c for c in chunk_pages_structured(pages, 200, 30) if c["page"] == 2]
    assert page2 and all(c["section"] == "2.1 Setup" for c in page2)


def test_a_new_section_starts_a_new_chunk_once_the_current_one_is_well_used():
    page = "ALPHA\n" + PROSE + "\nBETA\n" + PROSE
    chunks = chunk_pages_structured([(1, page)], 400, 0)
    beta = next(c for c in chunks if c["text"].startswith("BETA"))
    assert beta["section"] == "BETA"


def test_a_heading_is_never_left_at_the_end_of_a_chunk():
    page = (PROSE + " ") * 3 + "\nResults\n" + PROSE
    for c in chunk_pages_structured([(1, page)], 300, 0):
        assert not c["text"].rstrip().endswith("RESULTS")


def test_prose_lines_are_not_mistaken_for_headings():
    page = "Training took three and a half days on eight\nGPUs which was cheaper than before. We then evaluated."
    assert [b.kind for b in split_blocks(page)] == ["paragraph"]


# ---------- tables ----------

TABLE = "\n".join(["Table 2: Results"] + ["Model", "BLEU", "Cost"] + ["Base", "27.3", "3.3", "Big", "28.4", "2.3"])


def test_a_flattened_table_is_one_block_with_its_caption():
    blocks = split_blocks("Intro text here that is long enough to be a sentence.\n" + TABLE)
    tables = [b for b in blocks if b.kind == "table"]
    assert len(tables) == 1 and tables[0].text.startswith("Table 2: Results") and "Big\n28.4\n2.3" in tables[0].text


def test_a_table_is_not_split_across_chunks_when_it_fits():
    page = PROSE * 3 + "\n" + TABLE + "\n" + PROSE
    holders = [c for c in chunk_pages_structured([(1, page)], 400, 0) if "Base" in c["text"]]
    assert len(holders) == 1 and "Big" in holders[0]["text"] and "2.3" in holders[0]["text"]


def test_an_oversized_table_is_split_between_rows_and_repeats_its_header():
    big = "\n".join(["Table 9: Big"] + [f"row{i}" for i in range(80)])
    chunks = chunk_pages_structured([(1, big)], 150, 0)
    assert len(chunks) > 1 and all(c["text"].startswith("Table 9: Big") or c["text"].startswith("[") for c in chunks)
    assert all(len(c["text"]) <= 150 for c in chunks)


# ---------- sentences, overlap ----------


def test_sentences_are_not_cut_in_the_middle():
    chunks = chunk_pages_structured([(1, PROSE * 4)], 220, 0)
    for c in chunks:
        assert c["text"].rstrip().endswith((".", "!", "?"))  # always ends on a sentence boundary
        assert c["text"].lstrip("[").lstrip()[0].isupper() or c["text"].startswith("[")


def test_abbreviations_and_decimals_do_not_end_sentences():
    parts = split_sentences("Dr. Smith et al. found 3.5 days, e.g. on eight GPUs. Then it ended.")
    assert parts == ["Dr. Smith et al. found 3.5 days, e.g. on eight GPUs.", "Then it ended."]


def test_one_enormous_sentence_is_broken_at_word_boundaries():
    sentence = "word " * 300
    chunks = chunk_pages_structured([(1, sentence)], 200, 0)
    assert all(len(c["text"]) <= 200 for c in chunks) and all(not c["text"].endswith("wor") for c in chunks)


def test_an_unbroken_string_is_hard_cut_not_dropped():
    blob = "A" * 1000
    chunks = chunk_pages_structured([(1, blob)], 100, 0)
    assert sum(len(c["text"]) for c in chunks) == 1000 and all(len(c["text"]) <= 100 for c in chunks)


def test_the_last_sentences_are_repeated_as_overlap():
    chunks = chunk_pages_structured([(1, PROSE * 3)], 220, 120)
    for a, b in zip(chunks, chunks[1:]):
        tail = split_sentences(a["text"].split("\n", 1)[-1])[-1]
        assert tail.strip() in b["text"], (tail, b["text"])


def test_wrapped_lines_are_joined_and_hyphenated_breaks_rejoined():
    block = split_blocks("A model that is trans-\nduction based works well and is fast to\ntrain on modern hardware.")[0]
    assert block.text == "A model that is trans-duction based works well and is fast to train on modern hardware."


# ---------- real document ----------


def test_the_real_paper_chunks_cleanly_and_matches_the_old_chunker_on_content():
    from pdf_loader import load_pdf_pages

    pages = load_pdf_pages((ROOT / "sample_docs" / "dev_real_world_document.pdf").read_bytes())
    chunks = chunk_pages_structured(pages, 800, 150)
    assert all(len(c["text"]) <= 800 for c in chunks)
    pages_seen = {c["page"] for c in chunks}
    assert pages_seen == {p for p, _ in pages}
    sections = {c["section"] for c in chunks if c["section"]}
    assert any("3.2.2 Multi-Head Attention" in s for s in sections)
    assert any("3.1 Encoder and Decoder Stacks > Encoder:" in s for s in sections)
    # the flattened Table 2 stays in one place
    assert any("Transformer (big)\n28.4\n41.8" in c["text"] for c in chunks)


# ---------- dispatcher ----------


def test_the_dispatcher_follows_configuration(monkeypatch):
    pages = [(1, "Heading\n" + PROSE * 4)]
    monkeypatch.setenv("CHUNKER", "structured")
    assert "section" in chunk_document(pages, 200, 30)[0]
    monkeypatch.setenv("CHUNKER", "char")
    assert "section" not in chunk_document(pages, 200, 30)[0]
    monkeypatch.setenv("CHUNKER", "nonsense")
    assert chunker.chunker_name() == chunker.DEFAULT_CHUNKER
    assert "section" in chunk_document(pages, 200, 30, strategy="structured")[0]


# ---------- strong vs weak headings, dates, letter-spaced caps ----------


def test_a_bare_title_case_line_never_names_a_section():
    page = "Results\n" + PROSE * 4  # could equally be an entry, not a heading
    first = split_blocks(page)[0]
    assert first.kind == "heading" and first.strong is False
    assert all(c["section"] is None for c in chunk_pages_structured([(1, page)], 200, 30))


def test_a_weak_heading_still_keeps_chunks_from_ending_on_it():
    page = (PROSE + " ") * 3 + "\nResults\n" + PROSE
    for c in chunk_pages_structured([(1, page)], 300, 0):
        assert not c["text"].rstrip().endswith("Results")


def test_dates_and_years_are_not_section_numbers():
    page = "2026 - 2030\nUniversity of Somewhere\n" + PROSE * 3
    assert all(b.kind != "heading" or not b.strong or not b.text.startswith("2026") for b in split_blocks(page))
    assert chunker.push_heading([], "10.2 Appendix") == "10.2 Appendix"


def test_letter_spaced_capitals_are_a_strong_heading_even_mid_paragraph():
    page = "E D U C A T I O N\n" + PROSE * 3
    blocks = split_blocks(page)
    assert blocks[0].kind == "heading" and blocks[0].strong
    assert chunk_pages_structured([(1, page)], 200, 30)[0]["section"] == "E D U C A T I O N"
    mid = "Class XII, CBSE 82.4% and some text\nT E C H N I C A L S K I L L S\nPython and C++"
    assert any(b.kind == "heading" and b.strong for b in split_blocks(mid))


def test_a_resume_like_page_keeps_one_section_per_real_heading():
    page = "P R O F I L E\n" + PROSE + "\nS K I L L S\n" + PROSE + "\nB.Tech, Artificial Intelligence\n" + PROSE
    sections = {c["section"] for c in chunk_pages_structured([(1, page)], 300, 0)}
    assert sections == {"P R O F I L E", "S K I L L S"}
