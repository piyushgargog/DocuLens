"""Property-based tests (Hypothesis): invariants that must hold for *any* input,
not just the examples someone thought of. A failing example is minimised and
should be added to the regression corpus in test_security_hardening.py."""

import json
import re

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import chunker
import conversation
import docstore
import document_loader
import llm_adapters
import llm_client
import main
import pdf_loader
from llm_errors import LLMRequestError

FAST = settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
SLOW = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])

# Text that looks like documents but includes every awkward character class.
awkward = st.text(
    alphabet=st.one_of(
        st.characters(blacklist_categories=("Cs",)),
        st.sampled_from(list("\n\n\r\t .,;:!?-()[]<>#|\u202e\u200b\ufeff\u00a0\u0000")),
    ),
    max_size=1500,
)
words = st.text(alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")), min_size=1, max_size=12)
prose = st.lists(words, min_size=1, max_size=120).map(" ".join)

CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f​-‏ -‮⁠-⁯﻿]")


# ---------- filenames ----------


@FAST
@given(st.one_of(st.none(), awkward))
def test_clean_filename_is_always_a_short_inert_basename(raw):
    name = main._clean_filename(raw)
    assert "/" not in name and "\\" not in name
    assert not CONTROL.search(name)
    assert len(name) <= main.MAX_FILENAME_CHARS
    assert main._clean_filename(name) == name  # idempotent


@FAST
@given(st.lists(st.sampled_from(["..", ".", "a", "b.pdf", "", "~", "%2e%2e"]), min_size=1, max_size=8), st.sampled_from(["/", "\\"]))
def test_path_traversal_attempts_never_survive(parts, sep):
    name = main._clean_filename(sep.join(parts))
    assert "/" not in name and "\\" not in name and name not in ("..", ".")  # never a traversal component


# ---------- chunking ----------


@SLOW
@given(st.lists(awkward, min_size=1, max_size=4), st.integers(60, 600), st.integers(0, 50))
def test_structured_chunks_respect_size_and_provenance(texts, size, overlap):
    overlap = min(overlap, size - 1)
    pages = [(i + 1, t) for i, t in enumerate(texts)]
    for c in chunker.chunk_pages_structured(pages, size, overlap):
        assert c["text"].strip() and len(c["text"]) <= size
        assert c["page"] in {p for p, _ in pages} and "section" in c


@SLOW
@given(st.lists(prose, min_size=1, max_size=4), st.integers(80, 500))
def test_structured_chunks_lose_no_words_without_overlap(texts, size):
    pages = [(i + 1, t) for i, t in enumerate(texts)]
    out = chunker.chunk_pages_structured(pages, size, 0)
    for page, text in pages:
        got = " ".join(c["text"] for c in out if c["page"] == page)
        for word in text.split():
            if len(word) < size - chunker.SECTION_PREFIX_MAX:  # words longer than a chunk are hard-cut by design
                assert word in got


@SLOW
@given(st.lists(awkward, min_size=1, max_size=3), st.integers(50, 400), st.integers(0, 40))
def test_char_chunks_are_substrings_of_their_page(texts, size, overlap):
    overlap = min(overlap, size - 1)
    pages = [(i + 1, t) for i, t in enumerate(texts)]
    by_page = dict(pages)
    for c in chunker.chunk_pages(pages, size, overlap):
        assert c["text"] in by_page[c["page"]] and len(c["text"]) <= max(size, len(by_page[c["page"]]) if len(by_page[c["page"]]) <= size else size)


@FAST
@given(awkward)
def test_sentence_splitting_loses_nothing(text):
    if not text.strip():
        return
    joined = re.sub(r"\s+", "", "".join(chunker.split_sentences(text)))
    assert joined == re.sub(r"\s+", "", text)


@FAST
@given(st.lists(st.text(max_size=40), max_size=12))
def test_section_stack_never_raises_and_stays_bounded(titles):
    stack = []
    for t in titles:
        path = chunker.push_heading(stack, t)
        assert isinstance(path, str) and len(stack) <= len(titles)


@SLOW
@given(awkward)
def test_block_splitting_never_raises_and_never_returns_empty_blocks(text):
    for block in chunker.split_blocks(text):
        assert block.text.strip() and block.kind in ("heading", "table", "paragraph")


# ---------- persistence ----------

texts_strategy = st.lists(st.text(min_size=1, max_size=200).filter(lambda s: s.strip()), min_size=1, max_size=12)


@SLOW
@given(texts_strategy, st.integers(4, 16), st.randoms(use_true_random=False))
def test_pack_unpack_roundtrips_and_any_flipped_byte_is_detected(texts, dim, rnd):
    rng = np.random.default_rng(rnd.randint(0, 2**31))
    emb = rng.normal(size=(len(texts), dim)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    chunks = [{"text": t, "page": i + 1} for i, t in enumerate(texts)]
    p = docstore.pack("d", "n.txt", 1, 800, 150, chunks, emb)
    got, vec = docstore.unpack(p.meta, p.chunks_blob, p.emb_blob)
    assert got == chunks and np.allclose(np.linalg.norm(vec, axis=1), 1.0, atol=1e-4)
    # flip one byte anywhere: it must be reported as corruption, never decoded or crashed on
    blob = bytearray(p.chunks_blob + p.emb_blob)
    pos = rnd.randrange(len(blob))
    blob[pos] ^= 0xFF
    cut = len(p.chunks_blob)
    with pytest.raises(docstore.DocumentCorruptError):
        docstore.unpack(p.meta, bytes(blob[:cut]), bytes(blob[cut:]))


@FAST
@given(st.text(max_size=200), st.text(max_size=200))
def test_owner_keys_are_fixed_width_and_distinct_for_distinct_users(a, b):
    ka, kb = docstore.owner_key(a), docstore.owner_key(b)
    assert re.fullmatch(r"[0-9a-f]{40}", ka) and (ka == kb) == (a == b)


@FAST
@given(st.text(max_size=300))
def test_only_url_safe_tokens_can_become_a_login_key(value):
    if main._TOKEN_RE.fullmatch(value):
        assert re.fullmatch(r"[A-Za-z0-9_-]{16,64}", value)


@FAST
@given(st.binary(max_size=2000))
def test_docx_safety_check_never_raises(data):
    assert document_loader.docx_is_safe(data) in (True, False)


@SLOW
@given(st.binary(max_size=4000), st.sampled_from(["a.pdf", "a.docx", "a.txt", "a.md", "a", "A.PDF", "x.exe"]))
def test_no_input_can_crash_document_loading(data, name):
    result = document_loader.load_document_ex(name, data)
    assert isinstance(result.pages, list) and all(isinstance(p, int) and isinstance(t, str) and t.strip() for p, t in result.pages)


# ---------- conversation ----------

questions = st.text(min_size=1, max_size=200)


@FAST
@given(questions, st.lists(st.fixed_dictionaries({"question": st.text(max_size=300), "answer": st.text(max_size=300)}), max_size=6))
def test_resolve_never_raises_and_always_searches_the_original_first(question, history):
    r = conversation.resolve(question, history, use_llm=False)
    assert r.queries[0] == question and len(r.queries) == len(set(r.queries)) and len(r.queries) <= 3


@FAST
@given(st.text(max_size=400), questions, st.lists(questions, max_size=3))
def test_a_rewrite_is_accepted_only_if_it_is_one_clean_line_of_known_words(rewrite, question, previous):
    out = conversation.validate_rewrite(rewrite, question, previous)
    if out is not None:
        assert "\n" not in out and len(out) <= conversation.MAX_REWRITE_CHARS and "<<<" not in out and ">>>" not in out
        known = {conversation._stem(w) for w in conversation._words(question)} | {conversation._stem(w) for q in previous for w in conversation._words(q)}
        for w in conversation._words(out):
            assert len(w) < 4 or w.lower() in conversation._STOP or conversation._stem(w) in known


# ---------- prompts ----------


@FAST
@given(awkward)
def test_sanitised_text_can_never_contain_a_fence_or_a_page_label(text):
    out = llm_client._sanitize_passage_text(text)
    assert not re.search("<{3,}|>{3,}", out)
    assert not re.search(r"\[[^\]\n]{0,80}?\bPages?\s*\d", out, re.I)
    assert llm_client._sanitize_passage_text(out) == out  # idempotent


@FAST
@given(awkward.filter(bool), st.integers(1, 999))  # an empty name falls back to the plain "[Page N]" label
def test_a_passage_label_is_always_a_single_bracketed_pair(doc, page):
    label = llm_client._passage_label({"doc": doc, "page": page})
    assert label.startswith("[") and label.endswith(f", Page {page}]") and label.count("[") == 1 and label.count("]") == 1


@SLOW
@given(awkward, awkward, awkward)
def test_whatever_the_document_says_the_prompt_has_exactly_one_fence_pair(passage, doc, question):
    prompt = llm_client.build_prompt(question.replace("<<<", "").replace(">>>", ""), [{"text": passage, "page": 1, "doc": doc}])
    assert prompt.count("<<<BEGIN PASSAGES>>>") == 1 and prompt.count("<<<END PASSAGES>>>") == 1


# ---------- OCR text ----------


@FAST
@given(awkward)
def test_normalised_ocr_text_has_no_control_characters_and_is_stable(text):
    out = pdf_loader.normalize_ocr(text)
    assert not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", out) and "\n\n\n" not in out
    assert pdf_loader.normalize_ocr(out) == out


# ---------- provider replies ----------

json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(max_size=20),
    lambda children: st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=4),
    max_leaves=12,
)


@FAST
@given(json_values, st.sampled_from(["openai", "anthropic", "gemini", "cohere"]))
def test_parsing_a_hostile_reply_raises_only_what_the_client_handles(data, api):
    try:
        out = llm_adapters.get(api).parse(data)
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return  # llm_client turns these into "could not read the LLM response"
    assert isinstance(out, str)


class _Lines:
    def __init__(self, lines):
        self.lines, self.encoding = lines, None

    def iter_lines(self, decode_unicode=False):
        yield from self.lines


@SLOW
@given(st.lists(st.text(max_size=80).map(lambda s: "data: " + s) | st.text(max_size=40), max_size=12), st.sampled_from(["openai", "anthropic", "gemini", "cohere"]))
def test_a_garbage_event_stream_yields_text_pieces_or_a_clean_error(lines, api):
    try:
        for kind, text in llm_adapters.get(api).stream(_Lines(lines)):
            assert kind in ("content", "reasoning") and isinstance(text, str)
    except LLMRequestError:
        pass


@FAST
@given(json_values)
def test_valid_event_json_of_any_shape_never_crashes_the_stream(event):
    for api in ("openai", "anthropic", "gemini", "cohere"):
        try:
            list(llm_adapters.get(api).stream(_Lines(["data: " + json.dumps(event)])))
        except LLMRequestError:
            pass
