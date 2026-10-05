"""Conversational retrieval: turn a follow-up into search queries that stand alone.

"Who founded it?" after "Tell me about the company." has nothing to search
for on its own. This module decides whether a question depends on earlier
turns and, if so, produces the extra query to search with. Three rules govern it:

* **A rewrite is a search query, never evidence.** Only retrieved document
  passages reach the answer prompt as evidence; the rewrite just steers which
  passages are fetched. The user's original question is always searched too.
* **History is untrusted and is used only for its questions.** Earlier
  assistant answers are never read here (they could carry injected text, and
  they must not become a source of facts); the question text is reduced to
  plain keywords before it is reused, so no sentence of history is ever pasted
  into a query or a prompt by the deterministic path.
* **Cheap and local first.** A deterministic resolver handles pronouns and
  ellipsis. An LLM rewrite is tried only for the hard cases (references to a
  turn earlier than the last, "the previous one", comparisons) when
  REWRITE_LLM=auto (the default), is validated, cached, and falls back to the
  deterministic query and then to the original concatenation if it fails.

REWRITE_LLM=off|auto|always.
"""

import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field

import llm_client
from observability import log

MAX_CARRIED_WORDS = 8
MAX_REWRITE_CHARS = 200
CACHE_SIZE = 256
HISTORY_QUESTIONS_FOR_REWRITE = 3
MAX_QUESTION_CHARS_FOR_REWRITE = 300

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")

# Words that carry no topic: question scaffolding, auxiliaries, determiners, pronouns.
_STOP = frozenset(
    """a an the and or but if then than so as at by for from in into of on onto to with without about over under between
    is are was were be been being am do does did done doing have has had having can could should would will shall may might must
    what whats which who whom whose when where why how many much more most less least very also just only again further
    tell me us you your yours i my mine we our ours please explain describe give show list summarize summarise say mention
    it its they them their theirs he him his she her hers this that these those there here such same one ones each
    any some all both either neither other another former latter previous next first second third last
    yes no not now new use used uses using get got make made does""".split()
)

_PRONOUNS = frozenset("it its they them their theirs he him his she her hers this that these those".split())
_ELLIPSIS_OPENERS = (
    "and ", "also ", "then ", "what about", "how about", "what else", "why", "how come", "so ", "but ",
    "same ", "more ", "any other", "anything else",
)
_REFERS_BACK = re.compile(
    r"\b(the (previous|former|latter|first|second|third|last|other) (one|ones|document|doc|file|paper|model|method|section|question|answer)"
    r"|the (previous|former|latter|other|same)\b"
    r"|(earlier|previous(ly)?|before|above|mentioned)"
    r"|compare|comparison|versus|\bvs\b|differ(ent|ence)? from"
    r"|as (you|we) (said|mentioned))",
    re.IGNORECASE,
)
_VERBISH = frozenset(
    "is are was were be been am do does did has have had can could should would will shall may might must use uses used using say says said".split()
)
_ABOUT = re.compile(r"\b(?:about|of|on|regarding)\s+((?:the\s+|a\s+|an\s+)?[A-Za-z0-9][\w' -]{1,60}?)\s*(?:[?.!]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class Resolution:
    queries: list[str]  # search queries; the original question is always first
    method: str  # "standalone" | "deterministic" | "llm" | "fallback"
    rewrite: str | None = None  # the standalone query, when one was produced
    notes: list[str] = field(default_factory=list)


# ---------- deterministic resolver ----------


def _words(text: str) -> list[str]:
    return _WORD.findall(text)


def _content_words(text: str) -> list[str]:
    """Topic-bearing words in order, de-duplicated, lower-case except that
    acronyms and capitalised names keep their form."""
    seen, out = set(), []
    for word in _words(text):
        low = word.lower().strip("'-_")
        if not low or low in _STOP or len(low) > 30 or low in seen:
            continue
        seen.add(low)
        out.append(word if (word.isupper() or word[0].isupper()) and len(word) > 1 else low)
    return out


def needs_context(question: str) -> bool:
    """Does this question depend on an earlier turn? True for pronouns and
    ellipsis ("it", "what about…", "and the…"), explicit back-references, and
    very short questions with no specific topic of their own."""
    q = question.strip()
    low = q.lower()
    words = {w.lower() for w in _words(q)}
    if words & _PRONOUNS:
        return True
    if low.startswith(_ELLIPSIS_OPENERS) or _REFERS_BACK.search(q):
        return True
    return len(_content_words(q)) <= 1 and len(_words(q)) <= 5


def is_hard(question: str, history: list[dict]) -> bool:
    """Needs more than carrying keywords from the last question: an explicit
    back-reference ("the previous one", a comparison) or a pronoun whose last
    question offers nothing to substitute."""
    if _REFERS_BACK.search(question) and len(history) >= 1:
        return True
    last = history[-1]["question"] if history else ""
    return bool({w.lower() for w in _words(question)} & _PRONOUNS) and len(_content_words(last)) == 0 and len(history) >= 2


def topic_phrase(previous_question: str) -> str | None:
    """The thing a previous question was about, only when it says so outright
    ("about/of/on/regarding X"; "for X" is not used: "used for training" names a purpose, not a topic): "Tell me about the company." -> "the company". The
    phrase stops before the first verb, so "of the encoder does it have" gives
    "the encoder". Anything less explicit is left to keyword carry-over."""
    match = _ABOUT.search(previous_question.strip())
    if not match:
        return None
    kept = []
    for word in match.group(1).split():
        if word.lower().strip("'") in _VERBISH:
            break
        kept.append(word)
    phrase = " ".join(kept).strip()
    return phrase if _content_words(phrase) else None


def deterministic_query(question: str, history: list[dict]) -> str | None:
    """A standalone version of `question`, built only from words the user
    already typed, or None if the last turns give nothing to build on.

    1. A subject/object pronoun is replaced by the previous question's topic
       phrase when that is plain ("Who founded it?" -> "Who founded the company?").
    2. Otherwise the previous question's topic keywords are appended
       ("What is the dimension of each of them?" -> "... attention heads Transformer")."""
    if not history:
        return None
    previous = str(history[-1].get("question", ""))[:MAX_QUESTION_CHARS_FOR_REWRITE]
    topic = topic_phrase(previous)
    if topic:
        pattern = re.compile(r"\b(it|them|they|its|their|he|him|his|she|her|this|that|these|those)\b", re.IGNORECASE)
        match = pattern.search(question)
        if match and match.group(1).lower() not in {"this", "that"} | ({"these", "those"} if " " in topic else set()):
            replacement = topic if match.group(1).lower() not in {"its", "their", "his", "her"} else f"{topic}'s"
            return pattern.sub(replacement, question, count=1)
    carried = [w for w in _content_words(previous) if w.lower() not in {x.lower() for x in _content_words(question)}]
    if not carried:
        return None
    return f"{question.strip()} {' '.join(carried[:MAX_CARRIED_WORDS])}"


def legacy_queries(question: str, history: list[dict]) -> list[str]:
    """The pre-v4 behaviour (kept as the fallback and as a benchmark baseline):
    always also search "<previous question> <question>"."""
    queries = [question]
    if history:
        queries.append(f"{history[-1]['question']} {question}")
    return queries


# ---------- LLM rewrite (hard cases only) ----------

_cache: "OrderedDict[tuple, str | None]" = OrderedDict()


def _cache_get(key):
    if key in _cache:
        _cache.move_to_end(key)
        return True, _cache[key]
    return False, None


def _cache_put(key, value) -> None:
    _cache[key] = value
    _cache.move_to_end(key)
    while len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)


def clear_cache() -> None:
    _cache.clear()


def rewrite_mode() -> str:
    mode = os.environ.get("REWRITE_LLM", "auto").strip().lower()
    return mode if mode in ("off", "auto", "always") else "auto"


def _stem(word: str) -> str:
    return word.lower()[:5]


def validate_rewrite(rewrite: str, question: str, previous_questions: list[str]) -> str | None:
    """Accept the model's rewrite only if it is one short line and introduces no
    term that is absent from the question and the earlier questions it was given.
    A rewrite that adds facts of its own is discarded."""
    text = rewrite.strip().strip("\"'`").strip()
    if not text or "\n" in text or len(text) > MAX_REWRITE_CHARS:
        return None
    if "<<<" in text or ">>>" in text:
        return None
    allowed = {_stem(w) for w in _words(question)} | {_stem(w) for q in previous_questions for w in _words(q)}
    for word in _words(text):
        if len(word) >= 4 and word.lower() not in _STOP and _stem(word) not in allowed:
            return None
    return text


def llm_rewrite(question: str, history: list[dict]) -> str | None:
    """Ask the model for a standalone query. Returns None on any failure or on a
    rewrite that fails validation. Only the earlier *questions* are sent."""
    previous = [str(t.get("question", ""))[:MAX_QUESTION_CHARS_FOR_REWRITE] for t in history[-HISTORY_QUESTIONS_FOR_REWRITE:]]
    key = (question, tuple(previous))
    hit, cached = _cache_get(key)
    if hit:
        return cached
    try:
        raw = llm_client.rewrite_query(question, previous)
    except Exception as e:  # provider failure, rate limit, misconfiguration: never break the request
        log.warning("Query rewrite unavailable (%s); using the deterministic query", type(e).__name__)
        return None  # not cached: a later attempt may succeed
    checked = validate_rewrite(str(raw), question, previous)
    if checked is None:
        log.warning("Query rewrite rejected by validation")
    _cache_put(key, checked)
    return checked


# ---------- entry point ----------


def resolve(question: str, history: list[dict], use_llm: bool | None = None) -> Resolution:
    """The queries to search for `question` given the recent conversation.

    standalone     no dependence on earlier turns: just the question.
    deterministic  pronoun/ellipsis resolved from the last question.
    llm            a validated model rewrite (hard cases only).
    fallback       no usable rewrite (none to build, or the model's failed).
    For every follow-up (anything but "standalone") the v3 concatenation query is
    searched too."""
    recent = [t for t in history if isinstance(t, dict) and isinstance(t.get("question"), str)]
    if not recent or not needs_context(question):
        return Resolution([question], "standalone")

    queries = [question]
    method, rewrite = "deterministic", deterministic_query(question, recent)
    if rewrite and rewrite != question:
        queries.append(rewrite)

    wants_llm = rewrite_mode() == "always" or (rewrite_mode() == "auto" and is_hard(question, recent))
    if use_llm is not None:
        wants_llm = use_llm
    if wants_llm and rewrite_mode() != "off":
        better = llm_rewrite(question, recent)
        if better:
            method, rewrite = "llm", better
            queries = [question, better] + [q for q in queries[1:] if q != better]
        else:
            method = "fallback"
    elif rewrite is None:
        method = "fallback"  # nothing deterministic to carry over
    # The v3 concatenation stays as a safety net for every follow-up: measured in
    # conversation_eval.py, the resolved query alone loses to it on page recall,
    # and the three rankings fuse better than any one.
    queries = list(dict.fromkeys(queries + legacy_queries(question, recent)))
    return Resolution(queries, method, rewrite)
