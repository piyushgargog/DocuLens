"""Phase 2: page-aware chunking.

Two strategies produce the same chunk shape, {"text", "page"} (+ "section" for
the structured one), so everything downstream is unchanged:

* `chunk_pages` -- the original: an overlapping character window per page.
  Predictable and cheap, but it cuts sentences, headings and table rows
  wherever the window happens to end.
* `chunk_pages_structured` -- reads each page as blocks (headings, tables,
  paragraphs), packs whole sentences up to the size limit, keeps a table's rows
  together, never leaves a heading stranded at the end of a chunk, repeats the
  last sentence(s) as overlap, and records the section a chunk belongs to.

`chunk_document` picks one by CHUNKER=char|structured. Which is the default is
decided by `retrieval_eval.py`, not by taste (see DECISIONS.md): an earlier
sentence-aware attempt lowered retrieval quality and was reverted.

Text from PDFs wraps lines mid-sentence and flattens tables to one cell per
line, so blocks are found from line shape, not only from blank lines.
"""

import os
import re
from dataclasses import dataclass

DEFAULT_CHUNKER = "structured"  # chosen from retrieval_eval.py results, see DECISIONS.md (CHUNKER=char restores the old one)

SECTION_PREFIX_MAX = 90  # most of a chunk the "[Section]" header line may take
TABLE_MIN_LINES = 5  # this many short lines in a row look like a flattened table
TABLE_LINE_MAX = 45
MIN_TAIL = 0.15  # a final chunk shorter than this share of the size limit is merged back if it fits
HEADING_MAX_CHARS = 80
HEADING_MAX_WORDS = 12
TITLE_SMALL_WORDS = frozenset({"of", "the", "and", "in", "for", "to", "a", "an", "on", "with", "is", "all", "you", "need"})


# ---------- original character chunker ----------


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[dict]:
    """Split each page's text into overlapping character-window chunks.

    Each chunk keeps a reference to the page it came from. Returns a list
    of {"text": str, "page": int} dicts, in document order.
    """
    _check(chunk_size, chunk_overlap)
    chunks = []
    step = chunk_size - chunk_overlap
    for page_num, text in pages:
        if len(text) <= chunk_size:
            chunks.append({"text": text, "page": page_num})
            continue
        start = 0
        while start < len(text):
            piece = text[start : start + chunk_size]
            if piece.strip():
                chunks.append({"text": piece, "page": page_num})
            if start + chunk_size >= len(text):
                break
            start += step
    return chunks


def _check(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")


# ---------- structure detection ----------


@dataclass
class Block:
    kind: str  # "heading" | "table" | "paragraph"
    text: str
    strong: bool = True  # headings only: confident enough to name a section (see _heading_strength)


# Section numbers have 1-2 digit parts ("3", "3.2.1"); a 4-digit "2026 - 2030" is a date.
_NUMBER_ONLY = re.compile(r"^\d{1,2}(\.\d{1,2})*\.?$")
_NUMBERED_HEADING = re.compile(r"^(\d{1,2}(\.\d{1,2})*\.?|[A-Z]\.|[IVX]+\.)\s+\S")
_MD_HEADING = re.compile(r"^#{1,6}\s+\S")
_CAPTION = re.compile(r"^(table|figure|fig\.)\s*\d+", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?:;][\"')\]]?$")
_ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "fig.", "eq.", "vs.", "cf.", "no.", "dr.", "mr.", "mrs.", "approx.", "sec.")
_SPLIT_AFTER = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")


def _heading_strength(line: str, continues: bool, previous_ended: bool) -> str | None:
    """Is this short standalone line a title for what follows? "strong" when the
    shape is unambiguous (markdown #, a section number, a trailing colon, ALL
    CAPS); "weak" for a bare Title Case line, which in a resume or a table of
    names looks identical to an entry ("B.Tech, Artificial Intelligence"). A weak
    heading still keeps a chunk from ending on it but never names a section.

    `continues`: there is text after it. `previous_ended`: the line before it
    finished a sentence (or there is none), so it is not a mid-sentence wrap."""
    s = line.strip()
    if not s or len(s) > HEADING_MAX_CHARS or not continues:
        return None
    if _MD_HEADING.match(s):
        return "strong"
    words = s.split()
    spaced = len(words) >= 4 and sum(len(w) == 1 for w in words) >= 0.6 * len(words)  # "E D U C A T I O N"
    if len(words) > (HEADING_MAX_WORDS * 2 if spaced else HEADING_MAX_WORDS):
        return None
    if spaced and all(w.isupper() for w in words if w[0].isalpha()):
        return "strong"  # letter-spaced capitals are unambiguous, whatever precedes them
    if _NUMBERED_HEADING.match(s):  # "3.2 Multi-Head Attention", "A. Appendix"
        return "strong" if not s.endswith((".", ",", ";")) else None
    if not previous_ended:
        return None
    if s.endswith(":"):  # "Encoder:"
        return "strong" if len(words) <= 5 else None
    if s.endswith((".", ",", ";", ")", "]", "?", "!")) or any(ch.isdigit() for ch in s) or len(words) > 8:
        return None
    letters = [w for w in words if w[0].isalpha()]
    if not letters or not all(w[0].isupper() or w.lower() in TITLE_SMALL_WORDS for w in letters):
        return None
    return "strong" if all(w.isupper() for w in letters) else "weak"


def _short(line: str) -> bool:
    s = line.strip()
    return 0 < len(s) <= TABLE_LINE_MAX and not _SENTENCE_END.search(s)


def split_blocks(text: str) -> list[Block]:
    """Read a page's text as a sequence of blocks.

    * a run of at least TABLE_MIN_LINES short, non-sentence lines (a flattened
      table or figure labels) is one `table` block; a "Table 2: ..." caption
      just before it is kept with it;
    * a numbered heading split over two lines ("3.1" / "Encoder") is joined;
    * other heading-shaped lines become `heading` blocks;
    * the rest, joined across line wraps and split at blank lines, are paragraphs.
    """
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n")]
    blocks: list[Block] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(Block("paragraph", _join_wrapped(paragraph)))
            paragraph.clear()

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line:
            flush()
            i += 1
            continue
        run_end = i
        while run_end < n and _short(lines[run_end]):
            run_end += 1
        if run_end - i >= TABLE_MIN_LINES:
            flush()
            rows = "\n".join(lines[i:run_end])
            if blocks and blocks[-1].kind == "paragraph" and _CAPTION.match(blocks[-1].text) and len(blocks[-1].text) < 400:
                rows = blocks.pop().text + "\n" + rows
            blocks.append(Block("table", rows))
            i = run_end
            continue
        title = lines[i + 1] if i + 1 < n else ""
        if (
            _NUMBER_ONLY.match(line)
            and title
            and len(title) <= HEADING_MAX_CHARS
            and title[0].isupper()
            and not title.endswith((".", ",", ";"))
        ):
            flush()
            blocks.append(Block("heading", f"{line} {title}"))
            i += 2
            continue
        following = next((ln for ln in lines[i + 1 :] if ln), None)
        previous_ended = not paragraph or bool(_SENTENCE_END.search(paragraph[-1]))
        strength = _heading_strength(line, following is not None, previous_ended)
        if strength:
            flush()
            blocks.append(Block("heading", line.lstrip("# ").strip(), strong=strength == "strong"))
            i += 1
            continue
        if _CAPTION.match(line) and paragraph:
            flush()
        paragraph.append(line)
        i += 1
    flush()
    return blocks


def _join_wrapped(lines: list[str]) -> str:
    """Join PDF-wrapped lines into running text; a trailing hyphen joins its two
    halves without a space (kept as written: "English-" + "to-German" cannot be
    told apart from a split word)."""
    out = ""
    for ln in lines:
        out = ln if not out else (out + ln if out.endswith("-") else out + " " + ln)
    return re.sub(r"[ \t]+", " ", out).strip()


def split_sentences(text: str) -> list[str]:
    """Split running text after . ! ? when a new sentence plausibly begins,
    without breaking on common abbreviations, initials or decimals."""
    pieces, start = [], 0
    for match in _SPLIT_AFTER.finditer(text):
        head = text[start : match.start() + 1]
        last = head.rsplit(None, 1)[-1].lower() if head.split() else ""
        if last.endswith(_ABBREVIATIONS) or re.search(r"(^|\s)[A-Z]\.$", head):
            continue
        pieces.append(text[start : match.end()].strip())
        start = match.end()
    rest = text[start:].strip()
    if rest:
        pieces.append(rest)
    return pieces or [text.strip()]


# ---------- structured chunker ----------


@dataclass
class Unit:
    text: str
    kind: str  # "heading" | "table" | "sentence"
    section: str | None  # the section path in force at this unit (a heading carries its own)


_LEVEL = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)(?!\d)")


def push_heading(stack: list[tuple[float, str, bool]], title: str) -> str:
    """Update the section stack for a new heading and return the section path,
    e.g. "3.1 Encoder and Decoder Stacks > Encoder:". A numbered heading's level
    is its number of parts ("3.2.1" is 3) and replaces anything at its level or
    deeper. An unnumbered heading ("Encoder:", "Residual Dropout") sits half a
    level under the last numbered one, replacing sibling unnumbered headings."""
    match = _LEVEL.match(title)
    numbered = bool(match) or bool(re.match(r"^[A-Z]\.\s|^[IVX]+\.\s", title))
    if match:
        level = float(len(match.group(1).split(".")))
    elif numbered:
        level = 1.0
    else:
        parents = [lv for lv, _, is_numbered in stack if is_numbered]
        level = (max(parents) + 0.5) if parents else 1.0
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title, numbered))
    return " > ".join(t for _, t, _ in stack)


def _split_words(text: str, limit: int) -> list[str]:
    """Break one over-long unit at word boundaries (hard-cut an unbroken string)."""
    parts, current = [], ""
    for word in text.split():
        while len(word) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.append(word[:limit])
            word = word[limit:]
        if current and len(current) + 1 + len(word) > limit:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        parts.append(current)
    return parts


def _split_table(table: str, limit: int) -> list[str]:
    """Break a table longer than `limit` between rows, repeating its first line
    (caption or header) on every piece so each says what it belongs to."""
    rows = table.split("\n")
    header = rows[0] if len(rows[0]) <= limit // 3 else ""
    body = rows[1:] if header else rows
    room = limit - (len(header) + 1 if header else 0)
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for row in body:
        row_parts = _split_words(row, room) if len(row) > room else [row]
        for part in row_parts:
            if current and size + len(part) + 1 > room:
                pieces.append("\n".join(([header] if header else []) + current))
                current, size = [], 0
            current.append(part)
            size += len(part) + 1
    if current:
        pieces.append("\n".join(([header] if header else []) + current))
    return pieces or [table[:limit]]


def _prefix(section: str | None, cap: int) -> str:
    return f"[{section[: cap - 4].rstrip()}]\n" if section else ""


def _page_units(
    blocks: list[Block], stack: list[tuple[float, str, bool]], section: str | None, limit: int
) -> tuple[list[Unit], str | None]:
    """Blocks -> packable units (headings, whole tables, sentences), each cut to
    at most `limit` characters, plus the section path in force after the page.
    `stack` (the open headings) is updated in place."""
    units: list[Unit] = []
    for block in blocks:
        if block.kind == "heading":
            if block.strong:
                section = push_heading(stack, block.text)
            units.append(Unit(block.text if len(block.text) <= limit else block.text[:limit], "heading", section))
        elif block.kind == "table":
            pieces = [block.text] if len(block.text) <= limit else _split_table(block.text, limit)
            units.extend(Unit(p, "table", section) for p in pieces)
        else:
            for sentence in split_sentences(block.text):
                pieces = [sentence] if len(sentence) <= limit else _split_words(sentence, limit)
                units.extend(Unit(p, "sentence", section) for p in pieces)
    return units, section


def _render(units: list[Unit]) -> str:
    out, prev_kind = "", ""
    for u in units:
        out = u.text if not out else out + ("\n" if "table" in (u.kind, prev_kind) else " ") + u.text
        prev_kind = u.kind
    return out


def _pack(units: list[Unit], page: int, size: int, overlap: int, prefix_cap: int) -> list[dict]:
    """Greedy packing: each chunk is at most `size` characters including its
    "[Section]" header line; overlap is whole trailing sentences."""
    chunks: list[dict] = []
    current: list[Unit] = []
    used = 0  # rendered length of `current`, without the header

    def header_for(first: Unit) -> str:
        return "" if first.kind == "heading" else _prefix(first.section, prefix_cap)

    def budget() -> int:
        return size - len(header_for(current[0])) if current else size

    def add(unit: Unit) -> None:
        nonlocal used
        used = used + 1 + len(unit.text) if current else len(unit.text)
        current.append(unit)

    def fits(unit: Unit) -> bool:
        if not current:
            return len(unit.text) + len(header_for(unit)) <= size
        return used + 1 + len(unit.text) <= budget()

    def close(carry: bool) -> None:
        nonlocal current, used
        if not current:
            return
        stranded = current.pop() if len(current) > 1 and current[-1].kind == "heading" else None
        first = current[0]
        chunks.append({"text": (header_for(first) + _render(current)).strip(), "page": page, "section": first.section})
        kept: list[Unit] = []
        if carry and overlap and stranded is None:
            total = 0
            for u in reversed(current):
                if u.kind != "sentence" or total + len(u.text) + 1 > overlap:
                    break
                kept.insert(0, u)
                total += len(u.text) + 1
        current = []
        used = 0
        for u in kept:
            add(u)
        if stranded is not None:
            add(stranded)

    for unit in units:
        if unit.kind == "heading" and current and used > budget() * 0.5:
            close(carry=False)  # a new section starts a new chunk once this one is well used
        if current and not fits(unit):
            close(carry=unit.kind != "table")
            if current and not fits(unit):
                current, used = [], 0  # the carried overlap leaves no room: drop it
        add(unit)
    close(carry=False)
    return chunks


def chunk_pages_structured(
    pages: list[tuple[int, str]],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[dict]:
    """Structure-aware chunks: {"text", "page", "section"}.

    Every chunk is at most `chunk_size` characters (its "[Section]" header line
    included) and belongs to exactly one page. A page that fits in one chunk
    stays whole, like the character chunker. Sentences are never cut unless a
    single sentence exceeds the limit; a table is kept together unless it
    exceeds the limit, and is then split between rows."""
    _check(chunk_size, chunk_overlap)
    prefix_cap = min(SECTION_PREFIX_MAX, max(8, chunk_size // 4))
    limit = chunk_size - prefix_cap
    chunks: list[dict] = []
    section: str | None = None
    stack: list[tuple[float, str, bool]] = []

    for page_num, text in pages:
        stripped = text.strip()
        if not stripped:
            continue
        blocks = split_blocks(stripped)
        entry_section = section
        if len(stripped) <= chunk_size:
            for block in blocks:
                if block.kind == "heading" and block.strong:
                    section = push_heading(stack, block.text)
            chunks.append({"text": stripped, "page": page_num, "section": entry_section})
            continue
        units, section = _page_units(blocks, stack, section, limit)
        packed = _pack(units, page_num, chunk_size, chunk_overlap, prefix_cap)
        if (
            len(packed) > 1
            and len(packed[-1]["text"]) < MIN_TAIL * chunk_size
            and len(packed[-2]["text"]) + 1 + len(packed[-1]["text"]) <= chunk_size
        ):
            packed[-2]["text"] += "\n" + packed.pop()["text"]
        chunks.extend(c for c in packed if c["text"].strip())
    return chunks


# ---------- dispatcher ----------


def chunker_name() -> str:
    setting = os.environ.get("CHUNKER", DEFAULT_CHUNKER).strip().lower()
    return setting if setting in ("char", "structured") else DEFAULT_CHUNKER


def chunk_document(
    pages: list[tuple[int, str]],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    strategy: str | None = None,
) -> list[dict]:
    """Chunk a document with the given strategy, or the configured one."""
    if (strategy or chunker_name()) == "structured":
        return chunk_pages_structured(pages, chunk_size, chunk_overlap)
    return chunk_pages(pages, chunk_size, chunk_overlap)
