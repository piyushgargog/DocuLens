"""Phase 1: page-aware PDF text extraction (PyMuPDF) with per-page OCR.

Each page is judged on its own:

* a page with a real text layer is read directly and never OCR'd;
* a page with no usable text *and* something to read (a scanned image) is OCR'd;
* a truly blank page (no text, no images) is skipped, not OCR'd;
* a page whose text layer is garbage (mostly symbols or replacement characters)
  is OCR'd too, and the better of the two readings is kept.

So a mixed document (some pages typed, some scanned) comes out right, and one
bad page costs only itself. OCR is bounded three ways, all configurable:
OCR_MAX_PAGES (pages OCR'd per document), OCR_TIME_BUDGET (seconds for the whole
document; per-page Tesseract calls get a timeout from what is left) and
OCR_MAX_PIXELS (a huge page is rendered smaller). Pages left unread by a limit
are reported in `LoadResult.warnings`, never silently dropped.

A scan is read at ~144 DPI first; if that comes out poor (low Tesseract
confidence or almost no words) the page is rotated if Tesseract's orientation
detection says so, cleaned up (grey-scale, contrast, threshold) and read again,
and the better reading wins. OCR text is normalised (Unicode NFKC, control
characters, line-wrap hyphens, noise lines). Handwriting is not supported:
Tesseract does not read it, and nothing here claims it does.

`PDF_MAX_PAGES` and `PDF_EXTRACT_SECONDS` bound plain extraction of a PDF with
an enormous page count.
"""

import math
import os
import re
import shutil
import time
import unicodedata
from dataclasses import dataclass, field

import pymupdf

from observability import log

# Defaults; each is overridden by the environment variable of the same name.
OCR_MAX_PAGES = 30
OCR_ZOOM = 2.0  # ~144 DPI
OCR_MIN_CHARS = 12  # fewer characters than this = no usable text
OCR_TIME_BUDGET = 60.0
OCR_PAGE_TIMEOUT = 25.0
OCR_MIN_CONFIDENCE = 45.0
OCR_MAX_PIXELS = 25_000_000
OCR_LANG = "eng"
PDF_MAX_PAGES = 2000
PDF_EXTRACT_SECONDS = 60.0

_POOR_WORDS = 5  # a first reading with fewer words than this is "poor"


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, default)))
    except ValueError:
        return default


def ocr_max_pages() -> int:
    return int(_env_float("OCR_MAX_PAGES", OCR_MAX_PAGES))


def ocr_time_budget() -> float:
    return _env_float("OCR_TIME_BUDGET", OCR_TIME_BUDGET, 1.0)


def ocr_zoom() -> float:
    return _env_float("OCR_ZOOM", OCR_ZOOM, 0.5)


def ocr_lang() -> str:
    lang = os.environ.get("OCR_LANG", OCR_LANG).strip()
    return lang if re.fullmatch(r"[A-Za-z_+]{2,40}", lang) else OCR_LANG  # goes to a subprocess argument: keep it inert


@dataclass
class LoadResult:
    pages: list[tuple[int, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------- text-layer checks ----------


def looks_garbled(text: str) -> bool:
    """A text layer that is mostly symbols, private-use or replacement
    characters (a broken font map) reads as text but means nothing."""
    sample = text[:4000]
    if not sample.strip():
        return False
    letters = sum(ch.isalpha() for ch in sample)
    junk = sum(ch == "�" or 0xE000 <= ord(ch) <= 0xF8FF for ch in sample)
    return junk > 0.2 * len(sample) or letters < 0.4 * len(sample.replace(" ", "").replace("\n", ""))


def _has_visible_content(page) -> bool:
    """Something an OCR pass could read: an image on the page."""
    try:
        return bool(page.get_images())
    except Exception as e:
        log.debug("Could not list page images (%s)", type(e).__name__)
        return False


# ---------- OCR engine (replaceable in tests) ----------


def ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401
    except Exception:
        return False
    return shutil.which("tesseract") is not None or bool(os.environ.get("TESSERACT_CMD"))


def _run_tesseract(image, lang: str, timeout: float) -> tuple[str, float]:
    """(text, mean word confidence 0-100) for one image. Raises on failure."""
    import pytesseract
    from pytesseract import Output

    data = pytesseract.image_to_data(image, lang=lang, output_type=Output.DICT, timeout=max(1, int(timeout)))
    lines: dict[tuple, list[str]] = {}
    confidences = []
    for i, word in enumerate(data["text"]):
        word = str(word).strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if not word or conf < 0:
            continue
        confidences.append(conf)
        lines.setdefault((data["block_num"][i], data["par_num"][i], data["line_num"][i]), []).append(word)
    text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
    return text, (sum(confidences) / len(confidences) if confidences else 0.0)


def _detect_rotation(image) -> int:
    """0, 90, 180 or 270: how far to rotate the image (counter-clockwise, as PIL
    does) to make the text upright, from Tesseract's orientation detection.
    0 when detection is unavailable or unsure."""
    try:
        import pytesseract
        from pytesseract import Output

        osd = pytesseract.image_to_osd(image, output_type=Output.DICT, timeout=15)
        if float(osd.get("orientation_conf", 0)) >= 2.0 and int(osd.get("rotate", 0)) in (90, 180, 270):
            return int(osd["rotate"])
    except Exception as e:  # OSD data missing, image too small, timeout: just do not rotate
        log.debug("Orientation detection unavailable (%s)", type(e).__name__)
    return 0


# ---------- image handling ----------


def _render(page, zoom: float):
    """The page as a grey-scale PIL image, rendered smaller if it would exceed OCR_MAX_PIXELS."""
    from PIL import Image

    width, height = page.rect.width * zoom, page.rect.height * zoom
    limit = _env_float("OCR_MAX_PIXELS", OCR_MAX_PIXELS, 100_000)
    if width * height > limit:
        zoom *= math.sqrt(limit / (width * height))
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
    return Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)


def _preprocess(image):
    """Stretch contrast and binarise around the mean: helps faint or uneven scans."""
    from PIL import ImageOps

    image = ImageOps.autocontrast(image, cutoff=2)
    mean = sum(i * c for i, c in enumerate(image.histogram())) / max(1, image.width * image.height)
    return image.point(lambda v: 255 if v > mean * 0.85 else 0)


def _rotate(image, angle: int):
    return image.rotate(angle, expand=True) if angle else image


# ---------- text normalisation ----------


def normalize_ocr(text: str) -> str:
    """Tidy OCR output: NFKC, no control characters, line-wrap hyphens joined
    ("trans-\\nformer" -> "transformer"), noise lines dropped, whitespace collapsed."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)
    kept = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            kept.append("")
            continue
        alnum = sum(ch.isalnum() for ch in line)
        if len(line) >= 4 and alnum < 0.3 * len(line):
            continue  # a line of dots, rules or speckle
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def _score(text: str, confidence: float) -> float:
    return len(text.split()) * max(confidence, 1.0) / 100.0


def _ocr_page(page, lang: str, timeout: float, run=None) -> tuple[str, float, bool]:
    """(text, confidence, rotated) for one page, trying a cleaned-up and
    re-oriented second reading when the first is poor."""
    run = run or _run_tesseract
    image = _render(page, ocr_zoom())
    text, conf = run(image, lang, timeout)
    text = normalize_ocr(text)
    rotated = False
    poor = conf < OCR_MIN_CONFIDENCE or len(text.split()) < _POOR_WORDS
    if poor:
        angle = _detect_rotation(image)
        retry = _preprocess(_rotate(image, angle))
        try:
            text2, conf2 = run(retry, lang, timeout)
            text2 = normalize_ocr(text2)
            if _score(text2, conf2) > _score(text, conf):
                text, conf, rotated = text2, conf2, bool(angle)
        except Exception as e:  # keep the first reading
            log.debug("OCR retry failed (%s)", type(e).__name__)
    return text, conf, rotated


# ---------- public API ----------


def _ranges(numbers: list[int]) -> str:
    """[3,4,5,9] -> "3-5, 9"."""
    out: list[str] = []
    ordered = sorted(numbers)
    if not ordered:
        return ""
    start = prev = ordered[0]
    for n in ordered[1:]:
        if n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if prev != start else str(start))
            start = prev = n
    out.append(f"{start}-{prev}" if prev != start else str(start))
    return ", ".join(out)


def load_pdf(pdf_bytes: bytes, run=None) -> LoadResult:
    """Extract (page_number, text) pairs, 1-indexed, plus warnings about anything
    that could not be read. Returns an empty result for an invalid, corrupt,
    encrypted or fully empty PDF instead of raising. `run` replaces the
    Tesseract call (tests)."""
    result = LoadResult()
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return result
    try:
        if doc.needs_pass or doc.is_encrypted:
            return result
        total = doc.page_count
        limit = min(total, int(_env_float("PDF_MAX_PAGES", PDF_MAX_PAGES, 1)))
        if limit < total:
            result.warnings.append(f"Only the first {limit} of {total} pages were read.")
        started = time.monotonic()
        extract_budget = _env_float("PDF_EXTRACT_SECONDS", PDF_EXTRACT_SECONDS, 1.0)
        text_pages: dict[int, str] = {}
        candidates: list[int] = []  # pages that need OCR
        garbled: dict[int, str] = {}
        for i in range(limit):
            if time.monotonic() - started > extract_budget:
                result.warnings.append(f"Reading stopped at page {i}: the time limit for extracting text was reached.")
                break
            try:
                page = doc[i]
                text = page.get_text().strip()
            except Exception as e:  # one malformed page must not sink the document
                log.warning("Skipping an unreadable PDF page (%s)", type(e).__name__)
                continue
            if len(text) >= OCR_MIN_CHARS and not looks_garbled(text):
                text_pages[i + 1] = text
            elif _has_visible_content(page):
                candidates.append(i)
                if text:
                    garbled[i + 1] = text
        ocr_pages = _ocr_candidates(doc, candidates, garbled, result, run) if candidates else {}
        merged = {**text_pages, **ocr_pages}
        result.pages = sorted(merged.items())
        return result
    except Exception as e:
        log.warning("PDF extraction failed (%s)", type(e).__name__)
        return LoadResult(pages=[], warnings=[])
    finally:
        doc.close()


def _ocr_candidates(doc, candidates: list[int], garbled: dict[int, str], result: LoadResult, run) -> dict[int, str]:
    """OCR the pages that need it, within the page and time limits."""
    if not candidates:
        return {}
    if run is None and not ocr_available():
        result.warnings.append(
            f"{len(candidates)} page(s) look scanned but text recognition (OCR) is not available on this server: pages {_ranges([i + 1 for i in candidates])}."
        )
        return {}
    page_limit, budget, lang = ocr_max_pages(), ocr_time_budget(), ocr_lang()
    started = time.monotonic()
    out: dict[int, str] = {}
    failed, skipped_pages, skipped_time, low = [], [], [], []
    for n, i in enumerate(candidates):
        page_no = i + 1
        if n >= page_limit:
            skipped_pages.append(page_no)
            continue
        remaining = budget - (time.monotonic() - started)
        if remaining <= 1:
            skipped_time.append(page_no)
            continue
        try:
            text, conf, _rotated = _ocr_page(doc[i], lang, min(OCR_PAGE_TIMEOUT, remaining), run)
        except Exception as e:
            if type(e).__name__ == "TesseractNotFoundError":
                result.warnings.append("Text recognition (OCR) is not installed on this server.")
                return out
            failed.append(page_no)
            if page_no in garbled:
                out[page_no] = garbled[page_no]
            continue
        if page_no in garbled and _score(text, conf) < _score(garbled[page_no], 60.0):
            out[page_no] = garbled[page_no]  # the broken text layer still beats a worse OCR reading
            continue
        if len(text) < OCR_MIN_CHARS or conf < 20:
            failed.append(page_no)
            continue
        out[page_no] = text
        if conf < OCR_MIN_CONFIDENCE:
            low.append(page_no)
    if failed:
        result.warnings.append(f"Pages {_ranges(failed)} could not be read (blank, too faint or too noisy).")
    if low:
        result.warnings.append(f"Pages {_ranges(low)} were hard to read; the text may contain errors.")
    if skipped_pages:
        result.warnings.append(f"Text recognition is limited to {page_limit} scanned pages; pages {_ranges(skipped_pages)} were skipped.")
    if skipped_time:
        result.warnings.append(f"Text recognition ran out of time; pages {_ranges(skipped_time)} were skipped.")
    return out


def load_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Extract text per page from PDF bytes: a list of (page_number, page_text)
    pairs, 1-indexed. Returns [] for an invalid, corrupt, encrypted or empty PDF
    instead of raising. See `load_pdf` for the warnings."""
    return load_pdf(pdf_bytes).pages
