"""Load several document types into the same page-aware shape the pipeline
expects: a list of (page_number, page_text) pairs, 1-indexed.

PDFs keep their real pages (with per-page OCR -- see pdf_loader). Plain
text, Markdown and Word documents have no fixed pages, so they are split into
even "pages" of roughly PSEUDO_PAGE_CHARS on paragraph boundaries, which keeps
citations ("Page 3") meaningful without pretending a precision they don't have.
"""

from pdf_loader import LoadResult, load_pdf

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".markdown", ".docx")
PSEUDO_PAGE_CHARS = 2500


def is_supported(filename: str) -> bool:
    return filename.lower().endswith(SUPPORTED_EXTENSIONS)


def load_document_ex(filename: str, data: bytes) -> LoadResult:
    """Extract (page, text) pairs from a supported document, plus warnings about
    anything that could not be read (see pdf_loader). An unreadable, empty or
    unsupported file gives an empty result rather than raising."""
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return load_pdf(data)
    if name.endswith(".docx"):
        return LoadResult(_paginate(_load_docx(data)))
    if name.endswith((".txt", ".md", ".markdown")):
        return LoadResult(_paginate(_load_text(data)))
    return LoadResult()


def load_document(filename: str, data: bytes) -> list[tuple[int, str]]:
    """Just the (page, text) pairs of `load_document_ex`."""
    return load_document_ex(filename, data).pages


def _load_text(data: bytes) -> list[str]:
    """Decode text/Markdown as UTF-8 (falling back to latin-1 so odd bytes
    never crash ingestion), split into paragraphs on blank lines."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="replace")
    return _paragraphs(text)


MAX_DOCX_ENTRIES = 5000
MAX_DOCX_UNPACKED_BYTES = 100 * 1024 * 1024  # all parts together
MAX_DOCX_PART_BYTES = 50 * 1024 * 1024  # any one part
MAX_DOCX_RATIO = 200  # uncompressed : compressed, for parts over 1 MB


def docx_is_safe(data: bytes) -> bool:
    """A .docx is a zip file, and a zip file a few KB long can claim to hold
    gigabytes. Check the declared sizes (cheap, from the directory only) before
    anything is decompressed: entry count, per-part and total size, and the
    compression ratio of large parts."""
    import io
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_DOCX_ENTRIES:
                return False
            total = 0
            for info in infos:
                total += info.file_size
                if info.file_size > MAX_DOCX_PART_BYTES or total > MAX_DOCX_UNPACKED_BYTES:
                    return False
                if info.file_size > 1024 * 1024 and info.file_size > MAX_DOCX_RATIO * max(info.compress_size, 1):
                    return False
            return True
    except (zipfile.BadZipFile, OSError, ValueError):
        return False


def _load_docx(data: bytes) -> list[str]:
    """Paragraphs and table cell text from a .docx, in document order."""
    try:
        import io

        from docx import Document
    except Exception:
        return []
    if not docx_is_safe(data):
        return []
    try:
        doc = Document(io.BytesIO(data))
    except Exception:
        return []
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                paras.append(" | ".join(cells))
    return paras


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]


def _paginate(paragraphs: list[str]) -> list[tuple[int, str]]:
    """Pack paragraphs into ~PSEUDO_PAGE_CHARS pages, keeping paragraphs whole."""
    pages: list[tuple[int, str]] = []
    buffer: list[str] = []
    length = 0
    for para in paragraphs:
        if buffer and length + len(para) > PSEUDO_PAGE_CHARS:
            pages.append((len(pages) + 1, "\n\n".join(buffer)))
            buffer, length = [], 0
        buffer.append(para)
        length += len(para) + 2
    if buffer:
        pages.append((len(pages) + 1, "\n\n".join(buffer)))
    return pages
