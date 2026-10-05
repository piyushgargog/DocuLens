from pdf_loader import load_pdf_pages


def test_valid_pdf_extracts_all_pages_with_correct_numbering(sample_pdf_bytes):
    pages = load_pdf_pages(sample_pdf_bytes)
    assert [p for p, _ in pages] == [1, 2, 3, 4]
    assert "Solar System" in pages[0][1]


def test_invalid_pdf_bytes_return_empty_list():
    assert load_pdf_pages(b"this is not a pdf") == []


def test_empty_bytes_return_empty_list():
    assert load_pdf_pages(b"") == []


# OCR behaviour (scans, mixed documents, limits, failures) is covered in tests/test_ocr.py.


def test_pdf_with_a_text_layer_never_triggers_ocr(monkeypatch, sample_pdf_bytes):
    import pdf_loader

    def boom(*a, **k):
        raise AssertionError("OCR must not run when a text layer exists")

    monkeypatch.setattr(pdf_loader, "_ocr_candidates", boom)
    pages = pdf_loader.load_pdf_pages(sample_pdf_bytes)
    assert len(pages) == 4
