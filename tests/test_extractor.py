import fitz
import pytest

from nlpen.pdf.extractor import PAGE_MARKER_PATTERN, _read_content, extract_sentences


@pytest.fixture()
def single_page_pdf(tmp_path):
    path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Das ist ein Testsatz. Und noch ein weiterer Satz.")
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture()
def two_page_pdf(tmp_path):
    path = tmp_path / "two_page.pdf"
    doc = fitz.open()
    for _ in range(2):
        page = doc.new_page()
        page.insert_text((72, 100), "Ein Satz auf dieser Seite.")
    doc.save(str(path))
    doc.close()
    return str(path)


def test_pattern_matches_basic_marker():
    text = "Letzter Satz der Seite.\n[ENDOFPAGE0]\nErster Satz der nächsten Seite."
    assert PAGE_MARKER_PATTERN.search(text) is not None


def test_pattern_matches_with_trailing_page_number():
    text = "Satz.\n[ENDOFPAGE2]\n   42\nNächster Satz."
    assert PAGE_MARKER_PATTERN.search(text) is not None


def test_pattern_does_not_match_plain_text():
    text = "Das ist ein normaler Satz ohne Seitenmarkierung."
    assert PAGE_MARKER_PATTERN.search(text) is None


def test_pattern_sub_removes_marker():
    text = "Vor der Seite.\n[ENDOFPAGE1]\nNach der Seite."
    cleaned = PAGE_MARKER_PATTERN.sub("", text)
    assert "[ENDOFPAGE" not in cleaned
    assert "Vor der Seite." in cleaned
    assert "Nach der Seite." in cleaned


def test_pattern_matches_all_page_indices():
    for i in range(5):
        text = f"Satz.\n[ENDOFPAGE{i}]\nNächster."
        assert PAGE_MARKER_PATTERN.search(text) is not None, (
            f"Failed for page index {i}"
        )


def test_read_content_returns_string(single_page_pdf):
    content = _read_content(single_page_pdf)
    assert isinstance(content, str)
    assert len(content) > 0


def test_read_content_inserts_page_marker_between_pages(two_page_pdf):
    content = _read_content(two_page_pdf)
    assert "[ENDOFPAGE0]" in content


def test_read_content_no_marker_for_single_page(single_page_pdf):
    content = _read_content(single_page_pdf)
    assert "[ENDOFPAGE" not in content


def test_extract_sentences_returns_list(single_page_pdf):
    sentences = extract_sentences(single_page_pdf)
    assert isinstance(sentences, list)
    assert len(sentences) > 0
