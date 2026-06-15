from nlpen.pdf.extractor import PAGE_MARKER_PATTERN


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
