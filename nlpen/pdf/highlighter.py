"""PDF annotation helpers: sentence highlighting, entity highlighting, legends, saving."""

import logging
from pathlib import Path

import fitz

from nlpen.config import Color, LEGEND_FONT, LEGEND_FONTSIZE, LEGEND_Y
from nlpen.pdf.extractor import PAGE_MARKER_PATTERN

logger = logging.getLogger(__name__)


def highlight_sentences(
    sentences: list[str], doc: fitz.Document, color: Color
) -> None:
    """Highlight every sentence in *sentences* with *color*."""
    for sentence in sentences:
        _highlight_sentence(sentence, doc, color)


def highlight_entities(
    pairs: list[tuple[str, str]], doc: fitz.Document, color: Color
) -> None:
    """Highlight the entity span within its containing sentence.

    *pairs* is a list of ``(sentence, entity_text)`` tuples.
    """
    for sentence, entity in pairs:
        _highlight_entity(sentence, entity, doc, color)


def add_legend(
    doc: fitz.Document,
    label_text: str,
    items: list[tuple[Color, int, int]],
    base_x_from_right: int,
) -> None:
    """Draw a color-keyed legend in the top-right corner of the first page.

    Args:
        doc: Open PDF document.
        label_text: Full legend string, e.g. ``"Positive    Negative"``.
        items: One entry per label — ``(color, x_offset, width)`` relative to
            ``base_x``.  The offsets must match the rendered positions of each
            label inside *label_text*.
        base_x_from_right: Distance from the right edge of the page in pts.
    """
    page = doc.load_page(0)
    base_x = page.rect.width - base_x_from_right
    page.insert_text(
        (base_x, LEGEND_Y),
        label_text,
        fontname=LEGEND_FONT,
        fontsize=LEGEND_FONTSIZE,
    )
    for color, x_offset, width in items:
        rect = fitz.Rect(
            base_x + x_offset,
            LEGEND_Y - 10,
            base_x + x_offset + width,
            LEGEND_Y + 2,
        )
        _annotate(page, rect, color)


def save_annotated(doc: fitz.Document, source_path: str, suffix: str) -> str:
    """Save the annotated document next to the source file and return the new path."""
    p = Path(source_path)
    output_path = p.with_name(f"{p.stem}_{suffix}.pdf")
    doc.save(str(output_path))
    logger.info("Saved annotated PDF: %s", output_path)
    return str(output_path)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _highlight_sentence(sentence: str, doc: fitz.Document, color: Color) -> None:
    """Highlight all occurrences of *sentence* across the document.

    Handles sentences that span a page break by splitting at the injected
    page marker and highlighting each half on its respective page.
    """
    match = PAGE_MARKER_PATTERN.search(sentence)
    if match:
        page_no = int(match.group(1))
        parts = PAGE_MARKER_PATTERN.split(sentence)
        first, second = parts[0].rstrip(), parts[-1]
        _annotate_text_lines(first, doc.load_page(page_no), color, reversed_order=True)
        _annotate_text_lines(second, doc.load_page(page_no + 1), color, reversed_order=False)
        return

    for i in range(doc.page_count):
        page = doc.load_page(i)
        for rect in _merge_line_rects(page.search_for(sentence)):
            _annotate(page, rect, color)


def _highlight_entity(
    sentence: str, entity: str, doc: fitz.Document, color: Color
) -> None:
    """Highlight the *entity* span within its containing *sentence*.

    Restricts highlighting to the rectangle of the sentence to avoid
    marking unrelated occurrences of the same entity text elsewhere.
    Handles cross-page sentences via the injected page marker.
    """
    match = PAGE_MARKER_PATTERN.search(sentence)
    if match:
        page_no = int(match.group(1))
        parts = PAGE_MARKER_PATTERN.split(sentence)
        first, second = parts[0].rstrip(), parts[-1]
        _annotate_entity_in_half(first, entity, doc.load_page(page_no), color, reversed_order=True)
        _annotate_entity_in_half(second, entity, doc.load_page(page_no + 1), color, reversed_order=False)
        return

    for i in range(doc.page_count):
        page = doc.load_page(i)
        sent_rects = _merge_line_rects(page.search_for(sentence))
        ent_rects = _merge_line_rects(page.search_for(entity))
        for s_rect in sent_rects:
            for e_rect in ent_rects:
                if s_rect.contains(e_rect):
                    _annotate(page, e_rect, color)


def _annotate_text_lines(
    text: str, page: fitz.Page, color: Color, *, reversed_order: bool
) -> None:
    """Highlight the lines of *text* on *page*.

    Uses ``reversed_order=True`` for the last lines of a cross-page sentence
    (they sit at the bottom of the preceding page) and ``False`` for the
    first lines on the following page.
    """
    instances = _merge_line_rects(page.search_for(text))
    n_lines = text.count("\n") + 1
    for i in range(n_lines):
        idx = -(i + 1) if reversed_order else i
        try:
            _annotate(page, instances[idx], color)
        except IndexError:
            break


def _annotate_entity_in_half(
    half: str,
    entity: str,
    page: fitz.Page,
    color: Color,
    *,
    reversed_order: bool,
) -> None:
    """Highlight *entity* within one half of a cross-page sentence on *page*."""
    sent_rects = _merge_line_rects(page.search_for(half))
    ent_rects = _merge_line_rects(page.search_for(entity))
    n_lines = half.count("\n") + 1
    for i in range(n_lines):
        idx = -(i + 1) if reversed_order else i
        try:
            s_rect = sent_rects[idx]
        except IndexError:
            break
        for e_rect in ent_rects:
            if s_rect.contains(e_rect):
                _annotate(page, e_rect, color)


def _merge_line_rects(rects: list[fitz.Rect]) -> list[fitz.Rect]:
    """Merge rects that share the same line into one spanning rect.

    Justified text causes search_for to return one rect per word because
    inter-word gaps don't match a plain space character.  Merging eliminates
    the visible gaps inside a highlight annotation.
    """
    if not rects:
        return []
    sorted_rects = sorted(rects, key=lambda r: (r.y0, r.x0))
    merged = [sorted_rects[0]]
    for rect in sorted_rects[1:]:
        last = merged[-1]
        if abs(rect.y0 - last.y0) < 3:  # same line (< ~1 mm)
            merged[-1] = fitz.Rect(
                min(last.x0, rect.x0), min(last.y0, rect.y0),
                max(last.x1, rect.x1), max(last.y1, rect.y1),
            )
        else:
            merged.append(rect)
    return merged


def _annotate(page: fitz.Page, rect: fitz.Rect, color: Color) -> None:
    """Add a highlight annotation with *color* to *rect* on *page*."""
    hl = page.add_highlight_annot(rect)
    hl.set_colors(stroke=color)
    hl.update()
