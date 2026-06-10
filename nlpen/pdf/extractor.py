"""PDF text extraction and sentence tokenisation."""

import logging
import re

import fitz
from textblob_de import TextBlobDE

logger = logging.getLogger(__name__)

# Injected between pages so cross-page sentences can be detected and split later.
# The pattern also tolerates optional page-number lines that some PDFs insert.
PAGE_MARKER_PATTERN: re.Pattern[str] = re.compile(
    r"\n\[ENDOFPAGE(\d+)\](\s*\n\s*\d+)?(\s*\n)?"
)


def extract_sentences(pdf_path: str) -> list[str]:
    """Return a list of tokenised sentences extracted from *pdf_path*."""
    logger.info("Extracting sentences from %s", pdf_path)
    content = _read_content(pdf_path)
    sentences = [str(s) for s in TextBlobDE(content).sentences]
    logger.debug("Extracted %d sentences", len(sentences))
    return sentences


def _read_content(pdf_path: str) -> str:
    """Read all pages and join their text, inserting page markers between pages."""
    parts: list[str] = []
    with fitz.open(pdf_path) as doc:
        last_page_index = doc.page_count - 1
        for i, page in enumerate(doc):
            parts.append(page.get_text())
            if i < last_page_index:
                # page.get_text() ends with \n, so the marker matches the pattern
                parts.append(f"[ENDOFPAGE{i}]")
    return "".join(parts)
