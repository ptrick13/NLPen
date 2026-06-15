"""Abstract base class shared by all NLP analyzers."""

import logging
from abc import ABC, abstractmethod

import fitz

from nlpen.pdf.highlighter import save_annotated

logger = logging.getLogger(__name__)


class BaseAnalyzer(ABC):
    """Template for the analyze → annotate → save pipeline.

    Subclasses implement :meth:`analyze` (pure NLP, no I/O) and
    :meth:`annotate` (mutates an open PDF document).  The concrete
    :meth:`run` method wires them together and handles file I/O.
    """

    output_suffix: str  # appended to the source filename, e.g. "Sentiment"

    @abstractmethod
    def analyze(self, sentences: list[str]):
        """Run NLP analysis and return structured results."""

    @abstractmethod
    def annotate(self, results, doc: fitz.Document) -> None:
        """Apply highlights to *doc* in place."""

    def run(self, sentences: list[str], pdf_path: str) -> str:
        """Full pipeline: analyze → annotate → save.

        Returns the path of the annotated output PDF.
        """
        logger.info("Running %s on %s", self.__class__.__name__, pdf_path)
        results = self.analyze(sentences)
        doc = fitz.open(pdf_path)
        try:
            self.annotate(results, doc)
            return save_annotated(doc, pdf_path, self.output_suffix)
        finally:
            doc.close()
