"""Combined multi-mode analysis: any two of summarisation, sentiment, and NER."""

import logging
from enum import Enum, auto

import fitz

from nlpen.config import (
    COLOR_IMPORTANT,
    COLOR_NER_LOC,
    COLOR_NER_MISC,
    COLOR_NER_ORG,
    COLOR_NER_PERSON,
    COLOR_NEGATIVE,
    COLOR_POSITIVE,
    LONG_DOC_THRESHOLD,
    TOP_RATIO_LONG,
    TOP_RATIO_SHORT,
)
from nlpen.nlp.ner import (
    NERAnalyzer,
    NER_LEGEND_TEXT,
    NER_LEGEND_ITEMS,
    NER_LEGEND_BASE_X_FROM_RIGHT,
)
from nlpen.nlp.sentiment import (
    SentimentAnalyzer,
    SENTIMENT_LEGEND_TEXT,
    SENTIMENT_LEGEND_ITEMS,
    SENTIMENT_LEGEND_BASE_X_FROM_RIGHT,
)
from nlpen.nlp.summarizer import SentenceSummarizer
from nlpen.pdf.highlighter import (
    add_legend,
    highlight_entities,
    highlight_sentences,
    save_annotated,
)

logger = logging.getLogger(__name__)

# Combined legend for the NER + sentiment overlay mode.
# The individual module constants (NER_LEGEND_*, SENTIMENT_LEGEND_*) are reused
# where only one mode contributes; this definition covers the case where both
# appear side by side in a single legend bar.
_COMBINED_LEGEND_TEXT = (
    "Positive Sätze    Negative Sätze    Person    Organisation    Ort    Andere"
)
_COMBINED_LEGEND_ITEMS = [
    (COLOR_POSITIVE, 0, 68),
    (COLOR_NEGATIVE, 80, 72),
    (COLOR_NER_PERSON, 164, 33),
    (COLOR_NER_ORG, 209, 62),
    (COLOR_NER_LOC, 283, 16),
    (COLOR_NER_MISC, 311, 35),
]
_COMBINED_LEGEND_BASE_X_FROM_RIGHT = 364


class CombinationMode(Enum):
    """Supported two-mode analysis combinations."""

    SENTIMENT_AND_SUMMARIZATION = auto()
    NER_AND_SUMMARIZATION = auto()
    NER_AND_SENTIMENT = auto()


class CombinedAnalyzer:
    """Coordinates two analysis modes on a single document pass.

    Rather than extending :class:`BaseAnalyzer`, this class owns instances of
    the individual analyzers and delegates analysis to them.  It manages the
    PDF document lifecycle directly so both modes share one open file handle.

    Only the two analyzers required for the selected :class:`CombinationMode`
    are instantiated; the third is left as ``None``.
    """

    output_suffix = "Kombiniert"

    def __init__(self, mode: CombinationMode) -> None:
        self.mode = mode
        needs_summarizer = mode in (
            CombinationMode.SENTIMENT_AND_SUMMARIZATION,
            CombinationMode.NER_AND_SUMMARIZATION,
        )
        needs_sentiment = mode in (
            CombinationMode.SENTIMENT_AND_SUMMARIZATION,
            CombinationMode.NER_AND_SENTIMENT,
        )
        needs_ner = mode in (
            CombinationMode.NER_AND_SUMMARIZATION,
            CombinationMode.NER_AND_SENTIMENT,
        )
        self._summarizer: SentenceSummarizer | None = (
            SentenceSummarizer() if needs_summarizer else None
        )
        self._sentiment: SentimentAnalyzer | None = (
            SentimentAnalyzer() if needs_sentiment else None
        )
        self._ner: NERAnalyzer | None = NERAnalyzer() if needs_ner else None

    def run(self, sentences: list[str], pdf_path: str) -> str:
        """Analyze, annotate, and save the combined result.  Returns output path."""
        logger.info("Running CombinedAnalyzer (%s) on %s", self.mode.name, pdf_path)
        doc = fitz.open(pdf_path)
        try:
            if self.mode == CombinationMode.SENTIMENT_AND_SUMMARIZATION:
                self._annotate_sentiment_summarization(sentences, doc)
            elif self.mode == CombinationMode.NER_AND_SUMMARIZATION:
                self._annotate_ner_summarization(sentences, doc)
            else:
                self._annotate_ner_sentiment(sentences, doc)
            return save_annotated(doc, pdf_path, self.output_suffix)
        finally:
            doc.close()

    # ------------------------------------------------------------------
    # Combination implementations
    # ------------------------------------------------------------------

    def _annotate_sentiment_summarization(
        self, sentences: list[str], doc: fitz.Document
    ) -> None:
        """Keep only the most important positive/negative sentences."""
        assert self._sentiment is not None and self._summarizer is not None
        sentiment = self._sentiment.analyze(sentences)
        all_sentiment = sentiment.positive + sentiment.negative
        # Rank the sentiment sentences among themselves, then cap at the
        # same threshold used for standalone summarisation.
        top = self._summarizer.rank(all_sentiment, top_ratio=1.0)
        cap = self._top_ratio(len(sentences))
        n = max(1, int(cap * len(sentences)))
        top = top[:n]

        top_positive = [s for s in top if s in sentiment.positive]
        top_negative = [s for s in top if s in sentiment.negative]
        highlight_sentences(top_positive, doc, COLOR_POSITIVE)
        highlight_sentences(top_negative, doc, COLOR_NEGATIVE)
        add_legend(
            doc,
            SENTIMENT_LEGEND_TEXT,
            SENTIMENT_LEGEND_ITEMS,
            SENTIMENT_LEGEND_BASE_X_FROM_RIGHT,
        )

    def _annotate_ner_summarization(
        self, sentences: list[str], doc: fitz.Document
    ) -> None:
        """Show NER entities overlaid with the most important sentences."""
        assert self._ner is not None and self._summarizer is not None
        ner = self._ner.analyze(sentences)
        highlight_entities(ner.persons, doc, COLOR_NER_PERSON)
        highlight_entities(ner.organizations, doc, COLOR_NER_ORG)
        highlight_entities(ner.locations, doc, COLOR_NER_LOC)
        highlight_entities(ner.misc, doc, COLOR_NER_MISC)

        ratio = self._top_ratio(len(sentences))
        top = self._summarizer.rank(sentences, ratio)
        highlight_sentences(top, doc, COLOR_IMPORTANT)
        add_legend(doc, NER_LEGEND_TEXT, NER_LEGEND_ITEMS, NER_LEGEND_BASE_X_FROM_RIGHT)

    def _annotate_ner_sentiment(self, sentences: list[str], doc: fitz.Document) -> None:
        """Show NER entities overlaid with sentiment highlights."""
        assert self._ner is not None and self._sentiment is not None
        ner = self._ner.analyze(sentences)
        highlight_entities(ner.persons, doc, COLOR_NER_PERSON)
        highlight_entities(ner.organizations, doc, COLOR_NER_ORG)
        highlight_entities(ner.locations, doc, COLOR_NER_LOC)
        highlight_entities(ner.misc, doc, COLOR_NER_MISC)

        sentiment = self._sentiment.analyze(sentences)
        highlight_sentences(sentiment.positive, doc, COLOR_POSITIVE)
        highlight_sentences(sentiment.negative, doc, COLOR_NEGATIVE)
        add_legend(
            doc,
            _COMBINED_LEGEND_TEXT,
            _COMBINED_LEGEND_ITEMS,
            _COMBINED_LEGEND_BASE_X_FROM_RIGHT,
        )

    @staticmethod
    def _top_ratio(n_sentences: int) -> float:
        return TOP_RATIO_LONG if n_sentences > LONG_DOC_THRESHOLD else TOP_RATIO_SHORT
