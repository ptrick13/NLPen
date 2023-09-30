"""Sentiment analysis via XLM-RoBERTa."""

import logging
import re
from dataclasses import dataclass, field
from functools import cached_property

import fitz
from transformers import pipeline

from nlpen.config import (
    Color,
    COLOR_NEGATIVE,
    COLOR_POSITIVE,
    SENTIMENT_MODEL,
)
from nlpen.nlp.base import BaseAnalyzer
from nlpen.pdf.extractor import PAGE_MARKER_PATTERN
from nlpen.pdf.highlighter import add_legend, highlight_sentences

logger = logging.getLogger(__name__)

# Public legend constants — imported by combined.py to avoid duplication.
SENTIMENT_LEGEND_TEXT = "Positive Sätze    Negative Sätze"
SENTIMENT_LEGEND_ITEMS: list[tuple[Color, int, int]] = [
    (COLOR_POSITIVE, 0, 68),
    (COLOR_NEGATIVE, 80, 72),
]
SENTIMENT_LEGEND_BASE_X_FROM_RIGHT = 170


@dataclass
class SentimentResult:
    positive: list[str] = field(default_factory=list)
    negative: list[str] = field(default_factory=list)


class SentimentAnalyzer(BaseAnalyzer):
    """Classifies each sentence as positive or negative with XLM-RoBERTa.

    The multilingual model (``cardiffnlp/twitter-xlm-roberta-base-sentiment``)
    handles German text without any language-specific fine-tuning.
    """

    output_suffix = "Sentiment"

    @cached_property
    def _classifier(self):
        logger.info("Loading sentiment model '%s'", SENTIMENT_MODEL)
        return pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            tokenizer=SENTIMENT_MODEL,
        )

    def analyze(self, sentences: list[str]) -> SentimentResult:
        result = SentimentResult()
        for sentence in sentences:
            cleaned = self._clean(sentence)
            label = self._classifier(cleaned)[0]["label"]
            if label == "positive" and sentence not in result.positive:
                result.positive.append(sentence)
            elif label == "negative" and sentence not in result.negative:
                result.negative.append(sentence)
        logger.debug(
            "Sentiment: %d positive, %d negative",
            len(result.positive),
            len(result.negative),
        )
        return result

    def annotate(self, results: SentimentResult, doc: fitz.Document) -> None:
        highlight_sentences(results.positive, doc, COLOR_POSITIVE)
        highlight_sentences(results.negative, doc, COLOR_NEGATIVE)
        add_legend(doc, SENTIMENT_LEGEND_TEXT, SENTIMENT_LEGEND_ITEMS, SENTIMENT_LEGEND_BASE_X_FROM_RIGHT)

    @staticmethod
    def _clean(sentence: str) -> str:
        sentence = PAGE_MARKER_PATTERN.sub("", sentence)
        return re.sub(r"\n", "", sentence)
