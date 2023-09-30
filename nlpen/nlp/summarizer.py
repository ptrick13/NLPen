"""Extractive summarisation via TF-IDF + TextRank."""

import logging
from functools import cached_property

import fitz
import networkx as nx
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlpen.config import (
    COLOR_IMPORTANT,
    LONG_DOC_THRESHOLD,
    SPACY_MODEL,
    TOP_RATIO_LONG,
    TOP_RATIO_SHORT,
)
from nlpen.nlp.base import BaseAnalyzer
from nlpen.pdf.highlighter import highlight_sentences

logger = logging.getLogger(__name__)


class SentenceSummarizer(BaseAnalyzer):
    """Ranks sentences by importance using TF-IDF cosine similarity + PageRank.

    Algorithm overview:
        1. Lemmatise and remove stopwords with SpaCy (``de_core_news_md``).
        2. Build a TF-IDF matrix over the processed sentences.
        3. Compute pairwise cosine similarity → weighted graph.
        4. Apply PageRank to score each sentence.
        5. Return the top *ratio* % of sentences (deduped).
    """

    output_suffix = "Wichtig"

    @cached_property
    def _nlp(self) -> spacy.language.Language:
        logger.info("Loading SpaCy model '%s'", SPACY_MODEL)
        return spacy.load(SPACY_MODEL)

    def analyze(self, sentences: list[str]) -> list[str]:
        ratio = TOP_RATIO_LONG if len(sentences) > LONG_DOC_THRESHOLD else TOP_RATIO_SHORT
        return self.rank(sentences, ratio)

    def annotate(self, results: list[str], doc: fitz.Document) -> None:
        highlight_sentences(results, doc, COLOR_IMPORTANT)

    def rank(self, sentences: list[str], top_ratio: float) -> list[str]:
        """Return the top *top_ratio* fraction of *sentences* by TextRank score.

        Public so that :class:`~nlpen.nlp.combined.CombinedAnalyzer` can
        reuse the ranking logic without accessing private internals.
        """
        processed = self._preprocess(sentences)
        tfidf = TfidfVectorizer().fit_transform(processed)
        similarity = cosine_similarity(tfidf, tfidf)
        scores = nx.pagerank(nx.from_numpy_array(similarity))
        ranked = sorted(
            ((scores[i], s) for i, s in enumerate(sentences)),
            reverse=True,
        )
        n = max(1, int(top_ratio * len(sentences)))
        # dict.fromkeys preserves order and removes duplicates
        return list(dict.fromkeys(s for _, s in ranked[:n]))

    def _preprocess(self, sentences: list[str]) -> list[str]:
        return [
            " ".join(
                tok.lemma_
                for tok in self._nlp(sentence)
                if tok.is_alpha and not tok.is_stop
            )
            for sentence in sentences
        ]
