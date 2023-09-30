"""Named Entity Recognition via Flair (German model)."""

import logging
import re
from dataclasses import dataclass, field
from functools import cached_property

import fitz
from flair.data import Sentence
from flair.models import SequenceTagger

from nlpen.config import (
    Color,
    COLOR_NER_LOC,
    COLOR_NER_MISC,
    COLOR_NER_ORG,
    COLOR_NER_PERSON,
    NER_MODEL,
)
from nlpen.nlp.base import BaseAnalyzer
from nlpen.pdf.extractor import PAGE_MARKER_PATTERN
from nlpen.pdf.highlighter import add_legend, highlight_entities

logger = logging.getLogger(__name__)

# Public legend constants — imported by combined.py to avoid duplication.
NER_LEGEND_TEXT = "Person    Organisation    Ort    Andere"
NER_LEGEND_ITEMS: list[tuple[Color, int, int]] = [
    (COLOR_NER_PERSON, 0, 33),
    (COLOR_NER_ORG, 45, 62),
    (COLOR_NER_LOC, 119, 16),
    (COLOR_NER_MISC, 147, 35),
]
NER_LEGEND_BASE_X_FROM_RIGHT = 200

EntityPairs = list[tuple[str, str]]  # (sentence, entity_text)


@dataclass
class NERResult:
    persons: EntityPairs = field(default_factory=list)
    locations: EntityPairs = field(default_factory=list)
    organizations: EntityPairs = field(default_factory=list)
    misc: EntityPairs = field(default_factory=list)


class NERAnalyzer(BaseAnalyzer):
    """Extracts named entities from German text using Flair's sequence tagger.

    Recognised entity types:
        * PER  — persons (mint green)
        * LOC  — locations (lavender)
        * ORG  — organisations (peach)
        * MISC — miscellaneous named entities (light blue)
    """

    output_suffix = "Namen"

    @cached_property
    def _tagger(self) -> SequenceTagger:
        logger.info("Loading NER model '%s'", NER_MODEL)
        return SequenceTagger.load(NER_MODEL)

    def analyze(self, sentences: list[str]) -> NERResult:
        result = NERResult()
        for sentence in sentences:
            cleaned = self._clean(sentence)
            flair_sentence = Sentence(cleaned)
            self._tagger.predict(flair_sentence)
            for entity in flair_sentence.get_spans("ner"):
                pair = (sentence, entity.text)
                if entity.tag.startswith("PER") and pair not in result.persons:
                    result.persons.append(pair)
                elif entity.tag.startswith("LOC") and pair not in result.locations:
                    result.locations.append(pair)
                elif entity.tag.startswith("ORG") and pair not in result.organizations:
                    result.organizations.append(pair)
                elif entity.tag.startswith("MISC") and pair not in result.misc:
                    result.misc.append(pair)
        logger.debug(
            "NER: %d PER, %d LOC, %d ORG, %d MISC",
            len(result.persons),
            len(result.locations),
            len(result.organizations),
            len(result.misc),
        )
        return result

    def annotate(self, results: NERResult, doc: fitz.Document) -> None:
        highlight_entities(results.persons, doc, COLOR_NER_PERSON)
        highlight_entities(results.organizations, doc, COLOR_NER_ORG)
        highlight_entities(results.locations, doc, COLOR_NER_LOC)
        highlight_entities(results.misc, doc, COLOR_NER_MISC)
        add_legend(doc, NER_LEGEND_TEXT, NER_LEGEND_ITEMS, NER_LEGEND_BASE_X_FROM_RIGHT)

    @staticmethod
    def _clean(sentence: str) -> str:
        sentence = PAGE_MARKER_PATTERN.sub("", sentence)
        return re.sub(r"\n", "", sentence)
