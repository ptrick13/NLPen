"""Application-wide constants.

Centralising model identifiers, color values, and tuneable thresholds here
makes it easy to swap models or adjust behaviour without touching analysis code.
"""

from typing import Final

# ---------------------------------------------------------------------------
# Type alias used throughout the codebase
# ---------------------------------------------------------------------------
Color = tuple[float, float, float]  # RGB in [0, 1] — PyMuPDF convention

# ---------------------------------------------------------------------------
# NLP model identifiers
# ---------------------------------------------------------------------------
SENTIMENT_MODEL: Final[str] = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
NER_MODEL: Final[str] = "flair/ner-german"
SPACY_MODEL: Final[str] = "de_core_news_md"

# ---------------------------------------------------------------------------
# Summarization thresholds
# ---------------------------------------------------------------------------
LONG_DOC_THRESHOLD: Final[int] = 30   # sentence count that separates "long" from "short"
TOP_RATIO_LONG: Final[float] = 0.2    # keep top 20 % for long documents
TOP_RATIO_SHORT: Final[float] = 0.3   # keep top 30 % for short documents

# ---------------------------------------------------------------------------
# Highlight colors
# ---------------------------------------------------------------------------
COLOR_IMPORTANT: Final[Color] = (1.0, 1.0, 0.6)   # yellow
COLOR_POSITIVE: Final[Color] = (0.8, 1.0, 0.8)    # green
COLOR_NEGATIVE: Final[Color] = (1.0, 0.8, 0.8)    # red
COLOR_NER_PERSON: Final[Color] = (0.6, 1.0, 0.85)  # mint green
COLOR_NER_ORG: Final[Color] = (1.0, 0.9, 0.7)     # peach
COLOR_NER_LOC: Final[Color] = (0.8, 0.8, 1.0)     # lavender
COLOR_NER_MISC: Final[Color] = (0.7, 0.95, 1.0)   # light blue

# ---------------------------------------------------------------------------
# PDF legend layout
# ---------------------------------------------------------------------------
LEGEND_FONT: Final[str] = "Times-Roman"
LEGEND_FONTSIZE: Final[int] = 12
LEGEND_Y: Final[int] = 30  # distance from the top of the first page in pts
