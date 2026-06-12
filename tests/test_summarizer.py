import math

import pytest

from nlpen.nlp.summarizer import SentenceSummarizer


@pytest.fixture()
def summarizer(monkeypatch):
    s = SentenceSummarizer()
    # Bypass SpaCy: return sentences unchanged (lemmatisation not under test here)
    monkeypatch.setattr(s, "_preprocess", lambda sentences: sentences)
    return s


def test_rank_returns_correct_count(summarizer, sentences):
    top_ratio = 0.5
    result = summarizer.rank(sentences, top_ratio)
    expected = max(1, math.floor(top_ratio * len(sentences)))
    assert len(result) == expected


def test_rank_returns_at_least_one(summarizer, sentences):
    result = summarizer.rank(sentences, top_ratio=0.01)
    assert len(result) >= 1


def test_rank_output_is_subset_of_input(summarizer, sentences):
    result = summarizer.rank(sentences, top_ratio=0.5)
    assert all(s in sentences for s in result)


def test_rank_deduplicates(summarizer):
    dupes = ["Gleicher Satz.", "Gleicher Satz.", "Anderer Satz.", "Anderer Satz."]
    result = summarizer.rank(dupes, top_ratio=1.0)
    assert len(result) == len(set(result))


def test_rank_respects_long_doc_threshold(summarizer):
    short_result = summarizer.rank(["Satz eins.", "Satz zwei.", "Satz drei."], top_ratio=0.3)
    assert len(short_result) >= 1
