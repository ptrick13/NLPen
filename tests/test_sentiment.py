import pytest

from nlpen.nlp.sentiment import SentimentAnalyzer


class _MockTokenizer:
    """Minimal tokenizer stub — satisfies _clean()'s encode/decode calls."""

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2, 3]}

    def decode(self, input_ids, **kwargs):
        return "mocked"


class _FixedClassifier:
    """Always returns the same sentiment label regardless of input."""

    def __init__(self, label: str):
        self.tokenizer = _MockTokenizer()
        self._label = label

    def __call__(self, text):
        return [{"label": self._label}]


class _AlternatingClassifier:
    """Returns 'positive' then 'negative' in alternation."""

    def __init__(self):
        self.tokenizer = _MockTokenizer()
        self._count = 0

    def __call__(self, text):
        label = "positive" if self._count % 2 == 0 else "negative"
        self._count += 1
        return [{"label": label}]


@pytest.fixture()
def positive_analyzer():
    a = SentimentAnalyzer()
    a._classifier = _FixedClassifier("positive")
    return a


@pytest.fixture()
def negative_analyzer():
    a = SentimentAnalyzer()
    a._classifier = _FixedClassifier("negative")
    return a


def test_positive_sentences_land_in_positive(positive_analyzer, sentences):
    result = positive_analyzer.analyze(sentences)
    assert result.positive == sentences
    assert result.negative == []


def test_negative_sentences_land_in_negative(negative_analyzer, sentences):
    result = negative_analyzer.analyze(sentences)
    assert result.negative == sentences
    assert result.positive == []


def test_neutral_label_is_ignored(sentences):
    analyzer = SentimentAnalyzer()
    analyzer._classifier = _FixedClassifier("neutral")
    result = analyzer.analyze(sentences)
    assert result.positive == []
    assert result.negative == []


def test_deduplication_positive():
    analyzer = SentimentAnalyzer()
    analyzer._classifier = _FixedClassifier("positive")
    result = analyzer.analyze(["Gleicher Satz.", "Gleicher Satz."])
    assert result.positive.count("Gleicher Satz.") == 1


def test_mixed_labels_populate_both_lists(sentences):
    analyzer = SentimentAnalyzer()
    analyzer._classifier = _AlternatingClassifier()
    result = analyzer.analyze(sentences)
    assert len(result.positive) > 0
    assert len(result.negative) > 0
    assert len(result.positive) + len(result.negative) == len(sentences)


def test_clean_strips_marker_and_newline_before_tokenization(positive_analyzer):
    received: list[str] = []

    class _RecordTokenizer:
        def __call__(self, text, **kwargs):
            received.append(text)
            return {"input_ids": []}

        def decode(self, ids, **kwargs):
            return ""

    positive_analyzer._classifier.tokenizer = _RecordTokenizer()
    positive_analyzer._clean("Satz.\n[ENDOFPAGE0]\nNächster Satz.")
    assert received
    assert "[ENDOFPAGE" not in received[0]
    assert "\n" not in received[0]
