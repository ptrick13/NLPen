import pytest
from unittest.mock import patch

from app import _build_analyzer
from nlpen.nlp.combined import CombinationMode


def test_summarize_only():
    with (
        patch("app.SentenceSummarizer") as MockSummarizer,
        patch("app.SentimentAnalyzer"),
        patch("app.NERAnalyzer"),
        patch("app.CombinedAnalyzer"),
    ):
        result = _build_analyzer(summarize=True, sentiment=False, ner=False)
        MockSummarizer.assert_called_once_with()
        assert result is MockSummarizer.return_value


def test_sentiment_only():
    with (
        patch("app.SentenceSummarizer"),
        patch("app.SentimentAnalyzer") as MockSentiment,
        patch("app.NERAnalyzer"),
        patch("app.CombinedAnalyzer"),
    ):
        result = _build_analyzer(summarize=False, sentiment=True, ner=False)
        MockSentiment.assert_called_once_with()
        assert result is MockSentiment.return_value


def test_ner_only():
    with (
        patch("app.SentenceSummarizer"),
        patch("app.SentimentAnalyzer"),
        patch("app.NERAnalyzer") as MockNER,
        patch("app.CombinedAnalyzer"),
    ):
        result = _build_analyzer(summarize=False, sentiment=False, ner=True)
        MockNER.assert_called_once_with()
        assert result is MockNER.return_value


def test_summarize_and_sentiment():
    with (
        patch("app.SentenceSummarizer"),
        patch("app.SentimentAnalyzer"),
        patch("app.NERAnalyzer"),
        patch("app.CombinedAnalyzer") as MockCombined,
    ):
        result = _build_analyzer(summarize=True, sentiment=True, ner=False)
        MockCombined.assert_called_once_with(
            CombinationMode.SENTIMENT_AND_SUMMARIZATION
        )
        assert result is MockCombined.return_value


def test_summarize_and_ner():
    with (
        patch("app.SentenceSummarizer"),
        patch("app.SentimentAnalyzer"),
        patch("app.NERAnalyzer"),
        patch("app.CombinedAnalyzer") as MockCombined,
    ):
        result = _build_analyzer(summarize=True, sentiment=False, ner=True)
        MockCombined.assert_called_once_with(CombinationMode.NER_AND_SUMMARIZATION)
        assert result is MockCombined.return_value


def test_sentiment_and_ner():
    with (
        patch("app.SentenceSummarizer"),
        patch("app.SentimentAnalyzer"),
        patch("app.NERAnalyzer"),
        patch("app.CombinedAnalyzer") as MockCombined,
    ):
        result = _build_analyzer(summarize=False, sentiment=True, ner=True)
        MockCombined.assert_called_once_with(CombinationMode.NER_AND_SENTIMENT)
        assert result is MockCombined.return_value


def test_invalid_combination_raises():
    with (
        patch("app.SentenceSummarizer"),
        patch("app.SentimentAnalyzer"),
        patch("app.NERAnalyzer"),
        patch("app.CombinedAnalyzer"),
    ):
        with pytest.raises(ValueError):
            _build_analyzer(summarize=False, sentiment=False, ner=False)


def test_all_three_raises():
    with (
        patch("app.SentenceSummarizer"),
        patch("app.SentimentAnalyzer"),
        patch("app.NERAnalyzer"),
        patch("app.CombinedAnalyzer"),
    ):
        with pytest.raises(ValueError):
            _build_analyzer(summarize=True, sentiment=True, ner=True)
