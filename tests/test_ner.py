import pytest
from flair.data import Sentence, Span

from nlpen.nlp.ner import NERAnalyzer


def _make_tagger(word_tag_pairs: list[tuple[str, str]]):
    """Return a mock SequenceTagger that tags specific words with given NER labels.

    Uses real flair.data.Sentence and flair.data.Span so the NERAnalyzer's
    get_spans / .tag / .text interface is exercised without loading a model.
    """

    class MockTagger:
        def predict(self, sentence: Sentence) -> None:
            for token in sentence.tokens:
                for word, tag in word_tag_pairs:
                    if token.text == word:
                        span = Span([token])
                        span.add_label("ner", tag, score=1.0)

    return MockTagger()


@pytest.fixture()
def analyzer():
    a = NERAnalyzer()
    return a


def test_per_entity_routed_to_persons(analyzer):
    analyzer._tagger = _make_tagger([("Angela", "PER")])
    result = analyzer.analyze(["Angela Merkel ist Politikerin."])
    assert any(entity == "Angela" for _, entity in result.persons)
    assert result.locations == []
    assert result.organizations == []


def test_loc_entity_routed_to_locations(analyzer):
    analyzer._tagger = _make_tagger([("Berlin", "LOC")])
    result = analyzer.analyze(["Berlin ist die Hauptstadt."])
    assert any(entity == "Berlin" for _, entity in result.locations)
    assert result.persons == []


def test_org_entity_routed_to_organizations(analyzer):
    analyzer._tagger = _make_tagger([("Siemens", "ORG")])
    result = analyzer.analyze(["Siemens ist ein Konzern."])
    assert any(entity == "Siemens" for _, entity in result.organizations)
    assert result.persons == []


def test_misc_entity_routed_to_misc(analyzer):
    analyzer._tagger = _make_tagger([("Europa", "MISC")])
    result = analyzer.analyze(["Europa ist ein Kontinent."])
    assert any(entity == "Europa" for _, entity in result.misc)


def test_deduplication(analyzer):
    analyzer._tagger = _make_tagger([("Angela", "PER")])
    sentence = "Angela Merkel ist Politikerin."
    result = analyzer.analyze([sentence, sentence])
    pairs = [(s, e) for s, e in result.persons if e == "Angela"]
    assert len(pairs) == 1
