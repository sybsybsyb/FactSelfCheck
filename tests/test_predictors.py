import pytest

from hallucinations_kg.models.predictor import (
    FactOccurrencePredictor,
    Graph,
    RandomFactPredictor,
    Triple,
)
from hallucinations_kg.models.sentence_predictor import MostFrequentSentencePredictor


def test_random_fact_predictor() -> None:
    predictor = RandomFactPredictor()
    fact = Triple(head="a", relation="b", tail="c")
    samples_graphs = [Graph(triples=[fact])]
    assert 0 <= predictor.predict(fact, samples_graphs) <= 1


def test_most_frequent_sentence_predictor() -> None:
    predictor = MostFrequentSentencePredictor([0, 1, 1, 1, 0])
    sentence_graph = Graph(triples=[Triple(head="a", relation="b", tail="c")])
    samples_graphs = [Graph(triples=[Triple(head="a", relation="b", tail="c")])]
    assert predictor.predict(sentence_graph, samples_graphs) == 1


def test_fact_occurrence_predictor() -> None:
    predictor = FactOccurrencePredictor()
    fact = Triple(head="a", relation="b", tail="c")
    sample_graph_0 = Graph(
        triples=[Triple(head="a", relation="b", tail="f"), Triple(head="a", relation="b", tail="p")]
    )
    sample_graph_1 = Graph(
        triples=[Triple(head="a", relation="b", tail="c"), Triple(head="a", relation="b", tail="p")]
    )
    samples_graphs = [sample_graph_0, sample_graph_1]
    assert predictor.predict(fact, samples_graphs) == 0.5

    samples_graph_2 = Graph(
        triples=[Triple(head="a", relation="b", tail="c"), Triple(head="a", relation="b", tail="p")]
    )
    samples_graphs = [sample_graph_0, sample_graph_1, samples_graph_2]
    assert predictor.predict(fact, samples_graphs) == pytest.approx(0.33333333333333)
