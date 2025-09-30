import random
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np

from hallucinations_kg.models.predictor import FactPredictor, Graph
from hallucinations_kg.models.prompt_predictor import FactPromptPredictor


class SentencePredictor(ABC):
    def __init__(self, **kwargs: Any):
        pass

    @abstractmethod
    def predict(
        self, sentence_graph: Graph, samples_graphs: list[Graph], **kwargs: Any
    ) -> float | None:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    def params(self) -> dict[str, Any]:
        return {"sentence_predictor": self.name()}


class RandomSentencePredictor(SentencePredictor):
    def __init__(self, random_state: int = 17):
        self.gen = random.Random(random_state)

    def predict(self, sentence_graph: Graph, samples_graphs: list[Graph], **kwargs: Any) -> float:
        return self.gen.random()

    def name(self) -> str:
        return "Random"


class MostFrequentSentencePredictor(SentencePredictor):
    def __init__(self, y_true: list[float]):
        self.most_frequent_y = float(np.argmax(np.bincount(y_true)))
        assert self.most_frequent_y in [0, 1]

    def predict(self, sentence_graph: Graph, samples_graphs: list[Graph], **kwargs: Any) -> float:
        return self.most_frequent_y

    def name(self) -> str:
        return "MostFrequent"


class SentenceAggregatePredictor(SentencePredictor):
    def __init__(
        self,
        fact_predictor: FactPredictor,
        aggregation_function: Callable[[list[float]], float],
        n_samples: int | None = None,
    ):
        self.fact_predictor = fact_predictor
        self.aggregation_function = aggregation_function
        self.n_samples = n_samples

    def predict(
        self, sentence_graph: Graph, samples_graphs: list[Graph], **kwargs: Any
    ) -> float | None:
        samples_graphs = samples_graphs[: self.n_samples]
        if isinstance(self.fact_predictor, FactPromptPredictor):
            assert "answers" in kwargs
            predictions = [
                self.fact_predictor.predict(
                    fact, samples_graphs, answers=fact_answers[: self.n_samples]
                )
                for fact, fact_answers in zip(
                    sentence_graph.triples, kwargs["answers"], strict=True
                )
            ]
        else:
            predictions = [
                self.fact_predictor.predict(fact, samples_graphs) for fact in sentence_graph.triples
            ]
        if predictions:
            return self.aggregation_function(predictions)
        else:
            return None

    def name(self) -> str:
        return f"{self.aggregation_function.__name__}({self.fact_predictor.name()})"

    def params(self) -> dict[str, Any]:
        return {
            **self.fact_predictor.params(),
            "sentence_predictor": "Aggregate",
            "agg_func": self.aggregation_function.__name__,
            "n_samples": self.n_samples,
        }
