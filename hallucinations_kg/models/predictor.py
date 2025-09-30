import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Triple:
    head: str
    relation: str
    tail: str

    def __post_init__(self) -> None:
        assert isinstance(self.head, str)
        assert isinstance(self.relation, str)
        assert isinstance(self.tail, str)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Triple):
            return False
        return (
            self.head == other.head and self.relation == other.relation and self.tail == other.tail
        )

    def to_str(self) -> str:
        return f"({self.head}, {self.relation}, {self.tail})"


@dataclass
class Graph:
    triples: list[Triple]

    def __post_init__(self) -> None:
        assert isinstance(self.triples, list)
        assert all(isinstance(t, Triple) for t in self.triples)

    @classmethod
    def from_triples(cls, triples: list[tuple[str, str, str]]) -> "Graph":
        return cls(triples=[Triple(*t) for t in triples])

    def to_str(self) -> str:
        return f"[{', '.join([t.to_str() for t in self.triples])}]"


class FactPredictor(ABC):
    @abstractmethod
    def predict(self, fact: Triple, samples_graphs: list[Graph], **kwargs: Any) -> float:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    def params(self) -> dict[str, Any]:
        return {
            "fact_predictor": self.name(),
        }


class RandomFactPredictor(FactPredictor):
    def __init__(self, random_state: int = 17) -> None:
        self.gen = random.Random(random_state)

    def predict(self, fact: Triple, samples_graphs: list[Graph], **kwargs: Any) -> float:
        return self.gen.random()

    def name(self) -> str:
        return "RandomFact"


class FactOccurrencePredictor(FactPredictor):
    def predict(self, fact: Triple, samples_graphs: list[Graph], **kwargs: Any) -> float:
        counter = 0
        for sample_graph in samples_graphs:
            if fact in sample_graph.triples:
                counter += 1
        return 1 - (counter / len(samples_graphs))

    def name(self) -> str:
        return "FactSelfCheck-KG (Frequency-based)"
