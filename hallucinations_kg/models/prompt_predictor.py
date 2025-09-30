import re
from statistics import mean
from typing import Any

from loguru import logger

from hallucinations_kg.defaults import retry_llm_call
from hallucinations_kg.models.predictor import FactPredictor, Graph, Triple
from hallucinations_kg.utils.langchain import BaseAgent


class FactPromptAgent(BaseAgent):
    def get_answers(self, fact: Triple, samples_graphs: list[Graph], **kwargs: Any) -> list[str]:
        answers = []
        for sample_graph in samples_graphs:
            answers.append(self.get_answer(fact, sample_graph))
        return answers

    @retry_llm_call
    def get_answer(self, fact: Triple, sample_graph: Graph) -> str:
        result = self.chain.invoke(
            {"input": fact.to_str(), "knowledge_graph": sample_graph.to_str()}
        ).lower()
        assert isinstance(result, str)
        return result


class FactTextPromptAgent(BaseAgent):
    def get_answers(self, fact: Triple, samples: list[str], **kwargs: Any) -> list[str]:
        answers = []
        for sample in samples:
            answers.append(self.get_answer(fact, sample))
        return answers

    @retry_llm_call
    def get_answer(self, fact: Triple, sample: str) -> str:
        result = self.chain.invoke({"input": fact.to_str(), "context": sample}).lower()
        assert isinstance(result, str)
        return result


class FactPromptPredictor(FactPredictor):
    def predict(self, fact: Triple, samples_graphs: list[Graph], **kwargs: Any) -> float:
        assert "answers" in kwargs
        assert len(samples_graphs) == len(kwargs["answers"])
        answers = kwargs["answers"]
        scores = []
        for answer in answers:
            answer = re.findall(r"[\w]+|[.,!?;\"']", answer)
            if "yes" in answer:
                scores.append(0)
            elif "no" in answer:
                scores.append(1)
            else:
                logger.warning(f"Answer is not valid: {answer.__repr__()}")
        return mean(scores)

    def name(self) -> str:
        return "FactSelfCheck-KG (LLM-based)"


class FactTextPromptPredictor(FactPromptPredictor):
    def name(self) -> str:
        return "FactSelfCheck-Text"
