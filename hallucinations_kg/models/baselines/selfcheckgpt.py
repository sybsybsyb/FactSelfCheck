import re
from statistics import mean
from typing import Any

from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from loguru import logger

from hallucinations_kg.defaults import retry_llm_call
from hallucinations_kg.models.predictor import Graph
from hallucinations_kg.models.sentence_predictor import SentencePredictor


class SelfCheckGPTAgent:
    def __init__(self, prompt: ChatPromptTemplate, llm: ChatOpenAI):
        self.chain = self._get_chain(prompt, llm)

    def _get_chain(self, prompt: ChatPromptTemplate, llm: ChatOpenAI) -> RunnableSerializable:
        return prompt | llm | StrOutputParser()

    def get_answers(self, sentence: str, samples: list[str], **kwargs: Any) -> list[str]:
        answers = []
        for sample in samples:
            answers.append(self.get_answer(sentence, sample))
        return answers

    @retry_llm_call
    def get_answer(self, sentence: str, sample: str) -> str:
        result = self.chain.invoke({"sentence": sentence, "context": sample}).lower()
        assert isinstance(result, str)
        return result


class SelfCheckGPTPredictor(SentencePredictor):
    """
    SelfCheckGPT (Prompt)
    """

    def __init__(self, n_samples: int | None = None):
        self.n_samples = n_samples

    def predict(self, sentence_graph: Graph, samples_graphs: list[Graph], **kwargs: Any) -> float:
        assert "answers" in kwargs
        samples_graphs = samples_graphs[: self.n_samples]
        answers = kwargs["answers"][: self.n_samples]
        assert len(samples_graphs) == len(answers)

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
        return "SelfCheckGPT-reproduced"

    def params(self) -> dict[str, Any]:
        return {
            "sentence_predictor": "SelfCheckGPT-reproduced",
            "n_samples": self.n_samples,
        }


class SelfCheckGPTNLIPredictor(SentencePredictor):
    """
    SelfCheckGPT (NLI)
    """

    def __init__(self, n_samples: int | None = None):
        self.n_samples = n_samples

    def predict(self, sentence_graph: Graph, samples_graphs: list[Graph], **kwargs: Any) -> float:
        assert "answers" in kwargs
        samples_graphs = samples_graphs[: self.n_samples]
        answers = kwargs["answers"][: self.n_samples]
        assert len(samples_graphs) == len(answers)

        scores = []
        for answer in answers:
            scores.append(answer)
        return mean(scores)

    def name(self) -> str:
        return "SelfCheckGPT-NLI-reproduced"

    def params(self) -> dict[str, Any]:
        return {
            "sentence_predictor": "SelfCheckGPT-NLI-reproduced",
            "n_samples": self.n_samples,
        }
