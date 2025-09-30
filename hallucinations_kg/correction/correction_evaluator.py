from typing import Any

from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class CorrectionEvaluationAgent:
    def __init__(self, prompt: ChatPromptTemplate, llm: ChatOpenAI):
        self.chain = self._get_chain(prompt, llm)

    def _get_chain(self, prompt: ChatPromptTemplate, llm: ChatOpenAI) -> RunnableSerializable:
        return prompt | llm | StrOutputParser()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=300))
    def evaluate(self, sentence: str, full_text: str, source: str, **kwargs: Any) -> str:
        result = self.chain.invoke(
            {"input": sentence, "full_text": full_text, "source": source}
        ).lower()
        assert isinstance(result, str)
        return result
