from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import (
    ChatPromptTemplate,
)

from hallucinations_kg.defaults import retry_llm_call


class CSVLLMGraphTransformer:
    def __init__(
        self,
        llm: BaseLanguageModel,
        prompt: ChatPromptTemplate,
    ) -> None:
        self.chain = (prompt | llm).with_retry(stop_after_attempt=5)

    @retry_llm_call
    async def aprocess_sentence(
        self,
        sentence: str,
        full_text: str,
        allowed_nodes: list[str],
        allowed_relationships: list[str],
    ) -> str:
        assert isinstance(sentence, str)
        assert isinstance(full_text, str)
        assert isinstance(allowed_nodes, list)
        assert isinstance(allowed_relationships, list)
        result = await self.chain.ainvoke(
            {
                "input_sentence": sentence,
                "input_text": full_text,
                "allowed_nodes": allowed_nodes,
                "allowed_relationships": allowed_relationships,
            }
        )
        assert isinstance(result.content, str)
        return result.content

    @retry_llm_call
    async def aprocess_doc(
        self,
        document: str,
        allowed_nodes: list[str],
        allowed_relationships: list[str],
    ) -> str:
        assert isinstance(document, str)
        assert isinstance(allowed_nodes, list)
        assert isinstance(allowed_relationships, list)
        result = await self.chain.ainvoke(
            {
                "input": document,
                "allowed_nodes": allowed_nodes,
                "allowed_relationships": allowed_relationships,
            }
        )
        return result.content
