import os
from abc import ABC
from pathlib import Path
from typing import Any

from langchain.schema import StrOutputParser
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from hallucinations_kg.defaults import LANGCHAIN_CACHE_PATH, LLM_RETRIES


class BaseAgent(ABC):
    def __init__(self, prompt: ChatPromptTemplate, llm: ChatOpenAI):
        self.chain = self._get_chain(prompt, llm)

    def _get_chain(self, prompt: ChatPromptTemplate, llm: ChatOpenAI) -> RunnableSerializable:
        return prompt | llm | StrOutputParser()


def setup_langchain_llm_cache(cache_path: Path = LANGCHAIN_CACHE_PATH) -> None:
    set_llm_cache(SQLiteCache(database_path=str(cache_path)))


def get_llm(
    env_base_url: str, env_key: str, temperature: float, model_id: str, max_tokens: int
) -> ChatOpenAI:
    base_url = os.getenv(env_base_url) if env_base_url else None
    api_key = os.getenv(env_key)
    assert api_key
    return ChatOpenAI(
        temperature=temperature,
        model=model_id,
        base_url=base_url,
        max_retries=LLM_RETRIES,
        api_key=SecretStr(api_key),
        max_tokens=max_tokens,
    )


def get_prompt(
    system_prompt: str,
    human_prompt: str,
    input_variables: list[str],
    partial_variables: dict[str, Any],
) -> ChatPromptTemplate:
    system_message = SystemMessage(content=system_prompt)
    human_prompt_template = PromptTemplate(
        template=human_prompt,
        input_variables=input_variables,
        partial_variables=partial_variables,
    )
    human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message_prompt])
    return chat_prompt
