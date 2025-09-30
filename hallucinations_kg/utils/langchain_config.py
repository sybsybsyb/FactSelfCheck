from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from omegaconf import DictConfig

from hallucinations_kg.utils.langchain import get_llm, get_prompt


def get_llm_from_config(cfg_llm: DictConfig) -> ChatOpenAI:
    return get_llm(
        env_base_url=cfg_llm.env_base_url,
        env_key=cfg_llm.env_key,
        temperature=cfg_llm.temperature,
        model_id=cfg_llm.id,
        max_tokens=cfg_llm.max_tokens,
    )


def get_prompt_from_config(
    cfg_prompt: DictConfig, input_variables: list[str], partial_variables: dict[str, Any]
) -> ChatPromptTemplate:
    return get_prompt(
        system_prompt=cfg_prompt.system_prompt,
        human_prompt=cfg_prompt.human_prompt,
        input_variables=input_variables,
        partial_variables=partial_variables,
    )
