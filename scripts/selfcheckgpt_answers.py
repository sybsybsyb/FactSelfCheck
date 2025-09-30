import logging
from functools import partial
from typing import Any

import hydra
from dotenv import load_dotenv
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain.prompts.chat import MessageLikeRepresentation
from langchain.schema import SystemMessage
from loguru import logger
from omegaconf import DictConfig

from hallucinations_kg.data.utils import get_processed_dataset
from hallucinations_kg.defaults import ROOT_PATH
from hallucinations_kg.models.baselines.selfcheckgpt import SelfCheckGPTAgent
from hallucinations_kg.utils.langchain import setup_langchain_llm_cache
from hallucinations_kg.utils.langchain_config import get_llm_from_config
from hallucinations_kg.utils.logging import setup_logger

setup_logger()

setup_langchain_llm_cache()

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


@hydra.main(
    version_base="1.3", config_path=str(ROOT_PATH / "config"), config_name="predict_selfcheckgpt"
)
def main(cfg: DictConfig) -> None:
    logger.info(cfg)
    load_dotenv()

    dataset = get_processed_dataset(cfg.processed_dataset)
    samples_column = cfg.processed_dataset.original_dataset.samples_column
    sentences_column = cfg.processed_dataset.original_dataset.response_sentences_column

    remove_columns = next(iter(dataset.column_names.values()))
    dataset = dataset.map(
        partial(
            add_prompt_answers,
            samples_column=samples_column,
            sentences_column=sentences_column,
            cfg=cfg,
        ),
        num_proc=cfg.llm.num_proc,
        remove_columns=remove_columns,
    )
    dataset.save_to_disk(cfg.output_dir)


def add_prompt_answers(
    entry: dict[str, Any], samples_column: str, sentences_column: str, cfg: DictConfig
) -> dict[str, Any]:
    agent = get_agent(cfg)

    predictions = []
    for sentence in entry[sentences_column]:
        sentence_predictions = agent.get_answers(sentence, entry[samples_column])
        predictions.append(sentence_predictions)
    entry["selfcheckgpt_answers"] = predictions
    return entry


def get_agent(cfg: DictConfig) -> SelfCheckGPTAgent:
    prompt = _get_prompt_from_config(cfg, ["context", "sentence"])
    llm = get_llm_from_config(cfg.llm)
    return SelfCheckGPTAgent(prompt=prompt, llm=llm)


def _get_prompt_from_config(cfg: DictConfig, input_variables: list[str]) -> ChatPromptTemplate:
    if cfg.prompt.system_prompt:
        system_message = SystemMessage(content=cfg.prompt.system_prompt)
    else:
        system_message = None
    human_prompt = PromptTemplate(
        template=cfg.prompt.human_prompt,
        input_variables=input_variables,
        partial_variables={
            "examples": cfg.prompt.examples,
        }
        if cfg.prompt.examples
        else {},
    )
    human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
    messages: list[MessageLikeRepresentation]
    if system_message:
        messages = [system_message, human_message_prompt]
    else:
        messages = [human_message_prompt]
    return ChatPromptTemplate.from_messages(messages)


if __name__ == "__main__":
    main()
