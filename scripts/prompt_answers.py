import logging
from functools import partial
from typing import Any

import hydra
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig

from hallucinations_kg.data.postprocessing import add_allowed_nodes_and_relationships, parse_graph
from hallucinations_kg.data.utils import get_processed_dataset
from hallucinations_kg.defaults import ROOT_PATH
from hallucinations_kg.models.predictor import (
    Graph,
)
from hallucinations_kg.models.prompt_predictor import FactPromptAgent
from hallucinations_kg.utils.langchain import setup_langchain_llm_cache
from hallucinations_kg.utils.langchain_config import get_llm_from_config, get_prompt_from_config
from hallucinations_kg.utils.logging import setup_logger

setup_logger()

setup_langchain_llm_cache()

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


@hydra.main(version_base="1.3", config_path=str(ROOT_PATH / "config"), config_name="predict_prompt")
def main(cfg: DictConfig) -> None:
    logger.info(cfg)
    load_dotenv()

    restrictions_column = cfg.processed_dataset.original_dataset[
        cfg.processed_dataset.restricted_by_column
    ]
    response_sentences_column = cfg.processed_dataset.original_dataset.response_sentences_column
    samples_column = cfg.processed_dataset.original_dataset.samples_column
    sentences_column = cfg.processed_dataset.original_dataset.response_sentences_column

    dataset = get_processed_dataset(cfg.processed_dataset)

    dataset = dataset.map(
        partial(
            add_allowed_nodes_and_relationships,
            restrictions_column=restrictions_column,
            response_sentences_column=response_sentences_column,
        )
    )
    dataset = dataset.map(
        partial(
            parse_graph,
            samples_column=samples_column,
            sentences_column=sentences_column,
            restrict=False,
        )
    )
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
    sentences_graphs = [Graph.from_triples(g) for g in entry[f"{sentences_column}_graph"]]
    samples_graphs = [Graph.from_triples(g) for g in entry[f"{samples_column}_graph"]]

    predictions = []
    for g in sentences_graphs:
        sentence_predictions = [agent.get_answers(fact, samples_graphs) for fact in g.triples]
        predictions.append(sentence_predictions)
    entry["fact_prompt_answers"] = predictions
    return entry


def get_agent(cfg: DictConfig) -> FactPromptAgent:
    prompt = get_prompt_from_config(
        cfg.prompt,
        input_variables=["input", "context"],
        partial_variables={"examples": cfg.prompt.examples},
    )
    llm = get_llm_from_config(cfg.llm)
    return FactPromptAgent(prompt=prompt, llm=llm)


if __name__ == "__main__":
    main()
