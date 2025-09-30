from functools import partial
from typing import Any

import hydra
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig

from hallucinations_kg.annotation.fact import FactAnnotationAgent
from hallucinations_kg.data.postprocessing import (
    add_allowed_nodes_and_relationships,
    parse_graph,
)
from hallucinations_kg.data.utils import get_processed_dataset
from hallucinations_kg.defaults import ROOT_PATH
from hallucinations_kg.utils.langchain import setup_langchain_llm_cache
from hallucinations_kg.utils.langchain_config import get_llm_from_config, get_prompt_from_config
from hallucinations_kg.utils.logging import setup_logger

setup_logger()

setup_langchain_llm_cache()


@hydra.main(
    version_base="1.3", config_path=str(ROOT_PATH / "config"), config_name="fact_level_annotate"
)
def main(cfg: DictConfig) -> None:
    logger.info(cfg)
    load_dotenv()

    restrictions_column = cfg.processed_dataset.original_dataset.response_column
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
            postprocess=False,
        )
    )
    remove_columns = next(iter(dataset.column_names.values()))
    dataset = dataset.map(
        partial(
            add_fact_annotation,
            cfg=cfg,
        ),
        num_proc=cfg.llm.num_proc,
        remove_columns=remove_columns,
    )

    dataset.save_to_disk(cfg.output_dir)


def add_fact_annotation(
    entry: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Any]:
    gpt3_sentences_graph = entry["gpt3_sentences_graph"]
    source = entry["wiki_bio_text"]

    agent = get_agent(cfg)
    annotations = []
    for sentence_graph in gpt3_sentences_graph:
        sentence_annotations = []
        for fact in sentence_graph:
            annotation = agent.annotate(tuple(fact), source)
            sentence_annotations.append(annotation)
        annotations.append(sentence_annotations)
    entry["gpt_sentences_fact_level_annotations"] = annotations
    return entry


def get_agent(cfg: DictConfig) -> FactAnnotationAgent:
    prompt = get_prompt_from_config(
        cfg.prompt,
        input_variables=["fact", "source"],
        partial_variables={"examples": cfg.prompt.examples},
    )
    llm = get_llm_from_config(cfg.llm_annotation)
    return FactAnnotationAgent(prompt=prompt, llm=llm)


if __name__ == "__main__":
    main()
