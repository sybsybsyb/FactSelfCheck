import asyncio
import logging
from functools import partial

import hydra
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from omegaconf import DictConfig
from rich import print

from hallucinations_kg.data.postprocessing import add_allowed_nodes_and_relationships
from hallucinations_kg.data.utils import get_processed_dataset
from hallucinations_kg.defaults import ROOT_PATH
from hallucinations_kg.kg_builder.langchain_builder import (
    CSVLangChainKGBuilder,
)
from hallucinations_kg.utils.config import omegaconf_register_resolvers
from hallucinations_kg.utils.langchain import setup_langchain_llm_cache
from hallucinations_kg.utils.langchain_config import get_llm_from_config, get_prompt_from_config
from hallucinations_kg.utils.logging import setup_logger

setup_logger()

omegaconf_register_resolvers()

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

setup_langchain_llm_cache()


@hydra.main(version_base="1.3", config_path=str(ROOT_PATH / "config"), config_name="kg_building")
def main(cfg: DictConfig) -> None:
    print(cfg)
    load_dotenv()

    if cfg.restricted:
        if cfg.level == "document":
            cfg_prompt = cfg.prompt.restricted
        elif cfg.level == "sentence":
            cfg_prompt = cfg.prompt.restricted_sentence
        else:
            raise ValueError
    else:
        raise NotImplementedError("Unrestricted graphs are deprecated")

    dataset = get_processed_dataset(cfg.processed_dataset)
    column = cfg.processed_dataset.original_dataset[cfg.column]

    llm = get_llm_from_config(cfg.llm)
    prompt = get_prompt_from_config(
        cfg_prompt, input_variables=["input"], partial_variables={"examples": cfg_prompt.examples}
    )

    output_dataset = DatasetDict()
    for subset, ds in dataset.items():
        if cfg.level == "sentence":
            sentences_graphs = get_restricted_graph_sentence(llm, ds, column, prompt, cfg)
            assert len(sentences_graphs) == len(ds)
            output_dataset[subset] = Dataset.from_dict({f"{column}_raw_graph": sentences_graphs})

        elif cfg.level == "document" and isinstance(ds[column][0], list):  # samples column
            allowed_nodes, allowed_relationships = get_allowed_nodes_and_relationships(cfg, ds)
            samples_graphs = get_restricted_graph_samples(
                llm, ds, column, prompt, cfg, allowed_nodes, allowed_relationships
            )
            assert len(samples_graphs) == len(ds)
            output_dataset[subset] = Dataset.from_dict({f"{column}_raw_graph": samples_graphs})
        else:
            raise NotImplementedError(
                "Only sentence level and document level for samples are supported."
            )

    output_dataset.save_to_disk(cfg.output_dir)


def get_restricted_graph_sentence(
    llm: ChatOpenAI,
    dataset: Dataset,
    column: str,
    prompt: ChatPromptTemplate,
    cfg: DictConfig,
) -> list[list[str]]:
    org_dataset_cfg = cfg.processed_dataset.original_dataset
    restrictions_column = org_dataset_cfg[cfg.restricted_by_column]
    full_text_column = org_dataset_cfg[cfg.full_text_column]
    allowed_nodes = dataset[f"{restrictions_column}_entities"]
    allowed_relationships = dataset[f"{restrictions_column}_relations"]

    builder = CSVLangChainKGBuilder(
        llm=llm,
        prompt=prompt,
    )
    graphs = asyncio.run(
        builder.abuild_sentence(
            sentences=dataset[column],
            full_text=dataset[full_text_column],
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_relationships,
            n_jobs=cfg.llm.num_proc,
        )
    )
    return graphs


def get_allowed_nodes_and_relationships(
    cfg: DictConfig, dataset: Dataset
) -> tuple[list[list[str]], list[list[str]]]:
    restrictions_column = cfg.processed_dataset.original_dataset[cfg.restricted_by_column]
    org_dataset_cfg = cfg.processed_dataset.original_dataset
    response_sentences_column = org_dataset_cfg.response_sentences_column
    dataset = dataset.map(
        partial(
            add_allowed_nodes_and_relationships,
            restrictions_column=restrictions_column,
            response_sentences_column=response_sentences_column,
        )
    )
    return dataset["allowed_nodes"], dataset["allowed_relationships"]


def get_restricted_graph_samples(
    llm: ChatOpenAI,
    dataset: Dataset,
    column: str,
    prompt: ChatPromptTemplate,
    cfg: DictConfig,
    allowed_nodes: list[list[str]],
    allowed_relationships: list[list[str]],
) -> list[list[str]]:
    builder = CSVLangChainKGBuilder(
        llm=llm,
        prompt=prompt,
    )
    graphs = asyncio.run(
        builder.abuild_samples(
            samples=dataset[column],
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_relationships,
            n_jobs=cfg.llm.num_proc,
        )
    )
    return graphs


if __name__ == "__main__":
    main()
