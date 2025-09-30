import asyncio
from typing import Sequence

import hydra
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from loguru import logger
from omegaconf import DictConfig
from tqdm.asyncio import tqdm

from hallucinations_kg.data.langchain import StrictPydanticOutputParser, get_repair_json_runnable
from hallucinations_kg.data.utils import get_dataset
from hallucinations_kg.defaults import ROOT_PATH, retry_llm_call
from hallucinations_kg.kg_builder.utils import Entities, Relations
from hallucinations_kg.utils.config import omegaconf_register_resolvers
from hallucinations_kg.utils.langchain import setup_langchain_llm_cache
from hallucinations_kg.utils.langchain_config import get_llm_from_config, get_prompt_from_config
from hallucinations_kg.utils.logging import setup_logger

setup_logger()
omegaconf_register_resolvers()

setup_langchain_llm_cache()


@hydra.main(
    version_base="1.3",
    config_path=str(ROOT_PATH / "config"),
    config_name="entities_relationships_extraction",
)
def main(cfg: DictConfig) -> None:
    logger.info(cfg)
    load_dotenv(ROOT_PATH / ".env")

    dataset = get_dataset(cfg.dataset)
    column = cfg.dataset[cfg.column]

    llm = get_llm_from_config(cfg.llm)
    fixing_llm = get_llm_from_config(cfg.fixing_llm)
    repair_json = get_repair_json_runnable()
    parser_entities = StrictPydanticOutputParser(pydantic_object=Entities)
    parser_entities_fixing = OutputFixingParser.from_llm(parser=parser_entities, llm=fixing_llm)

    prompt_entities = _get_prompt_from_config(cfg.prompt_entities, parser_entities, ["input"])
    chain_entities = (prompt_entities | llm | repair_json | parser_entities_fixing).with_retry()

    parser_relations = StrictPydanticOutputParser(pydantic_object=Relations)
    parser_relations_fixing = OutputFixingParser.from_llm(parser=parser_relations, llm=fixing_llm)
    prompt_relations = _get_prompt_from_config(
        cfg.prompt_relationships, parser_relations, ["input", "entities"]
    )
    chain_relations = (prompt_relations | llm | repair_json | parser_relations_fixing).with_retry()

    output_dataset = DatasetDict()

    for subset, ds in dataset.items():
        generated_ent = asyncio.run(
            aget_entities_semaphore(chain_entities, ds[column], cfg.llm.num_proc)
        )
        entities = [[str(e) for e in x.entities] for x in generated_ent]
        generated_rels = asyncio.run(
            aget_relations_semaphore(chain_relations, ds[column], entities, cfg.llm.num_proc)
        )
        relations = [x.relationship_types for x in generated_rels]

        output_dataset[subset] = Dataset.from_dict(
            {
                f"{column}_entities": entities,
                f"{column}_relations": relations,
            }
        )

    output_dataset.save_to_disk(cfg.output_dir)


def _get_prompt_from_config(
    cfg_prompt: DictConfig, parser: PydanticOutputParser, input_variables: list[str]
) -> ChatPromptTemplate:
    partial_variables = {
        "format_instructions": parser.get_format_instructions(),
        "examples": cfg_prompt.examples,
    }
    return get_prompt_from_config(
        cfg_prompt=cfg_prompt,
        input_variables=input_variables,
        partial_variables=partial_variables,
    )


async def aget_entities_semaphore(
    chain: Runnable,
    documents: Sequence[str],
    max_concurrent_tasks: int,
) -> list[Entities]:
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    @retry_llm_call
    async def sem_task(document: str) -> Entities:
        async with semaphore:
            result = await chain.ainvoke({"input": document}, partial=False)
            assert result is not None, f"Result is None for document: {document}"
            return result

    tasks = [sem_task(document) for document in documents]
    results = await tqdm.gather(*tasks, desc="Extracting entities")
    return results


async def aget_relations_semaphore(
    chain: Runnable,
    documents: Sequence[str],
    entities: Sequence[list[str]],
    max_concurrent_tasks: int,
) -> list[Relations]:
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    @retry_llm_call
    async def sem_task(document: str, entities: list[str]) -> Relations:
        async with semaphore:
            return await chain.ainvoke({"input": document, "entities": entities})

    tasks = [sem_task(document, ent_) for document, ent_ in zip(documents, entities, strict=True)]
    results = await tqdm.gather(*tasks, desc="Extracting relations")
    return results


if __name__ == "__main__":
    main()
