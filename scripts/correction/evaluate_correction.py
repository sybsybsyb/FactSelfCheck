import re
from functools import partial
from typing import Any

import hydra
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig

from hallucinations_kg.correction.correction_evaluator import CorrectionEvaluationAgent
from hallucinations_kg.data.utils import get_processed_dataset
from hallucinations_kg.defaults import ROOT_PATH
from hallucinations_kg.utils.langchain import setup_langchain_llm_cache
from hallucinations_kg.utils.langchain_config import get_llm_from_config, get_prompt_from_config
from hallucinations_kg.utils.logging import setup_logger

setup_logger()

setup_langchain_llm_cache()


@hydra.main(
    version_base="1.3", config_path=str(ROOT_PATH / "config"), config_name="evaluate_correction"
)
def main(cfg: DictConfig) -> None:
    logger.info(cfg)
    load_dotenv()
    source_column = cfg.processed_dataset.original_dataset.source_column

    dataset = get_processed_dataset(cfg.processed_dataset)
    dataset["evaluation"] = dataset["evaluation"]

    remove_columns = next(iter(dataset.column_names.values()))
    for evaluation_column in cfg.evaluation_columns:
        dataset = dataset.map(
            partial(
                add_evaluation,
                cfg=cfg,
                source_column=source_column,
                evaluation_column=evaluation_column,
            ),
            num_proc=cfg.llm.num_proc,
        )
    dataset = dataset.remove_columns(remove_columns)
    dataset.save_to_disk(cfg.output_dir)


def add_evaluation(
    entry: dict[str, Any], cfg: DictConfig, source_column: str, evaluation_column: str
) -> dict[str, Any]:
    sentences = parse_sentences(entry[evaluation_column])
    agent = get_agent(cfg)
    full_text = " ".join(sentences)
    evaluations = []
    for sentence in sentences:
        evaluation = agent.evaluate(sentence, full_text, entry[source_column])
        evaluations.append(evaluation)
    entry[f"{evaluation_column}_evaluations"] = evaluations
    return entry


def parse_sentences(sentences_str: str) -> list[str]:
    pattern = r"^\d+\.\s*(.*)$"
    new_sentences = []
    for sentence in sentences_str.split("\n"):
        match = re.search(pattern, sentence)
        if match:
            content = match.group(1)
            new_sentences.append(content)
    return new_sentences


def get_agent(cfg: DictConfig) -> CorrectionEvaluationAgent:
    prompt = get_prompt_from_config(
        cfg.correction_evaluation_prompt,
        input_variables=["input", "source", "full_text"],
        partial_variables={"examples": cfg.correction_evaluation_prompt.examples},
    )
    llm = get_llm_from_config(cfg.llm_evaluation)
    return CorrectionEvaluationAgent(prompt=prompt, llm=llm)


if __name__ == "__main__":
    main()
