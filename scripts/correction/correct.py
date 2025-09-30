from functools import partial
from statistics import mean
from typing import Any, Callable

import hydra
import numpy as np
from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import precision_recall_curve

from hallucinations_kg.correction.corrector import CorrectorAgent
from hallucinations_kg.data.postprocessing import (
    add_allowed_nodes_and_relationships,
    parse_graph,
    postprocess_ents_rels,
)
from hallucinations_kg.data.utils import get_processed_dataset
from hallucinations_kg.defaults import ROOT_PATH
from hallucinations_kg.models.baselines.selfcheckgpt import SelfCheckGPTPredictor
from hallucinations_kg.models.predictor import FactPredictor
from hallucinations_kg.models.prompt_predictor import FactTextPromptPredictor
from hallucinations_kg.models.sentence_predictor import SentencePredictor
from hallucinations_kg.prediction.prediction import add_sentence_predictor_results
from hallucinations_kg.utils.langchain import setup_langchain_llm_cache
from hallucinations_kg.utils.langchain_config import get_llm_from_config, get_prompt_from_config
from hallucinations_kg.utils.logging import setup_logger

setup_logger()

setup_langchain_llm_cache()


@hydra.main(version_base="1.3", config_path=str(ROOT_PATH / "config"), config_name="correct_gpt")
def main(cfg: DictConfig) -> None:
    logger.info(cfg)
    load_dotenv()

    response_column = cfg.processed_dataset.original_dataset.response_column
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
            restrict=True,
        )
    )
    dataset = dataset.map(partial(postprocess_ents_rels, response_column=response_column))

    if cfg.processed_dataset.original_dataset.name == "wiki_bio":
        dataset["evaluation"] = dataset["evaluation"].add_column(
            "original_prompt", get_original_prompts(dataset["evaluation"]["wiki_bio_test_idx"])
        )

    thresholds = get_thresholds(dataset, sentences_column, samples_column)
    logger.info(f"{thresholds=}")
    dataset = add_incorrect_sentences_and_facts(dataset, thresholds)

    remove_columns = next(iter(dataset.column_names.values()))
    dataset = dataset.map(
        partial(
            add_corrected_sentences,
            cfg=cfg,
        ),
        num_proc=cfg.llm.num_proc,
        remove_columns=remove_columns,
    )

    dataset.save_to_disk(cfg.output_dir)


def get_original_prompts(wikibio_test_ids: list[int]) -> list[str]:
    ds = load_dataset("michaelauli/wiki_bio")
    names = [ds["test"][id_]["input_text"]["context"].strip() for id_ in wikibio_test_ids]
    prompts = [f"This is a Wikipedia passage about {name}: " for name in names]
    return prompts


def add_predictions(
    dataset: DatasetDict, sentences_column: str, samples_column: str
) -> DatasetDict:
    predictor_aggregation_functions: list[
        tuple[type[SentencePredictor] | type[FactPredictor], Callable[[list[float]], float] | None]
    ] = [
        (FactTextPromptPredictor, mean),
        (SelfCheckGPTPredictor, None),
    ]
    for name, ds in dataset.items():
        ds = ds.map(
            partial(
                add_sentence_predictor_results,
                predictor_aggregation_functions=predictor_aggregation_functions,
                sentences_column=sentences_column,
                samples_column=samples_column,
            )
        )
        dataset[name] = ds
    return dataset


def get_thresholds(
    dataset: DatasetDict, sentences_column: str, samples_column: str
) -> dict[str, float]:
    dataset = add_predictions(dataset, sentences_column, samples_column)
    df = dataset["evaluation"].to_pandas()
    sentence_score_columns = [col for col in df.columns if col.startswith("sent_score_")]
    assert len(sentence_score_columns) == 2
    sentence_df = df.explode(["gpt3_sentences", "binary_annotation"] + sentence_score_columns)

    result = {}
    for col in sentence_score_columns:
        y_pred = sentence_df[sentence_df[col].notna()][col]
        y_true = sentence_df[sentence_df[col].notna()].binary_annotation.to_numpy(dtype=float)
        prec, recall, thresholds = precision_recall_curve(y_true, y_pred)
        f1_scores = 2 * recall * prec / (recall + prec)
        best_threshold = thresholds[np.argmax(f1_scores)]
        if "SelfCheckGPT" in col:
            result["SelfCheckGPT"] = best_threshold
        elif "FactSelfCheck-Text" in col:
            result["FactSelfCheck-Text"] = best_threshold
        else:
            raise ValueError(f"Unknown column: {col}")
    return result


def add_incorrect_sentences_and_facts(
    dataset: DatasetDict, thresholds: dict[str, float]
) -> DatasetDict:
    sentence_score_columns = [
        col for col in dataset["evaluation"].column_names if col.startswith("sent_score_")
    ]
    [selfcheckgpt_column] = [col for col in sentence_score_columns if "SelfCheckGPT" in col]

    ds = dataset["evaluation"]

    incorrect_sentences = []
    for entry in ds:
        incorrect_sentences_ = []
        for i, (sentence, score) in enumerate(
            zip(entry["gpt3_sentences"], entry[selfcheckgpt_column])
        ):
            if score >= thresholds["SelfCheckGPT"]:
                incorrect_sentences_.append(f"{i+1}. {sentence}")
        incorrect_sentences.append("\n".join(incorrect_sentences_))
    dataset["evaluation"] = dataset["evaluation"].add_column(
        "incorrect_sentences", incorrect_sentences
    )

    answers = dataset["evaluation"]["fact_text_prompt_answers"]
    samples_graphs = dataset["evaluation"]["gpt3_text_samples_graph"]
    graphs = dataset["evaluation"]["gpt3_sentences_graph"]
    incorrect_facts = []
    predictor = FactTextPromptPredictor()
    for sentences_answers, sentences_graphs, example_samples_graphs in zip(
        answers, graphs, samples_graphs
    ):
        example_incorrect_facts = []
        for sentence_answers, sentence_facts in zip(sentences_answers, sentences_graphs):
            sentence_incorrect_facts = []
            for fact_answers, fact in zip(sentence_answers, sentence_facts):
                assert len(fact_answers) == 20 and len(fact) == 3
                prediction = predictor.predict(fact, example_samples_graphs, answers=fact_answers)
                if prediction >= thresholds["FactSelfCheck-Text"]:
                    sentence_incorrect_facts.append(f"({fact[0]}, {fact[1]}, {fact[2]})")
            example_incorrect_facts.append(f"[{' '.join(sentence_incorrect_facts)}]")
        incorrect_facts.append(
            "\n".join(f"{i+1}. {x}" for i, x in enumerate(example_incorrect_facts))
        )
    assert isinstance(incorrect_facts[0], str)
    dataset["evaluation"] = dataset["evaluation"].add_column("incorrect_facts", incorrect_facts)
    return dataset


def add_corrected_sentences(
    entry: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Any]:
    original_prompt = entry["original_prompt"]
    generated_sentences = entry["gpt3_sentences"]
    incorrect_sentences = entry["incorrect_sentences"]
    incorrect_facts = entry["incorrect_facts"]

    agent = get_agent(cfg)
    corrected_sentences = agent.correct(
        original_prompt=original_prompt,
        generated_sentences=generated_sentences,
        incorrect_sentences=incorrect_sentences,
        incorrect_facts=incorrect_facts,
    )
    entry[f"{cfg.prompt.name}_corrected_sentences"] = corrected_sentences
    return entry


def get_agent(cfg: DictConfig) -> CorrectorAgent:
    prompt = get_prompt_from_config(
        cfg.prompt,
        input_variables=["input", "knowledge_graph"],
        partial_variables={"examples": cfg.prompt.examples},
    )
    llm = get_llm_from_config(cfg.llm_correction)
    return CorrectorAgent(prompt=prompt, llm=llm)


if __name__ == "__main__":
    main()
