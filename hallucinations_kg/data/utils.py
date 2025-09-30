from typing import Any

import datasets
from datasets import Dataset, DatasetDict, load_dataset
from omegaconf import DictConfig

LABEL_MAPPING = {
    "accurate": 0.0,
    "minor_inaccurate": 0.5,
    "major_inaccurate": 1.0,
}


def get_dataset(cfg: DictConfig) -> DatasetDict | Dataset:
    if cfg.name == "wiki_bio":
        dataset = get_wiki_bio_gpt3_hallucination(cfg)
    elif cfg.name == "fava-sampling":
        dataset = get_fava_sampling(cfg)
    else:
        raise ValueError(f"Unknown dataset: {cfg.name}")

    return dataset


def get_wiki_bio_gpt3_hallucination(cfg: DictConfig) -> DatasetDict | Dataset:
    if cfg.local:
        ds = datasets.load_from_disk(cfg.path)
    else:
        ds = load_dataset(cfg.path)

    if "binary_annotation" not in ds.column_names:

        def to_binary_annotation(entry: dict[str, Any]) -> dict[str, Any]:
            entry["binary_annotation"] = [LABEL_MAPPING[x] >= 0.5 for x in entry["annotation"]]
            return entry

        ds = ds.map(to_binary_annotation)

    return ds


def get_fava_sampling(cfg: DictConfig) -> DatasetDict | Dataset:
    if cfg.local:
        ds = datasets.load_from_disk(cfg.path)
    else:
        ds = load_dataset(cfg.path)

    if "binary_annotation" not in ds.column_names:
        ds = ds.rename_column(cfg.binary_annotations_column, "binary_annotation")

    return ds


def get_processed_dataset(cfg: DictConfig) -> DatasetDict | Dataset:
    ds = get_dataset(cfg.original_dataset)
    other_datasets = []
    for path_name in [
        "ents_rels_path",
        "response_graphs_path",
        "samples_graphs_path",
        "fact_prompt_answers_path",
        "fact_text_prompt_answers_path",
        "selfcheckgpt_answers_path",
        "evaluation_baseline_path",
        "evaluation_sentences_path",
        "evaluation_facts_path",
        "fact_level_annotation_path",
        "selfcheckgpt_nli_scores_path",
    ]:
        if path_name in cfg:
            other_datasets.append(datasets.load_from_disk(cfg[path_name]))
    for subset, ds_ in ds.items():
        to_concat = [ds_]
        for other_dataset in other_datasets:
            to_concat.append(other_dataset[subset])
        ds[subset] = datasets.concatenate_datasets(to_concat, axis=1)
    return ds
