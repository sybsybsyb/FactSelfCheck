import json
from pathlib import Path
from statistics import mean

import hydra
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, disable_caching
from omegaconf import DictConfig
from transformers import AutoTokenizer

from hallucinations_kg.data.utils import get_dataset
from hallucinations_kg.defaults import ROOT_PATH
from hallucinations_kg.metrics import auc_pr
from hallucinations_kg.models.baselines.attention_score import AttnScorePredictor

disable_caching()


@hydra.main(
    version_base="1.3",
    config_path=str(ROOT_PATH / "config"),
    config_name="attention_score_eval",
)
def main(cfg: DictConfig) -> None:
    attn_diags = torch.load(Path(cfg.input_dir) / "attn_diags.pt")
    inputs = torch.load(Path(cfg.input_dir) / "inputs.pt")

    dataset = get_dataset(cfg.dataset)[cfg.dataset.test_split_name]
    sentences_col = cfg.dataset.response_sentences_column

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm.tokenizer_name)
    predictor = AttnScorePredictor(tokenizer)

    attn_score_absolute = []
    attn_score_relative = []

    for sentences, attn_diags_, inputs_ in zip(
        dataset[sentences_col], attn_diags, inputs, strict=True
    ):
        assert len(inputs_["input_ids"]) == 1
        input_ids_ = inputs_["input_ids"][0].cpu()
        attn_score_absolute_, attn_score_relative_ = predictor.predict_sentences(
            sentences, input_ids_, attn_diags_
        )

        attn_score_absolute.append(attn_score_absolute_)
        attn_score_relative.append(attn_score_relative_)

    dataset = dataset.add_column("attn_score_absolute", attn_score_absolute)
    dataset = dataset.add_column("attn_score_relative", attn_score_relative)

    dataset = dataset.map(add_attn_score_per_layer)

    sentence_results = get_sentence_results(dataset, sentences_col)
    sentence_best_relative_results = (
        sentence_results[sentence_results["version"] == "relative"]
        .sort_values(by="auc_hallucination", ascending=False)
        .iloc[0]
    )
    sentence_best_absolute_results = (
        sentence_results[sentence_results["version"] == "absolute"]
        .sort_values(by="auc_hallucination", ascending=False)
        .iloc[0]
    )

    to_save = {
        "sentence_best_relative_results": sentence_best_relative_results.to_dict(),
        "sentence_best_absolute_results": sentence_best_absolute_results.to_dict(),
    }
    with open(cfg.results_file, "w") as f:
        json.dump(to_save, f, indent=4)


def get_sentence_results(dataset: Dataset, sentences_column: str) -> pd.DataFrame:
    attn_score_columns = [col for col in dataset.column_names if "sent_score_" in col]
    columns_to_explode = [sentences_column, "binary_annotation"] + attn_score_columns

    df = dataset.to_pandas()
    sentence_df = df.explode(columns_to_explode)

    results = []
    for col in attn_score_columns:
        y_pred = sentence_df[col]
        y_true = sentence_df.binary_annotation.to_numpy(dtype=float)

        auc_pr_hallucination = auc_pr(y_true, y_pred) * 100
        y_true_factual = 1 - y_true
        y_pred_factual = 1 - np.array(y_pred)

        auc_pr_factual = auc_pr(y_true_factual, y_pred_factual) * 100

        layer = int(col.split("_")[-1]) if "all" not in col else None

        version = "relative" if "relative" in col else "absolute"
        results.append(
            {
                "col": col,
                "layer": layer,
                "version": version,
                "auc_hallucination": auc_pr_hallucination,
                "auc_factuality": auc_pr_factual,
                "auc_mean": mean([auc_pr_hallucination, auc_pr_factual]),
            }
        )

    return pd.DataFrame(results)


def add_attn_score_per_layer(entry: dict) -> dict:
    result = {}
    for attn_score in ["attn_score_absolute", "attn_score_relative"]:
        for i in range(len(entry[attn_score][0])):
            result[f"sent_score_{attn_score}_layer_{i}"] = [
                attn_sentence[i] for attn_sentence in entry[attn_score]
            ]
        result[f"sent_score_{attn_score}_layer_all"] = [
            sum(attn_sentence) for attn_sentence in entry[attn_score]
        ]
    return result


if __name__ == "__main__":
    main()
