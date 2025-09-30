import json
from pathlib import Path
from statistics import mean
from typing import Callable

import hydra
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, disable_caching
from omegaconf import DictConfig
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from hallucinations_kg.data.utils import get_dataset
from hallucinations_kg.defaults import ROOT_PATH
from hallucinations_kg.metrics import auc_pr
from hallucinations_kg.models.baselines.proba_score import ProbaScorePredictor

disable_caching()


@hydra.main(
    version_base="1.3",
    config_path=str(ROOT_PATH / "config"),
    config_name="proba_score_eval",
)
def main(cfg: DictConfig) -> None:
    inputs = torch.load(Path(cfg.input_dir) / "inputs.pt")
    neg_log_probs = torch.load(Path(cfg.input_dir) / "neg_log_probs.pt", weights_only=False)
    entropies = torch.load(Path(cfg.input_dir) / "entropies.pt", weights_only=False)

    dataset = get_dataset(cfg.dataset)[cfg.dataset.test_split_name]
    sentences_col = cfg.dataset.response_sentences_column

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm.tokenizer_name)

    mean_neg_log_probs = get_score(tokenizer, dataset, sentences_col, inputs, neg_log_probs, mean)
    max_neg_log_probs = get_score(tokenizer, dataset, sentences_col, inputs, neg_log_probs, max)

    mean_entropies = get_score(tokenizer, dataset, sentences_col, inputs, entropies, mean)
    max_entropies = get_score(tokenizer, dataset, sentences_col, inputs, entropies, max)

    dataset = dataset.add_column("sent_score_mean_neg_log_probs", mean_neg_log_probs)
    dataset = dataset.add_column("sent_score_max_neg_log_probs", max_neg_log_probs)
    dataset = dataset.add_column("sent_score_mean_entropies", mean_entropies)
    dataset = dataset.add_column("sent_score_max_entropies", max_entropies)

    sentence_results = get_sentence_results(dataset, sentences_col)

    with open(cfg.results_file, "w") as f:
        json.dump(sentence_results.to_dict(orient="records"), f, indent=4)


def get_score(
    tokenizer: Tokenizer,
    dataset: Dataset,
    sentences_col: str,
    inputs: list[dict],
    proba_metric: list[list[float]],
    agg: Callable[[torch.Tensor | list[float]], float],
) -> list[list[float]]:
    predictor = ProbaScorePredictor(tokenizer, agg)
    proba_scores = []
    for sentences, inputs_, proba_metric_ in zip(
        dataset[sentences_col], inputs, proba_metric, strict=True
    ):
        assert len(inputs_["input_ids"]) == 1
        input_ids_ = inputs_["input_ids"][0].cpu()
        proba_scores_ = predictor.predict_sentences(sentences, input_ids_, proba_metric_)

        proba_scores.append(proba_scores_)

    return proba_scores


def get_sentence_results(dataset: Dataset, sentences_column: str) -> pd.DataFrame:
    proba_score_columns = [col for col in dataset.column_names if "sent_score_" in col]
    columns_to_explode = [sentences_column, "binary_annotation"] + proba_score_columns

    df = dataset.to_pandas()
    sentence_df = df.explode(columns_to_explode)

    results = []
    for col in proba_score_columns:
        y_pred = sentence_df[col]
        y_true = sentence_df.binary_annotation.to_numpy(dtype=float)

        auc_pr_hallucination = auc_pr(y_true, y_pred) * 100
        y_true_factual = 1 - y_true
        y_pred_factual = 1 - np.array(y_pred)
        auc_pr_factual = auc_pr(y_true_factual, y_pred_factual) * 100

        results.append(
            {
                "col": col,
                "auc_hallucination": auc_pr_hallucination,
                "auc_factuality": auc_pr_factual,
                "auc_mean": mean([auc_pr_hallucination, auc_pr_factual]),
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    main()
