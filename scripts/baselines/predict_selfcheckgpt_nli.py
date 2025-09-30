from functools import partial
from pprint import pformat
from typing import Any

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from hallucinations_kg.data.utils import get_dataset
from hallucinations_kg.defaults import ROOT_PATH


@hydra.main(
    version_base="1.3",
    config_path=str(ROOT_PATH / "config"),
    config_name="predict_selfcheckgpt_nli",
)
def main(cfg: DictConfig) -> None:
    config = cfg
    logger.info(f"Config: {pformat(config)}")

    dataset = get_dataset(config.dataset)
    response_sentences_column = cfg.dataset.response_sentences_column
    samples_column = cfg.dataset.samples_column

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(config.model, device_map="auto")

    remove_columns = next(iter(dataset.column_names.values()))
    dataset = dataset.map(
        partial(
            predict,
            model=model,
            tokenizer=tokenizer,
            response_sentences_column=response_sentences_column,
            samples_column=samples_column,
        ),
        remove_columns=remove_columns,
    )
    dataset.save_to_disk(config.output_dir)


def predict(
    entry: dict[str, Any],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    response_sentences_column: str,
    samples_column: str,
) -> dict[str, Any]:
    scores: list[list[float]] = []
    for sentence in entry[response_sentences_column]:
        scores.append([])
        for sample in entry[samples_column]:
            inputs = tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=[(sentence, sample)],
                add_special_tokens=True,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=True,
                return_attention_mask=True,
            )
            inputs = inputs.to(model.device)
            logits = model(**inputs).logits  # neutral is already removed
            probs = torch.softmax(logits, dim=-1)
            prob_ = probs[0][1].item()  # prob(contradiction)
            scores[-1].append(prob_)
    return {"selfcheckgpt_nli_scores": scores}


if __name__ == "__main__":
    main()
