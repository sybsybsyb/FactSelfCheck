import gc
import os
import time
from pathlib import Path
from pprint import pformat
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from hallucinations_kg.data.utils import get_dataset
from hallucinations_kg.defaults import ROOT_PATH

NUM_PROC = int(os.getenv("NUM_PROC", 1))
NUM_SAVE_WORKERS = int(os.getenv("NUM_SAVE_WORKERS", 4))
MAX_MEMORY_GB = int(os.getenv("MAX_MEMORY_GB", 15))
MAX_MEMORY = {0: f"{MAX_MEMORY_GB}GIB"}


@hydra.main(
    version_base="1.3",
    config_path=str(ROOT_PATH / "config"),
    config_name="predict_proba_metrics",
)
def main(cfg: DictConfig) -> None:
    config = cfg
    logger.info(f"Config: {pformat(config)}")

    raw_ds, dataset = prepare_dataset(
        dataset_config=config.dataset,
        split=config.split,
        return_raw=True,
    )

    model, tokenizer = get_llm(
        config.llm,
        max_memory=MAX_MEMORY,
        device_map="auto",  # loads model in a balanced mode on all available GPUs
    )

    if cfg.llm.compile:
        # NOTE: using built-in hf compile results in wall of warnings
        model = torch.compile(model)  # type: ignore

    dataset.set_transform(SimpleEncoder(tokenizer))

    inputs, neg_log_probs, entropies = predict_with_llm(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=config.batch_size,
        num_proc=NUM_PROC,
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_file = output_dir / "inputs.pt"
    neg_log_probs_file = output_dir / "neg_log_probs.pt"
    entropies_file = output_dir / "entropies.pt"
    torch.save(inputs, inputs_file)
    logger.info(f"Saved ({inputs_file.stat().st_size * 1e-9:0.1f}GB) inputs to {inputs_file}")
    torch.save(neg_log_probs, neg_log_probs_file)
    logger.info(
        f"Saved ({neg_log_probs_file.stat().st_size * 1e-9:0.1f}GB) neg_log_probs to {neg_log_probs_file}"
    )
    torch.save(entropies, entropies_file)
    logger.info(
        f"Saved ({entropies_file.stat().st_size * 1e-9:0.1f}GB) entropies to {entropies_file}"
    )


def prepare_dataset(
    dataset_config: DictConfig,
    split: str | None,
    return_raw: bool = False,
) -> Dataset | tuple[Dataset, Dataset]:
    dataset = get_dataset(dataset_config)[split]

    if dataset_config.name == "wiki_bio":
        formatted_ds = dataset.map(
            function=WikiBioFormatter(),
            batched=False,
            desc="Formatting dataset",
        )
    elif dataset_config.name == "fava-sampling":
        formatted_ds = dataset.map(
            function=FavaSamplingFormatter(),
            batched=False,
            desc="Formatting dataset",
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_config.name}")

    if return_raw:
        return dataset, formatted_ds
    else:
        return formatted_ds


@torch.inference_mode()
def predict_with_llm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int,
    num_proc: int,
) -> tuple[list[dict[str, Any]], list[list[float]], list[list[float]]]:
    model.eval()  # type: ignore
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_proc,
        pin_memory=(num_proc > 1),
        shuffle=False,
    )

    device = next(model.parameters()).device  # type: ignore

    inputs = []
    neg_log_probs = []
    entropies = []

    with tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Generating predictions",
    ) as pbar:
        for i, batch in pbar:
            # Clear CUDA cache before processing each batch
            torch.cuda.empty_cache()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            input_length = input_ids.size(1)

            decoded_ids = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]

            assert len(decoded_ids) == len(input_ids)

            inputs.append(
                {
                    "input_ids": input_ids.cpu(),
                    "attention_mask": attention_mask,
                    "input_length": input_length,
                    "decoded_ids": decoded_ids,
                }
            )

            start_time = time.time()
            outputs = model(  # type: ignore
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                output_attentions=True,
            )
            duration = time.time() - start_time

            neg_log_probs_, entropies_ = calc_metrics(outputs, input_ids)
            neg_log_probs.append(neg_log_probs_)
            entropies.append(entropies_)

            stats = {
                "input_size": input_length,
                "throughput": f"{input_ids.numel() / duration:0.2f} tok/sec",
                "mean(#special_tokens)": f"{(1 - attention_mask).float().mean().item():0.3f}",
            }
            pbar.set_postfix(stats)

            del outputs, input_ids, attention_mask
            torch.cuda.empty_cache()
            gc.collect()

    return inputs, neg_log_probs, entropies


def calc_metrics(
    outputs: CausalLMOutputWithPast, input_ids: torch.Tensor
) -> tuple[list[float], list[float]]:
    logits = outputs.logits

    # Get probabilities (softmax over last dimension)
    assert logits is not None
    probs = F.softmax(logits, dim=-1)

    # Calculate token-level probabilities
    token_probs = []
    for i in range(1, input_ids.size(1)):
        token_id = input_ids[0, i]
        prob = probs[0, i - 1, token_id].item()
        token_probs.append(prob)

    # Calculate negatinp log probabilities
    neg_log_probs = [-np.log(p) for p in token_probs]

    # Calculate entropy for each position
    entropies = []
    for i in range(1, input_ids.size(1)):
        # Get probability distribution for current position
        prob_dist = probs[0, i - 1]
        # Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(
            prob_dist * torch.log(prob_dist + 1e-10)
        )  # Add small epsilon to avoid log(0)
        entropies.append(entropy.item())
    return neg_log_probs, entropies


def get_llm(llm_config: DictConfig, **kwargs: Any) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    if "Llama-3" in llm_config.full_name:
        return get_llama_3(llm_config, **kwargs)
    else:
        raise ValueError(f"Model {llm_config.full_name} not supported.")


def get_llama_3(
    llm_config: DictConfig, **kwargs: Any
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model = get_model(llm_config, **kwargs)
    tokenizer = get_tokenizer(llm_config)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def get_model(
    llm_config: DictConfig,
    **kwargs: Any,
) -> PreTrainedModel:
    if llm_config.quantization is not None:
        kwargs["quantization_config"] = BitsAndBytesConfig(**llm_config.quantization)

    model = AutoModelForCausalLM.from_pretrained(
        llm_config.full_name,
        torch_dtype=llm_config.torch_dtype,
        attn_implementation=llm_config.attn_implementation,
        **kwargs,
    )
    return model


def get_tokenizer(llm_config: DictConfig) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(llm_config.tokenizer_name)
    tokenizer.padding_side = llm_config.tokenizer_padding_side
    return tokenizer


class SimpleEncoder:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: dict[str, list[Any]]) -> dict[str, torch.Tensor]:
        try:
            final_input = self.tokenizer.apply_chat_template(
                batch["messages"],
                add_generation_prompt=False,
                tokenize=False,
            )
        except ValueError:
            assert all(
                len(item) == 1 for item in batch["messages"]
            ), f"Expected single message in batch, got {batch['messages']}"
            final_input = [item[0]["content"].rstrip() for item in batch["messages"]]

        return self.tokenizer(final_input, return_tensors="pt", padding="longest", truncation=False)


class WikiBioFormatter:
    def __init__(self) -> None:
        self.ds = load_dataset("michaelauli/wiki_bio")

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        id_ = item["wiki_bio_test_idx"]
        name = self.ds["test"][id_]["input_text"]["context"].strip()
        prompt = f"This is a Wikipedia passage about {name}:"
        messages = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role": "assistant",
                    "content": " ".join(item["gpt3_sentences"]),
                },
            ]
        }

        return messages


class FavaSamplingFormatter:
    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        messages = {
            "messages": [
                {
                    "role": "user",
                    "content": item["prompt"],
                },
                {
                    "role": "assistant",
                    "content": " ".join(item["sentences"]),
                },
            ]
        }
        return messages


if __name__ == "__main__":
    main()
