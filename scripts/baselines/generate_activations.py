import gc
import os
import time
from pathlib import Path
from pprint import pformat
from typing import Any

import hydra
import torch
from datasets import Dataset, load_dataset
from loguru import logger
from omegaconf import DictConfig
from torch import Tensor
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

from hallucinations_kg.data.attention_storage import (
    ActivationStorage,
    AttentionAndLaplacianDiagsFeatureStorage,
)
from hallucinations_kg.data.utils import get_dataset
from hallucinations_kg.defaults import ROOT_PATH

NUM_PROC = int(os.getenv("NUM_PROC", 1))
NUM_SAVE_WORKERS = int(os.getenv("NUM_SAVE_WORKERS", 4))
MAX_MEMORY_GB = int(os.getenv("MAX_MEMORY_GB", 15))
MAX_MEMORY = {0: f"{MAX_MEMORY_GB}GIB"}

if torch.cuda.device_count() != 1:
    logger.warning("This script was tested only on a single CUDA device.")


@hydra.main(
    version_base="1.3",
    config_path=str(ROOT_PATH / "config"),
    config_name="attention_score_generate_activations",
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

    if config.llm.compile:
        # NOTE: using built-in hf compile results in wall of warnings
        model = torch.compile(model)  # type: ignore

    dataset.set_transform(SimpleEncoder(tokenizer))

    activation_storage = AttentionAndLaplacianDiagsFeatureStorage(
        storage_path=Path(config.output_dir),
        max_save_workers=NUM_SAVE_WORKERS,
        pad_token_id=tokenizer.pad_token_id,
        verbose=True,
    )

    logger.info(f"Using activation storage: {activation_storage}")

    inputs = predict_with_llm(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        activation_storage=activation_storage,
        batch_size=config.batch_size,
        num_proc=NUM_PROC,
    )

    activation_storage.flush()

    # Save inputs to a PyTorch file
    inputs_file = Path(config.output_dir) / "inputs.pt"
    torch.save(inputs, inputs_file)
    logger.info(f"Saved ({inputs_file.stat().st_size * 1e-9:0.1f}GB) inputs to {inputs_file}")


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


@torch.inference_mode()
def predict_with_llm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    activation_storage: ActivationStorage | None,
    batch_size: int,
    num_proc: int,
) -> list[dict[str, Any]]:
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
                    "input_ids": input_ids,
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

            if isinstance(outputs, CausalLMOutputWithPast):
                assert (
                    activation_storage is not None
                ), "activation_storage must be provided for GenerateDecoderOnlyOutput"
                token_masks = get_token_masks(input_ids, tokenizer)
                activation_storage.update(
                    outputs=outputs,
                    attention_mask=attention_mask,
                    special_token_mask=token_masks["special_token_mask"],
                    decoder_added_token_mask=token_masks["decoder_added_token_mask"],
                    input_length=input_length,
                    batch_idx=i,
                )

            stats = {
                "input_size": input_length,
                "throughput": f"{input_ids.numel() / duration:0.2f} tok/sec",
                "mean(#special_tokens)": f"{(1 - attention_mask).float().mean().item():0.3f}",
            }
            pbar.set_postfix(stats)

            del outputs, input_ids, attention_mask
            torch.cuda.empty_cache()
            gc.collect()

    return inputs


def get_token_masks(token_ids: Tensor, tokenizer: PreTrainedTokenizer) -> dict[str, Tensor]:
    special_token_masks = torch.tensor(
        [
            tokenizer.get_special_tokens_mask(
                seq_tok_ids,
                already_has_special_tokens=True,
            )
            for seq_tok_ids in token_ids
        ]
    )

    decoder_added_token_mask = torch.tensor(
        [
            [tok_id in tokenizer.added_tokens_decoder.keys() for tok_id in seq_token_ids]
            for seq_token_ids in token_ids
        ]
    )

    return {
        "special_token_mask": special_token_masks,
        "decoder_added_token_mask": decoder_added_token_mask,
    }


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


if __name__ == "__main__":
    main()
