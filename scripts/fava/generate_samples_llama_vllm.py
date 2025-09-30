from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import typer
import yaml
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

NUM_WORKERS = 8
NUM_GPUS = torch.cuda.device_count()

HF_DATASET_PATH = "https://huggingface.co/datasets/fava-uw/fava-data/raw/main/annotations.json"
SEEDS = [42, 7, 13, 101, 999, 2023, 555, 888, 314, 11, 17, 29, 37, 53, 67, 83, 97, 113, 127, 131]


@dataclass
class Config:
    save_dir: Path
    dataset_file: str = HF_DATASET_PATH
    model_version: str = "meta-llama/Llama-2-70b-chat-hf"
    random_seeds: list[int] = field(default_factory=lambda: SEEDS)
    num_samples: int = 10
    temperature: float = 1.0
    max_new_tokens: int = 2048

    def __post_init__(self) -> None:
        self.results_file.parent.mkdir(parents=True, exist_ok=True)

        assert self.num_samples == len(self.random_seeds)

        if self.results_file.exists() or self.config_file.exists():
            raise ValueError(
                f"Results file {self.results_file} or config file "
                f"{self.config_file} already exists, remove them first"
            )

    @property
    def results_file(self) -> Path:
        return self.save_dir / "llama" / "results.json"

    @property
    def config_file(self) -> Path:
        return self.save_dir / "llama" / "config.yaml"


@torch.inference_mode()
def main(
    save_dir: Path = typer.Option(Path("results"), help="Directory to save results"),
) -> None:
    config = Config(save_dir=save_dir)

    ds = load_dataset("json", data_files=config.dataset_file)["train"]
    ds = ds.add_column("index", range(len(ds)))
    ds = ds.map(format_to_conversation, batched=False, num_proc=NUM_WORKERS)
    ds = ds.filter(lambda x: x["model"] == "llama")
    conversations = ds["messages"]

    llm = LLM(
        model=config.model_version,
        dtype="bfloat16",
        tensor_parallel_size=NUM_GPUS,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        max_num_batched_tokens=4096,
        trust_remote_code=True,
    )

    results = []
    for seed in tqdm(config.random_seeds, desc="Seeds", leave=False):
        params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
            seed=seed,
        )
        responses = llm.chat(
            messages=conversations,
            sampling_params=params,
            add_generation_prompt=True,
        )
        for idx, res in zip(ds["index"], responses, strict=True):
            assert len(res.outputs) == 1
            completion = res.outputs[0]
            results.append(
                {
                    "idx": idx,
                    "seed": seed,
                    "response": completion.text,
                    "num_tokens": len(completion.token_ids),
                    "stop_reason": completion.finish_reason,
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_json(config.results_file, index=False, orient="records", indent=4)

    with open(config.config_file, "w") as f:
        yaml.dump(asdict(config), f)


def format_to_conversation(item: dict[str, Any]) -> dict[str, list[dict]]:
    return {"messages": [{"role": "user", "content": item["prompt"]}]}


if __name__ == "__main__":
    typer.run(main)
