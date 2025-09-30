import asyncio
import pprint
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import aiohttp
import openai
import pandas as pd
import typer
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain.schema import BaseMessage, HumanMessage
from langchain_community.cache import SQLiteCache
from langchain_openai import ChatOpenAI
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

HF_DATASET_PATH = "https://huggingface.co/datasets/fava-uw/fava-data/raw/main/annotations.json"
SEEDS = [42, 7, 13, 101, 999, 2023, 555, 888, 314, 11, 17, 29, 37, 53, 67, 83, 97, 113, 127, 131]


@dataclass
class Config:
    save_dir: Path
    dataset_file: str = HF_DATASET_PATH
    model_version: str = "gpt-3.5-turbo-1106"
    random_seeds: List[int] = field(default_factory=lambda: SEEDS)
    num_samples: int = 10
    temperature: float = 1.0
    max_concurrent_calls: int = 20

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
        return self.save_dir / "chatgpt" / "results.json"

    @property
    def config_file(self) -> Path:
        return self.save_dir / "chatgpt" / "config.yaml"


def main(
    save_dir: Path = typer.Option(Path("results"), help="Directory to save results"),
) -> None:
    config = Config(save_dir=save_dir)
    asyncio.run(query_llm(config))


async def query_llm(config: Config) -> None:
    ds = load_dataset("json", data_files=config.dataset_file)["train"].to_pandas()
    logger.info(f"Loaded {len(ds)} rows")
    ds_info = {
        "llm_stats": ds["model"].value_counts().to_dict(),
        "dataset_stats": ds["dataset"].value_counts().to_dict(),
    }
    logger.info(f"Dataset info:\n{pprint.pformat(ds_info)}")
    ds = ds[ds["model"] == "chatgpt"]

    logger.info(f"Filtered to {len(ds)} chatgpt prompts")
    logger.info(f"Generating samples with {len(config.random_seeds)} random seeds")

    llm = LLMInterface(config)

    tasks = []
    for idx, row in ds.iterrows():
        for seed in config.random_seeds:
            tasks.append(llm(row["prompt"], idx, seed))

    results = await tqdm_asyncio.gather(*tasks)

    results_df = pd.DataFrame(results)
    results_df.to_json(config.results_file, index=False, orient="records", indent=4)

    with open(config.config_file, "w") as f:
        yaml.dump(asdict(config), f)


class LLMInterface:
    def __init__(self, config: Config) -> None:
        self.llm = ChatOpenAI(
            model=config.model_version,
            temperature=config.temperature,
        )
        self.semaphore = asyncio.Semaphore(config.max_concurrent_calls)

    async def __call__(self, prompt: str, idx: int, seed: int) -> dict:
        async with self.semaphore:
            messages = [HumanMessage(content=prompt)]
            response = await self._call_llm(messages, seed)
            return {
                "idx": idx,
                "seed": seed,
                "response": response.content,
                "system_fingerprint": response.response_metadata["system_fingerprint"],
            }

    @retry(
        retry=retry_if_exception_type(
            (
                aiohttp.ClientError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.RateLimitError,
                openai.APIError,
                asyncio.TimeoutError,
            )
        ),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _call_llm(self, messages: list[HumanMessage], seed: int) -> BaseMessage:
        return await self.llm.ainvoke(messages, seed=seed)


if __name__ == "__main__":
    typer.run(main)
