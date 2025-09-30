from pathlib import Path

import pandas as pd
import typer
from datasets import load_dataset

HF_DATASET_PATH = "https://huggingface.co/datasets/fava-uw/fava-data/raw/main/annotations.json"


def main(
    source_files: list[Path] = typer.Argument(..., help="Source files to merge"),
    output_file: Path = typer.Option(..., help="Output file"),
) -> None:
    dataset = load_dataset(
        "json",
        data_files=HF_DATASET_PATH,
    )["train"].to_pandas()

    model_results = []
    for source_file in source_files:
        df = pd.read_json(source_file)
        df = df.groupby("idx").agg(list)
        model_results.append(df)

    all_results = pd.concat(model_results, axis=0)
    dataset = dataset.join(all_results, how="left")

    dataset = dataset.rename(
        columns={
            "response": "text_samples",
            "seed": "generation_seeds",
            "system_fingerprint": "openai_system_fingerprint",
        }
    )

    assert dataset["text_samples"].notna().all()
    assert dataset["generation_seeds"].notna().all()
    assert dataset[dataset["openai_system_fingerprint"].isna()]["model"].unique().item() == "llama"
    assert (
        dataset[dataset["openai_system_fingerprint"].notna()]["model"].unique().item() == "chatgpt"
    )

    dataset.to_json(output_file, index=False, orient="records", indent=4)


if __name__ == "__main__":
    typer.run(main)
