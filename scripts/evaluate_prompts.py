"""
evaluate_prompts.py

This variant integrates with the repository's FactSelfCheck evaluation pipeline.
It expects the input prompts CSV to contain fact triples as columns: `head`, `relation`, `tail`.
It will construct a Triple from those columns, call the project's FactTextPromptAgent to get
LLM-based answers for provided sample contexts and then use the FactTextPromptPredictor to
compute a score. Only the framework's results (score & details) are written to the output CSV —
the raw LLM response is not saved.

Input CSV requirements (per-row):
- head, relation, tail : the fact triple components (strings)
- samples (optional): JSON array (string) of sample context texts to use for the prompt-style evaluation
  e.g. '["context1","context2"]'

If `samples` is missing or empty for a row, the script will skip scoring and write an appropriate note.

Configuration is loaded from .env / environment variables (see docs/EVALUATE_PROMPTS.md).
New env var (optional): FACT_CFG — path to the Hydra config to use to construct the FactTextPromptAgent
(default: config/predict_text_prompt.yaml)

Run:
  python scripts/evaluate_prompts.py

"""
import argparse
import csv
import datetime
import json
import os
import signal
import sys
import threading
from typing import Optional

import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

# imports from the project
from hallucinations_kg.models.predictor import Triple, Graph
from hallucinations_kg.models.prompt_predictor import FactTextPromptPredictor
from scripts.fact_text_prompt_answers import get_agent

# try to load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

stop_requested = False
stop_lock = threading.Lock()


def request_stop(signum, frame):
    global stop_requested
    with stop_lock:
        stop_requested = True


signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts_csv", help="CSV file with prompts")
    p.add_argument("--output_csv", default=None, help="CSV file to append results to")
    p.add_argument("--fact_cfg", default=None, help="Path to config YAML used to build FactTextPromptAgent")
    return p.parse_args()


def json_safe(obj):
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return ""


def load_prompts(path: str, delimiter: str = ","):
    df = pd.read_csv(path, delimiter=delimiter, dtype=str, keep_default_na=False)
    # require triple columns
    required = ["head", "relation", "tail"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in {path}. Columns: {df.columns.tolist()}")
    df = df.reset_index().rename(columns={"index": "prompt_index"})
    return df


def write_header_if_needed(path: str, header: list):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def append_row(path: str, row: list):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main():
    args = parse_args()

    prompts_csv = args.prompts_csv or os.getenv("PROMPTS_CSV") or "prompts.csv"
    output_csv = args.output_csv or os.getenv("OUTPUT_CSV") or "fs_results.csv"
    delimiter = os.getenv("DELIMITER") or ","
    fact_cfg_path = args.fact_cfg or os.getenv("FACT_CFG") or "config/predict_text_prompt.yaml"

    # load prompts
    try:
        df = load_prompts(prompts_csv, delimiter)
    except Exception as e:
        print(f"Failed to load prompts CSV '{prompts_csv}': {e}")
        sys.exit(1)

    # load config used to construct the agent
    try:
        cfg = OmegaConf.load(fact_cfg_path)
    except Exception as e:
        print(f"Failed to load config '{fact_cfg_path}': {e}")
        sys.exit(1)

    # construct FactTextPromptAgent using project's helper
    try:
        agent = get_agent(cfg)
    except Exception as e:
        print(f"Failed to construct FactTextPromptAgent from config {fact_cfg_path}: {e}")
        raise

    predictor = FactTextPromptPredictor()

    output_cols = [
        "prompt_index",
        "head",
        "relation",
        "tail",
        "fs_score",
        "fs_details",
        "status",
        "error",
        "ts",
    ]

    write_header_if_needed(output_csv, output_cols)

    processed = set()
    if os.path.exists(output_csv):
        try:
            already = pd.read_csv(output_csv, dtype=str)
            if "prompt_index" in already.columns:
                processed = set(int(x) for x in already["prompt_index"].dropna().unique())
        except Exception:
            processed = set()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="prompts"):
        with stop_lock:
            if stop_requested:
                print("Stop requested. Exiting loop and saving progress.")
                break

        idx = int(row["prompt_index"])
        if idx in processed:
            continue

        head = row["head"]
        relation = row["relation"]
        tail = row["tail"]

        # samples: optional JSON array in 'samples' column
        samples = []
        if "samples" in row and row["samples"]:
            try:
                samples = json.loads(row["samples"]) if isinstance(row["samples"], str) else row["samples"]
                if not isinstance(samples, list):
                    samples = [str(samples)]
            except Exception:
                samples = [str(row["samples"])]

        # prepare triple
        try:
            triple = Triple(head=head, relation=relation, tail=tail)
        except Exception as e:
            out_row = [idx, head, relation, tail, "", "", "error", f"Invalid triple: {e}", datetime.datetime.utcnow().isoformat()]
            append_row(output_csv, out_row)
            processed.add(idx)
            continue

        # if no samples provided, we cannot score with predictor - write note
        if not samples:
            out_row = [
                idx,
                head,
                relation,
                tail,
                "",
                json_safe({"error": "no_samples_provided"}),
                "skipped",
                "no samples provided",
                datetime.datetime.utcnow().isoformat(),
            ]
            append_row(output_csv, out_row)
            processed.add(idx)
            continue

        # call agent to get answers for the triple given samples
        try:
            answers = agent.get_answers(triple, samples)
        except Exception as e:
            out_row = [
                idx,
                head,
                relation,
                tail,
                "",
                json_safe({"error": repr(e)}),
                "error",
                repr(e),
                datetime.datetime.utcnow().isoformat(),
            ]
            append_row(output_csv, out_row)
            processed.add(idx)
            continue

        # predictor expects samples_graphs list of Graph with same length as answers; we can pass empty Graphs
        try:
            samples_graphs = [Graph(triples=[]) for _ in answers]
            score = predictor.predict(triple, samples_graphs, answers=answers)
            details = {"answers": answers}
            out_row = [
                idx,
                head,
                relation,
                tail,
                score,
                json_safe(details),
                "ok",
                "",
                datetime.datetime.utcnow().isoformat(),
            ]
        except Exception as e:
            out_row = [
                idx,
                head,
                relation,
                tail,
                "",
                json_safe({"error": repr(e), "answers": answers}),
                "error",
                repr(e),
                datetime.datetime.utcnow().isoformat(),
            ]

        append_row(output_csv, out_row)
        processed.add(idx)

    print(f"Finished. Results appended to {output_csv}")


if __name__ == "__main__":
    main()
