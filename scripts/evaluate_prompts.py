"""
evaluate_prompts.py

This version loads configuration from a .env file (if present) and from environment variables.
You can still override any setting via command-line arguments, but the script now runs with
no CLI parameters required as long as the required settings are present in .env or the environment.

Example .env (place in repository root or current working dir):

PROMPTS_CSV=prompts.csv
PROMPT_COL=prompt
OUTPUT_CSV=results.csv
DELIMITER=,
ENDPOINT=https://api.openai.com/v1/chat/completions
API_KEY=sk-...
MODEL=gpt-4o
CHECKPOINT_INTERVAL=5
TEMPERATURE=0.0
MAX_TOKENS=
MAX_RETRIES=3
BACKOFF=1.0
SYSTEM_PROMPT=
START_INDEX=

Dependencies (add): python-dotenv
pip install python-dotenv

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
import time
from typing import Optional

import pandas as pd
from tqdm import tqdm

from scripts.llm_client import LLMClient

# try to load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # no-op if dotenv not installed; env vars may still be set in the environment
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
    p.add_argument("--prompt_col", default=None, help="Column name that contains the prompt text")
    p.add_argument("--output_csv", default=None, help="CSV file to append results to")
    p.add_argument("--delimiter", default=None, help="CSV delimiter for prompts file")
    p.add_argument("--endpoint", help="LLM endpoint URL (e.g. https://api.openai.com/v1/chat/completions)")
    p.add_argument("--api_key", default=None, help="API key for the LLM endpoint")
    p.add_argument("--model", default=None, help="Model name to send in payload")
    p.add_argument("--checkpoint_interval", type=int, default=None, help="Save to CSV every N results")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--max_retries", type=int, default=None)
    p.add_argument("--backoff", type=float, default=None)
    p.add_argument("--system_prompt", default=None)
    p.add_argument("--start_index", type=int, default=None, help="Force start index (overrides resume)")
    return p.parse_args()


def env_get_int(name: str, default: Optional[int] = None) -> Optional[int]:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def env_get_float(name: str, default: Optional[float] = None) -> Optional[float]:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def env_get_str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v


def load_prompts(path: str, prompt_col: str, delimiter: str = ","):
    df = pd.read_csv(path, delimiter=delimiter, dtype=str, keep_default_na=False)
    if prompt_col not in df.columns:
        raise ValueError(f"Column {prompt_col} not found in {path}. Columns: {df.columns.tolist()}")
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


def json_safe(obj):
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return ""


def main():
    # load configuration from environment (possibly from .env) and CLI args (CLI overrides env)
    args = parse_args()

    # environment values (strings)
    env_prompts_csv = env_get_str("PROMPTS_CSV")
    env_prompt_col = env_get_str("PROMPT_COL")
    env_output_csv = env_get_str("OUTPUT_CSV")
    env_delimiter = env_get_str("DELIMITER")
    env_endpoint = env_get_str("ENDPOINT")
    # accept multiple common names for API key
    env_api_key = env_get_str("API_KEY") or env_get_str("LLM_API_KEY") or env_get_str("OPENAI_API_KEY")
    env_model = env_get_str("MODEL")
    env_checkpoint_interval = env_get_int("CHECKPOINT_INTERVAL")
    env_temperature = env_get_float("TEMPERATURE")
    env_max_tokens = env_get_int("MAX_TOKENS")
    env_max_retries = env_get_int("MAX_RETRIES")
    env_backoff = env_get_float("BACKOFF")
    env_system_prompt = env_get_str("SYSTEM_PROMPT")
    env_start_index = env_get_int("START_INDEX")

    # final values: CLI arg if provided else env else default
    prompts_csv = args.prompts_csv or env_prompts_csv or "prompts.csv"
    prompt_col = args.prompt_col or env_prompt_col or "prompt"
    output_csv = args.output_csv or env_output_csv or "prompt_results.csv"
    delimiter = args.delimiter or env_delimiter or ","
    endpoint = args.endpoint or env_endpoint
    api_key = args.api_key or env_api_key
    model = args.model or env_model
    checkpoint_interval = args.checkpoint_interval if args.checkpoint_interval is not None else (env_checkpoint_interval if env_checkpoint_interval is not None else 10)
    temperature = args.temperature if args.temperature is not None else (env_temperature if env_temperature is not None else 0.0)
    max_tokens = args.max_tokens if args.max_tokens is not None else env_max_tokens
    max_retries = args.max_retries if args.max_retries is not None else (env_max_retries if env_max_retries is not None else 3)
    backoff = args.backoff if args.backoff is not None else (env_backoff if env_backoff is not None else 1.0)
    system_prompt = args.system_prompt if args.system_prompt is not None else env_system_prompt
    start_idx = args.start_index if args.start_index is not None else env_start_index if env_start_index is not None else 0

    # endpoint is required; if not provided, show helpful message and exit
    if not endpoint:
        print("ERROR: No endpoint configured. Please set ENDPOINT in a .env file or pass --endpoint on the CLI.")
        sys.exit(2)

    # load prompts
    try:
        df = load_prompts(prompts_csv, prompt_col, delimiter)
    except Exception as e:
        print(f"Failed to load prompts CSV '{prompts_csv}': {e}")
        sys.exit(1)

    client = LLMClient(
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        timeout=60,
        max_retries=max_retries,
        backoff_factor=backoff,
    )

    output_cols = [
        "prompt_index",
        "prompt",
        "response",
        "status",
        "error",
        "usage",
        "model",
        "endpoint",
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

    rows_since_checkpoint = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="prompts"):
        with stop_lock:
            if stop_requested:
                print("Stop requested. Exiting loop and saving progress.")
                break
        idx = int(row["prompt_index"])
        if idx < start_idx:
            continue
        if idx in processed:
            continue

        prompt_text = row[prompt_col]

        try:
            res = client.generate(
                prompt_text,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = res.get("text")
            usage = res.get("usage")
            status = "ok"
            error = ""
        except Exception as e:
            text = ""
            usage = ""
            status = "error"
            error = repr(e)

        now = datetime.datetime.utcnow().isoformat()
        out_row = [
            idx,
            prompt_text,
            text,
            status,
            error,
            json_safe(usage),
            model or "",
            endpoint,
            now,
        ]

        append_row(output_csv, out_row)
        processed.add(idx)

        rows_since_checkpoint += 1
        if rows_since_checkpoint >= checkpoint_interval:
            rows_since_checkpoint = 0
            # additional checkpointing logic can be added here

    print(f"Finished. Results appended to {output_csv}")


if __name__ == "__main__":
    main()
