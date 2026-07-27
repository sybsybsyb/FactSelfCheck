"""
evaluate_prompts.py

Usage example:
python scripts/evaluate_prompts.py \
  --prompts_csv prompts.csv \
  --prompt_col prompt \
  --output_csv results.csv \
  --endpoint https://api.openai.com/v1/chat/completions \
  --api_key sk-... \
  --model gpt-4o \
  --checkpoint_interval 5

Features:
- Read prompts from CSV (supports custom column name)
- Resume if output CSV exists (skips already processed prompts)
- Append results incrementally to output CSV to avoid data loss
- Signal handler to save progress on SIGINT/SIGTERM
- Configurable endpoint, key and model
"""
import argparse
import csv
import datetime
import os
import signal
import sys
import threading
import time
from typing import Optional

import pandas as pd
from tqdm import tqdm

from scripts.llm_client import LLMClient


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
    p.add_argument("--prompts_csv", required=True, help="CSV file with prompts")
    p.add_argument("--prompt_col", default="prompt", help="Column name that contains the prompt text")
    p.add_argument("--output_csv", default="prompt_results.csv", help="CSV file to append results to")
    p.add_argument("--delimiter", default=",", help="CSV delimiter for prompts file")
    p.add_argument("--endpoint", required=True, help="LLM endpoint URL (e.g. https://api.openai.com/v1/chat/completions)")
    p.add_argument("--api_key", default=None, help="API key for the LLM endpoint")
    p.add_argument("--model", default=None, help="Model name to send in payload")
    p.add_argument("--checkpoint_interval", type=int, default=10, help="Save to CSV every N results")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--backoff", type=float, default=1.0)
    p.add_argument("--system_prompt", default=None)
    p.add_argument("--start_index", type=int, default=None, help="Force start index (overrides resume)")
    return p.parse_args()


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


def main():
    args = parse_args()

    df = load_prompts(args.prompts_csv, args.prompt_col, args.delimiter)

    client = LLMClient(
        endpoint=args.endpoint,
        api_key=args.api_key,
        model=args.model,
        timeout=60,
        max_retries=args.max_retries,
        backoff_factor=args.backoff,
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

    write_header_if_needed(args.output_csv, output_cols)

    processed = set()
    if os.path.exists(args.output_csv):
        try:
            already = pd.read_csv(args.output_csv, dtype=str)
            if "prompt_index" in already.columns:
                processed = set(int(x) for x in already["prompt_index"].dropna().unique())
        except Exception:
            # best-effort; if we can't parse, start from scratch
            processed = set()

    start_idx = args.start_index if args.start_index is not None else 0

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

        prompt_text = row[args.prompt_col]

        try:
            res = client.generate(
                prompt_text,
                system_prompt=args.system_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
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
            args.model or "",
            args.endpoint,
            now,
        ]

        append_row(args.output_csv, out_row)
        processed.add(idx)

        rows_since_checkpoint += 1
        # periodic flush (already appended), additional checkpointing logic could be added
        if rows_since_checkpoint >= args.checkpoint_interval:
            rows_since_checkpoint = 0
            # small flush/sync to disk handled by append

    print(f"Finished. Results appended to {args.output_csv}")


def json_safe(obj):
    try:
        import json

        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return ""


if __name__ == "__main__":
    main()
