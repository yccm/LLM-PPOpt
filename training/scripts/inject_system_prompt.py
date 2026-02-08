#!/usr/bin/env python
"""Inject the SFT system prompt into each record's input field in sft_train.jsonl.

Reads the system_prompt from config.yaml (sft section) and prepends it to each
record as a [SYSTEM] block, so the training data becomes self-contained.

Usage:
    python scripts/inject_system_prompt.py
    python scripts/inject_system_prompt.py --input data/sft_train.jsonl --output data/sft_train_with_sys.jsonl
    python scripts/inject_system_prompt.py --inplace  # overwrite the original file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def load_system_prompt(config_path: str) -> str:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prompt = cfg.get("sft", {}).get("system_prompt", "")
    if not prompt:
        print("Error: sft.system_prompt not found in config.yaml")
        sys.exit(1)
    return prompt.strip()


def main():
    parser = argparse.ArgumentParser(description="Inject system prompt into sft_train.jsonl")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    parser.add_argument("--input", "-i", default="data/sft_train.jsonl", help="Input JSONL file")
    parser.add_argument("--output", "-o", default=None, help="Output JSONL file (default: <input>_with_sys.jsonl)")
    parser.add_argument("--inplace", action="store_true", help="Overwrite the input file")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    config_path = root / args.config
    input_path = root / args.input

    if args.inplace:
        output_path = input_path
    elif args.output:
        output_path = root / args.output
    else:
        output_path = input_path.with_name(input_path.stem + "_with_sys.jsonl")

    system_prompt = load_system_prompt(str(config_path))

    print(f"System prompt ({len(system_prompt)} chars):")
    print(f"  {system_prompt[:80]}...")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    # Read all records
    with open(input_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    skipped = 0
    modified = 0

    for rec in records:
        inp = rec.get("input", "")
        # Skip if already has system prompt
        if inp.startswith("[SYSTEM]"):
            skipped += 1
            continue
        rec["input"] = f"[SYSTEM]\n{system_prompt}\n\n{inp}"
        modified += 1

    # Write
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone: {modified} modified, {skipped} skipped (already had system prompt)")
    print(f"Total: {len(records)} records -> {output_path}")


if __name__ == "__main__":
    main()
