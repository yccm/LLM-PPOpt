"""Create smaller SFT/RL subsets with no overlapping inputs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Sequence


def load_jsonl(path: Path) -> list[dict]:
    entries: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def write_jsonl(path: Path, entries: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def map_by_input(entries: Iterable[dict]) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for entry in entries:
        key = entry.get("input")
        if key is None:
            raise ValueError("Entry missing 'input' field")
        mapping[key] = entry
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pick disjoint subsets from SFT and RL JSONL files."
    )
    parser.add_argument(
        "--sft-input",
        type=Path,
        default=Path("data/sft_train.jsonl"),
        help="Original SFT JSONL file",
    )
    parser.add_argument(
        "--rl-input",
        type=Path,
        default=Path("data/rl_states.jsonl"),
        help="Original RL JSONL file",
    )
    parser.add_argument(
        "--sft-output",
        type=Path,
        default=Path("data/sft_train_30.jsonl"),
        help="Path to write the filtered SFT file",
    )
    parser.add_argument(
        "--rl-output",
        type=Path,
        default=Path("data/rl_states_70.jsonl"),
        help="Path to write the filtered RL file",
    )
    parser.add_argument(
        "--sft-count",
        type=int,
        default=30,
        help="Number of SFT samples to keep",
    )
    parser.add_argument(
        "--rl-count",
        type=int,
        default=70,
        help="Number of RL samples to keep",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for sampling",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting existing output files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    sft_entries = load_jsonl(args.sft_input)
    rl_entries = load_jsonl(args.rl_input)

    sft_map = map_by_input(sft_entries)
    rl_map = map_by_input(rl_entries)

    if args.sft_count > len(sft_map):
        raise SystemExit("Not enough unique SFT inputs to satisfy --sft-count")
    if args.rl_count > len(rl_map):
        raise SystemExit("Not enough unique RL inputs to satisfy --rl-count")

    available_inputs = list(sft_map.keys())
    selected_sft = random.sample(available_inputs, args.sft_count)

    available_rl = [inp for inp in rl_map.keys() if inp not in selected_sft]
    if len(available_rl) < args.rl_count:
        raise SystemExit(
            "Not enough RL inputs remain after removing SFT samples; reduce --sft-count or use a larger RL set"
        )

    selected_rl = random.sample(available_rl, args.rl_count)

    sft_out = [sft_map[inp] for inp in selected_sft]
    rl_out = [rl_map[inp] for inp in selected_rl]

    for path in (args.sft_output, args.rl_output):
        if path.exists() and not args.force:
            raise SystemExit(f"Output file {path} already exists; use --force to overwrite")

    write_jsonl(args.sft_output, sft_out)
    write_jsonl(args.rl_output, rl_out)

    print("Created disjoint subsets:")
    print(f"  {len(sft_out)} SFT entries -> {args.sft_output}")
    print(f"  {len(rl_out)} RL entries -> {args.rl_output}")
    print("All inputs are unique across the two outputs.")


if __name__ == "__main__":
    main()
