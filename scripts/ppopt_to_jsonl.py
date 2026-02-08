#!/usr/bin/env python3
"""
Convert PPOpt.json (list of samples) into JSONL.

Default behavior preserves each sample object exactly as one JSONL line.
Optionally, you can output a messages-only schema for training frameworks
that expect {"messages":[{"role","content"}, ...]}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def iter_items(data: Any) -> Iterable[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def build_messages(item: Dict[str, Any], keep_message_metadata: bool) -> list[Dict[str, Any]]:
    source = item.get("full_conversation") or item.get("messages") or []
    messages: list[Dict[str, Any]] = []
    for msg in source:
        if not isinstance(msg, dict):
            continue
        record = {
            "role": msg.get("role", ""),
            "content": msg.get("content", ""),
        }
        if keep_message_metadata and msg.get("metadata"):
            record["metadata"] = msg.get("metadata")
        messages.append(record)
    return messages


def convert_item(
    item: Any,
    schema: str,
    keep_message_metadata: bool,
) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    if schema == "full":
        return item
    if schema == "messages":
        return {"messages": build_messages(item, keep_message_metadata)}
    raise ValueError(f"Unknown schema: {schema}")


def default_output_path(input_path: Path) -> Path:
    if input_path.suffix.lower() == ".json":
        return input_path.with_suffix(".jsonl")
    return input_path.with_name(f"{input_path.name}.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert PPOpt.json to JSONL (one JSON object per line)."
    )
    parser.add_argument(
        "--input",
        default="output/training_data/PPOpt.json",
        help="Input JSON file (default: output/training_data/PPOpt.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: <input>.jsonl)",
    )
    parser.add_argument(
        "--schema",
        choices=["full", "messages"],
        default="full",
        help="Output schema: full (preserve sample) or messages (only role/content).",
    )
    parser.add_argument(
        "--keep-message-metadata",
        action="store_true",
        help="Keep per-message metadata when using --schema messages.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)

    if not input_path.exists():
        print(f"[error] input not found: {input_path}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as out:
        for item in iter_items(data):
            total += 1
            record = convert_item(item, args.schema, args.keep_message_metadata)
            if record is None:
                skipped += 1
                continue
            out.write(json.dumps(record, ensure_ascii=False))
            out.write("\n")
            written += 1

    print(
        f"[ok] wrote {written} records to {output_path} "
        f"(total: {total}, skipped: {skipped}, schema: {args.schema})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
