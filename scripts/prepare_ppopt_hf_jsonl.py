#!/usr/bin/env python3
"""Convert PPOpt.json into a richer Hugging Face friendly JSONL schema.

Output schema per line:
{
  "id": "...",
  "persona_id": "...",
  "persona": {...},
  "original_query": "...",
  "initial_query": "...",
  "full_conversation": [{"role": "...", "content": "..."}],
  "role_utterances": {
    "user": ["..."],
    "assistant": ["..."]
  },
  "num_turns": 2,
  "distractor_type": "semantic",
  "conversation_ended": true
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def normalize_persona(raw_persona: Any) -> dict[str, str]:
    if not isinstance(raw_persona, dict):
        return {}

    persona: dict[str, str] = {}
    for key, value in raw_persona.items():
        key_text = as_text(key)
        if isinstance(value, (dict, list)):
            # Keep schema stable for HF JSON loader by stringifying nested values.
            persona[key_text] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            persona[key_text] = as_text(value)
    return persona


def normalize_message(message: Any) -> dict[str, str] | None:
    if not isinstance(message, dict):
        return None
    return {
        "role": as_text(message.get("role")),
        "content": as_text(message.get("content")),
    }


def normalize_conversation(raw_conversation: Any) -> list[dict[str, str]]:
    if not isinstance(raw_conversation, list):
        return []
    conversation: list[dict[str, str]] = []
    for message in raw_conversation:
        normalized = normalize_message(message)
        if normalized is not None:
            conversation.append(normalized)
    return conversation


def split_role_utterances(conversation: list[dict[str, str]]) -> dict[str, list[str]]:
    users: list[str] = []
    assistants: list[str] = []
    for message in conversation:
        role = message.get("role", "").strip().lower()
        content = message.get("content", "")
        if role == "user":
            users.append(content)
        elif role == "assistant":
            assistants.append(content)
    return {"user": users, "assistant": assistants}


def infer_num_turns(record: dict[str, Any], role_utterances: dict[str, list[str]]) -> int:
    raw_num_turns = record.get("num_turns")
    if isinstance(raw_num_turns, int):
        return raw_num_turns
    return len(role_utterances["user"])


def transform_record(record: Any) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None

    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    persona = normalize_persona(record.get("persona_features"))

    conversation = normalize_conversation(record.get("full_conversation"))
    role_utterances = split_role_utterances(conversation)

    raw_conversation_ended = metadata.get("conversation_ended", False)
    if isinstance(raw_conversation_ended, bool):
        conversation_ended = raw_conversation_ended
    else:
        conversation_ended = False

    return {
        "id": as_text(record.get("sample_id")),
        "persona_id": as_text(record.get("persona_id")),
        "persona": persona,
        "original_query": as_text(record.get("original_query")),
        "initial_query": as_text(record.get("initial_query")),
        "full_conversation": conversation,
        "role_utterances": role_utterances,
        "num_turns": infer_num_turns(record, role_utterances),
        "distractor_type": as_text(metadata.get("distractor_type")),
        "conversation_ended": conversation_ended,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare PPOpt Hugging Face dataset JSONL with richer keys."
    )
    parser.add_argument(
        "--input",
        default="output/training_data/PPOpt.json",
        help="Input JSON path (default: output/training_data/PPOpt.json)",
    )
    parser.add_argument(
        "--output",
        default="output/training_data/PPOpt_hf_rich.jsonl",
        help="Output JSONL path (default: output/training_data/PPOpt_hf_rich.jsonl)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[error] input file not found: {input_path}")
        return 1

    if output_path.exists() and not args.overwrite:
        print(
            f"[error] output already exists: {output_path}. "
            "Use --overwrite to replace it."
        )
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        print(f"[error] unsupported JSON root type: {type(data).__name__}")
        return 1

    total = 0
    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as out:
        for item in items:
            total += 1
            transformed = transform_record(item)
            if transformed is None:
                skipped += 1
                continue
            out.write(json.dumps(transformed, ensure_ascii=False))
            out.write("\n")
            written += 1

    print(
        f"[ok] wrote {written} records to {output_path} "
        f"(total: {total}, skipped: {skipped})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
