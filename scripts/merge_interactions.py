#!/usr/bin/env python
"""
Merge interaction JSON files under output/interactions into one training data file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    except Exception:
        return {}


def get_collector(config: Dict[str, Any]):
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from src.training_data import TrainingDataCollector  # type: ignore
    except Exception:
        return None
    try:
        return TrainingDataCollector(config)
    except Exception:
        return None


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return (base / path).resolve()


def collect_files(input_dir: Path, use_index: bool) -> List[Path]:
    files: List[Path] = []
    index_path = input_dir / "index.json"
    if use_index and index_path.exists():
        try:
            with index_path.open("r", encoding="utf-8") as f:
                index = json.load(f)
            for entry in index.get("interactions", []):
                path_value = entry.get("filepath")
                if not path_value:
                    continue
                files.append(resolve_path(Path(path_value), REPO_ROOT))
        except Exception:
            files = []
    if not files:
        files = [p for p in input_dir.glob("*.json") if p.name != "index.json"]
        files.sort()
    return files


def manual_interaction_to_sample(interaction: Dict[str, Any]) -> Dict[str, Any]:
    messages = interaction.get("messages", []) or []
    prompt_trajectory = [m.get("content", "") for m in messages if m.get("role") == "user"]
    full_conversation = []
    for msg in messages:
        msg_dict = {
            "role": msg.get("role", ""),
            "content": msg.get("content", ""),
        }
        if msg.get("metadata"):
            msg_dict["metadata"] = msg.get("metadata")
        full_conversation.append(msg_dict)

    noisy_versions = interaction.get("noisy_versions", {}) or {}

    sample: Dict[str, Any] = {
        "sample_id": interaction.get("interaction_id", ""),
        "persona_id": interaction.get("persona_id", ""),
        "persona_features": interaction.get("persona_features", {}) or {},
        "original_query": interaction.get("original_query", interaction.get("initial_query", "")),
        "initial_query": interaction.get("initial_query", ""),
        "prompt_trajectory": prompt_trajectory,
        "full_conversation": full_conversation,
        "num_turns": interaction.get("num_turns", 0),
        "metadata": interaction.get("metadata", {}) or {},
    }

    if noisy_versions:
        if "initial_query" in noisy_versions:
            sample["noisy_initial_queries"] = noisy_versions.get("initial_query")
        if "followups" in noisy_versions:
            sample["noisy_trajectories"] = noisy_versions.get("followups")

    return sample


def normalize_item(item: Any, collector) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    if "sample_id" in item:
        return item
    if "interaction_id" in item:
        if collector is not None:
            try:
                sample = collector.interaction_to_training_sample(item)
                return sample.to_dict()
            except Exception:
                return manual_interaction_to_sample(item)
        return manual_interaction_to_sample(item)
    return None


def load_json_items(path: Path) -> Iterable[Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge interaction JSON files into one training data file."
    )
    parser.add_argument(
        "--input",
        default="output/interactions",
        help="Directory that contains interaction JSON files.",
    )
    parser.add_argument(
        "--output",
        default="output/training_data/merged_samples.json",
        help="Output training data JSON file.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Optional config file (used for conversion when needed).",
    )
    parser.add_argument(
        "--use-index",
        action="store_true",
        default=True,
        help="Use output/interactions/index.json when present (default: true).",
    )
    parser.add_argument(
        "--no-index",
        dest="use_index",
        action="store_false",
        help="Ignore index.json and scan the directory instead.",
    )
    parser.add_argument(
        "--no-dedupe",
        dest="dedupe",
        action="store_false",
        default=True,
        help="Keep duplicates (by sample_id) if present.",
    )
    args = parser.parse_args()

    input_dir = resolve_path(Path(args.input), REPO_ROOT)
    output_path = resolve_path(Path(args.output), REPO_ROOT)
    config_path = resolve_path(Path(args.config), REPO_ROOT)

    if not input_dir.exists():
        print(f"[error] input directory not found: {input_dir}")
        return 1

    config = load_config(config_path)
    collector = get_collector(config) if config else get_collector({})

    files = collect_files(input_dir, args.use_index)
    if not files:
        print(f"[error] no interaction json files found in: {input_dir}")
        return 1

    samples: List[Dict[str, Any]] = []
    seen_ids = set()
    skipped = 0
    converted = 0
    loaded_files = 0

    for path in files:
        if not path.exists():
            skipped += 1
            continue
        try:
            items = load_json_items(path)
        except Exception:
            skipped += 1
            continue

        loaded_files += 1
        for item in items:
            sample = normalize_item(item, collector)
            if sample is None:
                skipped += 1
                continue
            sample_id = sample.get("sample_id")
            if args.dedupe and sample_id:
                if sample_id in seen_ids:
                    continue
                seen_ids.add(sample_id)
            if "sample_id" not in item and "interaction_id" in item:
                converted += 1
            samples.append(sample)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(
        f"[ok] merged {len(samples)} samples from {loaded_files} files "
        f"(skipped: {skipped}, converted: {converted})"
    )
    print(f"[ok] output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
