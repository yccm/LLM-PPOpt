#!/usr/bin/env python3
"""
Export unused queries for selected datasets by comparing query.jsonl to train_samples.

Unused = entries in query.jsonl whose source_id does not appear in any train_samples query_id.
"""

import argparse
import json
import sys
from collections import Counter
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TARGET_DATASETS = {
    "HuggingFaceH4/ultrachat_200k": "unused_ultrachat_200k.jsonl",
    "allenai/ai2_arc": "unused_ai2_arc.jsonl",
    "Muennighoff/mbpp": "unused_mbpp.jsonl",
}


def iter_query_ids(obj: Any) -> Iterable[str]:
    if isinstance(obj, dict):
        if "query_id" in obj:
            yield obj["query_id"]
        for value in obj.values():
            yield from iter_query_ids(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_query_ids(item)


def extract_query_id(sample: Dict[str, Any]) -> Optional[str]:
    metadata = sample.get("metadata")
    if isinstance(metadata, dict) and metadata.get("query_id"):
        return metadata.get("query_id")
    for qid in iter_query_ids(sample):
        return qid
    return None


def parse_query_id(query_id: str) -> Tuple[str, Optional[str]]:
    if ":" in query_id:
        dataset, source_id = query_id.split(":", 1)
        return dataset, source_id
    return query_id, None


def resolve_inputs(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        if any(ch in pattern for ch in ("*", "?", "[")):
            files.extend(glob(pattern))
        else:
            files.append(pattern)
    return [str(Path(p)) for p in files if Path(p).exists()]


def load_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export unused queries for ultrachat_200k, ai2_arc, and mbpp"
    )
    parser.add_argument(
        "--query-file",
        default="input/query.jsonl",
        help="Path to query.jsonl",
    )
    parser.add_argument(
        "--train-samples",
        "-i",
        nargs="+",
        default=[
            "output/training_data/1_train_samples_ss.json",
            "output/training_data/train_samples_*.json",
        ],
        help="Input train_samples JSON file(s) (supports glob patterns)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="output/training_data",
        help="Directory to write unused dataset files",
    )

    args = parser.parse_args()

    query_path = Path(args.query_file)
    if not query_path.exists():
        print(f"Error: query file not found: {query_path}", file=sys.stderr)
        sys.exit(1)

    train_files = resolve_inputs(args.train_samples)
    if not train_files:
        print("Error: No train_samples files found.", file=sys.stderr)
        sys.exit(1)

    used_by_dataset = {ds: set() for ds in TARGET_DATASETS}
    for path in train_files:
        samples = load_samples(path)
        for sample in samples:
            qid = extract_query_id(sample)
            if not qid:
                continue
            dataset, source_id = parse_query_id(qid)
            if dataset in used_by_dataset and source_id is not None:
                used_by_dataset[dataset].add(str(source_id))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    writers = {}
    for dataset, filename in TARGET_DATASETS.items():
        writers[dataset] = open(output_dir / filename, "w", encoding="utf-8")

    total_counts = Counter()
    unused_counts = Counter()

    with open(query_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            dataset = obj.get("dataset")
            if dataset not in TARGET_DATASETS:
                continue
            total_counts[dataset] += 1
            source_id = obj.get("source_id")
            if source_id is None:
                continue
            if str(source_id) not in used_by_dataset[dataset]:
                json.dump(obj, writers[dataset], ensure_ascii=False)
                writers[dataset].write("\n")
                unused_counts[dataset] += 1

    for fp in writers.values():
        fp.close()

    print(f"Query file: {query_path}")
    print(f"Train samples: {train_files}")
    for dataset in TARGET_DATASETS:
        used = len(used_by_dataset[dataset])
        total = total_counts[dataset]
        unused = unused_counts[dataset]
        print(
            f"{dataset}: total={total} used={used} unused={unused} "
            f"-> {output_dir / TARGET_DATASETS[dataset]}"
        )


if __name__ == "__main__":
    main()
