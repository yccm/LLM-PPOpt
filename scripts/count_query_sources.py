#!/usr/bin/env python3
"""
Count query source datasets from train_samples JSON files.

The dataset is derived from query_id by taking the prefix before the first ":".
Counts are aggregated across all provided input files.
"""

import argparse
import json
import sys
from collections import Counter
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def dataset_from_query_id(query_id: str) -> str:
    if ":" in query_id:
        return query_id.split(":", 1)[0]
    return query_id


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


def format_table(rows: List[Tuple[str, int]], total: int) -> str:
    if not rows:
        return "No dataset counts found."
    name_width = min(max(len(name) for name, _ in rows), 60)
    lines = [f"{'dataset':<{name_width}}  {'count':>8}  {'pct':>6}"]
    for name, count in rows:
        pct = (count / total * 100) if total else 0.0
        display_name = name if len(name) <= name_width else name[: name_width - 3] + "..."
        lines.append(f"{display_name:<{name_width}}  {count:>8}  {pct:>6.2f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count query source datasets from train_samples JSON files"
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        default=[
            "output/training_data/1_train_samples_ss.json",
            "output/training_data/train_samples_*.json",
        ],
        help="Input train_samples JSON file(s) (supports glob patterns)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional output JSON path for aggregated counts",
    )

    args = parser.parse_args()
    input_files = resolve_inputs(args.input)
    if not input_files:
        print("Error: No input files found.", file=sys.stderr)
        sys.exit(1)

    counts: Counter[str] = Counter()
    total_samples = 0
    missing_query_id = 0

    for path in input_files:
        samples = load_samples(path)
        total_samples += len(samples)
        for sample in samples:
            qid = extract_query_id(sample)
            if not qid:
                missing_query_id += 1
                continue
            dataset = dataset_from_query_id(qid)
            counts[dataset] += 1

    rows = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    print(f"Input files: {input_files}")
    print(f"Total samples: {total_samples}")
    if missing_query_id:
        print(f"Missing query_id: {missing_query_id}")
    print(f"Datasets: {len(rows)}\n")
    print(format_table(rows, total_samples))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_files": input_files,
            "total_samples": total_samples,
            "missing_query_id": missing_query_id,
            "counts": dict(rows),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()
