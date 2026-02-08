#!/usr/bin/env python3
"""
Extract History + Current Query Script

For each persona:
- Selects the interaction with fewest turns as target (extracts original_query as current_query)
- Uses other interactions' full_conversation as history
- Outputs to JSONL format
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any


def load_samples(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Load samples from JSON files."""
    all_samples = []
    for file_path in file_paths:
        path = Path(file_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
                print(f"Loaded {len(samples)} samples from {path.name}")
                all_samples.extend(samples)
    return all_samples


def group_by_persona(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group samples by persona_id."""
    grouped = defaultdict(list)
    for sample in samples:
        persona_id = sample.get('persona_id', 'unknown')
        grouped[persona_id].append(sample)
    return dict(grouped)


def extract_for_persona(
    persona_id: str,
    interactions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Extract history and current_query for a single persona.

    Args:
        persona_id: The persona ID
        interactions: All interactions for this persona

    Returns:
        Dict with persona_id, persona_features, history, current_query
    """
    # Sort by num_turns to find the one with fewest turns
    sorted_interactions = sorted(interactions, key=lambda x: x.get('num_turns', 0))
    target = sorted_interactions[0]

    # Collect history from other interactions
    history = []
    for interaction in interactions:
        if interaction.get('sample_id') != target.get('sample_id'):
            conversation = interaction.get('full_conversation', [])
            if conversation:
                history.append(conversation)

    return {
        'persona_id': persona_id,
        'persona_features': target.get('persona_features', {}),
        'history': history,
        'current_query': target.get('original_query', '')
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract history and current_query from train samples"
    )
    parser.add_argument(
        '--input', '-i',
        nargs='+',
        default=['output/training_data/train_samples_*.json'],
        help='Input train_samples JSON file(s) (supports glob patterns)'
    )
    parser.add_argument(
        '--output', '-o',
        default='output/extracted/history_query.jsonl',
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--limit', '-n',
        type=int,
        default=None,
        help='Limit number of personas to process'
    )
    parser.add_argument(
        '--persona-id',
        default=None,
        help='Only process specific persona ID'
    )

    args = parser.parse_args()

    # Resolve glob patterns
    input_files = []
    for pattern in args.input:
        if '*' in pattern:
            input_files.extend(Path('.').glob(pattern))
        else:
            input_files.append(Path(pattern))
    input_files = [str(f) for f in input_files if f.exists()]

    if not input_files:
        print("Error: No input files found")
        sys.exit(1)

    print(f"Input files: {input_files}")

    # Load samples
    samples = load_samples(input_files)
    if not samples:
        print("Error: No samples loaded")
        sys.exit(1)

    # Group by persona
    grouped = group_by_persona(samples)
    print(f"\nFound {len(grouped)} unique personas")

    # Filter if specific persona requested
    if args.persona_id:
        if args.persona_id in grouped:
            grouped = {args.persona_id: grouped[args.persona_id]}
        else:
            print(f"Error: Persona '{args.persona_id}' not found")
            sys.exit(1)

    # Apply limit
    if args.limit:
        persona_ids = list(grouped.keys())[:args.limit]
        grouped = {pid: grouped[pid] for pid in persona_ids}
        print(f"Limited to {len(grouped)} personas")

    # Prepare output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process each persona and write to JSONL
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for persona_id, interactions in grouped.items():
            result = extract_for_persona(persona_id, interactions)
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
            count += 1

    print(f"\nExtracted {count} personas to {output_path}")


if __name__ == "__main__":
    main()
