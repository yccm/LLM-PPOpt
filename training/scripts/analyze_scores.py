#!/usr/bin/env python
"""
Analyze evaluation results in res-125-1 folder.
Compute statistics for every 25 samples (in order).
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def analyze_file(file_path: str, batch_size: int = 25):
    """Analyze a single file, compute stats for every batch_size samples."""

    results = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"\n{'='*80}")
    print(f"File: {os.path.basename(file_path)}")
    print(f"Total samples: {len(lines)}")
    print(f"{'='*80}")

    # Process each batch
    batch_idx = 0
    for start_idx in range(0, len(lines), batch_size):
        end_idx = min(start_idx + batch_size, len(lines))
        batch_lines = lines[start_idx:end_idx]

        # Collect scores for this batch
        original_persona_scores = []
        original_task_scores = []
        optimized_persona_scores = []
        optimized_task_scores = []

        for line in batch_lines:
            try:
                data = json.loads(line.strip())
                original_scores = data.get('original_scores', {})
                optimized_scores = data.get('optimized_scores', {})

                if original_scores:
                    original_persona_scores.append(original_scores.get('persona_score', 0))
                    original_task_scores.append(original_scores.get('task_score', 0))

                if optimized_scores:
                    optimized_persona_scores.append(optimized_scores.get('persona_score', 0))
                    optimized_task_scores.append(optimized_scores.get('task_score', 0))
            except json.JSONDecodeError:
                continue

        # Compute averages
        n = len(original_persona_scores)
        if n > 0:
            avg_orig_persona = sum(original_persona_scores) / n
            avg_orig_task = sum(original_task_scores) / n
            avg_opt_persona = sum(optimized_persona_scores) / n
            avg_opt_task = sum(optimized_task_scores) / n
            avg_orig_total = (avg_orig_persona + avg_orig_task) / 2
            avg_opt_total = (avg_opt_persona + avg_opt_task) / 2

            batch_result = {
                'batch': batch_idx + 1,
                'range': f"{start_idx + 1}-{end_idx}",
                'count': n,
                'original': {
                    'persona': avg_orig_persona,
                    'task': avg_orig_task,
                    'total': avg_orig_total
                },
                'optimized': {
                    'persona': avg_opt_persona,
                    'task': avg_opt_task,
                    'total': avg_opt_total
                },
                'delta': {
                    'persona': avg_opt_persona - avg_orig_persona,
                    'task': avg_opt_task - avg_orig_task,
                    'total': avg_opt_total - avg_orig_total
                }
            }
            results.append(batch_result)

            print(f"\nBatch {batch_idx + 1} (samples {start_idx + 1}-{end_idx}, n={n}):")
            print(f"  Original  - Persona: {avg_orig_persona:.2f}, Task: {avg_orig_task:.2f}, Total: {avg_orig_total:.2f}")
            print(f"  Optimized - Persona: {avg_opt_persona:.2f}, Task: {avg_opt_task:.2f}, Total: {avg_opt_total:.2f}")
            print(f"  Delta     - Persona: {avg_opt_persona - avg_orig_persona:+.2f}, Task: {avg_opt_task - avg_orig_task:+.2f}, Total: {avg_opt_total - avg_orig_total:+.2f}")

        batch_idx += 1

    # Overall stats
    if results:
        all_orig_persona = sum(r['original']['persona'] * r['count'] for r in results) / sum(r['count'] for r in results)
        all_orig_task = sum(r['original']['task'] * r['count'] for r in results) / sum(r['count'] for r in results)
        all_opt_persona = sum(r['optimized']['persona'] * r['count'] for r in results) / sum(r['count'] for r in results)
        all_opt_task = sum(r['optimized']['task'] * r['count'] for r in results) / sum(r['count'] for r in results)

        print(f"\n{'-'*40}")
        print(f"Overall Stats (total {sum(r['count'] for r in results)} samples):")
        print(f"  Original  - Persona: {all_orig_persona:.2f}, Task: {all_orig_task:.2f}, Total: {(all_orig_persona + all_orig_task)/2:.2f}")
        print(f"  Optimized - Persona: {all_opt_persona:.2f}, Task: {all_opt_task:.2f}, Total: {(all_opt_persona + all_opt_task)/2:.2f}")
        print(f"  Delta     - Persona: {all_opt_persona - all_orig_persona:+.2f}, Task: {all_opt_task - all_orig_task:+.2f}, Total: {(all_opt_persona + all_opt_task)/2 - (all_orig_persona + all_orig_task)/2:+.2f}")

    return results


def main():
    # Set paths
    base_dir = Path(__file__).parent.parent
    res_dir = base_dir / "res-125-3"

    if not res_dir.exists():
        print(f"Error: Directory {res_dir} does not exist")
        return

    # Get all jsonl files
    jsonl_files = list(res_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"Error: No jsonl files found in {res_dir}")
        return

    print(f"Found {len(jsonl_files)} files:")
    for f in jsonl_files:
        print(f"  - {f.name}")

    all_results = {}

    # Analyze each file
    for file_path in sorted(jsonl_files):
        results = analyze_file(str(file_path), batch_size=25)
        all_results[file_path.name] = results

    # Summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)

    for file_name, results in all_results.items():
        print(f"\n{file_name}:")
        print(f"{'Batch':<8} {'Range':<12} {'Orig_Persona':<14} {'Orig_Task':<12} {'Opt_Persona':<14} {'Opt_Task':<12} {'D_Persona':<12} {'D_Task':<10}")
        print("-"*100)
        for r in results:
            print(f"{r['batch']:<8} {r['range']:<12} {r['original']['persona']:<14.2f} {r['original']['task']:<12.2f} {r['optimized']['persona']:<14.2f} {r['optimized']['task']:<12.2f} {r['delta']['persona']:<+12.2f} {r['delta']['task']:<+10.2f}")


if __name__ == "__main__":
    main()
