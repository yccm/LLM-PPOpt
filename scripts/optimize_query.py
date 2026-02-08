#!/usr/bin/env python3
"""
Query Optimization Script

This script optimizes user queries based on persona preferences and conversation history.
For each persona:
- Uses N-1 interactions as history
- Optimizes the initial_query of the interaction with fewest turns
- Outputs reasoning chain + optimized_query via GPT API
"""

import argparse
import json
import sys
import yaml
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import create_llm_client, LLMClient


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


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


def select_target_and_history(
    interactions: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Select target interaction (fewest turns) and history interactions.

    Args:
        interactions: List of interactions for a persona

    Returns:
        Tuple of (target_interaction, history_interactions)
    """
    # Sort by num_turns to find the one with fewest turns
    sorted_interactions = sorted(interactions, key=lambda x: x.get('num_turns', 0))
    target = sorted_interactions[0]

    # For single interaction: use its own conversation as history
    # For multiple: use other interactions as history
    if len(interactions) == 1:
        history = []  # History will be built from target's full_conversation
    else:
        history = [i for i in interactions if i.get('sample_id') != target.get('sample_id')]

    return target, history


def format_persona(persona_features: Dict[str, str]) -> str:
    """Format persona features for prompt."""
    lines = []
    for key, value in sorted(persona_features.items()):
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def format_conversation(conversation: List[Dict[str, Any]], max_chars: int = 300, assistant_placeholder: str = "[Assistant responded]") -> str:
    """Format a single conversation for prompt.

    Args:
        conversation: List of message dictionaries with 'role' and 'content'
        max_chars: Maximum characters per message before truncation
        assistant_placeholder: Placeholder text for assistant messages
    """
    lines = []
    for msg in conversation:
        role = msg.get('role', 'unknown').lower()
        if role == 'user':
            content = msg.get('content', '')
            # Truncate long messages
            if len(content) > max_chars:
                content = content[:max_chars] + "..."
            lines.append(f"**User:** {content}")
        elif role == 'assistant':
            lines.append(f"**Assistant:** {assistant_placeholder}")
    return "\n".join(lines)


def format_history(
    target: Dict[str, Any],
    history_interactions: List[Dict[str, Any]],
    max_interactions: int = None,
    max_chars_per_msg: int = 500
) -> str:
    """
    Format conversation history for prompt.
    User messages shown in full, assistant messages replaced with placeholder.

    For single interaction: use target's full_conversation
    For multiple: concatenate history interactions' conversations (limited)
    """
    if not history_interactions:
        # Single interaction case: use target's own conversation as context
        conversation = target.get('full_conversation', [])
        if conversation:
            formatted = format_conversation(conversation, max_chars_per_msg)
            if formatted:
                return f"### Previous conversation in this session:\n{formatted}"
        return "(No conversation history available)"

    # Multiple interactions case
    limited_interactions = history_interactions if max_interactions is None else history_interactions[:max_interactions]
    parts = []
    for i, interaction in enumerate(limited_interactions, 1):
        conversation = interaction.get('full_conversation', [])
        if conversation:
            formatted = format_conversation(conversation, max_chars_per_msg)
            if formatted:
                parts.append(f"### Session {i}:\n{formatted}")

    if max_interactions is not None and len(history_interactions) > max_interactions:
        parts.append(f"\n(... {len(history_interactions) - max_interactions} more sessions omitted for brevity)")

    return "\n\n".join(parts) if parts else "(No conversation history available)"


def load_prompt_template(template_path: str) -> str:
    """Load prompt template from file."""
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def build_optimization_prompt(
    template: str,
    persona_features: Dict[str, str],
    history: str,
    next_query: str
) -> str:
    """Build the optimization prompt."""
    prompt = template.replace('{persona}', format_persona(persona_features))
    prompt = prompt.replace('{history}', history)
    prompt = prompt.replace('{next_query}', next_query)
    return prompt


def make_llm_client_provider(api_config: Dict[str, Any], dry_run: bool) -> Callable[[], Optional[LLMClient]]:
    """Return a callable that provides a thread-local LLM client."""
    if dry_run:
        return lambda: None

    thread_local = threading.local()

    def provider() -> Optional[LLMClient]:
        client = getattr(thread_local, 'llm_client', None)
        if client is None:
            client = create_llm_client(api_config)
            thread_local.llm_client = client
        return client

    return provider


def parse_gpt_response(response: str) -> Dict[str, Any]:
    """Parse GPT response to extract reasoning and optimized_query."""
    # Try to parse as JSON
    try:
        # Clean up response if needed
        clean_response = response.strip()
        if clean_response.startswith('```'):
            # Remove markdown code blocks
            lines = clean_response.split('\n')
            clean_response = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])

        result = json.loads(clean_response)

        return {
            'reasoning': result.get('reasoning', ''),
            'optimized_query': result.get('optimized_query', ''),
            'parse_success': True
        }
    except json.JSONDecodeError as e:
        return {
            'reasoning': '',
            'optimized_query': '',
            'raw_response': response,
            'parse_error': str(e),
            'parse_success': False
        }


def process_persona(
    persona_id: str,
    interactions: List[Dict[str, Any]],
    llm_client: LLMClient,
    template: str,
    model_name: str,
    dry_run: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Process a single persona: build prompt, call API, parse response.

    Args:
        persona_id: The persona ID
        interactions: All interactions for this persona
        llm_client: LLM client for API calls
        template: Prompt template
        model_name: Model name for logging
        dry_run: If True, skip API call

    Returns:
        Result dictionary or None on failure
    """
    # Select target and history
    target, history_interactions = select_target_and_history(interactions)

    persona_features = target.get('persona_features', {})
    next_query = target.get('initial_query', '')
    history = format_history(target, history_interactions)

    # Build prompt
    prompt = build_optimization_prompt(template, persona_features, history, next_query)

    result = {
        'persona_id': persona_id,
        'persona_features': persona_features,
        'target_interaction_id': target.get('sample_id', ''),
        'target_num_turns': target.get('num_turns', 0),
        'history_interaction_ids': [i.get('sample_id', '') for i in history_interactions],
        'num_history_interactions': len(history_interactions),
        'original_next_query': next_query,
        'model_used': model_name,
        'timestamp': datetime.now().isoformat()
    }

    if dry_run:
        result['dry_run'] = True
        result['prompt_length'] = len(prompt)
        result['prompt_preview'] = prompt[:500] + "..." if len(prompt) > 500 else prompt
        print(f"\n[DRY RUN] Persona: {persona_id}")
        print(f"  - Target interaction: {target.get('sample_id', '')[:50]}...")
        print(f"  - Target turns: {target.get('num_turns', 0)}")
        print(f"  - History interactions: {len(history_interactions)}")
        print(f"  - Next query: {next_query[:80]}...")
        print(f"  - Prompt length: {len(prompt)} chars")
        return result

    # Call GPT API
    try:
        print(f"\nProcessing persona: {persona_id}")
        print(f"  - Target turns: {target.get('num_turns', 0)}, History: {len(history_interactions)} interactions")
        print(f"  - Prompt length: {len(prompt)} chars")

        response = llm_client.generate(prompt)
        print(f"  - Response length: {len(response) if response else 0} chars")
        print(f"  - Response preview: {repr(response[:200]) if response else 'EMPTY'}")

        if not response or not response.strip():
            result['error'] = "Empty response from API"
            result['success'] = False
            print(f"  [ERROR] Empty response from API")
            return result

        parsed = parse_gpt_response(response)

        result['gpt_response'] = parsed
        result['success'] = parsed.get('parse_success', False)

        if parsed.get('parse_success'):
            print(f"  [OK] Optimized: {parsed['optimized_query'][:80]}...")
        else:
            print(f"  [FAIL] Parse failed: {parsed.get('parse_error', 'Unknown error')}")

    except Exception as e:
        result['error'] = str(e)
        result['success'] = False
        print(f"  [ERROR] API error: {str(e)}")

    return result


def print_summary(results: List[Dict[str, Any]]):
    """Print summary statistics."""
    total = len(results)
    successful = sum(1 for r in results if r.get('success', False))
    failed = sum(1 for r in results if not r.get('success', False) and not r.get('dry_run', False))
    dry_run = sum(1 for r in results if r.get('dry_run', False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total personas processed: {total}")
    if dry_run:
        print(f"Dry run (no API calls): {dry_run}")
    else:
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/total*100:.1f}%" if total > 0 else "N/A")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize user queries based on persona and conversation history"
    )
    parser.add_argument(
        '--input', '-i',
        nargs='+',
        default=['output/training_data/train_samples_*.json'],
        help='Input train_samples JSON file(s) (supports glob patterns)'
    )
    parser.add_argument(
        '--output', '-o',
        default='output/optimized_queries',
        help='Output directory for results'
    )
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Config file path'
    )
    parser.add_argument(
        '--model', '-m',
        default=None,
        help='Model to use (overrides config)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without calling API'
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
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Number of personas to process in parallel (default: CPU count)'
    )

    args = parser.parse_args()

    # Resolve glob patterns
    input_files = []
    for pattern in args.input:
        if '*' in pattern:
            input_files.extend(Path('.').glob(pattern))
        else:
            input_files.append(Path(pattern))
    # Filter to only train_samples_*.json files (exclude statistics, token_usage, etc.)
    input_files = [
        str(f) for f in input_files
        if f.exists() and f.name.startswith('train_samples_')
    ]

    if not input_files:
        print("Error: No input files found")
        sys.exit(1)

    print(f"Input files: {input_files}")

    # Load config
    config = load_config(args.config)

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

    # Setup LLM client config
    api_config = dict(config.get('api', {}))
    if args.model:
        api_config['model'] = args.model
    model_name = api_config.get('model', 'gpt-4o-mini')

    if args.dry_run:
        print("Dry run mode - no API calls will be made")
    else:
        print(f"Using model: {model_name}")
    client_provider = make_llm_client_provider(api_config, args.dry_run)

    # Load prompt template
    template_path = Path(__file__).parent.parent / "prompts" / "query_optimization.txt"
    template = load_prompt_template(str(template_path))

    # Prepare incremental output file
    output_dir_path = Path(args.output)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir_path / f"optimized_queries_{timestamp}.jsonl"
    print(f"\nIncremental results will be appended to: {output_path}")

    # Process each persona (with optional concurrency)
    persona_items = list(grouped.items())
    results: List[Dict[str, Any]] = []
    worker_count = args.workers if args.workers and args.workers > 0 else (os.cpu_count() or 4)
    write_lock = threading.Lock()

    def persist_result(result: Dict[str, Any]):
        with write_lock:
            with open(output_path, 'a', encoding='utf-8') as wf:
                json.dump(result, wf, indent=2, ensure_ascii=False)
                wf.write("\n")
            results.append(result)

    def process_worker(persona_entry: Tuple[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        pid, interactions = persona_entry
        client = client_provider()
        return process_persona(
            persona_id=pid,
            interactions=interactions,
            llm_client=client,
            template=template,
            model_name=model_name,
            dry_run=args.dry_run
        )

    if worker_count <= 1:
        for entry in persona_items:
            result = process_worker(entry)
            if result:
                persist_result(result)
    else:
        print(f"\nProcessing {len(persona_items)} personas across {worker_count} workers...")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_persona = {
                executor.submit(process_worker, entry): entry[0] for entry in persona_items
            }
            for future in as_completed(future_to_persona):
                persona_id = future_to_persona[future]
                try:
                    result = future.result()
                    if result:
                        persist_result(result)
                except Exception as exc:
                    print(f"[ERROR] Persona {persona_id} failed: {exc}")

    # Print summary
    print_summary(results)
    print(f"\nIncremental output file: {output_path}")


if __name__ == "__main__":
    main()
