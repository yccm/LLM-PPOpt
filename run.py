"""
Unified Entry Point for LLM_PPOpt Pipeline

This script provides a command-line interface for running the pipeline.
Supports both basic persona generation and full enhanced pipeline.
"""

import argparse
import sys
from pathlib import Path
import yaml
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import PersonaGenerationPipeline
from src.enhanced_pipeline import EnhancedPersonaGenerationPipeline
from src.config_validation import validate_config_report


def run_basic_pipeline(args):
    """Run basic persona generation pipeline."""
    pipeline = PersonaGenerationPipeline(args.config)

    if args.reset:
        pipeline.reset()
        print("Pipeline state reset")

    print("\nStarting persona generation...")
    stats = pipeline.run(
        num_personas=args.num_personas,
        generate_prompts=not args.no_prompts,
        export_dataset=args.export_dataset,
        dataset_path=args.dataset_path,
        export_all_stored=args.export_all_stored,
    )

    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Total personas generated: {stats['total_generated']}")
    print(f"Personas with system prompts: {stats['with_prompts']}")
    print(f"Sampling strategy: {stats['config']['sampling_strategy']}")
    print(f"LLM provider: {stats['config']['llm_provider']}")
    print(f"Storage directory: {stats['storage_stats']['output_directory']}")
    print("=" * 80)


def run_enhanced_pipeline(args):
    """Run enhanced pipeline with full functionality."""
    pipeline = EnhancedPersonaGenerationPipeline(args.config)

    if args.skip_query_transfer:
        pipeline.config['query_generation']['style_transfer']['enabled'] = False

    if args.skip_distractor:
        pipeline.config['distractor']['enabled'] = False

    # Set stages based on args
    if args.stage != 'all':
        all_stages = {
            'persona_generation': False,
            'query_generation': False,
            'interaction_generation': False,
            'distractor_application': False,
            'training_data_export': False
        }

        stage_map = {
            'persona': 'persona_generation',
            'query': 'query_generation',
            'interaction': 'interaction_generation',
            'distractor': 'distractor_application',
            'training': 'training_data_export'
        }

        if args.stage in stage_map:
            all_stages[stage_map[args.stage]] = True

        pipeline.config['experiment']['stages'] = all_stages

    print("\nStarting enhanced pipeline...")
    results = pipeline.run(num_personas=args.num_personas)

    print("\n" + "=" * 80)
    print("ENHANCED PIPELINE SUMMARY")
    print("=" * 80)

    if 'personas' in results:
        print(f"Personas generated: {len(results['personas'])}")

    if 'queries' in results:
        total_queries = sum(len(q) for q in results['queries'].values())
        print(f"Queries generated: {total_queries}")

    if 'interactions' in results:
        print(f"Interactions generated: {len(results['interactions'])}")

    if 'training_data' in results:
        td = results['training_data']
        print(f"Training data:")
        print(f"  - Total samples: {td['total_samples']}")
        print(f"  Files exported:")
        for key, path in td['sample_files'].items():
            print(f"    - {key}: {path}")

    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='LLM_PPOpt: Generate synthetic training data for personalized prompt optimization'
    )

    # Common arguments
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--num-personas',
        type=int,
        default=None,
        help='Number of personas to generate (overrides config)'
    )

    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        choices=['basic', 'enhanced'],
        default='enhanced',
        help='Pipeline mode: basic (persona only) or enhanced (full pipeline)'
    )

    # Enhanced mode arguments
    parser.add_argument(
        '--stage',
        type=str,
        choices=['all', 'persona', 'query', 'interaction', 'distractor', 'training'],
        default='all',
        help='Which pipeline stage to run (enhanced mode only)'
    )

    parser.add_argument(
        '--skip-query-transfer',
        action='store_true',
        help='Skip query style transfer'
    )

    parser.add_argument(
        '--skip-distractor',
        action='store_true',
        help='Skip distractor/noise application'
    )

    # Basic mode arguments
    parser.add_argument(
        '--no-prompts',
        action='store_true',
        help='Skip system prompt generation (basic mode only)'
    )

    parser.add_argument(
        '--export-dataset',
        action='store_true',
        help='Export personas to dataset file (basic mode only)'
    )

    parser.add_argument(
        '--export-all-stored',
        action='store_true',
        help='Include all stored personas in export (basic mode only)'
    )

    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Path for dataset export (basic mode only)'
    )

    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset pipeline state before running (basic mode only)'
    )

    args = parser.parse_args()

    # Check if config file exists
    if not Path(args.config).exists():
        print(f"Error: Configuration file '{args.config}' not found")
        sys.exit(1)

    try:
        def supports_color() -> bool:
            if os.environ.get("NO_COLOR"):
                return False
            return sys.stdout.isatty()

        def c(text: str, color_code: str) -> str:
            if not supports_color():
                return text
            return f"\033[{color_code}m{text}\033[0m"

        GREEN = "32"
        RED = "31"
        YELLOW = "33"
        DIM = "2"

        def fmt_status(status: str, width: int) -> str:
            # Keep width stable by padding BEFORE applying ANSI color codes.
            s = status.ljust(width)
            if status == "PASS":
                return c(s, GREEN)
            if status == "FAIL":
                return c(s, RED)
            if status == "WARN":
                return c(s, YELLOW)
            return s

        def print_validation_table(checks, issues):
            # Simple terminal table (no external deps)
            status_width = max(len("Status"), 4)
            rows = []
            for chk in checks:
                details = chk.details or ""
                rows.append([chk.name, fmt_status(chk.status, status_width), details])

            headers = ["Check", "Status", "Details"]
            widths = [len(h) for h in headers]
            for r in rows:
                widths[0] = max(widths[0], len(r[0]))
                widths[1] = max(widths[1], status_width)
                widths[2] = max(widths[2], len(r[2]))

            def line(char: str = "-") -> str:
                return "+" + "+".join(char * (w + 2) for w in widths) + "+"

            print(line("-"))
            print(f"| {headers[0].ljust(widths[0])} | {headers[1].ljust(widths[1])} | {headers[2].ljust(widths[2])} |")
            print(line("="))
            for name, status, details in rows:
                # status is already padded to a stable width (4)
                print(f"| {name.ljust(widths[0])} | {status} | {details.ljust(widths[2])} |")
            print(line("-"))

            warn_count = sum(1 for i in issues if i.level == "warning")
            err_count = sum(1 for i in issues if i.level == "error")
            summary = f"Validation summary: {err_count} error(s), {warn_count} warning(s)"
            if err_count:
                print(c(summary, RED))
            elif warn_count:
                print(c(summary, YELLOW))
            else:
                print(c(summary, GREEN))

        # Visible, structured config validation (PASS/FAIL table)
        with open(args.config, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
        _cfg, checks, issues = validate_config_report(raw_cfg, config_path=args.config)
        print_validation_table(checks, issues)

        if any(i.level == "error" for i in issues):
            raise ValueError("Config validation failed (see table above).")

        print(f"Initializing {args.mode} pipeline...")

        if args.mode == 'basic':
            run_basic_pipeline(args)
        else:
            run_enhanced_pipeline(args)

        print("\n[OK] Pipeline completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
