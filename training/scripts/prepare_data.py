"""Prepare SFT and RL training data from optimized queries.

Usage:
    python scripts/prepare_data.py --config config.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ppopt.data.prepare import build_sft_and_rl, DEFAULT_OPTIMIZER_SYSTEM_PROMPT
from ppopt.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT and RL training data")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})

    optimized_queries_path = data_cfg.get("optimized_queries_file", "data/optimized_queries.jsonl")
    train_samples_path = data_cfg.get("train_samples_file", "data/train_samples.json")
    sft_out_path = data_cfg.get("sft_train_file", "data/sft_train.jsonl")
    rl_out_path = data_cfg.get("rl_train_file", "data/rl_states.jsonl")

    print(f"[INFO] Preparing training data...")
    print(f"  Optimized queries: {optimized_queries_path}")
    print(f"  Train samples: {train_samples_path}")
    print(f"  SFT output: {sft_out_path}")
    print(f"  RL output: {rl_out_path}")
    print()

    result = build_sft_and_rl(
        optimized_queries_path=optimized_queries_path,
        train_samples_path=train_samples_path,
        sft_out_path=sft_out_path,
        rl_out_path=rl_out_path,
        max_history_sessions=data_cfg.get("max_history_sessions", 4),
        include_persona=data_cfg.get("include_persona", True),
        use_persona_in_input=data_cfg.get("use_persona_in_input", False),
        require_history=data_cfg.get("require_history", True),
        min_history_sessions=data_cfg.get("min_history_sessions", 1),
        include_assistant_content=data_cfg.get("include_assistant_content", True),
        system_prompt=DEFAULT_OPTIMIZER_SYSTEM_PROMPT,
    )

    print(f"[SUCCESS] Data preparation complete!")
    print(f"  SFT samples: {result['sft']}")
    print(f"  RL samples: {result['rl']}")
    print(f"  Skipped (no history): {result['skipped_no_history']}")


if __name__ == "__main__":
    main()
