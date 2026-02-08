"""
One-click training pipeline: SFT -> RL

Usage:
    python scripts/train_pipeline.py --config config.yaml

This script automatically:
1. Runs SFT training
2. Uses SFT checkpoint paths for RL training
3. Runs RL (GRPO) training

All output (stdout, stderr, errors, interrupts) is logged to a file.
"""

import argparse
import json
import sys
import os
import signal
import traceback
import atexit
from pathlib import Path
from datetime import datetime
import logging

sys.path.append(str(Path(__file__).resolve().parents[1]))


class TeeStream:
    """Stream that writes to both a file and original stream."""

    def __init__(self, original_stream, log_file):
        self.original_stream = original_stream
        self.log_file = log_file
        self.encoding = getattr(original_stream, 'encoding', 'utf-8') or 'utf-8'

    def write(self, message):
        if message:
            self.original_stream.write(message)
            self.original_stream.flush()
            try:
                self.log_file.write(message)
                self.log_file.flush()
            except Exception:
                pass

    def flush(self):
        self.original_stream.flush()
        try:
            self.log_file.flush()
        except Exception:
            pass

    def fileno(self):
        return self.original_stream.fileno()

    def isatty(self):
        return self.original_stream.isatty()


class TrainingLogger:
    """Logger that captures all output to both terminal and log file."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create log filename with timestamp
        self.start_time = datetime.now()
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"training_{timestamp}.log"

        self.log_file = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self._setup_done = False

    def setup(self, command_args: list):
        """Setup logging - redirect stdout/stderr to both terminal and file."""
        # Open log file
        self.log_file = open(self.log_path, "w", encoding="utf-8")

        # Write header
        self._write_header(command_args)

        # Redirect stdout and stderr
        sys.stdout = TeeStream(self.original_stdout, self.log_file)
        sys.stderr = TeeStream(self.original_stderr, self.log_file)

        # Setup signal handlers for interrupts
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Register cleanup on exit
        atexit.register(self.cleanup)

        self._setup_done = True

        print(f"[LOG] Logging to: {self.log_path}")
        print()

    def _write_header(self, command_args: list):
        """Write log header with system info and command."""
        header = [
            "=" * 80,
            "TRAINING LOG",
            "=" * 80,
            f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Command: python {' '.join(command_args)}",
            f"Working Directory: {os.getcwd()}",
            f"Python Version: {sys.version}",
            f"Log File: {self.log_path}",
            "=" * 80,
            "",
        ]
        self.log_file.write("\n".join(header) + "\n")
        self.log_file.flush()

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals (Ctrl+C, SIGTERM)."""
        signal_name = signal.Signals(signum).name
        msg = f"\n\n{'=' * 80}\n[INTERRUPTED] Received signal: {signal_name}\n"
        msg += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        msg += "=" * 80 + "\n"

        # Write to both streams
        self.original_stderr.write(msg)
        if self.log_file:
            self.log_file.write(msg)
            self.log_file.flush()

        self.cleanup()
        sys.exit(1)

    def log_exception(self, exc_type, exc_value, exc_tb):
        """Log exception details."""
        msg = f"\n\n{'=' * 80}\n"
        msg += f"[ERROR] Exception occurred!\n"
        msg += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        msg += f"Type: {exc_type.__name__}\n"
        msg += f"Message: {exc_value}\n"
        msg += "-" * 80 + "\n"
        msg += "Traceback:\n"
        msg += "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        msg += "=" * 80 + "\n"

        # Write to stderr (which is tee'd to log file)
        sys.stderr.write(msg)

    def cleanup(self):
        """Cleanup and write footer."""
        if not self._setup_done:
            return

        end_time = datetime.now()
        duration = end_time - self.start_time

        footer = [
            "",
            "=" * 80,
            "TRAINING LOG END",
            "=" * 80,
            f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {duration}",
            "=" * 80,
        ]

        footer_text = "\n".join(footer) + "\n"

        # Restore original streams first
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # Write footer and close
        if self.log_file and not self.log_file.closed:
            self.log_file.write(footer_text)
            self.log_file.close()

        print(footer_text)
        print(f"[LOG] Log saved to: {self.log_path}")

        self._setup_done = False


# Global logger instance
_logger = None


def get_logger() -> TrainingLogger:
    global _logger
    return _logger


def run_sft(cfg: dict, run_dir: str) -> dict:
    """Run SFT training and return checkpoint paths."""
    from ppopt.training.sft_trainer import SFTTrainer, SFTConfig

    print("=" * 60)
    print("Stage 1: SFT Training")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {run_dir}")
    print()

    api_cfg = cfg.get("api", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    sft_cfg = cfg.get("sft", {})

    # Log config
    print("[CONFIG] SFT Configuration:")
    print(f"  Base Model: {model_cfg['base_model']}")
    print(f"  LoRA Rank: {model_cfg.get('lora_rank', 32)}")
    print(f"  Train File: {data_cfg['sft_train_file']}")
    print(f"  Batch Size: {sft_cfg.get('batch_size', 4)}")
    print(f"  Epochs: {sft_cfg.get('num_epochs', 1)}")
    print(f"  Learning Rate: {sft_cfg.get('learning_rate', 1e-4)}")
    print()

    trainer_cfg = SFTConfig(
        train_file=data_cfg["sft_train_file"],
        eval_file=data_cfg.get("sft_eval_file"),
        max_seq_length=data_cfg.get("max_seq_length", 4096),
        batch_size=sft_cfg.get("batch_size", 4),
        num_epochs=sft_cfg.get("num_epochs", 1),
        learning_rate=sft_cfg.get("learning_rate", 1e-4),
        min_learning_rate=sft_cfg.get("min_learning_rate", 1e-6),
        warmup_steps=sft_cfg.get("warmup_steps", 100),
        weight_decay=sft_cfg.get("weight_decay", 0.0),
        grad_clip_norm=sft_cfg.get("grad_clip_norm", 0.0),
        gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 1),
        system_prompt=sft_cfg.get("system_prompt"),
        base_model=model_cfg["base_model"],
        lora_rank=model_cfg.get("lora_rank", 32),
        chat_template=model_cfg.get("chat_template"),
        output_dir=run_dir,  # Use run directory
        checkpoint_name=cfg.get("checkpoint", {}).get("sft_name", "sft_checkpoint"),
    )

    print("[INFO] Initializing SFT trainer...")
    trainer = SFTTrainer(
        trainer_cfg,
        api_key=api_cfg.get("tinker_api_key"),
        base_url=api_cfg.get("tinker_base_url"),
    )

    print("[INFO] Setting up Tinker client...")
    trainer.setup()

    print("[INFO] Starting SFT training...")
    result = trainer.train()

    print()
    print(f"[SUCCESS] SFT training completed!")
    print(f"  State checkpoint: {result['state_path']}")
    print(f"  Sampler checkpoint: {result['sampler_path']}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return result


def run_rl(cfg: dict, sft_state_path: str, sft_sampler_path: str, run_dir: str) -> dict:
    """Run RL training using SFT checkpoint paths."""
    from ppopt.training.rl_trainer import RLTrainer, RLConfig

    print()
    print("=" * 60)
    print("Stage 2: RL (GRPO) Training")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {run_dir}")
    print()
    print(f"[INFO] Using SFT checkpoints:")
    print(f"  State path: {sft_state_path}")
    print(f"  Sampler path: {sft_sampler_path}")
    print()

    api_cfg = cfg.get("api", {})
    openai_cfg = cfg.get("openai", {})
    openrouter_cfg = cfg.get("openrouter", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    rl_cfg = cfg.get("rl", {})
    assistant_cfg = cfg.get("assistant", {})
    judge_cfg = cfg.get("judge", {})

    # Log config
    print("[CONFIG] RL Configuration:")
    print(f"  Base Model: {model_cfg['base_model']}")
    print(f"  Train File: {data_cfg['rl_train_file']}")
    print(f"  Batch Size: {rl_cfg.get('batch_size', 1)}")
    print(f"  Epochs: {rl_cfg.get('num_epochs', 1)}")
    print(f"  Learning Rate: {rl_cfg.get('learning_rate', 1e-6)}")
    print(f"  Clip Epsilon: {rl_cfg.get('clip_eps', 0.2)}")
    print(f"  KL Coefficient: {rl_cfg.get('kl_coef', 0.1)}")
    print(f"  Rollouts per State: {rl_cfg.get('rollouts_per_state', 4)}")
    print(f"  Lambda Prof: {rl_cfg.get('lambda_prof', 0.3)}")
    print(f"  Lambda Task: {rl_cfg.get('lambda_task', 0.7)}")
    print(f"  Log Interactions: {rl_cfg.get('log_interactions', True)}")
    print()

    trainer_cfg = RLConfig(
        train_file=data_cfg["rl_train_file"],
        batch_size=rl_cfg.get("batch_size", 1),
        num_epochs=rl_cfg.get("num_epochs", 1),
        learning_rate=rl_cfg.get("learning_rate", 1e-6),
        weight_decay=rl_cfg.get("weight_decay", 0.0),
        grad_clip_norm=rl_cfg.get("grad_clip_norm", 0.0),
        clip_eps=rl_cfg.get("clip_eps", 0.2),
        kl_coef=rl_cfg.get("kl_coef", 0.1),
        rollouts_per_state=rl_cfg.get("rollouts_per_state", 4),
        max_new_tokens=rl_cfg.get("max_new_tokens", 512),
        temperature=rl_cfg.get("temperature", 0.7),
        top_p=rl_cfg.get("top_p", 0.9),
        base_model=model_cfg["base_model"],
        lora_rank=model_cfg.get("lora_rank", 32),
        chat_template=model_cfg.get("chat_template"),
        system_prompt=rl_cfg.get("system_prompt"),
        output_dir=run_dir,  # Use run directory
        checkpoint_name=cfg.get("checkpoint", {}).get("rl_name", "rl_checkpoint"),
        # Use SFT paths from previous stage (override config)
        sft_checkpoint_path=sft_state_path,
        sft_sampler_path=sft_sampler_path,
        # OpenRouter config for assistant
        openrouter_api_key=openrouter_cfg.get("api_key", ""),
        openrouter_base_url=openrouter_cfg.get("base_url", "https://openrouter.ai/api/v1"),
        openrouter_model=openrouter_cfg.get("model", "anthropic/claude-3-haiku"),
        openrouter_max_tokens=openrouter_cfg.get("max_tokens", 512),
        openrouter_temperature=openrouter_cfg.get("temperature", 0.7),
        openrouter_top_p=openrouter_cfg.get("top_p", 0.9),
        # Assistant system prompt
        assistant_system_prompt=assistant_cfg.get("system_prompt", "You are a helpful assistant."),
        # Judge config
        judge_model=judge_cfg.get("model", model_cfg["base_model"]),
        judge_max_tokens=judge_cfg.get("max_tokens", 128),
        judge_temperature=judge_cfg.get("temperature", 0.0),
        judge_top_p=judge_cfg.get("top_p", 1.0),
        lambda_prof=rl_cfg.get("lambda_prof", 0.3),
        lambda_task=rl_cfg.get("lambda_task", 0.7),
        openai_api_key=openai_cfg.get("api_key", ""),
        openai_base_url=openai_cfg.get("base_url", "https://api.openai.com/v1"),
        # Interaction logging config
        log_interactions=rl_cfg.get("log_interactions", True),
        log_interval=rl_cfg.get("log_interval", 1),
        interactions_log_file=rl_cfg.get("interactions_log_file", "grpo_interactions.jsonl"),
        # Concurrency / pipeline config
        pipeline=rl_cfg.get("pipeline", True),
        assistant_concurrency=rl_cfg.get("assistant_concurrency", 4),
        judge_concurrency=rl_cfg.get("judge_concurrency", 4),
        prefetch_logprobs=rl_cfg.get("prefetch_logprobs", True),
        judge_parallel_profile_task=rl_cfg.get("judge_parallel_profile_task", True),
        # Interaction logging performance
        log_async=rl_cfg.get("log_async", False),
        log_buffer_size=rl_cfg.get("log_buffer_size", 1),
        log_flush_interval=rl_cfg.get("log_flush_interval", 0.0),
        log_fields=rl_cfg.get("log_fields"),
        log_state_fields=rl_cfg.get("log_state_fields"),
        log_rollout_fields=rl_cfg.get("log_rollout_fields"),
        # API fault tolerance
        api_max_retries=rl_cfg.get("api_max_retries", 3),
        api_retry_base_delay=rl_cfg.get("api_retry_base_delay", 1.0),
        api_fail_threshold=rl_cfg.get("api_fail_threshold", 0.8),
        api_fail_window=rl_cfg.get("api_fail_window", 5),
    )

    print("[INFO] Initializing RL trainer...")
    trainer = RLTrainer(
        trainer_cfg,
        api_key=api_cfg.get("tinker_api_key"),
        base_url=api_cfg.get("tinker_base_url"),
    )

    print("[INFO] Setting up Tinker client and loading SFT checkpoint...")
    trainer.setup()

    print("[INFO] Starting RL (GRPO) training...")
    result = trainer.train()

    print()
    print(f"[SUCCESS] RL training completed!")
    print(f"  State checkpoint (for training): {result['state_path']}")
    print(f"  Sampler checkpoint (for inference): {result['sampler_path']}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save checkpoint paths for automatic inference
    ckpt_file = Path(__file__).resolve().parent.parent / "latest_checkpoint.json"
    with open(ckpt_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Checkpoint paths saved to: {ckpt_file}")

    return result


def main():
    global _logger

    parser = argparse.ArgumentParser(description="One-click SFT -> RL training pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--sft-only", action="store_true", help="Only run SFT training")
    parser.add_argument("--rl-only", action="store_true", help="Only run RL training (requires sft paths in config)")
    parser.add_argument("--log-dir", default="logs", help="Directory to save log files")
    parser.add_argument("--no-log", action="store_true", help="Disable logging to file")
    args = parser.parse_args()

    # Setup logging
    if not args.no_log:
        _logger = TrainingLogger(log_dir=args.log_dir)
        _logger.setup(sys.argv)

    try:
        from ppopt.utils.config import load_config, create_run_dir

        print(f"[INFO] Loading config from: {args.config}")
        cfg = load_config(args.config)
        print(f"[INFO] Config loaded successfully")
        print()

        # Create timestamped run directory for this training session
        base_output_dir = cfg.get("checkpoint", {}).get("output_dir", "checkpoints")
        run_dir = str(create_run_dir(base_output_dir, prefix="run"))
        print(f"[INFO] Created run directory: {run_dir}")
        print()

        # Determine training mode from args and config
        # Priority: command line args > config file
        skip_rl = args.sft_only or cfg.get("pipeline", {}).get("skip_rl", False)

        if args.rl_only:
            # Use paths from config
            rl_cfg = cfg.get("rl", {})
            sft_state_path = rl_cfg["sft_checkpoint_path"]
            sft_sampler_path = rl_cfg["sft_sampler_path"]
            print(f"[INFO] RL-only mode: using paths from config")
            rl_result = run_rl(cfg, sft_state_path, sft_sampler_path, run_dir)

            print()
            print("=" * 60)
            print("[COMPLETE] RL Training Finished")
            print("=" * 60)
            print(f"  Run directory: {run_dir}")
            print(f"  RL checkpoint (state): {rl_result['state_path']}")
            print(f"  RL checkpoint (sampler): {rl_result['sampler_path']}")

        elif skip_rl:
            # SFT only (from --sft-only flag or config pipeline.skip_rl)
            print(f"[INFO] SFT-only mode (skip_rl={skip_rl})")
            sft_result = run_sft(cfg, run_dir)

            # Save SFT checkpoint paths for later use
            ckpt_file = Path(__file__).resolve().parent.parent / "latest_checkpoint.json"
            with open(ckpt_file, "w") as f:
                json.dump(sft_result, f, indent=2)
            print(f"  Checkpoint paths saved to: {ckpt_file}")

            print()
            print("=" * 60)
            print("[COMPLETE] SFT Training Finished")
            print("=" * 60)
            print(f"  Run directory: {run_dir}")
            print(f"  SFT checkpoint (state): {sft_result['state_path']}")
            print(f"  SFT checkpoint (sampler): {sft_result['sampler_path']}")
            print()
            print("To run RL later, update config.yaml with:")
            print(f"  sft_checkpoint_path: \"{sft_result['state_path']}\"")
            print(f"  sft_sampler_path: \"{sft_result['sampler_path']}\"")
            print("Then run: python scripts/train_pipeline.py --rl-only")

        else:
            # Full pipeline: SFT -> RL
            sft_result = run_sft(cfg, run_dir)
            rl_result = run_rl(cfg, sft_result["state_path"], sft_result["sampler_path"], run_dir)

            print()
            print("=" * 60)
            print("[COMPLETE] Full Training Pipeline Finished")
            print("=" * 60)
            print(f"  Run directory: {run_dir}")
            print(f"  SFT checkpoint (state): {sft_result['state_path']}")
            print(f"  SFT checkpoint (sampler): {sft_result['sampler_path']}")
            print(f"  RL checkpoint (state): {rl_result['state_path']}")
            print(f"  RL checkpoint (sampler): {rl_result['sampler_path']}")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training interrupted by user (Ctrl+C)")
        raise

    except Exception as e:
        if _logger:
            _logger.log_exception(type(e), e, e.__traceback__)
        else:
            traceback.print_exc()
        raise

    finally:
        if _logger:
            _logger.cleanup()


if __name__ == "__main__":
    main()
