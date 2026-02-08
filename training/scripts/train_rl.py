import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ppopt.training.rl_trainer import RLTrainer, RLConfig
from ppopt.utils.config import load_config, create_run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    api_cfg = cfg.get("api", {})
    openai_cfg = cfg.get("openai", {})
    openrouter_cfg = cfg.get("openrouter", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    rl_cfg = cfg.get("rl", {})
    ckpt_cfg = cfg.get("checkpoint", {})
    assistant_cfg = cfg.get("assistant", {})
    judge_cfg = cfg.get("judge", {})

    # Create timestamped run directory
    base_output_dir = ckpt_cfg.get("output_dir", "checkpoints")
    run_dir = str(create_run_dir(base_output_dir, prefix="run"))
    print(f"[INFO] Created run directory: {run_dir}")

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
        checkpoint_name=ckpt_cfg.get("rl_name", "rl_checkpoint"),
        sft_checkpoint_path=rl_cfg["sft_checkpoint_path"],
        sft_sampler_path=rl_cfg["sft_sampler_path"],
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
        use_profile_reward=rl_cfg.get("use_profile_reward", True),
        openai_api_key=openai_cfg.get("api_key", ""),
        openai_base_url=openai_cfg.get("base_url", "https://api.openai.com/v1"),
        # Interaction logging
        log_interactions=rl_cfg.get("log_interactions", True),
        log_interval=rl_cfg.get("log_interval", 1),
        interactions_log_file=rl_cfg.get("interactions_log_file", "grpo_interactions.jsonl"),
        # Concurrency / pipeline
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
        # Checkpoint save interval
        sampler_save_interval=rl_cfg.get("sampler_save_interval", 200),
    )

    trainer = RLTrainer(
        trainer_cfg,
        api_key=api_cfg.get("tinker_api_key"),
        base_url=api_cfg.get("tinker_base_url"),
    )
    trainer.setup()
    result = trainer.train()
    print(f"RL training completed!")
    print(f"  Run directory: {run_dir}")
    print(f"  State checkpoint (for training continuation): {result['state_path']}")
    print(f"  Sampler checkpoint (for inference): {result['sampler_path']}")

    # Save checkpoint paths for automatic inference
    ckpt_file = Path(__file__).resolve().parent.parent / "latest_checkpoint.json"
    with open(ckpt_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Checkpoint paths saved to: {ckpt_file}")


if __name__ == "__main__":
    main()
