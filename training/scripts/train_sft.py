import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ppopt.training.sft_trainer import SFTTrainer, SFTConfig
from ppopt.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    api_cfg = cfg.get("api", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    sft_cfg = cfg.get("sft", {})
    ckpt_cfg = cfg.get("checkpoint", {})

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
        output_dir=ckpt_cfg.get("output_dir", "checkpoints"),
        checkpoint_name=ckpt_cfg.get("sft_name", "sft_checkpoint"),
    )

    trainer = SFTTrainer(
        trainer_cfg,
        api_key=api_cfg.get("tinker_api_key"),
        base_url=api_cfg.get("tinker_base_url"),
    )
    trainer.setup()
    result = trainer.train()
    print(f"SFT training completed!")
    print(f"  State checkpoint (for RL training): {result['state_path']}")
    print(f"  Sampler checkpoint (for reference model): {result['sampler_path']}")
    print()
    print("Update config.yaml with these paths before running RL training:")
    print(f"  sft_checkpoint_path: \"{result['state_path']}\"")
    print(f"  sft_sampler_path: \"{result['sampler_path']}\"")


if __name__ == "__main__":
    main()
