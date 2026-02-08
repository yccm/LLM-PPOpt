"""SFT trainer using Tinker TrainingClient."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import math
import os

import numpy as np
import torch

from ppopt.training.metrics import create_sft_tracker, MetricsTracker
from ppopt.utils.tinker_compat import ensure_tinker_compat
ensure_tinker_compat()

import tinker
from tinker import types

from ppopt.utils.chat import get_chat_template, build_io_prompt


DEFAULT_SYSTEM_PROMPT = (
    "You are a personalized prompt optimizer. Your task is to rewrite a user's query so that "
    "a general-purpose assistant produces a response better aligned with the user's latent "
    "preferences, style, and constraints.\n\n"
    "## Input\n"
    "You will receive:\n"
    "- [HISTORY]: Past interaction sessions between the user and an assistant. Each session "
    "contains the user's queries, the assistant's responses, and optionally the user's "
    "follow-up feedback. Use these to infer the user's preferences.\n"
    "- [CURRENT_QUERY]: The user's new query that you need to optimize.\n\n"
    "## Output\n"
    "You must output exactly two sections:\n"
    "1. <REASONING>: Analyze the interaction history to infer the user's profile, including "
    "communication style (e.g., formal/casual, concise/verbose), domain expertise level, "
    "formatting preferences, content constraints, and any latent priorities or trade-offs.\n"
    "2. <PROMPT>: Rewrite the current query incorporating the inferred preferences. The "
    "rewritten prompt should preserve the original intent while adding context, constraints, "
    "or style cues so that the assistant's response better satisfies this specific user.\n\n"
    "Output format:\n"
    "<REASONING>your analysis of user preferences</REASONING>"
    "<PROMPT>the optimized query</PROMPT>"
)


@dataclass
class SFTConfig:
    train_file: str
    eval_file: str | None
    max_seq_length: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float
    grad_clip_norm: float
    gradient_accumulation_steps: int
    system_prompt: str
    base_model: str
    lora_rank: int
    chat_template: str | None
    output_dir: str
    checkpoint_name: str


class SFTTrainer:
    def __init__(self, cfg: SFTConfig, api_key: str | None, base_url: str | None):
        self.cfg = cfg
        self.api_key = api_key
        self.base_url = base_url
        self.client: tinker.TrainingClient | None = None
        self.tokenizer = None
        self.chat_template = get_chat_template(cfg.base_model, cfg.chat_template)
        self._metrics_tracker: MetricsTracker | None = None

    def setup(self) -> None:
        if self.api_key:
            os.environ["TINKER_API_KEY"] = self.api_key
        if self.base_url:
            os.environ["TINKER_BASE_URL"] = self.base_url
        service = tinker.ServiceClient(api_key=self.api_key, base_url=self.base_url)
        self.client = service.create_lora_training_client(
            base_model=self.cfg.base_model,
            rank=self.cfg.lora_rank,
        )
        self.tokenizer = self.client.get_tokenizer()
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        # Initialize metrics tracker for loss curve plotting
        self._metrics_tracker = create_sft_tracker(self.cfg.output_dir, plot_interval=50)

    def load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items

    def _truncate_prompt(self, prompt_tokens: List[int], completion_tokens: List[int]) -> Tuple[List[int], List[int]]:
        max_len = self.cfg.max_seq_length
        if len(prompt_tokens) + len(completion_tokens) <= max_len:
            return prompt_tokens, completion_tokens
        if len(completion_tokens) >= max_len:
            return [], completion_tokens[-max_len:]
        keep_prompt = max_len - len(completion_tokens)
        return prompt_tokens[-keep_prompt:], completion_tokens

    def tokenize_io_sample(self, sample: Dict[str, Any]) -> types.Datum:
        # 数据中的 input 字段已经包含系统提示，不再额外添加
        user_input = sample["input"]
        output = sample["output"]

        prompt = build_io_prompt(self.chat_template, "", user_input)
        stop_token = self.chat_template["stop"][0] if self.chat_template.get("stop") else ""
        completion = f"{output}{stop_token}"

        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        completion_tokens = self.tokenizer.encode(completion, add_special_tokens=False)
        prompt_tokens, completion_tokens = self._truncate_prompt(prompt_tokens, completion_tokens)

        tokens = prompt_tokens + completion_tokens
        weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
        if len(tokens) < 2:
            raise ValueError("Sample too short after tokenization")

        return types.Datum(
            model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
            loss_fn_inputs={
                "weights": torch.tensor(weights[1:], dtype=torch.float32),
                "target_tokens": torch.tensor(tokens[1:], dtype=torch.int64),
            },
        )

    def _lr(self, step: int, total: int) -> float:
        if step < self.cfg.warmup_steps:
            return self.cfg.learning_rate * step / max(1, self.cfg.warmup_steps)
        progress = (step - self.cfg.warmup_steps) / max(1, total - self.cfg.warmup_steps)
        return self.cfg.min_learning_rate + (self.cfg.learning_rate - self.cfg.min_learning_rate) * 0.5 * (1 + math.cos(math.pi * progress))

    def train(self) -> dict:
        """Train and return checkpoint paths.

        Returns:
            dict with keys:
                - 'state_path': checkpoint path with optimizer state (for RL training continuation)
                - 'sampler_path': checkpoint path for sampling (for reference model in RL)
        """
        assert self.client is not None
        train_items = self.load_jsonl(self.cfg.train_file)
        train_data = [self.tokenize_io_sample(x) for x in train_items]

        total_steps = math.ceil(len(train_data) / self.cfg.batch_size) * self.cfg.num_epochs
        step = 0

        for epoch in range(self.cfg.num_epochs):
            np.random.shuffle(train_data)
            for i in range(0, len(train_data), self.cfg.batch_size):
                batch = train_data[i:i + self.cfg.batch_size]
                lr = self._lr(step, total_steps)

                # Calculate total tokens in batch (for converting sum loss to mean)
                batch_num_tokens = 0
                for datum in batch:
                    weights = datum.loss_fn_inputs.get("weights")
                    if weights is not None:
                        # weights could be torch.Tensor or TensorData
                        if hasattr(weights, 'sum'):
                            batch_num_tokens += float(weights.sum())
                        elif hasattr(weights, 'to_numpy'):
                            batch_num_tokens += float(weights.to_numpy().sum())
                        elif hasattr(weights, 'data'):
                            data = weights.data
                            if isinstance(data, list):
                                batch_num_tokens += sum(data)
                            else:
                                batch_num_tokens += float(data)

                fwdbwd = self.client.forward_backward(batch, "cross_entropy")
                fwdbwd_result = fwdbwd.result()
                opt = self.client.optim_step(
                    types.AdamParams(
                        learning_rate=lr,
                        beta1=0.9,
                        beta2=0.95,
                        eps=1e-12,
                        weight_decay=self.cfg.weight_decay,
                        grad_clip_norm=self.cfg.grad_clip_norm,
                    )
                )
                _ = opt.result()
                step += 1

                # Record loss metric from ForwardBackwardOutput
                if self._metrics_tracker is not None:
                    loss_value = None
                    loss_key_used = None
                    is_sum_loss = False

                    # Method 1: Try metrics dict with possible suffix variations
                    if hasattr(fwdbwd_result, 'metrics') and isinstance(fwdbwd_result.metrics, dict):
                        metrics = fwdbwd_result.metrics
                        # Debug: print available keys on first step
                        if step == 1:
                            print(f"[DEBUG] Tinker metrics keys available: {list(metrics.keys())}")
                        # Tinker may add suffix like ':mean' or ':sum' to metric names
                        # Prioritize mean over sum to get per-token average loss
                        for key in ['loss:mean', 'cross_entropy:mean']:
                            if key in metrics:
                                loss_value = metrics[key]
                                loss_key_used = key
                                is_sum_loss = False
                                break
                        # If no mean available, try sum and convert to mean
                        if loss_value is None:
                            for key in ['loss:sum', 'cross_entropy:sum', 'loss', 'cross_entropy']:
                                if key in metrics:
                                    loss_value = metrics[key]
                                    loss_key_used = key
                                    is_sum_loss = ':sum' in key or key in ['loss', 'cross_entropy']
                                    break

                    # Convert sum loss to mean loss
                    if loss_value is not None and is_sum_loss and batch_num_tokens > 0:
                        mean_loss = loss_value / batch_num_tokens
                        if step == 1:
                            print(f"[DEBUG] Converting sum loss to mean: {loss_value} / {batch_num_tokens} = {mean_loss:.4f}")
                        loss_value = mean_loss

                    # Method 2: Try loss_fn_outputs list (List[Dict[str, TensorData]])
                    if loss_value is None and hasattr(fwdbwd_result, 'loss_fn_outputs'):
                        outputs = fwdbwd_result.loss_fn_outputs
                        if outputs and isinstance(outputs, list):
                            losses = []
                            for out_dict in outputs:
                                if isinstance(out_dict, dict):
                                    # Look for 'loss' key in each output dict
                                    for key in ['loss', 'cross_entropy']:
                                        if key in out_dict:
                                            tensor_data = out_dict[key]
                                            # TensorData has .data attribute or .to_numpy()/.tolist()
                                            if hasattr(tensor_data, 'to_numpy'):
                                                arr = tensor_data.to_numpy()
                                                losses.append(float(arr.mean()) if hasattr(arr, 'mean') else float(arr))
                                            elif hasattr(tensor_data, 'tolist'):
                                                val = tensor_data.tolist()
                                                if isinstance(val, list):
                                                    losses.append(float(np.mean(val)))
                                                else:
                                                    losses.append(float(val))
                                            elif hasattr(tensor_data, 'data'):
                                                data = tensor_data.data
                                                if isinstance(data, list):
                                                    losses.append(float(np.mean(data)))
                                                else:
                                                    losses.append(float(data))
                                            break
                            if losses:
                                loss_value = float(np.mean(losses))

                    if loss_value is not None:
                        if step == 1:
                            print(f"[DEBUG] Final loss value: {loss_value:.4f} (key: '{loss_key_used}')")
                        self._metrics_tracker.record(step=step, loss=float(loss_value))
                    elif step == 1:
                        # Debug: print all available attributes on first step
                        attrs = [a for a in dir(fwdbwd_result) if not a.startswith('_')]
                        print(f"[WARNING] Could not extract loss from forward_backward result.")
                        print(f"  Result type: {type(fwdbwd_result)}")
                        print(f"  Available attributes: {attrs}")
                        if hasattr(fwdbwd_result, 'metrics'):
                            print(f"  metrics: {fwdbwd_result.metrics}")
                        if hasattr(fwdbwd_result, 'loss_fn_outputs'):
                            print(f"  loss_fn_outputs: {fwdbwd_result.loss_fn_outputs[:2] if fwdbwd_result.loss_fn_outputs else 'empty'}")

        # Close metrics tracker and save final plots
        if self._metrics_tracker is not None:
            self._metrics_tracker.close()

        # Save checkpoint with optimizer state (for RL training continuation)
        save_name = f"{self.cfg.checkpoint_name}"
        save_future = self.client.save_state(save_name)
        save_resp = save_future.result()
        state_path = save_resp.path

        # Save checkpoint for sampling (for reference model in RL)
        sampler_save = self.client.save_weights_for_sampler(f"{self.cfg.checkpoint_name}_sampler")
        sampler_resp = sampler_save.result()
        sampler_path = sampler_resp.path

        return {
            "state_path": state_path,
            "sampler_path": sampler_path,
        }
