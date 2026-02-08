"""GRPO (Group Relative Policy Optimization) trainer using Tinker TrainingClient."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import asyncio
import json
import math
import os
import queue
import random
import logging
import threading
import time

import numpy as np
import torch

from ppopt.utils.tinker_compat import ensure_tinker_compat
ensure_tinker_compat()

import tinker
from tinker import types

from ppopt.utils.chat import get_chat_template, build_io_prompt
from ppopt.utils.formatting import parse_optimizer_output, persona_to_text, safe_parse_score
from ppopt.training.metrics import create_rl_tracker, MetricsTracker
from ppopt.training.rewards import (
    RewardComputer,
    RewardConfig,
    PROFILE_JUDGE_SYSTEM,
    PROFILE_JUDGE_USER,
    TASK_JUDGE_SYSTEM,
    TASK_JUDGE_USER,
)

logger = logging.getLogger(__name__)


def _extract_status_code(exc: Exception) -> str:
    """Extract HTTP status code from API exceptions (e.g., '429', '502')."""
    code = getattr(exc, "status_code", None)
    if code is not None:
        return str(code)
    resp = getattr(exc, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        if code is not None:
            return str(code)
    return ""


class InteractionLogger:
    """Buffered/async JSONL logger for GRPO interaction records."""

    def __init__(
        self,
        path: Path,
        async_write: bool = False,
        buffer_size: int = 1,
        flush_interval: float = 0.0,
        fields: List[str] | None = None,
        state_fields: List[str] | None = None,
        rollout_fields: List[str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", encoding="utf-8")
        self._async = async_write
        self._buffer_size = max(1, int(buffer_size))
        self._flush_interval = max(0.0, float(flush_interval))
        self._fields = fields
        self._state_fields = state_fields
        self._rollout_fields = rollout_fields
        self._buffer: List[Dict[str, Any]] = []
        self._last_flush = time.monotonic()
        self._sentinel = object()
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None

        if self._async:
            self._queue = queue.Queue()
            self._thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._thread.start()

    def _filter_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if self._fields:
            filtered = {k: record.get(k) for k in self._fields if k in record}
        else:
            filtered = dict(record)

        if "states" in filtered and isinstance(filtered["states"], list):
            states = []
            for state in filtered["states"]:
                if not isinstance(state, dict):
                    continue
                if self._state_fields:
                    state_out = {k: state.get(k) for k in self._state_fields if k in state}
                else:
                    state_out = dict(state)
                if "rollouts" in state_out and isinstance(state_out["rollouts"], list) and self._rollout_fields:
                    rollouts = []
                    for roll in state_out["rollouts"]:
                        if not isinstance(roll, dict):
                            continue
                        rollouts.append({k: roll.get(k) for k in self._rollout_fields if k in roll})
                    state_out["rollouts"] = rollouts
                states.append(state_out)
            filtered["states"] = states

        return filtered

    def write(self, record: Dict[str, Any]) -> None:
        record = self._filter_record(record)
        if self._async and self._queue is not None:
            self._queue.put(record)
            return
        self._buffer.append(record)
        self._maybe_flush()

    def _maybe_flush(self) -> None:
        now = time.monotonic()
        if len(self._buffer) >= self._buffer_size:
            self._flush()
        elif self._flush_interval > 0.0 and (now - self._last_flush) >= self._flush_interval:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        for record in self._buffer:
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
        self._buffer.clear()
        self._last_flush = time.monotonic()

    def _writer_loop(self) -> None:
        buffer: List[Dict[str, Any]] = []
        last_flush = time.monotonic()
        while True:
            try:
                item = self._queue.get(timeout=0.1)  # type: ignore[union-attr]
            except queue.Empty:
                item = None
            if item is self._sentinel:
                break
            if item is not None:
                buffer.append(item)
            now = time.monotonic()
            if buffer and (len(buffer) >= self._buffer_size or (
                self._flush_interval > 0.0 and (now - last_flush) >= self._flush_interval
            )):
                for record in buffer:
                    self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
                self._file.flush()
                buffer.clear()
                last_flush = now
        if buffer:
            for record in buffer:
                self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._async and self._queue is not None and self._thread is not None:
            self._queue.put(self._sentinel)
            self._thread.join()
        else:
            self._flush()
        self._file.close()

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
class RLConfig:
    train_file: str
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    clip_eps: float
    kl_coef: float
    rollouts_per_state: int
    max_new_tokens: int
    temperature: float
    top_p: float
    base_model: str
    lora_rank: int
    chat_template: str | None
    system_prompt: str
    output_dir: str
    checkpoint_name: str
    sft_checkpoint_path: str  # For training continuation (with optimizer state)
    sft_sampler_path: str     # For reference model sampling (frozen SFT weights)
    # OpenRouter config for assistant
    openrouter_api_key: str
    openrouter_base_url: str
    openrouter_model: str
    openrouter_max_tokens: int
    openrouter_temperature: float
    openrouter_top_p: float
    # Assistant system prompt
    assistant_system_prompt: str
    # Judge config
    judge_model: str
    judge_max_tokens: int
    judge_temperature: float
    judge_top_p: float
    lambda_prof: float
    lambda_task: float
    openai_api_key: str
    openai_base_url: str
    # Reward config
    use_profile_reward: bool = True  # false = only task reward, true = combined reward
    # Interaction logging config
    log_interactions: bool = True  # Whether to log detailed GRPO interactions
    log_interval: int = 1  # Log every N steps (1 = log all)
    interactions_log_file: str = "grpo_interactions.jsonl"  # Log file name
    # Concurrency / pipeline config
    pipeline: bool = True  # Enable async pipeline for assistant/judge/logprobs
    assistant_concurrency: int = 4  # Max concurrent assistant calls per state
    judge_concurrency: int = 4  # Max concurrent judge calls per state
    prefetch_logprobs: bool = True  # Start logprob requests before rewards finish
    judge_parallel_profile_task: bool = True  # Parallelize profile/task judge within rollout
    # Interaction logging performance
    log_async: bool = False  # Write logs on a background thread
    log_buffer_size: int = 1  # Flush after N records
    log_flush_interval: float = 0.0  # Flush after T seconds (0 = disabled)
    log_fields: List[str] | None = None  # Top-level fields to keep
    log_state_fields: List[str] | None = None  # Per-state fields to keep
    log_rollout_fields: List[str] | None = None  # Per-rollout fields to keep
    # API fault tolerance config
    api_max_retries: int = 3  # Max retries per API call
    api_retry_base_delay: float = 1.0  # Base delay (seconds) for exponential backoff
    api_fail_threshold: float = 0.8  # Abort training if failure rate exceeds this
    api_fail_window: int = 5  # Sliding window size for failure rate tracking
    # Checkpoint save interval
    sampler_save_interval: int = 200  # Save sampler checkpoint every N steps (0 = disabled)


class RLTrainer:
    """GRPO trainer: samples a group of outputs per state, computes group-relative
    advantages from rewards, and updates the policy with clipped surrogate + KL."""

    def __init__(self, cfg: RLConfig, api_key: str | None, base_url: str | None):
        self.cfg = cfg
        self.api_key = api_key
        self.base_url = base_url
        self.chat_template = get_chat_template(cfg.base_model, cfg.chat_template)
        self.client: tinker.TrainingClient | None = None
        self.tokenizer = None
        self.service: tinker.ServiceClient | None = None
        self.rewarder: RewardComputer | None = None
        self.assistant_client = None  # OpenRouter client for assistant
        self.assistant_client_async = None  # Async OpenRouter client for assistant
        self.judge_client_async = None  # Async OpenAI client for judge
        self._interaction_logger: InteractionLogger | None = None
        self._metrics_tracker: MetricsTracker | None = None
        self._api_fail_history: List[float] = []  # Per-step failure rates

    def setup(self) -> None:
        if self.api_key:
            os.environ["TINKER_API_KEY"] = self.api_key
        if self.base_url:
            os.environ["TINKER_BASE_URL"] = self.base_url
        self.service = tinker.ServiceClient(api_key=self.api_key, base_url=self.base_url)
        # Load SFT checkpoint as starting point
        self.client = self.service.create_training_client_from_state_with_optimizer(
            self.cfg.sft_checkpoint_path
        )
        self.tokenizer = self.client.get_tokenizer()
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

        # Setup interaction log file
        self._interaction_logger = None
        if self.cfg.log_interactions:
            log_path = Path(self.cfg.output_dir) / self.cfg.interactions_log_file
            self._interaction_logger = InteractionLogger(
                log_path,
                async_write=self.cfg.log_async,
                buffer_size=self.cfg.log_buffer_size,
                flush_interval=self.cfg.log_flush_interval,
                fields=self.cfg.log_fields,
                state_fields=self.cfg.log_state_fields,
                rollout_fields=self.cfg.log_rollout_fields,
            )
            logger.info("GRPO interaction log: %s", log_path)

        # Initialize metrics tracker for loss/reward curve plotting
        self._metrics_tracker = create_rl_tracker(self.cfg.output_dir, plot_interval=50)

        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("OpenAI SDK not installed. Please install 'openai'.") from exc
        try:
            from openai import AsyncOpenAI
        except Exception:
            AsyncOpenAI = None

        # Initialize OpenRouter client for assistant (OpenAI-compatible API)
        if not self.cfg.openrouter_api_key:
            raise RuntimeError("OpenRouter API key missing in config (openrouter.api_key).")
        self.assistant_client = OpenAI(
            api_key=self.cfg.openrouter_api_key,
            base_url=self.cfg.openrouter_base_url,
        )
        if self.cfg.pipeline and AsyncOpenAI is not None:
            try:
                self.assistant_client_async = AsyncOpenAI(
                    api_key=self.cfg.openrouter_api_key,
                    base_url=self.cfg.openrouter_base_url,
                )
            except Exception as exc:
                logger.warning("Async assistant client init failed, falling back to sync: %s", exc)
                self.assistant_client_async = None
        elif self.cfg.pipeline and AsyncOpenAI is None:
            logger.warning("AsyncOpenAI not available; assistant calls will run in threads.")

        # Initialize OpenAI client for judge/reward
        if not self.cfg.openai_api_key:
            raise RuntimeError("OpenAI API key missing in config (openai.api_key).")
        openai_client = OpenAI(
            api_key=self.cfg.openai_api_key,
            base_url=self.cfg.openai_base_url,
        )
        if self.cfg.pipeline and AsyncOpenAI is not None:
            try:
                self.judge_client_async = AsyncOpenAI(
                    api_key=self.cfg.openai_api_key,
                    base_url=self.cfg.openai_base_url,
                )
            except Exception as exc:
                logger.warning("Async judge client init failed, falling back to sync: %s", exc)
                self.judge_client_async = None
        elif self.cfg.pipeline and AsyncOpenAI is None:
            logger.warning("AsyncOpenAI not available; judge calls will run in threads.")
        self.rewarder = RewardComputer(
            openai_client,
            RewardConfig(
                lambda_prof=self.cfg.lambda_prof,
                lambda_task=self.cfg.lambda_task,
                max_output_tokens=self.cfg.judge_max_tokens,
                temperature=self.cfg.judge_temperature,
                top_p=self.cfg.judge_top_p,
                model=self.cfg.judge_model,
                max_retries=self.cfg.api_max_retries,
                retry_base_delay=self.cfg.api_retry_base_delay,
                use_profile_reward=self.cfg.use_profile_reward,
            ),
        )

    def load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items

    def _log_interaction(self, record: Dict[str, Any]) -> None:
        """Write a single interaction record to the JSONL log."""
        if self._interaction_logger is not None:
            self._interaction_logger.write(record)

    def _call_assistant(self, query: str) -> str:
        """Call OpenRouter API to get assistant response with retry."""
        messages = [
            {"role": "system", "content": self.cfg.assistant_system_prompt},
            {"role": "user", "content": query},
        ]
        last_exc: Exception | None = None
        for attempt in range(1, self.cfg.api_max_retries + 1):
            try:
                response = self.assistant_client.chat.completions.create(
                    model=self.cfg.openrouter_model,
                    messages=messages,
                    max_tokens=self.cfg.openrouter_max_tokens,
                    temperature=self.cfg.openrouter_temperature,
                    top_p=self.cfg.openrouter_top_p,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_exc = e
                status = _extract_status_code(e)
                status_str = f" HTTP {status}" if status else ""
                if attempt < self.cfg.api_max_retries:
                    delay = self.cfg.api_retry_base_delay * (2 ** (attempt - 1))
                    print(f"[Assistant] Retry {attempt}/{self.cfg.api_max_retries}{status_str} {type(e).__name__}: {e} (wait {delay:.0f}s)")
                    time.sleep(delay)
        status = _extract_status_code(last_exc) if last_exc else ""
        status_str = f" HTTP {status}" if status else ""
        print(f"[Assistant] FAILED after {self.cfg.api_max_retries} retries{status_str}: {last_exc}")
        return ""

    async def _call_assistant_async(self, query: str, semaphore: asyncio.Semaphore | None) -> str:
        """Async assistant call with optional concurrency limit and retry."""
        if self.assistant_client_async is None:
            if semaphore is None:
                return await asyncio.to_thread(self._call_assistant, query)
            async with semaphore:
                return await asyncio.to_thread(self._call_assistant, query)
        messages = [
            {"role": "system", "content": self.cfg.assistant_system_prompt},
            {"role": "user", "content": query},
        ]
        last_exc: Exception | None = None
        for attempt in range(1, self.cfg.api_max_retries + 1):
            try:
                if semaphore is None:
                    response = await self.assistant_client_async.chat.completions.create(
                        model=self.cfg.openrouter_model,
                        messages=messages,
                        max_tokens=self.cfg.openrouter_max_tokens,
                        temperature=self.cfg.openrouter_temperature,
                        top_p=self.cfg.openrouter_top_p,
                    )
                else:
                    async with semaphore:
                        response = await self.assistant_client_async.chat.completions.create(
                            model=self.cfg.openrouter_model,
                            messages=messages,
                            max_tokens=self.cfg.openrouter_max_tokens,
                            temperature=self.cfg.openrouter_temperature,
                            top_p=self.cfg.openrouter_top_p,
                        )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_exc = e
                status = _extract_status_code(e)
                status_str = f" HTTP {status}" if status else ""
                if attempt < self.cfg.api_max_retries:
                    delay = self.cfg.api_retry_base_delay * (2 ** (attempt - 1))
                    print(f"[Assistant] Retry {attempt}/{self.cfg.api_max_retries}{status_str} {type(e).__name__}: {e} (wait {delay:.0f}s)")
                    await asyncio.sleep(delay)
        status = _extract_status_code(last_exc) if last_exc else ""
        status_str = f" HTTP {status}" if status else ""
        print(f"[Assistant] FAILED after {self.cfg.api_max_retries} retries{status_str}: {last_exc}")
        return ""

    async def _judge_async(
        self,
        system_prompt: str,
        user_prompt: str,
        semaphore: asyncio.Semaphore | None,
        judge_type: str = "unknown",
    ) -> float | None:
        """Async judge call with optional concurrency limit and retry.

        Returns None if all retries are exhausted (API failure).
        """
        if self.judge_client_async is None:
            # Fallback to sync path (which already has retry via RewardComputer._judge)
            if semaphore is None:
                raw = await asyncio.to_thread(self.rewarder._judge, system_prompt, user_prompt)  # type: ignore[union-attr]
            else:
                async with semaphore:
                    raw = await asyncio.to_thread(self.rewarder._judge, system_prompt, user_prompt)  # type: ignore[union-attr]
            if not raw:
                return None
            return safe_parse_score(raw)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_exc: Exception | None = None
        for attempt in range(1, self.cfg.api_max_retries + 1):
            try:
                if semaphore is None:
                    response = await self.judge_client_async.chat.completions.create(
                        model=self.cfg.judge_model,
                        messages=messages,
                        max_tokens=self.cfg.judge_max_tokens,
                        temperature=self.cfg.judge_temperature,
                        top_p=self.cfg.judge_top_p,
                    )
                else:
                    async with semaphore:
                        response = await self.judge_client_async.chat.completions.create(
                            model=self.cfg.judge_model,
                            messages=messages,
                            max_tokens=self.cfg.judge_max_tokens,
                            temperature=self.cfg.judge_temperature,
                            top_p=self.cfg.judge_top_p,
                        )
                text = response.choices[0].message.content or ""
                return safe_parse_score(text)
            except Exception as e:
                last_exc = e
                status = _extract_status_code(e)
                status_str = f" HTTP {status}" if status else ""
                if attempt < self.cfg.api_max_retries:
                    delay = self.cfg.api_retry_base_delay * (2 ** (attempt - 1))
                    print(f"[Judge/{judge_type}] Retry {attempt}/{self.cfg.api_max_retries}{status_str} {type(e).__name__}: {e} (wait {delay:.0f}s)")
                    await asyncio.sleep(delay)
        status = _extract_status_code(last_exc) if last_exc else ""
        status_str = f" HTTP {status}" if status else ""
        print(f"[Judge/{judge_type}] FAILED after {self.cfg.api_max_retries} retries{status_str}: {last_exc}")
        return None

    async def _total_reward_async(
        self,
        inferred_profile: str,
        persona_text: str,
        query: str,
        base: str,
        opt: str,
        semaphore: asyncio.Semaphore | None,
    ) -> Dict[str, Any]:
        if self.judge_client_async is None:
            if semaphore is None:
                return await asyncio.to_thread(
                    self.rewarder.total_reward, inferred_profile, persona_text, query, base, opt  # type: ignore[union-attr]
                )
            async with semaphore:
                return await asyncio.to_thread(
                    self.rewarder.total_reward, inferred_profile, persona_text, query, base, opt  # type: ignore[union-attr]
                )

        task_prompt = TASK_JUDGE_USER.format(query=query, persona=persona_text, base=base, opt=opt)

        if self.cfg.use_profile_reward:
            # Combined reward: profile + task
            profile_prompt = PROFILE_JUDGE_USER.format(profile=inferred_profile, persona=persona_text)

            if self.cfg.judge_parallel_profile_task:
                prof_task = asyncio.create_task(
                    self._judge_async(PROFILE_JUDGE_SYSTEM, profile_prompt, semaphore, judge_type="profile")
                )
                task_task = asyncio.create_task(
                    self._judge_async(TASK_JUDGE_SYSTEM, task_prompt, semaphore, judge_type="task")
                )
                r_prof, r_task = await asyncio.gather(prof_task, task_task)
            else:
                r_prof = await self._judge_async(PROFILE_JUDGE_SYSTEM, profile_prompt, semaphore, judge_type="profile")
                r_task = await self._judge_async(TASK_JUDGE_SYSTEM, task_prompt, semaphore, judge_type="task")

            failed = r_prof is None or r_task is None
            if failed:
                return {"r_prof": 0.0, "r_task": 0.0, "total": 0.0, "failed": True}
            total = self.cfg.lambda_prof * r_prof + self.cfg.lambda_task * r_task
            return {"r_prof": r_prof, "r_task": r_task, "total": total, "failed": False}
        else:
            # Only use task reward
            r_task = await self._judge_async(TASK_JUDGE_SYSTEM, task_prompt, semaphore, judge_type="task")
            if r_task is None:
                return {"r_prof": None, "r_task": 0.0, "total": 0.0, "failed": True}
            return {"r_prof": None, "r_task": r_task, "total": r_task, "failed": False}

    def _sample_text(self, sampler, prompt_text: str, max_tokens: int, temperature: float, top_p: float) -> str:
        prompt_tokens = self.tokenizer.encode(prompt_text)
        params = types.SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        resp = sampler.sample(
            prompt=types.ModelInput.from_ints(prompt_tokens),
            num_samples=1,
            sampling_params=params,
        ).result()
        return self.tokenizer.decode(resp.sequences[0].tokens)

    def _sample_optimizer_outputs(self, sampler, state_text: str) -> List[Dict[str, Any]]:
        """Sample G completions from the current policy for a given state."""
        # 数据中的 state_text (input字段) 已经包含系统提示，不再额外添加
        prompt = build_io_prompt(self.chat_template, "", state_text)
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        params = types.SamplingParams(
            max_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )
        resp = sampler.sample(
            prompt=types.ModelInput.from_ints(prompt_tokens),
            num_samples=self.cfg.rollouts_per_state,
            sampling_params=params,
        ).result()
        outputs = []
        for seq in resp.sequences:
            out_tokens = seq.tokens
            out_text = self.tokenizer.decode(out_tokens)
            outputs.append({
                "prompt_tokens": prompt_tokens,
                "output_tokens": out_tokens,
                "output_text": out_text,
            })
        return outputs

    def _compute_logprobs(self, sampler, tokens: List[int]) -> List[float]:
        prompt = types.ModelInput.from_ints(tokens)
        logprobs = sampler.compute_logprobs(prompt).result()
        return [0.0 if v is None else float(v) for v in logprobs]

    def _start_logprobs(self, sampler, tokens: List[int]):
        prompt = types.ModelInput.from_ints(tokens)
        return sampler.compute_logprobs(prompt)

    @staticmethod
    def _finish_logprobs(logprob_future) -> List[float]:
        logprobs = logprob_future.result()
        return [0.0 if v is None else float(v) for v in logprobs]

    def _build_grpo_datum(
        self,
        prompt_tokens: List[int],
        output_tokens: List[int],
        advantage: float,
        old_logprobs: List[float],
        ref_logprobs: List[float],
    ) -> Tuple[types.Datum, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a training datum for GRPO and return extra tensors for custom loss."""
        full_tokens = prompt_tokens + output_tokens
        n_prompt = len(prompt_tokens)
        n_output = len(output_tokens)

        # Weights: 0 for prompt tokens, 1 for output tokens (shifted by 1 for next-token prediction)
        weights = [0.0] * n_prompt + [1.0] * n_output
        target_tokens = full_tokens[1:]
        weights = weights[1:]

        # Advantage vector: only apply to output tokens
        adv_vec = [0.0] * (n_prompt - 1) + [advantage] * n_output

        # Old and ref logprobs (shifted: predict token t from context 0..t-1)
        old_lp = old_logprobs[1:]
        ref_lp = ref_logprobs[1:]

        datum = types.Datum(
            model_input=types.ModelInput.from_ints(full_tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": torch.tensor(target_tokens, dtype=torch.int64),
                "weights": torch.tensor(weights, dtype=torch.float32),
            },
        )
        return (
            datum,
            torch.tensor(adv_vec, dtype=torch.float32),
            torch.tensor(old_lp, dtype=torch.float32),
            torch.tensor(ref_lp, dtype=torch.float32),
        )

    @staticmethod
    def _group_relative_advantages(rewards: List[float]) -> List[float]:
        """Compute GRPO group-relative advantages: (r_i - mean) / std."""
        if len(rewards) <= 1:
            return [0.0] * len(rewards)
        mean_r = float(np.mean(rewards))
        std_r = float(np.std(rewards)) + 1e-8
        return [(r - mean_r) / std_r for r in rewards]

    def _evaluate_rollouts_sync(
        self,
        outputs: List[Dict[str, Any]],
        persona_text: str,
        original_query: str,
    ) -> Tuple[List[Dict[str, Any]], List[float], int]:
        """Evaluate rollouts synchronously.

        Returns: (rollout_records, rewards, failed_count)
        """
        base_response = self._call_assistant(original_query)
        base_failed = not base_response  # Empty string means assistant API failed
        rewards: List[float] = []
        rollout_records: List[Dict[str, Any]] = []
        failed_count = 0
        for out in outputs:
            parsed_out = parse_optimizer_output(out["output_text"])
            opt_query = parsed_out["prompt"]
            inferred_profile = parsed_out["reasoning"]

            opt_response = self._call_assistant(opt_query)
            opt_failed = not opt_response  # Empty string means assistant API failed

            # Skip judge if assistant API failed (empty response is unreliable)
            if base_failed or opt_failed:
                reward = {"r_prof": 0.0, "r_task": 0.0, "total": 0.0, "failed": True}
            else:
                reward = self.rewarder.total_reward(
                    inferred_profile=inferred_profile,
                    persona_text=persona_text,
                    query=original_query,
                    base=base_response,
                    opt=opt_response,
                )
            rollout_records.append({
                "optimizer_output": out["output_text"],
                "reasoning": inferred_profile,
                "optimized_prompt": opt_query,
                "base_response": base_response,
                "opt_response": opt_response,
                "reward": reward,
            })
            if reward.get("failed", False):
                failed_count += 1
                rewards.append(None)  # Placeholder, will be filtered out
            else:
                rewards.append(reward["total"])
        return rollout_records, rewards, failed_count

    async def _evaluate_rollouts_async(
        self,
        outputs: List[Dict[str, Any]],
        persona_text: str,
        original_query: str,
    ) -> Tuple[List[Dict[str, Any]], List[float], int]:
        """Evaluate rollouts asynchronously.

        Returns: (rollout_records, rewards, failed_count)
        """
        assistant_sem = None
        judge_sem = None
        if self.cfg.assistant_concurrency and self.cfg.assistant_concurrency > 0:
            assistant_sem = asyncio.Semaphore(self.cfg.assistant_concurrency)
        if self.cfg.judge_concurrency and self.cfg.judge_concurrency > 0:
            judge_sem = asyncio.Semaphore(self.cfg.judge_concurrency)

        parsed = [parse_optimizer_output(out["output_text"]) for out in outputs]
        opt_queries = [p["prompt"] for p in parsed]
        inferred_profiles = [p["reasoning"] for p in parsed]

        base_task = asyncio.create_task(self._call_assistant_async(original_query, assistant_sem))
        opt_tasks = [
            asyncio.create_task(self._call_assistant_async(opt_query, assistant_sem))
            for opt_query in opt_queries
        ]
        base_response = await base_task
        opt_responses = await asyncio.gather(*opt_tasks)
        base_failed = not base_response  # Empty string means assistant API failed

        # Only call judge for rollouts where assistant succeeded
        reward_tasks = []
        for inferred_profile, opt_response in zip(inferred_profiles, opt_responses):
            if base_failed or not opt_response:
                # Mark as failed immediately without calling judge
                reward_tasks.append(None)
            else:
                reward_tasks.append(
                    asyncio.create_task(
                        self._total_reward_async(
                            inferred_profile=inferred_profile,
                            persona_text=persona_text,
                            query=original_query,
                            base=base_response,
                            opt=opt_response,
                            semaphore=judge_sem,
                        )
                    )
                )

        # Gather reward results
        reward_results = []
        for rt in reward_tasks:
            if rt is None:
                reward_results.append({"r_prof": 0.0, "r_task": 0.0, "total": 0.0, "failed": True})
            else:
                reward_results.append(await rt)

        rewards: List[float | None] = []
        rollout_records: List[Dict[str, Any]] = []
        failed_count = 0
        for out, opt_query, inferred_profile, opt_response, reward in zip(
            outputs, opt_queries, inferred_profiles, opt_responses, reward_results
        ):
            rollout_records.append({
                "optimizer_output": out["output_text"],
                "reasoning": inferred_profile,
                "optimized_prompt": opt_query,
                "base_response": base_response,
                "opt_response": opt_response,
                "reward": reward,
            })
            if reward.get("failed", False):
                failed_count += 1
                rewards.append(None)
            else:
                rewards.append(reward["total"])
        return rollout_records, rewards, failed_count

    def train(self) -> str:
        assert self.client is not None
        assert self.service is not None
        assert self.rewarder is not None
        assert self.assistant_client is not None

        # Reference model (frozen SFT checkpoint) for KL penalty
        # Use sft_sampler_path which was saved with save_weights_for_sampler()
        ref_sampler = self.service.create_sampling_client(model_path=self.cfg.sft_sampler_path)

        data = self.load_jsonl(self.cfg.train_file)

        step = 0

        loss_meta: List[Dict[str, torch.Tensor]] = []

        # Mutable container to capture metrics from grpo_loss closure
        step_metrics: Dict[str, float] = {}

        def grpo_loss(data_batch, new_logprobs_list):
            """GRPO loss: clipped surrogate objective + KL penalty against reference."""
            losses = []
            kls = []
            if len(data_batch) != len(loss_meta):
                raise ValueError("Loss metadata length mismatch with data batch")
            for idx, (datum, new_lp) in enumerate(zip(data_batch, new_logprobs_list)):
                meta = loss_meta[idx]
                weights = datum.loss_fn_inputs["weights"]
                advantages = meta["advantages"]
                old_lp = meta["old_logprobs"]
                ref_lp = meta["ref_logprobs"]

                if not isinstance(weights, torch.Tensor):
                    weights = torch.as_tensor(getattr(weights, "data", weights), dtype=torch.float32)
                if not isinstance(advantages, torch.Tensor):
                    advantages = torch.as_tensor(getattr(advantages, "data", advantages), dtype=torch.float32)
                if not isinstance(old_lp, torch.Tensor):
                    old_lp = torch.as_tensor(getattr(old_lp, "data", old_lp), dtype=torch.float32)
                if not isinstance(ref_lp, torch.Tensor):
                    ref_lp = torch.as_tensor(getattr(ref_lp, "data", ref_lp), dtype=torch.float32)

                # Policy ratio: π_θ / π_θ_old
                ratio = torch.exp(new_lp - old_lp)
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps)

                # Clipped surrogate objective (negative because we minimize)
                pg_loss = -torch.minimum(ratio * advantages, clipped * advantages)

                # KL divergence penalty: approximate KL(π_θ || π_ref)
                # Using the unbiased estimator: (π_ref/π_θ) - log(π_ref/π_θ) - 1
                log_ratio = new_lp - ref_lp  # log(π_θ / π_ref)
                kl = torch.exp(-log_ratio) - (-log_ratio) - 1.0

                loss = (pg_loss + self.cfg.kl_coef * kl) * weights
                denom = torch.clamp(weights.sum(), min=1.0)
                losses.append(loss.sum() / denom)
                kls.append((kl * weights).sum() / denom)

            total = torch.mean(torch.stack(losses))
            metrics = {
                "grpo_loss": float(total.item()),
                "kl_mean": float(torch.mean(torch.stack(kls)).item()),
            }
            step_metrics.update(metrics)  # Capture metrics for tracker
            return total, metrics

        for epoch in range(self.cfg.num_epochs):
            random.shuffle(data)
            for i in range(0, len(data), self.cfg.batch_size):
                batch = data[i:i + self.cfg.batch_size]

                # Snapshot current policy weights for sampling (π_θ_old)
                policy_sampler = self.client.save_weights_and_get_sampling_client()

                batch_datums: List[types.Datum] = []
                loss_meta.clear()
                step_interactions: List[Dict[str, Any]] = []
                step_total_rollouts = 0
                step_failed_rollouts = 0
                for state in batch:
                    persona_text = persona_to_text(state.get("persona_features") or {})
                    history_texts = state.get("history_texts") or []
                    original_query = state.get("original_query") or ""
                    state_text = state.get("input") or ""
                    if not state_text:
                        continue

                    # GRPO Step 1: Sample G completions from policy_old
                    outputs = self._sample_optimizer_outputs(policy_sampler, state_text)
                    if not outputs:
                        continue

                    # Optional pipeline: start logprob requests early
                    logprob_futures = None
                    if self.cfg.pipeline and self.cfg.prefetch_logprobs:
                        logprob_futures = []
                        for out in outputs:
                            full_tokens = out["prompt_tokens"] + out["output_tokens"]
                            old_future = self._start_logprobs(policy_sampler, full_tokens)
                            ref_future = self._start_logprobs(ref_sampler, full_tokens)
                            logprob_futures.append((old_future, ref_future))

                    # GRPO Step 2: Compute rewards for each completion
                    if self.cfg.pipeline:
                        rollout_records, rewards, failed_count = asyncio.run(
                            self._evaluate_rollouts_async(outputs, persona_text, original_query)
                        )
                    else:
                        rollout_records, rewards, failed_count = self._evaluate_rollouts_sync(
                            outputs, persona_text, original_query
                        )

                    step_total_rollouts += len(outputs)
                    step_failed_rollouts += failed_count

                    # Filter out failed rollouts (those with reward=None)
                    valid_indices = [idx for idx, r in enumerate(rewards) if r is not None]
                    failed_indices = [idx for idx, r in enumerate(rewards) if r is None]
                    if not valid_indices:
                        print(f"[GRPO step {step+1}] All {len(outputs)} rollouts failed, skipping state")
                        continue

                    if failed_indices:
                        print(
                            f"[GRPO step {step+1}] {len(failed_indices)}/{len(outputs)} rollouts failed "
                            f"(idx={failed_indices}), using {len(valid_indices)} valid"
                        )

                    # Filter data to only valid rollouts
                    valid_outputs = [outputs[idx] for idx in valid_indices]
                    valid_rewards = [rewards[idx] for idx in valid_indices]
                    valid_rollout_records = [rollout_records[idx] for idx in valid_indices]
                    if logprob_futures is not None:
                        valid_logprob_futures = [logprob_futures[idx] for idx in valid_indices]
                    else:
                        valid_logprob_futures = None

                    # GRPO Step 3: Compute group-relative advantages (Eq. in GRPO)
                    advantages = self._group_relative_advantages(valid_rewards)
                    for idx, adv in enumerate(advantages):
                        valid_rollout_records[idx]["advantage"] = adv

                    # GRPO Step 4: Build training data with logprobs
                    for idx, (out, adv) in enumerate(zip(valid_outputs, advantages)):
                        full_tokens = out["prompt_tokens"] + out["output_tokens"]
                        if valid_logprob_futures is not None:
                            old_logprobs = self._finish_logprobs(valid_logprob_futures[idx][0])
                            ref_logprobs = self._finish_logprobs(valid_logprob_futures[idx][1])
                        else:
                            old_logprobs = self._compute_logprobs(policy_sampler, full_tokens)
                            ref_logprobs = self._compute_logprobs(ref_sampler, full_tokens)

                        datum, adv_vec, old_lp, ref_lp = self._build_grpo_datum(
                            out["prompt_tokens"],
                            out["output_tokens"],
                            adv,
                            old_logprobs,
                            ref_logprobs,
                        )
                        batch_datums.append(datum)
                        loss_meta.append(
                            {
                                "advantages": adv_vec,
                                "old_logprobs": old_lp,
                                "ref_logprobs": ref_lp,
                            }
                        )

                    step_interactions.append({
                        "state_input": state_text,
                        "original_query": original_query,
                        "persona_text": persona_text,
                        "rollouts": rollout_records,  # Keep all records for logging
                        "rewards": rewards,  # Keep all (including None for failed)
                        "advantages": advantages,
                        "valid_rollout_count": len(valid_indices),
                        "failed_rollout_count": failed_count,
                    })

                if not batch_datums:
                    continue

                # GRPO Step 5: Update policy with clipped surrogate + KL
                fwdbwd = self.client.forward_backward_custom(batch_datums, grpo_loss)
                _ = fwdbwd.result()
                opt = self.client.optim_step(
                    types.AdamParams(
                        learning_rate=self.cfg.learning_rate,
                        beta1=0.9,
                        beta2=0.95,
                        eps=1e-12,
                        weight_decay=self.cfg.weight_decay,
                        grad_clip_norm=self.cfg.grad_clip_norm,
                    )
                )
                _ = opt.result()
                step += 1

                # Calculate and track failure rate
                step_fail_rate = step_failed_rollouts / step_total_rollouts if step_total_rollouts > 0 else 0.0
                valid_rollouts_this_step = step_total_rollouts - step_failed_rollouts

                print(
                    f"[GRPO step {step}] done. Rollouts: {valid_rollouts_this_step}/{step_total_rollouts} valid "
                    f"({(1.0 - step_fail_rate)*100:.0f}% success)"
                )

                # Update sliding window failure rate history
                self._api_fail_history.append(step_fail_rate)
                if len(self._api_fail_history) > self.cfg.api_fail_window:
                    self._api_fail_history.pop(0)

                # Check if failure rate exceeds threshold
                if len(self._api_fail_history) >= self.cfg.api_fail_window:
                    avg_fail_rate = sum(self._api_fail_history) / len(self._api_fail_history)
                    if avg_fail_rate > self.cfg.api_fail_threshold:
                        rates_str = ", ".join(f"{r*100:.0f}%" for r in self._api_fail_history)
                        print(
                            f"[GRPO] ABORT: API failure rate {avg_fail_rate*100:.0f}% > threshold "
                            f"{self.cfg.api_fail_threshold*100:.0f}% over last {self.cfg.api_fail_window} steps. "
                            f"Per-step rates: [{rates_str}]"
                        )
                        raise RuntimeError(
                            f"API failure rate {avg_fail_rate*100:.1f}% exceeds threshold "
                            f"{self.cfg.api_fail_threshold*100:.1f}%. Training aborted."
                        )

                # Record metrics for plotting
                if self._metrics_tracker is not None:
                    all_rewards = [r for si in step_interactions for r in si.get("rewards", []) if r is not None]
                    mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
                    self._metrics_tracker.record(
                        step=step,
                        grpo_loss=step_metrics.get("grpo_loss", 0.0),
                        kl_mean=step_metrics.get("kl_mean", 0.0),
                        mean_reward=mean_reward,
                    )
                    step_metrics.clear()

                # Save sampler checkpoint periodically
                if self.cfg.sampler_save_interval > 0 and step % self.cfg.sampler_save_interval == 0:
                    # Save state checkpoint (for training continuation)
                    interval_state_name = f"{self.cfg.checkpoint_name}_step{step}"
                    interval_state_save = self.client.save_state(interval_state_name)
                    interval_state_resp = interval_state_save.result()
                    print(f"[GRPO step {step}] Saved state checkpoint: {interval_state_resp.path}")
                    # Save sampler checkpoint (for inference)
                    interval_sampler_name = f"{self.cfg.checkpoint_name}_sampler_step{step}"
                    interval_save = self.client.save_weights_for_sampler(interval_sampler_name)
                    interval_resp = interval_save.result()
                    print(f"[GRPO step {step}] Saved sampler checkpoint: {interval_resp.path}")

                # Log interaction record
                if step_interactions and step % self.cfg.log_interval == 0:
                    self._log_interaction({
                        "timestamp": datetime.now().isoformat(),
                        "epoch": epoch,
                        "step": step,
                        "states": step_interactions,
                    })

            # Save checkpoints at end of each epoch
            # Save state checkpoint (for training continuation)
            epoch_state_name = f"{self.cfg.checkpoint_name}_epoch{epoch + 1}"
            epoch_state_save = self.client.save_state(epoch_state_name)
            epoch_state_resp = epoch_state_save.result()
            print(f"[GRPO epoch {epoch + 1}] Saved state checkpoint: {epoch_state_resp.path}")
            # Save sampler checkpoint (for inference)
            epoch_sampler_name = f"{self.cfg.checkpoint_name}_sampler_epoch{epoch + 1}"
            epoch_save = self.client.save_weights_for_sampler(epoch_sampler_name)
            epoch_resp = epoch_save.result()
            print(f"[GRPO epoch {epoch + 1}] Saved sampler checkpoint: {epoch_resp.path}")

        # Save checkpoint with optimizer state (for training continuation)
        save_name = f"{self.cfg.checkpoint_name}"
        save_future = self.client.save_state(save_name)
        save_resp = save_future.result()
        state_path = save_resp.path

        # Save checkpoint for sampling (for inference)
        sampler_save = self.client.save_weights_for_sampler(f"{self.cfg.checkpoint_name}_sampler")
        sampler_resp = sampler_save.result()
        sampler_path = sampler_resp.path

        # Close metrics tracker and save final plots
        if self._metrics_tracker is not None:
            self._metrics_tracker.close()
            self._metrics_tracker = None

        # Close interaction log file
        if self._interaction_logger is not None:
            self._interaction_logger.close()
            self._interaction_logger = None

        return {
            "state_path": state_path,
            "sampler_path": sampler_path,
        }
