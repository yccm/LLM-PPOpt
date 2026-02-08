 #!/usr/bin/env python
"""Combined inference + evaluation script for PPOpt.

Modes:
    --eval        Full evaluation: optimize -> assistant -> judge (default)
    --infer       Inference only: optimize queries and output results
    --interactive Interactive mode: optimize queries one by one
    --parallel    Parallel evaluation of multiple checkpoints

Usage:
    python scripts/run_eval.py --eval -i prompt.jsonl -o eval_result.jsonl
    python scripts/run_eval.py --infer -i prompt.jsonl -o output.jsonl
    python scripts/run_eval.py --interactive
    python scripts/run_eval.py --parallel -i prompt.jsonl

Config is read from config.yaml (eval section).
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppopt.utils.tinker_compat import ensure_tinker_compat
ensure_tinker_compat()

import tinker
from tinker import types
from transformers import AutoTokenizer
from openai import OpenAI

from ppopt.utils.chat import get_chat_template, build_io_prompt
from ppopt.utils.config import load_config
from ppopt.utils.formatting import (
    build_state_text,
    conversation_to_text,
    persona_to_text,
)


# ============================================================================
# CHECKPOINT CONFIGURATION - Add checkpoint URLs here for parallel evaluation
# ============================================================================
CHECKPOINT_URLS = [
    # llama 1 epoch sft - rlc - full
    "tinker://0e248612-c93f-51c0-809d-857fd5da97a0:train:0/sampler_weights/ppopt_sft_llama8b_sampler",
    "tinker://cd569d0e-1522-588f-89ea-a9bba2ff69d6:train:0/sampler_weights/ppopt_rl_llama8b_sampler",
    "tinker://cb1866d3-8a10-5700-ae66-5454e86f4fe0:train:0/sampler_weights/ppopt_rl_llama8b_sampler",
    # qwen  1 epoch sft - rlc - full
    "tinker://7831890b-7fbe-545d-8b92-2eb711162334:train:0/sampler_weights/ppopt_sft_qwen8b_sampler",
    "tinker://9c2b454a-1900-593f-8f72-d7dd6a756e7f:train:0/sampler_weights/ppopt_rl_qwen8b_sampler",
    "tinker://e01d9a15-98a0-5e3e-a4b0-b571f1e0b498:train:0/sampler_weights/ppopt_rl_qwen8b_sampler",

    # gpt-oss 1 epoch sft - rlc - full
    "tinker://bbad408f-c2ab-5de2-9101-2f7413d72fc0:train:0/sampler_weights/ppopt_sft_sampler",
    "tinker://e8db22e2-c7b5-54c7-959b-f100ea362ac1:train:0/sampler_weights/ppopt_rl_sampler_epoch1",
    "tinker://4e21c90e-2c89-5f57-a4cf-e95ac4ff485b:train:0/sampler_weights/ppopt_sft_sampler",
    
    # second checkpoint for qwen sft epoch for 1
    "tinker://bbf2ec57-0875-5672-8cb3-c2b17620fe7d:train:0/sampler_weights/ppopt_sft_qwen8b_sampler"
    # qwen rl constraint
    "tinker://e49f6b9e-d648-5739-854f-f870486fd5d3:train:0/sampler_weights/ppopt_rl_qwen8b_sampler",
    # oss full
    "tinker://3a36038e-a612-5892-add3-b3f3b90afb23:train:0/sampler_weights/ppopt_rl_sampler", 
    # llama 2 epochs full
    "tinker://d01955d1-f320-5408-8029-5f26b47d43d9:train:0/sampler_weights/ppopt_rl_llama8b_sampler",
]

# Number of parallel workers (set to len(CHECKPOINT_URLS) for full parallelism)
PARALLEL_WORKERS = 5

# API concurrency settings
API_CONCURRENCY = 10  # Number of concurrent API calls (assistant + judge)


# ============================================================================
# Judge Prompts for Independent Scoring
# ============================================================================

PERSONALIZATION_JUDGE_PROMPT = """You are a strict judge of PERSONALIZATION.

User Profile:
{persona}

Query:
{query}

Response:
{response}

Judge ONLY whether the response is explicitly personalized to the user profile.
Being helpful or correct is NOT sufficient.

Rubric:
- 9-10: Clearly and specifically tailored to multiple concrete profile traits; would be different without the profile.
- 7-8: Clearly tailored to at least one non-trivial profile trait.
- 5-6: Weak or generic alignment; personalization is shallow or implicit.
- 3-4: Mostly generic; profile has little effect.
- 1-2: No personalization or contradicts the profile.

Output ONLY an integer from 1 to 10.

Score:
"""

TASK_JUDGE_PROMPT = """You are a lenient judge of TASK COMPLETION.

Query:
{query}

Response:
{response}

Judge ONLY whether the response completes the task.
Ignore tone, style, and personalization.
Be forgiving of minor errors or missing details.

Rubric:
- 8-10: Task is clearly completed; response is usable.
- 5-7: Task is mostly completed; minor gaps are acceptable.
- 3-4: Task is partially addressed.
- 1-2: Task is not completed or irrelevant.

Output ONLY an integer from 1 to 10.

Score:
"""


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_score_int(text: str) -> int:
    """Parse integer score from judge output."""
    numbers = re.findall(r"\b(\d+)\b", text.strip())
    if numbers:
        score = int(numbers[0])
        return max(1, min(10, score))
    return 5  # default


def get_tokenizer_model_from_checkpoint(checkpoint_path: str) -> str:
    """Determine the tokenizer model name from checkpoint path.

    Maps checkpoint naming conventions to their base model tokenizers:
    - llama checkpoints -> meta-llama/Llama-3.1-8B-Instruct
    - qwen checkpoints -> Qwen/Qwen3-8B
    - harmony/default -> openai/gpt-oss-20b
    """
    if not checkpoint_path:
        return "openai/gpt-oss-20b"

    ckpt_lower = checkpoint_path.lower()
    if "llama" in ckpt_lower:
        return "meta-llama/Llama-3.1-8B-Instruct"
    elif "qwen" in ckpt_lower:
        return "Qwen/Qwen3-8B"
    else:
        return "openai/gpt-oss-20b"


def clean_optimizer_output(text: str) -> str:
    """Clean optimizer output by removing <REASONING> tags and extracting only the prompt.

    Rules:
    - If no <REASONING> tag exists, return text as-is (no cleaning needed)
    - Handle incomplete blocks (only start tag or only end tag)
    - Extract content from <PROMPT> tags if present

    Returns:
        Cleaned prompt text, or empty string if nothing left after cleaning.
    """
    if not text:
        return ""

    cleaned = text.strip()

    # Check if any REASONING tag exists (complete or incomplete)
    has_reasoning_start = bool(re.search(r"<REASONING>", cleaned, re.IGNORECASE))
    has_reasoning_end = bool(re.search(r"</REASONING>", cleaned, re.IGNORECASE))

    # If no REASONING tags at all, return as-is
    if not has_reasoning_start and not has_reasoning_end:
        return cleaned

    # Handle complete block: <REASONING>...</REASONING>
    cleaned = re.sub(r"<REASONING>.*?</REASONING>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Handle incomplete block: only <REASONING> (remove everything from <REASONING> to end)
    if has_reasoning_start and not has_reasoning_end:
        cleaned = re.sub(r"<REASONING>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Handle incomplete block: only </REASONING> (remove everything from start to </REASONING>)
    if has_reasoning_end and not has_reasoning_start:
        cleaned = re.sub(r"^.*?</REASONING>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    cleaned = cleaned.strip()

    # If there's a <PROMPT>...</PROMPT> block, extract just that content
    prompt_match = re.search(r"<PROMPT>(.*?)</PROMPT>", cleaned, re.DOTALL | re.IGNORECASE)
    if prompt_match:
        cleaned = prompt_match.group(1).strip()
    else:
        # Handle incomplete <PROMPT> tag (only start tag)
        prompt_start_match = re.search(r"<PROMPT>(.*)$", cleaned, re.DOTALL | re.IGNORECASE)
        if prompt_start_match:
            cleaned = prompt_start_match.group(1).strip()
        # Handle incomplete </PROMPT> tag (only end tag)
        prompt_end_match = re.search(r"^(.*?)</PROMPT>", cleaned, re.DOTALL | re.IGNORECASE)
        if prompt_end_match:
            cleaned = prompt_end_match.group(1).strip()

    return cleaned


def load_checkpoint_path(eval_cfg: dict) -> tuple[str, str | None]:
    """Resolve checkpoint path from eval config.

    If checkpoint_path is "auto", read from latest_checkpoint.json.
    Otherwise use the value directly.

    Returns: (sampler_path, state_path or None)
    """
    ckpt_path = eval_cfg.get("checkpoint_path", "auto")

    if ckpt_path == "auto":
        ckpt_file = PROJECT_ROOT / "latest_checkpoint.json"
        if not ckpt_file.exists():
            print(f"[ERROR] checkpoint_path is 'auto' but {ckpt_file} not found.")
            print("  Run training first, or set eval.checkpoint_path in config.yaml.")
            sys.exit(1)
        with open(ckpt_file, "r") as f:
            data = json.load(f)
        sampler_path = data.get("sampler_path") or data.get("state_path")
        state_path = data.get("state_path")
        print(f"[INFO] Auto-loaded checkpoint: {sampler_path}")
        return sampler_path, state_path

    return ckpt_path, None


def resolve_sampler_path(service: tinker.ServiceClient, checkpoint_path: str) -> str:
    """If checkpoint_path is a state checkpoint (/weights/), create a sampler.
    If it's already a sampler path (/sampler_weights/), return as-is."""
    if "/sampler_weights/" in checkpoint_path:
        return checkpoint_path

    print(f"[INFO] Detected state checkpoint, creating sampler...")
    print(f"  State: {checkpoint_path}")
    training_client = service.create_training_client_from_state_with_optimizer(checkpoint_path)
    result = training_client.save_weights_for_sampler("eval_sampler").result()
    print(f"  Sampler: {result.path}")
    return result.path


def convert_history(history: List[Any]) -> List[str]:
    """Convert history sessions to text."""
    texts = []
    for session in history:
        if isinstance(session, str):
            texts.append(session)
        elif isinstance(session, list):
            texts.append(conversation_to_text(session))
    return texts


class PPOptRunner:
    """Unified runner for inference and evaluation."""

    def __init__(self, cfg: dict, skip_optimizer: bool = False, checkpoint_override: str | None = None):
        """Initialize runner.

        Args:
            cfg: Configuration dict
            skip_optimizer: If True, skip tinker sampler/tokenizer init.
                Useful for baseline-only computation (assistant + judge only).
            checkpoint_override: If provided, use this checkpoint path instead of config.
                Also used to infer chat_template from checkpoint name.
        """
        self.cfg = cfg

        # Get eval config section
        eval_cfg = cfg.get("eval", {})
        self.system_prompt = eval_cfg.get("system_prompt", "")
        self.max_tokens = eval_cfg.get("max_tokens", 512)
        self.temperature = eval_cfg.get("temperature", 0.7)

        # Determine chat template: from checkpoint name, config override, or base_model
        model_cfg = cfg.get("model", {})
        chat_template_override = eval_cfg.get("chat_template") or model_cfg.get("chat_template")

        # Infer chat template from checkpoint name if available
        if checkpoint_override and not chat_template_override:
            ckpt_lower = checkpoint_override.lower()
            if "llama" in ckpt_lower:
                chat_template_override = "llama"
            elif "qwen" in ckpt_lower:
                chat_template_override = "qwen"
            else:
                # Default to harmony (gpt-oss) for checkpoints without llama/qwen in name
                # e.g., ppopt_sft_sampler -> gpt-oss model
                chat_template_override = "harmony"

        base_model = model_cfg.get("base_model", "")
        self.chat_template = get_chat_template(base_model, chat_template_override)

        # Tinker for inference (skip if only doing baseline)
        self.sampler = None
        self.tokenizer = None
        if not skip_optimizer:
            api_cfg = cfg.get("api", {})
            self.service = tinker.ServiceClient(api_key=api_cfg["tinker_api_key"])

            # Use checkpoint_override if provided, otherwise read from config
            if checkpoint_override:
                checkpoint_path = checkpoint_override
                state_path = None
            else:
                checkpoint_path, state_path = load_checkpoint_path(eval_cfg)

            sampler_path = resolve_sampler_path(self.service, checkpoint_path)
            self.sampler = self.service.create_sampling_client(model_path=sampler_path)

            # Load tokenizer based on checkpoint name (SamplingClient doesn't have get_tokenizer)
            tokenizer_model = get_tokenizer_model_from_checkpoint(checkpoint_path)
            print(f"[INFO] Loading tokenizer from: {tokenizer_model}")
            # Use HuggingFace token if available (for gated models like Llama)
            hf_token = cfg.get("huggingface", {}).get("token")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, token=hf_token)
            print(f"[INFO] Loaded tokenizer: {type(self.tokenizer).__name__}")

        # Assistant (use openrouter config)
        openrouter_cfg = cfg.get("openrouter", {})
        if openrouter_cfg.get("api_key"):
            self.assistant_client = OpenAI(
                api_key=openrouter_cfg["api_key"],
                base_url=openrouter_cfg.get("base_url", "https://openrouter.ai/api/v1"),
            )
            self.assistant_model = openrouter_cfg.get("model", "meta-llama/llama-3.3-70b-instruct:free")
            self.assistant_max_tokens = openrouter_cfg.get(
                "max_tokens", eval_cfg.get("assistant_max_tokens", 1024)
            )
            self.assistant_temperature = openrouter_cfg.get("temperature", 0.7)
            self.assistant_top_p = openrouter_cfg.get("top_p", 1.0)
            self.assistant_system_prompt = cfg.get("assistant", {}).get("system_prompt", "You are a helpful assistant.")
        else:
            self.assistant_client = None

        # Judge (use openai + judge config)
        openai_cfg = cfg.get("openai", {})
        judge_cfg = cfg.get("judge", {})
        if openai_cfg.get("api_key"):
            self.judge_client = OpenAI(
                api_key=openai_cfg["api_key"],
                base_url=openai_cfg.get("base_url", "https://api.openai.com/v1"),
            )
            self.judge_model = judge_cfg.get("model", "gpt-4o-mini")
            self.judge_max_tokens = judge_cfg.get("max_tokens", eval_cfg.get("judge_max_tokens", 1024))
            self.judge_temperature = judge_cfg.get("temperature", 0.0)
            self.judge_top_p = judge_cfg.get("top_p", 1.0)
        else:
            self.judge_client = None

    def optimize_query(self, current_query: str, history_texts: List[str] | None = None, input_text: str | None = None, max_retries: int = 3) -> dict:
        """Optimize a query using the trained model.

        Args:
            current_query: The original query (for output tracking)
            history_texts: Historical conversation texts (used if input_text not provided)
            input_text: Pre-built input prompt (if provided, skip state_text building)
            max_retries: Maximum number of retries if output is empty after cleaning
        """
        if input_text:
            state_text = input_text
            prompt = build_io_prompt(self.chat_template, "", state_text)
        else:
            state_text = build_state_text("", history_texts or [], current_query)
            prompt = build_io_prompt(self.chat_template, self.system_prompt, state_text)

        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        params = types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        # Retry logic: if cleaned output is empty, retry up to max_retries times
        for attempt in range(max_retries):
            response = self.sampler.sample(
                prompt=types.ModelInput.from_ints(prompt_tokens),
                num_samples=1,
                sampling_params=params,
            ).result()

            output_text = self.tokenizer.decode(response.sequences[0].tokens)

            # Clean the output: remove <REASONING> tags and extract prompt
            cleaned_prompt = clean_optimizer_output(output_text)

            if cleaned_prompt:
                # Successfully got a non-empty prompt
                return {
                    "original_query": current_query,
                    "optimized_prompt": cleaned_prompt,
                    "reasoning": "",  # reasoning is stripped
                    "optimizer_output": output_text,
                }

            # Empty after cleaning, retry
            if attempt < max_retries - 1:
                print(f"    [RETRY {attempt + 1}/{max_retries}] Output empty after cleaning, retrying...")

        # All retries exhausted, return original query as fallback
        print(f"    [WARNING] All {max_retries} attempts failed, using original query as fallback")
        return {
            "original_query": current_query,
            "optimized_prompt": current_query,  # fallback to original
            "reasoning": "",
            "optimizer_output": output_text,
        }

    def get_assistant_response(self, query: str) -> str:
        """Get response from assistant model."""
        if not self.assistant_client:
            return ""
        try:
            resp = self.assistant_client.chat.completions.create(
                model=self.assistant_model,
                messages=[
                    {"role": "system", "content": self.assistant_system_prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=self.assistant_max_tokens,
                temperature=self.assistant_temperature,
                top_p=self.assistant_top_p,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"    Assistant error: {e}")
            return ""

    def _judge_call(self, prompt: str) -> str:
        """Raw judge call with prompt only (no system message)."""
        if not self.judge_client:
            return ""
        try:
            resp = self.judge_client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.judge_max_tokens,
                temperature=self.judge_temperature,
                top_p=self.judge_top_p,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"    Judge error: {e}")
            return ""

    def score_response(self, persona_text: str, query: str, response: str) -> Dict[str, Any]:
        """Score a response independently on personalization and task (concurrent)."""
        if not self.judge_client:
            return {"persona_score": 0, "task_score": 0, "persona_raw": "", "task_raw": ""}

        response_snippet = response[:2000] if response else ""

        pers_prompt = PERSONALIZATION_JUDGE_PROMPT.format(
            persona=persona_text or "(No persona)",
            query=query,
            response=response_snippet,
        )
        task_prompt = TASK_JUDGE_PROMPT.format(
            query=query,
            response=response_snippet,
        )

        # Run persona and task scoring in parallel
        with ThreadPoolExecutor(max_workers=2) as pool:
            pers_future = pool.submit(self._judge_call, pers_prompt)
            task_future = pool.submit(self._judge_call, task_prompt)
            pers_raw = pers_future.result()
            task_raw = task_future.result()

        return {
            "persona_score": parse_score_int(pers_raw),
            "task_score": parse_score_int(task_raw),
            "persona_raw": pers_raw,
            "task_raw": task_raw,
        }

    def run_eval(self, item: dict, original_baseline: dict | None = None) -> dict:
        """Run full pipeline with comparison scoring (concurrent API calls).

        Args:
            item: Input data item
            original_baseline: Pre-computed baseline for original query (optional).
                If provided, skips computing original response/scores.
                Should contain: original_response, original_scores
        """
        persona = item.get("persona_features", {})

        original_query = item.get("original_query") or item.get("current_query") or item.get("query") or item.get("prompt", "")

        input_text = item.get("input")

        if not input_text:
            history_raw = item.get("history", [])
            history_texts = convert_history(history_raw) if history_raw else item.get("history_texts", [])
        else:
            history_texts = None

        optimize_out = self.optimize_query(original_query, history_texts, input_text)
        optimized_prompt = optimize_out["optimized_prompt"]

        persona_text = persona_to_text(persona)

        if original_baseline:
            # Use pre-computed baseline
            original_response = original_baseline["original_response"]
            original_scores = original_baseline["original_scores"]
            # Only get optimized response and scores
            optimized_response = self.get_assistant_response(optimized_prompt)
            optimized_scores = self.score_response(persona_text, optimized_prompt, optimized_response)
        else:
            # Compute both original and optimized (first time or single checkpoint mode)
            # Get responses for both queries in parallel
            with ThreadPoolExecutor(max_workers=2) as pool:
                orig_resp_future = pool.submit(self.get_assistant_response, original_query)
                opt_resp_future = pool.submit(self.get_assistant_response, optimized_prompt)
                original_response = orig_resp_future.result()
                optimized_response = opt_resp_future.result()

            # Score BOTH responses in parallel
            with ThreadPoolExecutor(max_workers=2) as pool:
                orig_score_future = pool.submit(self.score_response, persona_text, original_query, original_response)
                opt_score_future = pool.submit(self.score_response, persona_text, optimized_prompt, optimized_response)
                original_scores = orig_score_future.result()
                optimized_scores = opt_score_future.result()

        # Calculate deltas
        persona_delta = optimized_scores["persona_score"] - original_scores["persona_score"]
        task_delta = optimized_scores["task_score"] - original_scores["task_score"]

        return {
            "original_query": original_query,
            "optimized_prompt": optimized_prompt,
            "original_response": original_response,
            "optimized_response": optimized_response,
            "original_scores": {
                "persona_score": original_scores["persona_score"],
                "task_score": original_scores["task_score"],
            },
            "optimized_scores": {
                "persona_score": optimized_scores["persona_score"],
                "task_score": optimized_scores["task_score"],
            },
            "delta": {
                "persona_delta": persona_delta,
                "task_delta": task_delta,
                "total_delta": (persona_delta + task_delta) / 2.0,
            },
            "persona_features": persona,
        }

    def compute_original_baseline(self, item: dict) -> dict:
        """Compute baseline for original query (response + scores).

        This can be computed once and reused across multiple checkpoints.
        """
        persona = item.get("persona_features", {})
        original_query = item.get("original_query") or item.get("current_query") or item.get("query") or item.get("prompt", "")

        original_response = self.get_assistant_response(original_query)
        persona_text = persona_to_text(persona)
        original_scores = self.score_response(persona_text, original_query, original_response)

        return {
            "original_query": original_query,
            "original_response": original_response,
            "original_scores": {
                "persona_score": original_scores["persona_score"],
                "task_score": original_scores["task_score"],
            },
        }


def run_eval_mode(runner: PPOptRunner, args, checkpoint_url: str = "auto"):
    """Full evaluation: optimize -> assistant -> judge."""
    if not args.input:
        print("[ERROR] --input/-i is required for eval mode")
        sys.exit(1)
    if not args.output:
        print("[ERROR] --output/-o is required for eval mode")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    n = len(items)
    if args.limit:
        n = min(n, args.limit)
        items = items[:n]

    print(f"Running PPOpt Evaluation")
    print(f"  Checkpoint: {checkpoint_url}")
    print(f"  Input:  {args.input} ({n} samples)")
    print(f"  Output: {args.output}")
    print()

    results = []
    total_orig_persona = 0
    total_orig_task = 0
    total_opt_persona = 0
    total_opt_task = 0

    for i, item in enumerate(items):
        print(f"[{i+1}/{n}] Processing...")
        result = runner.run_eval(item)
        # Add checkpoint URL identifier
        result["checkpoint_url"] = checkpoint_url
        results.append(result)

        orig = result["original_scores"]
        opt = result["optimized_scores"]
        delta = result["delta"]

        total_orig_persona += orig["persona_score"]
        total_orig_task += orig["task_score"]
        total_opt_persona += opt["persona_score"]
        total_opt_task += opt["task_score"]

        print(f"    Query: {result['original_query'][:50]}...")
        print(f"    Optimized: {result['optimized_prompt'][:50]}...")
        print(f"    Original:  persona={orig['persona_score']}, task={orig['task_score']}")
        print(f"    Optimized: persona={opt['persona_score']}, task={opt['task_score']}")
        print(f"    Delta:     persona={delta['persona_delta']:+d}, task={delta['task_delta']:+d}")

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {args.output}")

    evaluated = len(results)
    if evaluated > 0:
        print(f"\n{'='*60}")
        print(f"SUMMARY ({evaluated} samples)")
        print(f"{'='*60}")
        print(f"\nOriginal Scores:")
        print(f"  Avg Persona: {total_orig_persona/evaluated:.2f}")
        print(f"  Avg Task:    {total_orig_task/evaluated:.2f}")
        print(f"\nOptimized Scores:")
        print(f"  Avg Persona: {total_opt_persona/evaluated:.2f}")
        print(f"  Avg Task:    {total_opt_task/evaluated:.2f}")
        print(f"\nDelta (Improvement):")
        print(f"  Avg Persona Delta: {(total_opt_persona - total_orig_persona)/evaluated:+.2f}")
        print(f"  Avg Task Delta:    {(total_opt_task - total_orig_task)/evaluated:+.2f}")


def run_infer_mode(runner: PPOptRunner, args):
    """Inference only: optimize queries."""
    if not args.input:
        print("[ERROR] --input/-i is required for infer mode")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    n = len(items)
    if args.limit:
        n = min(n, args.limit)
        items = items[:n]

    print(f"Running PPOpt Inference")
    print(f"  Input:  {args.input} ({n} samples)")
    if args.output:
        print(f"  Output: {args.output}")
    print()

    results = []
    for i, item in enumerate(items):
        query = item.get("original_query") or item.get("current_query") or item.get("query") or item.get("prompt", "")
        input_text = item.get("input")

        if not input_text:
            history_raw = item.get("history", [])
            history = convert_history(history_raw) if history_raw else item.get("history_texts", [])
        else:
            history = None

        result = runner.optimize_query(query, history, input_text)
        results.append(result)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved to {args.output}")
    else:
        for r in results:
            print(json.dumps(r, ensure_ascii=False))


def run_interactive_mode(runner: PPOptRunner):
    """Interactive mode."""
    print("\n=== PPOpt Interactive Mode ===")
    print("Commands: /history, /clear, /show, quit\n")

    history_texts: List[str] = []

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower() == "/history":
            print("Enter history (empty line to finish):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            if lines:
                history_texts.append("\n".join(lines))
                print(f"Added session {len(history_texts)}")
            continue
        if query.lower() == "/clear":
            history_texts = []
            print("History cleared.")
            continue
        if query.lower() == "/show":
            for i, h in enumerate(history_texts, 1):
                print(f"[SESSION {i}]\n{h}\n")
            continue

        print("Optimizing...")
        result = runner.optimize_query(query, history_texts)
        print(f"\n[Reasoning]\n{result['reasoning']}\n")
        print(f"[Optimized Query]\n{result['optimized_prompt']}\n")


def run_single_checkpoint_inference(
    checkpoint_url: str,
    cfg: dict,
    items: List[dict],
    tinker_concurrency: int = 5,
) -> List[dict]:
    """Run Tinker inference for a single checkpoint (Phase 2).

    Only performs optimization, no API calls.
    Returns list of {item_idx, optimized_prompt, optimizer_output, ...}
    """
    print(f"  [TINKER] Initializing {checkpoint_url}...")
    # Use checkpoint_override to auto-detect chat template and use the correct checkpoint
    runner = PPOptRunner(cfg, checkpoint_override=checkpoint_url)

    def optimize_item(idx_item):
        idx, item = idx_item
        original_query = item.get("original_query") or item.get("current_query") or item.get("query") or item.get("prompt", "")
        input_text = item.get("input")

        if not input_text:
            history_raw = item.get("history", [])
            history_texts = convert_history(history_raw) if history_raw else item.get("history_texts", [])
        else:
            history_texts = None

        result = runner.optimize_query(original_query, history_texts, input_text)
        return {
            "item_idx": idx,
            "checkpoint_url": checkpoint_url,
            "original_query": original_query,
            "optimized_prompt": result["optimized_prompt"],
            "optimizer_output": result["optimizer_output"],
            "persona_features": item.get("persona_features", {}),
        }

    # Concurrent Tinker inference within this checkpoint
    results = []
    with ThreadPoolExecutor(max_workers=tinker_concurrency) as pool:
        futures = [pool.submit(optimize_item, (i, item)) for i, item in enumerate(items)]
        for future in as_completed(futures):
            results.append(future.result())

    # Sort by item_idx to maintain order
    results.sort(key=lambda x: x["item_idx"])
    print(f"  [TINKER] {checkpoint_url}: {len(results)} items optimized")
    return results


def run_parallel_eval(args, cfg: dict):
    """Run evaluation on multiple checkpoints in parallel.

    Three-phase optimization:
    - Phase 1: Compute original baselines (API: assistant + judge) - concurrent
    - Phase 2: Run Tinker inference for all checkpoints - concurrent (both across checkpoints and within each checkpoint)
    - Phase 3: Compute optimized responses and scores (API: assistant + judge) - concurrent
    """
    checkpoints = CHECKPOINT_URLS if CHECKPOINT_URLS else ["auto"]

    if not args.input:
        print("[ERROR] --input/-i is required for parallel mode")
        sys.exit(1)

    # Load items
    with open(args.input, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        items = items[:args.limit]

    print(f"Running parallel evaluation on {len(checkpoints)} checkpoints")
    print(f"  Checkpoint workers: {PARALLEL_WORKERS}")
    print(f"  API concurrency: {API_CONCURRENCY}")
    print(f"  Input: {args.input} ({len(items)} samples)")
    print(f"  Checkpoints: {checkpoints}")
    print()

    # =========================================================================
    # Phase 1: Compute original baselines (API calls only, no Tinker)
    # =========================================================================
    print("[PHASE 1] Computing original baselines (one-time)...")
    baseline_runner = PPOptRunner(cfg, skip_optimizer=True)

    original_baselines = [None] * len(items)
    with ThreadPoolExecutor(max_workers=API_CONCURRENCY) as pool:
        futures = {pool.submit(baseline_runner.compute_original_baseline, item): i for i, item in enumerate(items)}
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            original_baselines[idx] = future.result()
            completed += 1
            if completed % 10 == 0:
                print(f"  Baselines: {completed}/{len(items)}")

    print(f"  Computed {len(original_baselines)} baselines")
    print()

    # =========================================================================
    # Phase 2: Tinker inference for all checkpoints (concurrent)
    # =========================================================================
    print("[PHASE 2] Running Tinker inference (all checkpoints parallel)...")

    # Each checkpoint runs with internal concurrency for items
    tinker_per_checkpoint = max(1, API_CONCURRENCY // len(checkpoints))

    all_inference_results = {}  # checkpoint_url -> list of results
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {
            executor.submit(
                run_single_checkpoint_inference,
                ckpt, cfg, items, tinker_per_checkpoint
            ): ckpt
            for ckpt in checkpoints
        }
        for future in as_completed(futures):
            ckpt = futures[future]
            try:
                all_inference_results[ckpt] = future.result()
            except Exception as e:
                print(f"  [ERROR] Tinker inference failed for {ckpt}: {e}")
                all_inference_results[ckpt] = []

    total_inferences = sum(len(v) for v in all_inference_results.values())
    print(f"  Total inferences: {total_inferences}")
    print()

    # =========================================================================
    # Phase 3: Get optimized responses and scores (API calls, high concurrency)
    # =========================================================================
    print("[PHASE 3] Computing optimized responses and scores (API parallel)...")

    # Flatten all (checkpoint, item) pairs for batch API calls
    api_tasks = []
    for ckpt, infer_results in all_inference_results.items():
        for infer_res in infer_results:
            api_tasks.append({
                "checkpoint_url": ckpt,
                "item_idx": infer_res["item_idx"],
                "original_query": infer_res["original_query"],
                "optimized_prompt": infer_res["optimized_prompt"],
                "optimizer_output": infer_res["optimizer_output"],
                "persona_features": infer_res["persona_features"],
            })

    def process_api_task(task):
        """Get assistant response and judge scores for optimized prompt."""
        optimized_prompt = task["optimized_prompt"]
        persona_features = task["persona_features"]
        persona_text = persona_to_text(persona_features)

        # Get optimized response
        optimized_response = baseline_runner.get_assistant_response(optimized_prompt)

        # Score the response
        optimized_scores = baseline_runner.score_response(persona_text, optimized_prompt, optimized_response)

        return {
            "checkpoint_url": task["checkpoint_url"],
            "item_idx": task["item_idx"],
            "optimized_response": optimized_response,
            "optimized_scores": {
                "persona_score": optimized_scores["persona_score"],
                "task_score": optimized_scores["task_score"],
            },
        }

    api_results = []
    with ThreadPoolExecutor(max_workers=API_CONCURRENCY) as pool:
        futures = [pool.submit(process_api_task, task) for task in api_tasks]
        completed = 0
        for future in as_completed(futures):
            api_results.append(future.result())
            completed += 1
            if completed % 20 == 0:
                print(f"  API calls: {completed}/{len(api_tasks)}")

    print(f"  Completed {len(api_results)} API calls")
    print()

    # =========================================================================
    # Phase 4: Merge results and save
    # =========================================================================
    print("[PHASE 4] Merging results and saving...")

    # Build lookup: (checkpoint_url, item_idx) -> api_result
    api_lookup = {(r["checkpoint_url"], r["item_idx"]): r for r in api_results}

    # Build final results per checkpoint
    summaries = []
    for ckpt in checkpoints:
        infer_results = all_inference_results.get(ckpt, [])
        final_results = []

        for infer_res in infer_results:
            item_idx = infer_res["item_idx"]
            api_res = api_lookup.get((ckpt, item_idx))
            baseline = original_baselines[item_idx]

            if not api_res or not baseline:
                continue

            # Calculate deltas
            orig_scores = baseline["original_scores"]
            opt_scores = api_res["optimized_scores"]
            persona_delta = opt_scores["persona_score"] - orig_scores["persona_score"]
            task_delta = opt_scores["task_score"] - orig_scores["task_score"]

            final_results.append({
                "original_query": infer_res["original_query"],
                "optimized_prompt": infer_res["optimized_prompt"],
                "original_response": baseline["original_response"],
                "optimized_response": api_res["optimized_response"],
                "original_scores": orig_scores,
                "optimized_scores": opt_scores,
                "delta": {
                    "persona_delta": persona_delta,
                    "task_delta": task_delta,
                    "total_delta": (persona_delta + task_delta) / 2.0,
                },
                "persona_features": infer_res["persona_features"],
                "checkpoint_url": ckpt,
            })

        # Generate output filename
        if ckpt == "auto":
            output_file = args.output or "eval_result_auto.jsonl"
        else:
            # For tinker:// URLs, extract UUID and sampler name
            # e.g., tinker://cb1866d3-8a10-5700-ae66-5454e86f4fe0:train:0/sampler_weights/ppopt_rl_llama8b_sampler
            if ckpt.startswith("tinker://"):
                # Extract UUID (first 8 chars) and sampler name
                import re as re_module
                uuid_match = re_module.search(r"tinker://([a-f0-9-]+):", ckpt)
                uuid_short = uuid_match.group(1)[:8] if uuid_match else ""
                sampler_name = Path(ckpt).stem  # ppopt_rl_llama8b_sampler
                if uuid_short:
                    ckpt_name = f"{sampler_name}_{uuid_short}"
                else:
                    ckpt_name = sampler_name
            else:
                # Local path: use run_ or step_ parts
                parts = Path(ckpt).parts
                ckpt_name = "_".join(p for p in parts if p.startswith("run_") or p.startswith("step_"))
                if not ckpt_name:
                    ckpt_name = Path(ckpt).stem
            output_file = f"eval_result_{ckpt_name}.jsonl"

        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            for r in final_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Calculate summary
        if final_results:
            avg_persona_delta = sum(r["delta"]["persona_delta"] for r in final_results) / len(final_results)
            avg_task_delta = sum(r["delta"]["task_delta"] for r in final_results) / len(final_results)
            avg_opt_persona = sum(r["optimized_scores"]["persona_score"] for r in final_results) / len(final_results)
            avg_opt_task = sum(r["optimized_scores"]["task_score"] for r in final_results) / len(final_results)
        else:
            avg_persona_delta = avg_task_delta = avg_opt_persona = avg_opt_task = 0.0

        summary = {
            "checkpoint": ckpt,
            "output_file": output_file,
            "num_samples": len(final_results),
            "avg_persona_delta": avg_persona_delta,
            "avg_task_delta": avg_task_delta,
            "avg_opt_persona": avg_opt_persona,
            "avg_opt_task": avg_opt_task,
        }
        summaries.append(summary)

        print(f"  [{ckpt}] Saved {len(final_results)} results to {output_file}")

    # Print final summary
    print(f"\n{'='*60}")
    print("PARALLEL EVALUATION SUMMARY")
    print(f"{'='*60}")

    # Calculate original baseline averages
    if original_baselines:
        avg_orig_persona = sum(b["original_scores"]["persona_score"] for b in original_baselines) / len(original_baselines)
        avg_orig_task = sum(b["original_scores"]["task_score"] for b in original_baselines) / len(original_baselines)
        print(f"\nOriginal Baseline (shared):")
        print(f"  Avg Persona: {avg_orig_persona:.2f}")
        print(f"  Avg Task:    {avg_orig_task:.2f}")

    for s in summaries:
        print(f"\n{s['checkpoint']}:")
        print(f"  Output: {s['output_file']}")
        print(f"  Samples: {s['num_samples']}")
        print(f"  Optimized Scores: persona={s['avg_opt_persona']:.2f}, task={s['avg_opt_task']:.2f}")
        print(f"  Delta: persona={s['avg_persona_delta']:+.2f}, task={s['avg_task_delta']:+.2f}")

    return summaries


def main():
    parser = argparse.ArgumentParser(description="PPOpt Inference + Evaluation")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--eval", action="store_true", default=True, help="Full evaluation (default)")
    mode_group.add_argument("--infer", action="store_true", help="Inference only")
    mode_group.add_argument("--interactive", action="store_true", help="Interactive mode")
    mode_group.add_argument("--parallel", "-p", action="store_true", help="Parallel evaluation on CHECKPOINT_URLS")

    # I/O
    parser.add_argument("--input", "-i", help="Input JSONL file")
    parser.add_argument("--output", "-o", help="Output JSONL file")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limit number of samples")

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    if args.parallel:
        run_parallel_eval(args, cfg)
    elif args.interactive:
        runner = PPOptRunner(cfg)
        run_interactive_mode(runner)
    elif args.infer:
        runner = PPOptRunner(cfg)
        run_infer_mode(runner, args)
    else:
        # Get checkpoint URL from config for result tracking
        checkpoint_url = cfg.get("eval", {}).get("checkpoint_path", "auto")
        runner = PPOptRunner(cfg)
        run_eval_mode(runner, args, checkpoint_url)


if __name__ == "__main__":
    main()
