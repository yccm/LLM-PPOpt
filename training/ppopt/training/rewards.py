"""Reward computation using LLM-as-judge via OpenAI Chat Completions API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import logging
import time

from ppopt.utils.formatting import safe_parse_score

logger = logging.getLogger(__name__)


def _extract_status_code(exc: Exception) -> str:
    """Extract HTTP status code from OpenAI API exceptions (e.g., '429', '502')."""
    # openai.APIStatusError has .status_code
    code = getattr(exc, "status_code", None)
    if code is not None:
        return str(code)
    # httpx.HTTPStatusError has .response.status_code
    resp = getattr(exc, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        if code is not None:
            return str(code)
    return ""


PROFILE_JUDGE_SYSTEM = (
    "You are a precise evaluation assistant. Evaluate how well an inferred user profile "
    "matches the ground-truth persona using the following 1-5 rubric:\n\n"
    "## Scoring Rubric\n"
    "- **5 (Excellent)**: The inferred profile captures all key aspects of the persona accurately, "
    "including communication style, expertise level, preferences, and constraints. No significant omissions or errors.\n"
    "- **4 (Good)**: The inferred profile captures most key aspects correctly with minor omissions "
    "or slight inaccuracies that don't fundamentally misrepresent the user.\n"
    "- **3 (Moderate)**: The inferred profile captures some aspects correctly but misses important "
    "preferences or contains noticeable inaccuracies. Partially useful for personalization.\n"
    "- **2 (Poor)**: The inferred profile captures only superficial aspects or contains significant "
    "errors that would lead to inappropriate personalization.\n"
    "- **1 (Very Poor)**: The inferred profile is mostly incorrect, irrelevant, or completely misses "
    "the user's actual preferences and characteristics.\n\n"
    "Output only a single integer (1-5) and nothing else."
)

PROFILE_JUDGE_USER = (
    "Ground-truth persona:\n{persona}\n\n"
    "Inferred profile:\n{profile}\n\n"
    "Score (1-5):"
)

TASK_JUDGE_SYSTEM = (
    "You are a precise evaluation assistant. You compare two assistant responses (A and B) "
    "for a user query, grounded in the user's persona and preferences. "
    "If Response B is clearly better than Response A, reply with exactly the word \"win\". "
    "If Response A is clearly better, reply with exactly \"lose\". "
    "If the responses are equivalent, reply with exactly \"tie\". "
    "Consider task correctness, personalization alignment, and constraint satisfaction, "
    "and output only one of win, lose, or tie."
)

TASK_JUDGE_USER = (
    "User persona:\n{persona}\n\n"
    "User query:\n{query}\n\n"
    "Response A (original prompt):\n{base}\n\n"
    "Response B (optimized prompt):\n{opt}\n\n"
    "Outcome (win/lose/tie):"
)


@dataclass
class RewardConfig:
    lambda_prof: float
    lambda_task: float
    max_output_tokens: int
    temperature: float
    top_p: float
    model: str
    max_retries: int = 3
    retry_base_delay: float = 1.0
    use_profile_reward: bool = True  # false = only task reward, true = combined reward


class RewardComputer:
    def __init__(self, openai_client, cfg: RewardConfig):
        self.client = openai_client
        self.cfg = cfg

    def _judge(self, system_prompt: str, user_prompt: str) -> str:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_exc: Exception | None = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    max_tokens=self.cfg.max_output_tokens,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                )
                text = response.choices[0].message.content or ""
                return text.strip()
            except Exception as e:
                last_exc = e
                status = _extract_status_code(e)
                status_str = f" HTTP {status}" if status else ""
                if attempt < self.cfg.max_retries:
                    delay = self.cfg.retry_base_delay * (2 ** (attempt - 1))
                    print(f"[Judge] Retry {attempt}/{self.cfg.max_retries}{status_str} {type(e).__name__}: {e} (wait {delay:.0f}s)")
                    time.sleep(delay)
        status = _extract_status_code(last_exc) if last_exc else ""
        status_str = f" HTTP {status}" if status else ""
        print(f"[Judge] FAILED after {self.cfg.max_retries} retries{status_str}: {last_exc}")
        return ""

    def profile_reward(self, inferred_profile: str, persona_text: str) -> Optional[float]:
        user_prompt = PROFILE_JUDGE_USER.format(profile=inferred_profile, persona=persona_text)
        raw = self._judge(PROFILE_JUDGE_SYSTEM, user_prompt)
        if not raw:
            return None
        return safe_parse_score(raw)

    def _parse_task_outcome(self, text: str) -> float:
        normalized = text.strip().lower()
        if normalized == "win":
            return 1.0
        if normalized == "lose":
            return 0.0
        if normalized == "tie":
            return 0.0  # tie also maps to 0
        if "win" in normalized:
            return 1.0
        if "lose" in normalized or "loss" in normalized:
            return 0.0
        if "tie" in normalized or "draw" in normalized:
            return 0.0  # tie also maps to 0
        return 0.0  # default to 0 (was 0.5)

    def task_reward(self, query: str, persona_text: str, base: str, opt: str) -> Optional[float]:
        user_prompt = TASK_JUDGE_USER.format(query=query, persona=persona_text, base=base, opt=opt)
        verdict = self._judge(TASK_JUDGE_SYSTEM, user_prompt)
        if not verdict:
            return None
        return self._parse_task_outcome(verdict)

    def total_reward(self, inferred_profile: str, persona_text: str, query: str, base: str, opt: str) -> Dict[str, Any]:
        if self.cfg.use_profile_reward:
            r_prof = self.profile_reward(inferred_profile, persona_text)
            r_task = self.task_reward(query, persona_text, base, opt)
            failed = r_prof is None or r_task is None
            if failed:
                return {"r_prof": 0.0, "r_task": 0.0, "total": 0.0, "failed": True}
            total = self.cfg.lambda_prof * r_prof + self.cfg.lambda_task * r_task
            return {"r_prof": r_prof, "r_task": r_task, "total": total, "failed": False}
        else:
            # Only use task reward
            r_task = self.task_reward(query, persona_text, base, opt)
            if r_task is None:
                return {"r_prof": None, "r_task": 0.0, "total": 0.0, "failed": True}
            return {"r_prof": None, "r_task": r_task, "total": r_task, "failed": False}
