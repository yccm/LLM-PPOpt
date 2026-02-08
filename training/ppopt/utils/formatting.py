"""Formatting helpers for PPOpt inputs/outputs."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Any


def persona_to_text(persona: Dict[str, Any] | None) -> str:
    if not persona:
        return ""
    lines = []
    for k in sorted(persona.keys()):
        lines.append(f"- {k}: {persona[k]}")
    return "\n".join(lines)


def conversation_to_text(messages: List[Dict[str, Any]], include_assistant_content: bool = True) -> str:
    lines: List[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "assistant":
            if include_assistant_content:
                if not content:
                    continue
                lines.append(content)
            else:
                lines.append("[ASSISTANT_RESPONSE]")
            lines.append("")  # blank line after assistant response
        elif role == "user":
            if not content:
                continue
            lines.append(f"User: {content}")
        else:
            if not content:
                continue
            lines.append(f"{role.capitalize()}: {content}")
    # Remove trailing blank lines
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def build_state_text(
    persona_text: str,
    history_texts: List[str],
    current_query: str,
    system_prompt: str = ""
) -> str:
    sections: List[str] = []

    sections.append(
        "Your task is to analyze the user's historical conversations, "
        "infer their preferences and communication style, and then optimize "
        "the user's current query."
    )

    # Historical Conversations
    if history_texts:
        sections.append("## Historical Conversations")
        sections.append(
            "Below are the user's past conversation logs. Each conversation "
            "includes the user's messages and the assistant's replies "
            "(assistant replies are replaced with the placeholder [ASSISTANT_RESPONSE])."
        )
        for i, h in enumerate(history_texts, start=1):
            sections.append(f"### Conversation {i}\n\n{h}\n\n---")

    # Current Query
    sections.append(f"## Current Query\n\nUser: {current_query}")

    # Task
    sections.append(
        "## Your Task\n\n"
        "Based on the historical conversations above, analyze the user's "
        "communication style, preferences, and expectations.\n\n"
        "Then rewrite the \"Current Query\" into a more precise prompt that "
        "better matches the user's inferred style.\n\n"
        "**CRITICAL INSTRUCTIONS:**\n"
        "- You are a PROMPT OPTIMIZER, NOT a general assistant.\n"
        "- You MUST NOT answer the user's question directly.\n"
        "- You MUST NOT include any reasoning, analysis, or explanations.\n"
        "- You MUST output ONLY the optimized query text.\n"
        "- Do NOT wrap the output in any tags or markers.\n\n"
        "Output ONLY the optimized prompt:"
    )

    return "\n\n".join(sections)


def parse_optimizer_output(text: str) -> Dict[str, str]:
    """Parse <REASONING> and <PROMPT> blocks from model output."""
    reasoning = ""
    prompt = ""
    m = re.search(r"<REASONING>(.*?)</REASONING>", text, re.DOTALL | re.IGNORECASE)
    if m:
        reasoning = m.group(1).strip()
    m = re.search(r"<PROMPT>(.*?)</PROMPT>", text, re.DOTALL | re.IGNORECASE)
    if m:
        prompt = m.group(1).strip()
    if not prompt:
        # fallback: treat entire output as prompt
        prompt = text.strip()
    return {"reasoning": reasoning, "prompt": prompt}


def safe_parse_score(text: str, min_val: float = 0.0, max_val: float = 1.0, scale: int = 5) -> float:
    """Parse a numeric score from judge output and normalize to [min_val, max_val].

    Args:
        text: Raw text output from judge
        min_val: Minimum value of output range (default 0.0)
        max_val: Maximum value of output range (default 1.0)
        scale: The scale used by the judge (default 5 for 1-5 scale)

    Returns:
        Normalized score in [min_val, max_val]
    """
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return 0.0
    val = float(numbers[0])
    # Normalize from 1-scale to 0-1 range
    # e.g., scale=5: score 1 -> 0.0, score 5 -> 1.0
    # e.g., scale=10: score 1 -> 0.0, score 10 -> 1.0
    if val >= 1.0 and val <= scale and max_val <= 1.0:
        val = (val - 1.0) / (scale - 1.0)
    if val < min_val:
        val = min_val
    if val > max_val:
        val = max_val
    return val


def to_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
