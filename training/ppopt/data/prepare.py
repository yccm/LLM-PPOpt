"""Data preparation for PPOpt SFT/RL."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import json

from ppopt.utils.formatting import conversation_to_text, persona_to_text, build_state_text


# Default system prompt for optimizer
DEFAULT_OPTIMIZER_SYSTEM_PROMPT = """You are a PROMPT OPTIMIZER, NOT an assistant. Your ONLY job is to analyze and rewrite queries.

CRITICAL: You must NEVER answer the user's question directly. You must ONLY output the analysis and rewritten query.

RULES:
1. Output ONLY the <REASONING>...</REASONING><PROMPT>...</PROMPT> format
2. NEVER answer the query content itself
3. NEVER output JSON, lists, stories, or any other format
4. If the original query is already good, keep it mostly unchanged in <PROMPT>"""


def load_json_array(path: str | Path) -> List[Dict[str, Any]]:
    """Load JSON array or JSONL file.

    Supports:
    1. JSON array: [{"key": "value"}, ...]
    2. Standard JSONL: one JSON object per line
    3. Pretty-printed concatenated JSON objects (auto-detected)
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw_text = f.read()

    # Try to parse as JSON array first
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            return data
        # Single object, wrap in list
        return [data]
    except json.JSONDecodeError:
        pass

    # Try to parse as JSONL (one JSON object per line)
    lines = raw_text.splitlines()
    items: List[Dict[str, Any]] = []

    # Check if it's standard JSONL (each line is a complete JSON)
    first_non_empty = next((l.strip() for l in lines if l.strip()), "")
    if first_non_empty.startswith("{") and first_non_empty.endswith("}"):
        # Standard JSONL format
        for line in lines:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
        return items

    # Handle pretty-printed concatenated JSON objects
    # Use JSONDecoder to parse multiple objects from the stream
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(raw_text):
        # Skip whitespace
        while idx < len(raw_text) and raw_text[idx] in ' \t\n\r':
            idx += 1
        if idx >= len(raw_text):
            break
        try:
            obj, end_idx = decoder.raw_decode(raw_text, idx)
            items.append(obj)
            idx = end_idx  # raw_decode returns absolute position, not offset
        except json.JSONDecodeError:
            break

    return items


def save_jsonl(items: List[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_interaction_map(train_samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for item in train_samples:
        sample_id = item.get("sample_id")
        if sample_id:
            mapping[sample_id] = item
    return mapping


def build_history_texts(
    history_ids: List[str],
    interaction_map: Dict[str, Dict[str, Any]],
    max_sessions: int,
    include_assistant_content: bool = True
) -> List[str]:
    texts: List[str] = []
    for idx, sid in enumerate(history_ids[:max_sessions]):
        item = interaction_map.get(sid)
        if not item:
            continue
        conv = item.get("full_conversation") or []
        texts.append(conversation_to_text(conv, include_assistant_content))
    return texts


def build_sft_and_rl(
    optimized_queries_path: str,
    train_samples_path: str,
    sft_out_path: str,
    rl_out_path: str,
    max_history_sessions: int = 4,
    include_persona: bool = True,
    use_persona_in_input: bool = False,
    require_history: bool = True,
    min_history_sessions: int = 1,
    include_assistant_content: bool = True,
    system_prompt: str = "",
) -> Dict[str, int]:
    optimized = load_json_array(optimized_queries_path)
    train_samples = load_json_array(train_samples_path)
    interaction_map = build_interaction_map(train_samples)

    sft_items: List[Dict[str, Any]] = []
    rl_items: List[Dict[str, Any]] = []
    skipped_no_history = 0

    for item in optimized:
        if not item.get("success"):
            continue
        gpt_resp = item.get("gpt_response") or {}
        if not gpt_resp.get("parse_success"):
            continue
        original_query = item.get("original_next_query") or ""
        if not original_query:
            continue

        persona = item.get("persona_features") if include_persona else None
        persona_text = persona_to_text(persona)
        history_ids = item.get("history_interaction_ids") or []
        history_texts = build_history_texts(history_ids, interaction_map, max_history_sessions, include_assistant_content)
        if require_history and len(history_texts) < min_history_sessions:
            skipped_no_history += 1
            continue
        input_persona = persona_text if use_persona_in_input else ""
        state_text = build_state_text(input_persona, history_texts, original_query, system_prompt)

        reasoning = gpt_resp.get("reasoning", "")
        optimized_query = gpt_resp.get("optimized_query", "")
        if not optimized_query:
            continue

        sft_items.append({
            "input": state_text,
            "output": f"<REASONING>{reasoning}</REASONING><PROMPT>{optimized_query}</PROMPT>",
        })
        rl_items.append({
            "state_id": item.get("target_interaction_id"),
            "persona_features": persona or {},
            "history_texts": history_texts,
            "original_query": original_query,
            "input": state_text,
        })

    save_jsonl(sft_items, sft_out_path)
    save_jsonl(rl_items, rl_out_path)
    return {"sft": len(sft_items), "rl": len(rl_items), "skipped_no_history": skipped_no_history}
