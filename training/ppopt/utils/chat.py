"""Chat formatting helpers."""

from __future__ import annotations

from typing import Dict


CHAT_TEMPLATES: Dict[str, Dict[str, object]] = {
    "qwen": {
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
        "assistant_prefix": "<|im_start|>assistant\n",
        "stop": ["<|im_end|>"]
    },
    "llama": {
        "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant_prefix": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "stop": ["<|eot_id|>"]
    },
    "harmony": {
        "system": "<|start|>developer<|message|>{content}<|end|>",
        "user": "<|start|>user<|message|>{content}<|end|>",
        "assistant": "<|start|>assistant<|channel|>final<|message|>{content}<|end|>",
        "assistant_prefix": "<|start|>assistant<|channel|>final<|message|>",
        "stop": ["<|end|>", "<|return|>"]
    }
}


def get_chat_template(model_name: str, override: str | None = None) -> Dict[str, object]:
    if override and override in CHAT_TEMPLATES:
        return CHAT_TEMPLATES[override]
    name = model_name.lower()
    if "llama" in name:
        return CHAT_TEMPLATES["llama"]
    if "gpt-oss" in name:
        return CHAT_TEMPLATES["harmony"]
    return CHAT_TEMPLATES["qwen"]


def build_io_prompt(template: Dict[str, object], system_prompt: str, user_input: str) -> str:
    sys_block = template["system"].format(content=system_prompt) if system_prompt else ""
    user_block = template["user"].format(content=user_input)
    return f"{sys_block}{user_block}{template['assistant_prefix']}"
