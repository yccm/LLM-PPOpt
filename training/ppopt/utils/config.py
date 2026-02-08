"""Config helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import os
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def apply_env(cfg: Dict[str, Any]) -> None:
    api_cfg = cfg.get("api", {})
    api_key = api_cfg.get("tinker_api_key")
    base_url = api_cfg.get("tinker_base_url")
    if api_key:
        os.environ["TINKER_API_KEY"] = api_key
    if base_url:
        os.environ["TINKER_BASE_URL"] = base_url

    # HuggingFace token for gated models (e.g., Llama)
    hf_cfg = cfg.get("huggingface", {})
    hf_token = hf_cfg.get("token")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token


def create_run_dir(base_dir: str | Path, prefix: str = "run") -> Path:
    """Create a timestamped run directory for training outputs.

    Args:
        base_dir: Base checkpoint directory (e.g., "checkpoints")
        prefix: Prefix for run directory name (default: "run")

    Returns:
        Path to the created run directory (e.g., "checkpoints/run_20260201_143025")

    Example:
        run_dir = create_run_dir("checkpoints")
        # Returns: Path("checkpoints/run_20260201_143025")
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_path / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir
