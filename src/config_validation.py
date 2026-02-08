"""
Configuration Validation Module

Provides early, fail-fast validation for config.yaml to prevent mid-run crashes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os


@dataclass
class ValidationIssue:
    level: str  # "error" | "warning"
    message: str
    path: Optional[str] = None


@dataclass
class ValidationCheck:
    name: str
    status: str  # "PASS" | "FAIL" | "WARN"
    details: str = ""
    path: Optional[str] = None


def _get(config: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = config
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _file_exists(base_dir: Path, maybe_path: Optional[str]) -> bool:
    if not maybe_path:
        return False
    p = Path(maybe_path)
    if not p.is_absolute():
        p = base_dir / p
    return p.exists()


_PLACEHOLDER_KEYS = {
    "YOUR_API_KEY_HERE",
    "YOUR_OPENAI_API_KEY_HERE",
    "YOUR_OPENROUTER_API_KEY_HERE",
    "YOUR_ANTHROPIC_API_KEY_HERE",
    "",
}


def _is_placeholder_key(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    v = value.strip()
    return v in _PLACEHOLDER_KEYS or v.upper() in _PLACEHOLDER_KEYS


def _resolve_api_key(provider: str, explicit_key: Optional[str]) -> Tuple[Optional[str], Optional[str], str]:
    """
    Returns (api_key, source, expected_env_var).
    """
    expected_env_var = f"{provider.upper()}_API_KEY"

    if explicit_key and not _is_placeholder_key(explicit_key):
        return explicit_key, "config", expected_env_var

    env_val = os.environ.get(expected_env_var)
    if env_val and not _is_placeholder_key(env_val):
        return env_val, "env", expected_env_var

    # Common aliases
    if provider == "openai":
        env_val = os.environ.get("OPENAI_API_KEY")
        if env_val and not _is_placeholder_key(env_val):
            return env_val, "env", expected_env_var
    if provider == "openrouter":
        env_val = os.environ.get("OPENROUTER_API_KEY")
        if env_val and not _is_placeholder_key(env_val):
            return env_val, "env", expected_env_var
    if provider == "anthropic":
        env_val = os.environ.get("ANTHROPIC_API_KEY")
        if env_val and not _is_placeholder_key(env_val):
            return env_val, "env", expected_env_var

    return None, None, expected_env_var


def validate_config_report(
    config: Dict[str, Any],
    *,
    config_path: str
) -> Tuple[Dict[str, Any], List[ValidationCheck], List[ValidationIssue]]:
    """
    Validate config and return a report (checks + issues) WITHOUT raising.
    """
    issues: List[ValidationIssue] = []
    checks: List[ValidationCheck] = []
    base_dir = Path(config_path).resolve().parent

    def ok(name: str, details: str = "", path: Optional[str] = None):
        checks.append(ValidationCheck(name=name, status="PASS", details=details, path=path))

    def warn(name: str, details: str, path: Optional[str] = None):
        checks.append(ValidationCheck(name=name, status="WARN", details=details, path=path))
        issues.append(ValidationIssue("warning", details, path))

    def fail(name: str, details: str, path: Optional[str] = None):
        checks.append(ValidationCheck(name=name, status="FAIL", details=details, path=path))
        issues.append(ValidationIssue("error", details, path))

    # ---- Basic structure ----
    if not isinstance(config, dict):
        fail("Config is a YAML mapping", "Config must be a mapping (YAML dict).")
        return config, checks, issues
    ok("Config is a YAML mapping")

    if "paths" not in config:
        fail("Required section: paths", "Missing required section: paths", "paths")
    else:
        ok("Required section: paths")

    if "interaction_generation" not in config:
        fail("Required section: interaction_generation", "Missing required section: interaction_generation", "interaction_generation")
    else:
        ok("Required section: interaction_generation")

    # ---- Required files exist ----
    persona_path = _get(config, "paths.persona_bank.persona")
    if persona_path and _file_exists(base_dir, str(persona_path)):
        ok("Persona bank file exists", str(persona_path), "paths.persona_bank.persona")
    else:
        fail("Persona bank file exists", f"Persona bank file not found: {persona_path}", "paths.persona_bank.persona")

    query_datasets = _get(config, "paths.query_datasets", [])
    if isinstance(query_datasets, list) and query_datasets:
        missing = [str(qp) for qp in query_datasets if not _file_exists(base_dir, str(qp))]
        if missing:
            fail("Query datasets exist", f"Missing dataset files: {missing}", "paths.query_datasets")
        else:
            ok("Query datasets exist", f"{len(query_datasets)} file(s)", "paths.query_datasets")
    else:
        fail("Query datasets exist", "paths.query_datasets must be a non-empty list", "paths.query_datasets")

    # Templates
    templates = [
        ("Persona->system prompt template exists", "formulation.system_prompt_template"),
        ("Query style transfer template exists", "query_generation.style_transfer.template"),
        ("User feedback template exists", "interaction_generation.user_model.system_prompt_template"),
    ]
    for check_name, dotted in templates:
        pth = _get(config, dotted)
        if pth and _file_exists(base_dir, str(pth)):
            ok(check_name, str(pth), dotted)
        else:
            fail(check_name, f"Template file not found: {pth}", dotted)

    distractor_llm_enabled = bool(_get(config, "distractor.llm.enabled", False))
    if distractor_llm_enabled:
        pth = _get(config, "distractor.llm.template")
        if pth and _file_exists(base_dir, str(pth)):
            ok("Distractor LLM template exists", str(pth), "distractor.llm.template")
        else:
            fail("Distractor LLM template exists", f"Distractor LLM template not found: {pth}", "distractor.llm.template")
    else:
        ok("Distractor LLM template exists", "Skipped (distractor.llm.enabled=false)")

    # ---- Experiment batch size ----
    batch_size = _get(config, "experiment.batch_size")
    if batch_size is not None:
        try:
            bs = int(batch_size)
            if bs <= 0:
                fail("Experiment batch_size valid", "experiment.batch_size must be a positive integer", "experiment.batch_size")
            else:
                ok("Experiment batch_size valid", str(bs), "experiment.batch_size")
        except Exception:
            fail("Experiment batch_size valid", "experiment.batch_size must be an integer", "experiment.batch_size")

    # ---- API key checks (not placeholder; config or env) ----
    primary_provider = str(_get(config, "api.provider", "openai")).lower()
    primary_key = _get(config, "api.api_key")
    resolved, src, env_var = _resolve_api_key(primary_provider, primary_key)
    if resolved:
        ok("Primary provider API key present", f"{primary_provider} ({src})", "api.api_key")
        if primary_provider == "openai" and isinstance(resolved, str) and not resolved.startswith("sk-"):
            warn("Primary provider API key format", "OpenAI keys usually start with 'sk-'. This may still work depending on your setup.", "api.api_key")
    else:
        fail("Primary provider API key present", f"Missing/placeholder API key for provider '{primary_provider}'. Set api.api_key or env {env_var}.", "api.api_key")

    # ---- Assistant model pool ----
    assistant_cfg = _get(config, "interaction_generation.assistant_model", {})
    if not isinstance(assistant_cfg, dict):
        fail("assistant_model is a mapping", "interaction_generation.assistant_model must be a mapping", "interaction_generation.assistant_model")
        assistant_cfg = {}
    else:
        ok("assistant_model is a mapping")

    model_pool = assistant_cfg.get("model_pool")
    uses_openrouter = False
    if isinstance(model_pool, list):
        uses_openrouter = any(isinstance(e, dict) and e.get("provider") == "openrouter" for e in model_pool)

    if model_pool is not None:
        if not isinstance(model_pool, list) or not model_pool:
            fail("assistant model_pool non-empty", "assistant_model.model_pool must be a non-empty list", "interaction_generation.assistant_model.model_pool")
        else:
            ok("assistant model_pool non-empty", f"{len(model_pool)} model(s)", "interaction_generation.assistant_model.model_pool")
            weights: List[float] = []
            for idx, entry in enumerate(model_pool):
                base_path = f"interaction_generation.assistant_model.model_pool[{idx}]"
                if not isinstance(entry, dict):
                    fail("model_pool entry is a mapping", "Each model_pool entry must be a mapping", base_path)
                    continue
                provider = entry.get("provider")
                model = entry.get("model")
                if not provider or not isinstance(provider, str):
                    fail("model_pool entry provider", "Missing/invalid provider", f"{base_path}.provider")
                if not model or not isinstance(model, str):
                    fail("model_pool entry model", "Missing/invalid model", f"{base_path}.model")

                w = entry.get("weight")
                if w is None:
                    fail("model_pool entry weight", "Missing weight (weights must sum to 1.0)", f"{base_path}.weight")
                else:
                    try:
                        wf = float(w)
                        if wf < 0:
                            fail("model_pool entry weight", "Weight must be >= 0", f"{base_path}.weight")
                        else:
                            weights.append(wf)
                    except Exception:
                        fail("model_pool entry weight", "Weight must be a number", f"{base_path}.weight")

            if weights:
                s = sum(weights)
                if s <= 0:
                    fail("model_pool weights sum", "Sum of model_pool weights must be > 0", "interaction_generation.assistant_model.model_pool")
                elif abs(s - 1.0) > 1e-6:
                    fail("model_pool weights sum", f"Sum of model_pool weights must be 1.0 (got {s}).", "interaction_generation.assistant_model.model_pool")
                else:
                    ok("model_pool weights sum", "1.0", "interaction_generation.assistant_model.model_pool")
    else:
        # Legacy single-model config
        if assistant_cfg.get("provider") and assistant_cfg.get("model"):
            ok("assistant single model configured", f"{assistant_cfg.get('provider')}/{assistant_cfg.get('model')}", "interaction_generation.assistant_model")
        else:
            fail("assistant configured", "assistant_model must define either model_pool OR (provider + model)", "interaction_generation.assistant_model")

    # ---- OpenRouter config (if used) ----
    if uses_openrouter or _get(config, "openrouter", None) is not None:
        or_key = _get(config, "openrouter.api_key")
        resolved_or, src_or, env_or = _resolve_api_key("openrouter", or_key)
        if resolved_or:
            ok("OpenRouter API key present", f"openrouter ({src_or})", "openrouter.api_key")
            if isinstance(resolved_or, str) and not resolved_or.startswith("sk-"):
                warn("OpenRouter API key format", "OpenRouter keys often start with 'sk-'. This may still work.", "openrouter.api_key")
        else:
            fail("OpenRouter API key present", f"Missing/placeholder OpenRouter key. Set openrouter.api_key or env {env_or}.", "openrouter.api_key")

        endpoint = _get(config, "openrouter.endpoint")
        if endpoint:
            if str(endpoint).endswith("/chat/completions"):
                ok("OpenRouter endpoint format", str(endpoint), "openrouter.endpoint")
            else:
                fail("OpenRouter endpoint format", f"openrouter.endpoint must end with /chat/completions (got: {endpoint})", "openrouter.endpoint")
        else:
            if uses_openrouter:
                fail("OpenRouter endpoint present", "Missing openrouter.endpoint (required when using provider=openrouter)", "openrouter.endpoint")
            else:
                ok("OpenRouter endpoint present", "Skipped (not used)")

    return config, checks, issues


def validate_config(config: Dict[str, Any], *, config_path: str) -> Tuple[Dict[str, Any], List[ValidationIssue]]:
    """
    Validate config integrity & compliance.

    Returns:
      (config, issues)

    Raises:
      ValueError if any error-level issues are found.
    """
    _cfg, _checks, issues = validate_config_report(config, config_path=config_path)

    errors = [i for i in issues if i.level == "error"]
    if errors:
        details = "\n".join(
            f"- {i.path + ': ' if i.path else ''}{i.message}"
            for i in errors
        )
        raise ValueError("Config validation failed:\n" + details)

    return _cfg, issues

