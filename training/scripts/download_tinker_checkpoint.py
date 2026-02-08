#!/usr/bin/env python
"""Download a Tinker checkpoint archive.

Fill the placeholders below or pass values via CLI args or env vars.

Examples:
  python scripts/download_tinker_checkpoint.py \
    --api-key YOUR_TINKER_API_KEY \
    --checkpoint tinker://RUN_ID/weights/checkpoint-001 \
    --out checkpoint_archive.tar

Env vars (override defaults if CLI args not provided):
  TINKER_API_KEY, TINKER_BASE_URL, TINKER_CHECKPOINT
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

# Placeholders (fill these or pass via args/env)
PLACEHOLDER_API_KEY = "YOUR_TINKER_API_KEY"
PLACEHOLDER_TINKER_PATH = "tinker://RUN_ID/weights/checkpoint-001"

# Defaults (can be real values or left as placeholders)
DEFAULT_API_KEY = "xxx"
DEFAULT_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod"
DEFAULT_TINKER_PATH = "tinker://cb1866d3-8a10-5700-ae66-5454e86f4fe0:train:0/sampler_weights/ppopt_rl_llama8b_sampler"
DEFAULT_OUT = "checkpoint_archive.tar"


def _resolve_value(cli_value: str | None, env_key: str, default: str) -> str:
    return cli_value or os.getenv(env_key) or default


def _make_service_client(tinker_mod, api_key: str, base_url: str):
    """Create ServiceClient, tolerating different constructor signatures."""
    try:
        return tinker_mod.ServiceClient(api_key=api_key, base_url=base_url)
    except TypeError:
        # Fallback to env-based configuration
        if api_key:
            os.environ["TINKER_API_KEY"] = api_key
        if base_url:
            os.environ["TINKER_BASE_URL"] = base_url
        return tinker_mod.ServiceClient()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a Tinker checkpoint archive.")
    parser.add_argument("--api-key", help="Tinker API key")
    parser.add_argument("--base-url", help="Tinker base URL")
    parser.add_argument("--checkpoint", help="tinker:// checkpoint path (state or sampler)")
    parser.add_argument("--out", help="Output archive path")
    parser.add_argument("--print-url", action="store_true", help="Only print signed URL (no download)")
    args = parser.parse_args()

    api_key = _resolve_value(args.api_key, "TINKER_API_KEY", DEFAULT_API_KEY)
    base_url = _resolve_value(args.base_url, "TINKER_BASE_URL", DEFAULT_BASE_URL)
    tinker_path = _resolve_value(args.checkpoint, "TINKER_CHECKPOINT", DEFAULT_TINKER_PATH)
    out_path = _resolve_value(args.out, "TINKER_OUT", DEFAULT_OUT)

    if api_key == PLACEHOLDER_API_KEY:
        print("ERROR: Please set your Tinker API key (placeholder is still in use).", file=sys.stderr)
        return 2
    if tinker_path == PLACEHOLDER_TINKER_PATH:
        print("ERROR: Please set your tinker:// checkpoint path (placeholder is still in use).", file=sys.stderr)
        return 2

    try:
        import tinker  # type: ignore
    except Exception as exc:
        print("ERROR: Missing Python package 'tinker'. Install it first.", file=sys.stderr)
        print(f"Details: {exc}", file=sys.stderr)
        return 3

    sc = _make_service_client(tinker, api_key, base_url)
    rc = sc.create_rest_client()
    future = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path)
    resp = future.result()
    signed_url = resp.url

    if args.print_url:
        print(signed_url)
        return 0

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(signed_url, out_file.as_posix())
    except Exception as exc:
        print(f"ERROR: Download failed: {exc}", file=sys.stderr)
        return 4

    print(f"Downloaded: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
