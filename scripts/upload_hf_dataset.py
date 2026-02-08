#!/usr/bin/env python3
"""Upload a local dataset file/folder to a Hugging Face dataset repository.

Examples:
  python scripts/upload_hf_dataset.py \
    --repo-id your_name/ppopt \
    --file output/training_data/PPOpt_hf.jsonl

  HF_TOKEN=hf_xxx python scripts/upload_hf_dataset.py \
    --repo-id your_name/ppopt \
    --file output/training_data/PPOpt_hf.jsonl \
    --private
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload JSONL dataset artifacts to Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help='Target dataset repo id, e.g. "username/repo_name".',
    )
    parser.add_argument(
        "--file",
        default="output/training_data/PPOpt_hf.jsonl",
        help="Local file or folder to upload (default: output/training_data/PPOpt_hf.jsonl).",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Destination path in repo when uploading a single file (default: local filename).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token. If omitted, use HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/use private dataset repository.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload dataset artifact",
        help='Commit message (default: "Upload dataset artifact").',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[error] huggingface_hub is not installed. Run: pip install huggingface_hub")
        return 1

    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("[error] missing token. Provide --token or set HF_TOKEN.")
        return 1

    local_path = Path(args.file)
    if not local_path.exists():
        print(f"[error] file/folder not found: {local_path}")
        return 1

    api = HfApi(token=token)
    whoami = api.whoami()
    username = whoami.get("name", "unknown")
    print(f"[info] authenticated as: {username}")

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    print(f"[info] dataset repo ready: {args.repo_id}")

    if local_path.is_dir():
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=args.commit_message,
        )
        print(f"[ok] uploaded folder: {local_path}")
    else:
        target_path = args.path_in_repo or local_path.name
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=target_path,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=args.commit_message,
        )
        print(f"[ok] uploaded file: {local_path} -> {target_path}")

    print(f"[ok] dataset URL: https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
