"""
upload_to_hf.py
===============
Pushes the built datasets to the HuggingFace Hub. Each ends up as its own repo.

Repos created (PRIVATE by default):
    {user}/printfarm-sft         ← data/sft/{train,eval}.jsonl
    {user}/printfarm-dpo         ← data/dpo/preferences.jsonl
    {user}/printfarm-baselines   ← data/baselines/** (summaries + trajectories)

Outputs:
    data/uploaded_manifest.json    # repo URLs + commit SHAs for provenance

Prereqs:
    1. pip install huggingface_hub datasets
    2. export HF_TOKEN=hf_...     (or run `huggingface-cli login`)

Usage:
    python scripts/upload_to_hf.py --user YOUR_HF_USERNAME
    python scripts/upload_to_hf.py --user YOUR_HF_USERNAME --public
    python scripts/upload_to_hf.py --user YOUR_HF_USERNAME --skip-baselines
    python scripts/upload_to_hf.py --user YOUR_HF_USERNAME --only sft
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _require_hf():
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder  # noqa
        return True
    except ImportError:
        print("ERROR: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)


def _ensure_repo(user: str, name: str, public: bool, token: str):
    from huggingface_hub import create_repo
    repo_id = f"{user}/{name}"
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=not public,
        exist_ok=True,
        token=token,
    )
    return repo_id


def _upload(user: str, name: str, folder: Path, public: bool, token: str) -> dict:
    from huggingface_hub import HfApi
    repo_id = _ensure_repo(user, name, public, token)
    api = HfApi(token=token)
    commit = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload {name} dataset",
    )
    url = f"https://huggingface.co/datasets/{repo_id}"
    return {"repo_id": repo_id, "url": url, "commit": str(commit), "private": not public}


def main():
    ap = argparse.ArgumentParser(description="Upload printfarm datasets to HF")
    ap.add_argument("--user", required=True, help="HuggingFace username or org")
    ap.add_argument("--public", action="store_true",
                    help="Create public repos (default: private)")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--only", choices=["sft", "dpo", "baselines"], default=None)
    ap.add_argument("--skip-sft", action="store_true")
    ap.add_argument("--skip-dpo", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    _require_hf()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: export HF_TOKEN (or run `huggingface-cli login`)", file=sys.stderr)
        sys.exit(1)

    data_root = Path(args.data_root).resolve()
    manifest = {"user": args.user, "public": args.public, "uploads": {}}

    targets = []
    if args.only:
        targets = [args.only]
    else:
        if not args.skip_sft:       targets.append("sft")
        if not args.skip_dpo:       targets.append("dpo")
        if not args.skip_baselines: targets.append("baselines")

    for which in targets:
        folder = data_root / which
        if not folder.exists():
            print(f"  skip: {folder} missing")
            continue
        repo_name = f"printfarm-{which}"
        print(f"Uploading {folder} → {args.user}/{repo_name} "
              f"({'public' if args.public else 'private'})...")
        info = _upload(args.user, repo_name, folder, args.public, token)
        manifest["uploads"][which] = info
        print(f"  → {info['url']}")

    out_path = data_root / "uploaded_manifest.json"
    with out_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {out_path}")


if __name__ == "__main__":
    main()
