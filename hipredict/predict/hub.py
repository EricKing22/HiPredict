"""Locate a matched HiPredict model release on Hugging Face."""

import hashlib
import json
import os
from pathlib import Path

RELEASE_FILES = ("model.pt", "meta_scalar.joblib", "config.yaml", "global_stats.json")

# Published HiPredict weights, pinned to the commit this code version was validated against.
DEFAULT_REPO = "ardan-S/HiPredict"
DEFAULT_REVISION = "d0dff1aa08f30730ff02f75297193d866c578f1f"


def validate_release(directory):
    directory = Path(directory)
    missing = [name for name in RELEASE_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete HiPredict release in {directory}: missing {', '.join(missing)}")
    manifest = directory / "provenance.json"
    if manifest.is_file():
        hashes = json.loads(manifest.read_text())["sha256"]
        for name in RELEASE_FILES:
            if name not in hashes:
                raise ValueError(f"Release provenance has no checksum for {name}")
            actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            if actual != hashes[name]:
                raise ValueError(f"Release checksum mismatch: {name}")
    return directory


def download_model(repo_id=None, revision=None, local_files_only=False):
    """Download (or reuse from the cache) one matched HiPredict release.

    The repository and revision come from the arguments, then the
    HIPREDICT_HF_REPO / HIPREDICT_HF_REVISION environment variables, then the
    published release (DEFAULT_REPO at DEFAULT_REVISION). The default revision
    applies only to the default repository. See docs/model.md.
    """
    repo_id = repo_id or os.environ.get("HIPREDICT_HF_REPO") or DEFAULT_REPO
    revision = revision or os.environ.get("HIPREDICT_HF_REVISION") or (
        DEFAULT_REVISION if repo_id == DEFAULT_REPO else None
    )
    from huggingface_hub import snapshot_download
    directory = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_files_only=local_files_only,
        allow_patterns=[*RELEASE_FILES, "provenance.json"],
    )
    return validate_release(directory)
