"""Canonical filesystem locations for the hipredict package.

All paths are resolved relative to this file (via ``__file__``) so the package
behaves identically regardless of the current working directory. This replaces
the old ``../../`` relative paths, which only resolved when the interpreter was
launched from a specific directory inside the source tree.
"""
import os
from pathlib import Path

# .../hipredict  (the importable package directory)
PACKAGE_ROOT = Path(__file__).resolve().parent

# Repository root (one level above the package). Used to locate assets that
# live outside the importable package, such as trained model weights.
REPO_ROOT = PACKAGE_ROOT.parent

# YAML/CSV configs shipped inside the package.
CONFIGS_DIR = PACKAGE_ROOT / "configs"


def files_dir() -> Path:
    """Directory holding runtime data and outputs (isis_imgs, records, figs).

    Defaults to ``<repo_root>/files``. Set the ``HIPREDICT_FILES_DIR``
    environment variable to point at data stored outside the repository
    (useful for installed deployments where ``files/`` is not co-located
    with the package).
    """
    env = os.environ.get("HIPREDICT_FILES_DIR")
    return Path(env).expanduser().resolve() if env else REPO_ROOT / "files"


def models_dir() -> Path:
    """Directory holding trained model weights (``<name>/latest/...``).

    Defaults to ``<repo_root>/models``. Set the ``HIPREDICT_MODELS_DIR``
    environment variable to point at weights stored outside the repository
    (useful for installed deployments where ``models/`` is not co-located
    with the package).
    """
    env = os.environ.get("HIPREDICT_MODELS_DIR")
    return Path(env).expanduser().resolve() if env else REPO_ROOT / "models"


def resolve_path(path, root=None) -> Path:
    """Resolve a config path against data.root (the repository by default)."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    base = Path(root).expanduser() if root else REPO_ROOT
    if not base.is_absolute():
        base = REPO_ROOT / base
    return (base / path).resolve()
