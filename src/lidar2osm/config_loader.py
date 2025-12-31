from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_REPO_MARKERS = (".git", "pyproject.toml", "README.md")


def find_repo_root(start: Path | None = None) -> Path:
    """
    Find the repository root by walking upward from `start` until a marker is found.

    Markers: `.git/`, `pyproject.toml`, or `README.md`.
    """
    cur = (start or Path(__file__)).resolve()
    if cur.is_file():
        cur = cur.parent

    for p in (cur, *cur.parents):
        for marker in _REPO_MARKERS:
            if (p / marker).exists():
                return p
    # Fallback: best-effort (keeps code usable in unusual layouts)
    return cur


def get_config_dir(repo_root: Path | None = None) -> Path:
    # Highest priority: explicit override.
    env_override = os.environ.get("LIDAR2OSM_CONFIG_DIR")
    if env_override:
        cfg = Path(expand_vars(env_override)).expanduser().resolve()
        if cfg.exists():
            return cfg
        raise FileNotFoundError(f"LIDAR2OSM_CONFIG_DIR is set but does not exist: {cfg}")

    root = repo_root or find_repo_root()
    cfg = root / "config"
    if cfg.exists():
        return cfg

    # Fallback: allow running in environments where repo markers aren't present,
    # but a sibling/parent `config/` exists.
    for p in (root, *root.parents):
        candidate = p / "config"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Config directory not found starting from {root}. "
        "Expected repo-root `config/` or set LIDAR2OSM_CONFIG_DIR."
    )


def expand_vars(value: str) -> str:
    """Expand `${VARS}` / `$VARS` using environment variables."""
    return os.path.expandvars(value)


def resolve_path(path: str | Path, *, base_dir: Path) -> Path:
    p = Path(expand_vars(str(path)))
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping at {p}, got {type(data).__name__}")
    return data


def load_repo_config(rel_path: str | Path) -> dict[str, Any]:
    """
    Load a YAML file from repo-root `config/` by relative path.

    Example: `load_repo_config('training.yaml')`
    """
    cfg_dir = get_config_dir()
    p = resolve_path(rel_path, base_dir=cfg_dir)
    return load_yaml(p)


