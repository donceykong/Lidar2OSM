from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from lidar2osm.config_loader import find_repo_root, load_yaml, resolve_path
from lidar2osm.pipelines.data import run as run_data
from lidar2osm.pipelines.infer import run as run_infer
from lidar2osm.pipelines.train import run as run_train


def _load_cli_config(config_path: str | Path | None) -> tuple[dict[str, Any], Path | None]:
    if config_path is None:
        repo_root = find_repo_root()
        default = repo_root / "config.yaml"
        if default.exists():
            return load_yaml(default), default
        return {}, None

    repo_root = find_repo_root()
    p = resolve_path(config_path, base_dir=repo_root)
    return load_yaml(p), p


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lidar2osm")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a CLI config YAML (default: repo-root `config.yaml` if present).",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--data_pipeline", action="store_true", help="Run relabeling scans pipeline.")
    g.add_argument("--training_pipeline", action="store_true", help="Run training pipeline.")
    g.add_argument("--inference_pipeline", action="store_true", help="Run inference pipeline.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg, cfg_path = _load_cli_config(args.config)

    if args.data_pipeline:
        run_data(cfg)
    elif args.training_pipeline:
        run_train(cfg, config_path=cfg_path)
    elif args.inference_pipeline:
        run_infer(cfg, config_path=cfg_path)
    else:
        parser.error("No pipeline selected")

    return 0


