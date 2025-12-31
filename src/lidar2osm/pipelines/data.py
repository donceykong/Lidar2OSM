from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from lidar2osm.config_loader import find_repo_root


def _run_script(repo_root: Path, script_rel: str, args: list[str]) -> None:
    script_path = (repo_root / script_rel).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    env = os.environ.copy()
    # Keep behavior consistent with the documented usage.
    env["PYTHONPATH"] = str(repo_root)

    cmd = [sys.executable, str(script_path), *args]
    subprocess.run(cmd, env=env, cwd=str(repo_root), check=True)


def run(config: dict[str, Any]) -> None:
    """
    Run the relabeling scans pipeline in the same order as documented:

    - create_global_sem_map_octree
    - extend_building_offsets
    - relabel_maps
    - relabel_dense_semantics
    - relabel_scans
    """
    repo_root = find_repo_root()
    dp = config.get("data_pipeline", {}) if isinstance(config, dict) else {}

    dataset_path = dp.get("dataset_path") or config.get("dataset_path")
    if not dataset_path:
        raise ValueError("Missing `data_pipeline.dataset_path` (or top-level `dataset_path`) in config.")

    num_scans_global = str(dp.get("create_global_sem_map_octree", {}).get("num_scans", 1))
    num_scans_relabel = str(dp.get("relabel_scans", {}).get("num_scans", 2))

    _run_script(
        repo_root,
        "src/core/data/create_global_sem_map_octree.py",
        ["--num_scans", num_scans_global, "--dataset_path", str(dataset_path)],
    )
    _run_script(
        repo_root,
        "src/core/data/extend_building_offsets.py",
        ["--dataset_path", str(dataset_path)],
    )
    _run_script(
        repo_root,
        "src/core/data/relabel_maps.py",
        ["--dataset_path", str(dataset_path)],
    )
    _run_script(
        repo_root,
        "src/core/data/relabel_dense_semantics.py",
        ["--dataset_path", str(dataset_path)],
    )
    _run_script(
        repo_root,
        "src/core/data/relabel_scans.py",
        ["--dataset_path", str(dataset_path), "--num_scans", num_scans_relabel],
    )


