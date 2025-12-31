from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from lidar2osm.config_loader import find_repo_root, resolve_path


def run(config: dict[str, Any], *, config_path: Path | None = None) -> None:
    repo_root = find_repo_root()
    # Train script consumes `training` from the provided config.yaml.
    train_cfg_path = config_path or (repo_root / "config.yaml")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    cmd = [sys.executable, str(repo_root / "src/core/models/train.py"), "--config", str(train_cfg_path)]
    subprocess.run(cmd, env=env, cwd=str(repo_root), check=True)


