from pathlib import Path

from .config_loader import find_repo_root

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(PACKAGE_DIR)
CONFIG_DIR = REPO_ROOT / "config"