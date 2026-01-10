"""
Dataset binarize package initialization.

This package contains scripts for binarizing ROS2 bag files and visualizing the results.
"""

import os
import sys
from pathlib import Path

# Add the project root to sys.path so lidar2osm imports work
# This allows scripts in dataset_binarize to import from lidar2osm
_package_dir = Path(__file__).resolve().parent
_project_root = _package_dir.parent

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

