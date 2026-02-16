#!/usr/bin/env python3
"""
Visualize a .bki map file with Open3D.

Reads the .bki (version 2) and config to build a point cloud of voxel centers,
colored by predicted semantic label. Requires --config for class count and label mapping.
"""

import argparse
import sys
import numpy as np
import open3d as o3d
from pathlib import Path

# Ensure scripts dir is on path so bki_to_bin_label can be imported
_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from bki_to_bin_label import read_bki

# MCD label colors (same as visualize_voxel_map / visualize_osm)
MCD_LABEL_COLORS = {
    0: [0.5, 0.5, 0.5],    # barrier
    1: [0.0, 0.0, 1.0],    # bike
    2: [0.8, 0.2, 0.2],    # building
    7: [0.6, 0.3, 0.1],    # fence
    13: [0.4, 0.4, 0.4],   # parkinglot
    14: [1.0, 0.5, 0.0],   # pedestrian
    15: [0.7, 0.7, 0.0],   # pole
    16: [0.2, 0.2, 0.2],   # road
    18: [0.7, 0.7, 0.7],   # sidewalk
    22: [0.0, 0.8, 0.8],   # traffic-sign
    24: [0.4, 0.3, 0.1],   # treetrunk
    25: [0.2, 0.8, 0.2],   # vegetation
    26: [0.0, 0.5, 1.0],   # vehicle-dynamic
}

KITTI_LABEL_COLORS = {
    0: [0.0, 0.0, 0.0],    # unlabeled
    10: [0.0, 0.0, 1.0],   # car
    30: [1.0, 0.5, 0.0],   # person
    40: [0.2, 0.2, 0.2],   # road
    44: [0.4, 0.4, 0.4],   # parking
    48: [0.7, 0.7, 0.7],   # sidewalk
    50: [0.8, 0.2, 0.2],   # building
    51: [0.6, 0.3, 0.1],   # fence
    70: [0.2, 0.8, 0.2],   # vegetation
    71: [0.4, 0.3, 0.1],   # trunk
    80: [0.7, 0.7, 0.0],   # pole
    81: [0.0, 0.8, 0.8],   # traffic-sign
}


def get_label_colors(labels, label_format="auto"):
    """(N,) uint32 labels -> (N, 3) RGB."""
    colors = np.zeros((len(labels), 3))
    if label_format == "auto":
        unique = np.unique(labels)
        if np.max(unique) > 30 or any(l in (40, 44, 48, 50, 70, 80, 81) for l in unique):
            label_format = "kitti"
        else:
            label_format = "mcd"
    color_map = KITTI_LABEL_COLORS if label_format == "kitti" else MCD_LABEL_COLORS
    for i, label in enumerate(labels):
        label = int(label)
        if label in color_map:
            colors[i] = color_map[label]
        else:
            colors[i] = [1.0, 0.0, 1.0]
    return colors


def main():
    parser = argparse.ArgumentParser(description="Visualize a .bki map with Open3D.")
    parser.add_argument("bki", help="Path to .bki map file")
    parser.add_argument("--config", required=True, help="Path to YAML config used when building the map")
    parser.add_argument("--min-alpha", type=float, default=1e-6, help="Min alpha sum per voxel to show (default: 1e-6)")
    parser.add_argument("--format", choices=["auto", "mcd", "kitti"], default="auto", help="Label format for colors")
    parser.add_argument("--point-size", type=float, default=2.0, help="Point size (default: 2.0)")
    args = parser.parse_args()

    bki_path = Path(args.bki)
    if not bki_path.exists():
        raise SystemExit(f"File not found: {args.bki}")
    if not Path(args.config).exists():
        raise SystemExit(f"Config not found: {args.config}")

    print(f"Loading {bki_path}...")
    points_xyz, labels_raw = read_bki(str(bki_path), args.config, min_alpha_sum=args.min_alpha)
    n = len(points_xyz)
    if n == 0:
        raise SystemExit("No voxels to show (try lowering --min-alpha).")

    colors = get_label_colors(labels_raw, args.format)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Showing {n} voxels. Close the window to exit.")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"BKI Map - {bki_path.name}", width=1280, height=720)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = args.point_size
    vis.run()
    vis.destroy_window()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
