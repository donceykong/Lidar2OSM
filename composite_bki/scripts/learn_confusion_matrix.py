#!/usr/bin/env python3
"""
Learn a confusion matrix from GT labels, points, and OSM features.

The learned matrix matches the config convention:
  - rows = predicted classes (label_to_matrix_idx)
  - cols = prior classes (osm_class_map)

Example:
  python scripts/learn_confusion_matrix.py \
      --scan example_data/mcd_scan/0000000011_transformed.bin \
      --gt example_data/mcd_scan/0000000011_transformed.labels \
      --osm example_data/mcd_scan/kth_day_06_osm_geometries.bin \
      --config configs/mcd_config.yaml \
      --output learned_confusion.yaml
"""

from __future__ import annotations

import argparse
import ast
import os
import struct
from collections import defaultdict

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree


def load_simple_yaml(path: str) -> dict:
    """
    Minimal YAML loader for the project config files.
    Supports top-level keys with list or dict values and inline lists.
    """
    data = {}
    current_key = None
    current_container = None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue

            if not line.startswith(" "):  # top-level key
                key = line.strip().rstrip(":")
                current_key = key
                current_container = None
                data[current_key] = None
                continue

            if current_key is None:
                continue

            entry = line.strip()
            if entry.startswith("- "):  # list entry
                if current_container is None:
                    current_container = []
                    data[current_key] = current_container
                value = ast.literal_eval(entry[2:].strip())
                current_container.append(value)
            else:  # dict entry
                if current_container is None:
                    current_container = {}
                    data[current_key] = current_container
                if ":" not in entry:
                    continue
                key_str, value_str = entry.split(":", 1)
                key_str = key_str.strip()
                value_str = value_str.strip()
                key_val = int(key_str) if key_str.isdigit() else key_str
                if value_str == "":
                    value_val = None
                else:
                    try:
                        value_val = ast.literal_eval(value_str)
                    except (ValueError, SyntaxError):
                        value_val = value_str
                current_container[key_val] = value_val

    return data


def load_osm_bin(bin_file: str) -> dict:
    """
    Load OSM geometries from binary file.

    Binary format (from create_map_OSM_BEV_GEOM.py):
    - uint32_t: number of buildings
    - For each building: uint32_t point count, then float[2*n_points] (x,y pairs)
    - uint32_t: number of roads
    - For each road: uint32_t point count, then float[2*n_points] (x,y pairs)
    - uint32_t: number of grasslands
    - For each grassland: uint32_t point count, then float[2*n_points] (x,y pairs)
    - uint32_t: number of trees
    - For each tree: uint32_t point count, then float[2*n_points] (x,y pairs)
    - uint32_t: number of wood/forests
    - For each wood: uint32_t point count, then float[2*n_points] (x,y pairs)
    """
    if not os.path.exists(bin_file):
        raise FileNotFoundError(f"OSM bin file not found: {bin_file}")

    data = {}
    categories = ["buildings", "roads", "grasslands", "trees", "wood"]

    with open(bin_file, "rb") as f:
        for cat in categories:
            try:
                num_items_bytes = f.read(4)
                if not num_items_bytes:
                    break
                num_items = struct.unpack("I", num_items_bytes)[0]
                items = []
                for _ in range(num_items):
                    n_pts_bytes = f.read(4)
                    if not n_pts_bytes:
                        break
                    n_pts = struct.unpack("I", n_pts_bytes)[0]
                    bytes_data = f.read(n_pts * 2 * 4)
                    floats = struct.unpack(f"{n_pts * 2}f", bytes_data)
                    poly_coords = list(zip(floats[0::2], floats[1::2]))
                    items.append(poly_coords)
                data[cat] = items
            except struct.error:
                print(f"Warning: Failed to load {cat}")
                break

    return data


def build_osm_geometries(osm_data: dict, osm_class_map: dict, num_prior: int) -> dict:
    geoms = {k: [] for k in range(num_prior)}
    for cat_name, items in osm_data.items():
        if cat_name in {"bounds", "world_bounds"}:
            continue
        idx = osm_class_map.get(cat_name)
        if idx is None:
            continue
        for coords in items:
            if len(coords) < 2:
                continue
            if len(coords) < 3:
                geoms[idx].append(LineString(coords))
            else:
                geoms[idx].append(Polygon(coords))
    return geoms


def build_rtrees(geoms_by_class: dict) -> dict:
    trees = {}
    for k, geom_list in geoms_by_class.items():
        if geom_list:
            trees[k] = STRtree(geom_list)
    return trees


def nearest_geometry(tree: STRtree, geoms: list, point: Point):
    nearest = tree.nearest(point)
    if hasattr(tree, "geometries"):
        return tree.geometries.take(nearest)
    if isinstance(nearest, (int, np.integer)):
        return geoms[nearest]
    return nearest


def get_osm_prior(x: float, y: float, geoms_by_class: dict, trees: dict, prior_delta: float) -> np.ndarray:
    p = Point(x, y)
    k_prior = len(geoms_by_class)
    scores = np.zeros(k_prior, dtype=np.float64)

    for k in range(k_prior):
        geoms = geoms_by_class[k]
        tree = trees.get(k)
        if not geoms or tree is None:
            dist = 50.0
        else:
            nearest_geom = nearest_geometry(tree, geoms, p)
            dist = p.distance(nearest_geom)
            if isinstance(nearest_geom, Polygon) and dist == 0.0:
                if nearest_geom.contains(p):
                    dist = -1.0 * nearest_geom.boundary.distance(p)
        scores[k] = 1.0 / (1.0 + np.exp(dist / prior_delta))

    denom = np.sum(scores)
    if denom <= 0:
        return np.full(k_prior, 1.0 / k_prior, dtype=np.float64)
    return scores / denom


def build_row_names(label_to_idx: dict, labels: dict, k_pred: int) -> list[str]:
    grouped = defaultdict(list)
    for raw_id, row_idx in label_to_idx.items():
        name = labels.get(raw_id, str(raw_id)) if labels else str(raw_id)
        grouped[row_idx].append(name)
    row_names = []
    for idx in range(k_pred):
        names = grouped.get(idx, [f"row_{idx}"])
        row_names.append("/".join(sorted(names)))
    return row_names


def build_col_names(osm_class_map: dict, k_prior: int) -> list[str]:
    grouped = defaultdict(list)
    for name, idx in osm_class_map.items():
        grouped[idx].append(name)
    col_names = []
    for idx in range(k_prior):
        names = grouped.get(idx, [f"prior_{idx}"])
        col_names.append("/".join(sorted(names)))
    return col_names


def normalize_matrix(counts: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return counts
    if mode == "cols":
        col_sum = counts.sum(axis=0, keepdims=True)
        col_sum[col_sum == 0] = 1.0
        return counts / col_sum
    row_sum = counts.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return counts / row_sum


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn confusion matrix from GT labels and OSM features")
    parser.add_argument("--scan", required=True, help="Path to .bin point cloud (x,y,z,intensity)")
    parser.add_argument("--gt", required=True, help="Path to ground truth .label/.labels file")
    parser.add_argument("--osm", required=True, help="Path to OSM .bin file")
    parser.add_argument("--config", default="configs/mcd_config.yaml", help="Path to config YAML")
    parser.add_argument("--output", default=None, help="Optional output file (.yaml/.csv/.npy)")
    parser.add_argument("--normalize", choices=["rows", "cols", "none"], default="rows",
                        help="Normalization mode for confusion matrix")
    parser.add_argument("--prior_to_gt", action="store_true",
                        help="Learn P(gt | prior): forces column-normalized output")
    parser.add_argument("--prior_delta", type=float, default=5.0, help="OSM prior delta for distance weighting")
    parser.add_argument("--hard_prior", action="store_true",
                        help="Use hard argmax prior assignment instead of soft weighting")
    parser.add_argument("--max_points", type=int, default=None, help="Optional cap on points processed")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")

    args = parser.parse_args()

    if not os.path.exists(args.scan):
        raise FileNotFoundError(f"Scan not found: {args.scan}")
    if not os.path.exists(args.gt):
        raise FileNotFoundError(f"GT labels not found: {args.gt}")
    if not os.path.exists(args.osm):
        raise FileNotFoundError(f"OSM not found: {args.osm}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    config = load_simple_yaml(args.config)
    label_to_idx = config.get("label_to_matrix_idx")
    osm_class_map = config.get("osm_class_map")
    labels = config.get("labels", {})

    if not label_to_idx or not osm_class_map:
        raise ValueError("Config must include label_to_matrix_idx and osm_class_map")

    k_pred = max(label_to_idx.values()) + 1
    k_prior = max(osm_class_map.values()) + 1

    print("Loading point cloud...")
    scan = np.fromfile(args.scan, dtype=np.float32).reshape((-1, 4))
    points = scan[:, :3]

    print("Loading GT labels...")
    gt_raw = np.fromfile(args.gt, dtype=np.uint32).reshape((-1))
    gt_labels = gt_raw & 0xFFFF

    if len(points) != len(gt_labels):
        n = min(len(points), len(gt_labels))
        print(f"Warning: point/label count mismatch ({len(points)} vs {len(gt_labels)}). Using {n}.")
        points = points[:n]
        gt_labels = gt_labels[:n]

    if args.max_points and args.max_points < len(points):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(points), size=args.max_points, replace=False)
        points = points[idx]
        gt_labels = gt_labels[idx]

    print("Loading OSM geometries...")
    osm_data = load_osm_bin(args.osm)
    geoms_by_class = build_osm_geometries(osm_data, osm_class_map, k_prior)
    trees = build_rtrees(geoms_by_class)

    counts = np.zeros((k_pred, k_prior), dtype=np.float64)
    row_totals = np.zeros(k_pred, dtype=np.float64)

    print(f"Computing priors for {len(points)} points...")
    for i in range(len(points)):
        raw_label = int(gt_labels[i])
        row_idx = label_to_idx.get(raw_label)
        if row_idx is None:
            continue
        m_i = get_osm_prior(points[i, 0], points[i, 1], geoms_by_class, trees, args.prior_delta)
        if args.hard_prior:
            col = int(np.argmax(m_i))
            counts[row_idx, col] += 1.0
        else:
            counts[row_idx, :] += m_i
        row_totals[row_idx] += 1.0

    normalize_mode = "cols" if args.prior_to_gt else args.normalize
    confusion = normalize_matrix(counts, normalize_mode)

    row_names = build_row_names(label_to_idx, labels, k_pred)
    col_names = build_col_names(osm_class_map, k_prior)

    print("\nLearned confusion matrix:")
    print(f"  rows (pred): {row_names}")
    print(f"  cols (prior): {col_names}")
    if args.prior_to_gt:
        print("  mode: P(gt | prior) (column-normalized)")
    np.set_printoptions(precision=4, suppress=True)
    print(confusion)

    if args.output:
        out_ext = os.path.splitext(args.output)[1].lower()
        if out_ext in {".npy"}:
            np.save(args.output, confusion)
        elif out_ext in {".csv"}:
            import csv
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["pred_class"] + col_names)
                for idx, row in enumerate(confusion):
                    writer.writerow([row_names[idx]] + list(row))
        elif out_ext in {".yaml", ".yml"}:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write("confusion_matrix:\n")
                for row in confusion:
                    row_str = ", ".join(f"{v:.6f}" for v in row)
                    f.write(f"  - [{row_str}]\n")
        else:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(repr(confusion))

        print(f"Saved confusion matrix to {args.output}")


if __name__ == "__main__":
    main()
