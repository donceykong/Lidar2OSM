#!/usr/bin/env python3
"""
Create a global semantic map for MCD sequences.

This script consolidates the historical implementations behind two strategies:
- **octree** (default): accumulate `.bin` scans + inferred labels into a colored `.ply` via `pyoctomap`
- **voxel**: merge GT labeled `.pcd` scans into a single merged `.npy` (x, y, z, label)

Example usage:
python create_global_gt_map.py octree --dataset-path /mnt/semkitti/MCD-finalized-dataset/MCD-finalized-dataset/ --max-scans 5 --output ./outie
python create_global_gt_map.py voxel --dataset-path /mnt/semkitti/MCD-finalized-dataset/MCD-finalized-dataset/ --max-scans 5  --output ./outie.ply
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# Body to LiDAR transformation matrix
BODY_TO_LIDAR_TF = np.array(
    [
        [0.9999135040741837, -0.011166365511073898, -0.006949579221822984, -0.04894521120494695],
        [-0.011356389542502144, -0.9995453006865824, -0.02793249526856565, -0.03126929060348084],
        [-0.006634514801117132, 0.02800900135032654, -0.999585653686922, -0.01755515794222565],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

DEFAULT_SEQ_NAMES = ["kth_day_06", "kth_day_09", "kth_night_05"]


@dataclass
class Label:
    name: str
    id: int
    color: tuple[int, int, int]


SEMANTIC_LABELS = [
    Label("unlabeled", 0, (0, 0, 0)),
    Label("outlier", 1, (0, 0, 0)),
    Label("car", 10, (0, 0, 142)),
    Label("bicycle", 11, (119, 11, 32)),
    Label("bus", 13, (250, 80, 100)),
    Label("motorcycle", 15, (0, 0, 230)),
    Label("on-rails", 16, (255, 0, 0)),
    Label("truck", 18, (0, 0, 70)),
    Label("other-vehicle", 20, (51, 0, 51)),
    Label("person", 30, (220, 20, 60)),
    Label("bicyclist", 31, (200, 40, 255)),
    Label("motorcyclist", 32, (90, 30, 150)),
    Label("road", 40, (128, 64, 128)),
    Label("parking", 44, (250, 170, 160)),
    Label("OSM BUILDING", 45, (0, 0, 255)),  # OSM
    Label("OSM ROAD", 46, (255, 0, 0)),  # OSM
    Label("sidewalk", 48, (244, 35, 232)),
    Label("other-ground", 49, (81, 0, 81)),
    Label("building", 50, (0, 100, 0)),
    Label("fence", 51, (190, 153, 153)),
    Label("other-structure", 52, (0, 150, 255)),
    Label("lane-marking", 60, (170, 255, 150)),
    Label("vegetation", 70, (107, 142, 35)),
    Label("trunk", 71, (0, 60, 135)),
    Label("terrain", 72, (152, 251, 152)),
    Label("pole", 80, (153, 153, 153)),
    Label("traffic-sign", 81, (0, 0, 255)),
    Label("other-object", 99, (255, 255, 50)),
]


class GlobalMapBuilder(ABC):
    """
    Template/strategy base class for building a global map.

    Subclasses implement `run()`; shared I/O + geometry helpers live here.
    """

    def __init__(
        self,
        dataset_path: str,
        seq_names: list[str],
        max_scans: Optional[int],
        downsample_factor: int,
        body_to_lidar_tf: np.ndarray = BODY_TO_LIDAR_TF,
    ) -> None:
        self.dataset_path = dataset_path
        self.seq_names = seq_names
        self.max_scans = max_scans
        self.downsample_factor = downsample_factor
        self.body_to_lidar_tf = body_to_lidar_tf

    @abstractmethod
    def run(self) -> int:
        raise NotImplementedError

    def read_bin_file(self, file_path: str, dtype, shape=None) -> np.ndarray:
        data = np.fromfile(file_path, dtype=dtype)
        if shape:
            return data.reshape(shape)
        return data

    def transform_points_to_world(
        self, points_xyz: np.ndarray, position, quaternion
    ) -> np.ndarray:
        body_rotation_matrix = R.from_quat(quaternion).as_matrix()
        body_to_world = np.eye(4, dtype=np.float64)
        body_to_world[:3, :3] = body_rotation_matrix
        body_to_world[:3, 3] = np.asarray(position, dtype=np.float64)

        lidar_to_body = np.linalg.inv(self.body_to_lidar_tf)
        transform_matrix = body_to_world @ lidar_to_body

        points_h = np.hstack(
            [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)]
        )
        world_points = (transform_matrix @ points_h.T).T
        return world_points[:, :3].astype(np.float32, copy=False)

    def load_poses(self, poses_file: str):
        """Load poses from CSV file."""
        try:
            df = self._read_poses_csv(poses_file)
            if df is None:
                return {}, {}
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return {}, {}

        poses: dict[float, list[float]] = {}
        index_to_timestamp: dict[int, float] = {}

        col_map = self._build_col_map(df)
        for _, row in df.iterrows():
            try:
                parsed = self._parse_pose_row(row, col_map)
                if parsed:
                    ts, pose = parsed
                    poses[float(ts)] = pose
                    if pose[0] is not None:
                        index_to_timestamp[int(pose[0])] = float(ts)
            except Exception:
                continue

        print(f"Loaded {len(poses)} poses from {os.path.basename(poses_file)}")
        return poses, index_to_timestamp

    def _read_poses_csv(self, poses_file: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(poses_file, comment="#", skipinitialspace=True)
            cols = [
                str(c).strip().lower().replace("#", "").replace(" ", "")
                for c in df.columns
            ]
            has_ts = any("timestamp" in c or c == "t" for c in cols)
            has_pose = all(
                any(k in c for c in cols) for k in ["x", "y", "z", "qx", "qy", "qz", "qw"]
            )
            if has_ts and has_pose and len(df.columns) >= 8:
                return df
        except Exception:
            pass

        try:
            df = pd.read_csv(
                poses_file, comment="#", header=None, skipinitialspace=True
            )
            return df if len(df.columns) >= 8 else None
        except Exception:
            return None

    def _yield_scans(self, seq_name: str, offset: np.ndarray | None = None):
        """
        Yields (world_points, labels) for a sequence, abstracting format (BIN/PCD).
        """
        root = os.path.join(self.dataset_path, seq_name)
        poses_file = os.path.join(root, "pose_inW.csv")
        if not os.path.exists(poses_file):
            return

        poses_dict, idx_to_ts = self.load_poses(poses_file)
        if not poses_dict:
            return

        # Check for BIN data
        bin_dir = os.path.join(root, "lidar_bin", "data")
        ts_file = os.path.join(root, "lidar_bin", "timestamps.txt")
        labels_dir = os.path.join(root, "gt_labels")

        if os.path.exists(bin_dir) and os.path.exists(ts_file):
            yield from self._yield_bin_scans(
                bin_dir, labels_dir, ts_file, poses_dict, idx_to_ts, offset
            )
            return

        # Check for PCD data
        pcd_dir = os.path.join(root, "gt_labels", "inL_labelled")
        if os.path.exists(pcd_dir):
            yield from self._yield_pcd_scans(
                pcd_dir, poses_dict, idx_to_ts, offset
            )

    def _yield_bin_scans(
        self, bin_dir, labels_dir, ts_file, poses_dict, idx_to_ts, offset
    ):
        timestamps = np.loadtxt(ts_file)
        bin_files = sorted([f for f in os.listdir(bin_dir) if f.endswith(".bin")])
        
        # Sync lengths
        min_len = min(len(bin_files), len(timestamps))
        bin_files = bin_files[:min_len]
        timestamps = timestamps[:min_len]

        sorted_indices = sorted(idx_to_ts.keys())
        if self.downsample_factor > 1:
            sorted_indices = sorted_indices[:: self.downsample_factor]
        if self.max_scans:
            sorted_indices = sorted_indices[: self.max_scans]

        for pose_num in tqdm(sorted_indices, desc="Processing BIN", unit="scan"):
            scan_idx = pose_num - 1
            if scan_idx < 0 or scan_idx >= len(bin_files):
                continue

            pose_ts = idx_to_ts[pose_num]
            if abs(timestamps[scan_idx] - pose_ts) > 0.1:
                continue

            bin_file = bin_files[scan_idx]
            try:
                pts = self.read_bin_file(
                    os.path.join(bin_dir, bin_file), dtype=np.float32, shape=(-1, 4)
                )[:, :3]
                
                label_path = os.path.join(labels_dir, bin_file)
                if not os.path.exists(label_path):
                    label_path = label_path.replace(".bin", ".label")
                    if not os.path.exists(label_path):
                        continue

                labels = np.fromfile(label_path, dtype=np.int32)
                if len(labels) != len(pts):
                    labels = np.fromfile(label_path, dtype=np.uint32).astype(np.int32)
                
                if len(labels) != len(pts):
                    continue
                
                labels = labels & 0xFFFF # SemanticKITTI mask
            except Exception:
                continue

            yield self._transform_and_offset(pts, poses_dict[pose_ts], offset), labels, self._get_pos(poses_dict[pose_ts], offset)

    def _yield_pcd_scans(self, pcd_dir, poses_dict, idx_to_ts, offset):
        try:
            from pypcd import pypcd
        except ImportError:
            print("ERROR: pypcd not found")
            return

        pcd_files = {
            int(f.replace("cloud_", "").replace(".pcd", "")): f
            for f in os.listdir(pcd_dir)
            if f.startswith("cloud_") and f.endswith(".pcd")
        }

        sorted_indices = sorted(idx_to_ts.keys())
        if self.downsample_factor > 1:
            sorted_indices = sorted_indices[:: self.downsample_factor]
        if self.max_scans:
            sorted_indices = sorted_indices[: self.max_scans]

        for pose_num in tqdm(sorted_indices, desc="Processing PCD", unit="scan"):
            if pose_num not in pcd_files:
                continue
            
            pcd_path = os.path.join(pcd_dir, pcd_files[pose_num])
            try:
                pc = pypcd.PointCloud.from_path(pcd_path)
                pts = np.column_stack(
                    [pc.pc_data["x"], pc.pc_data["y"], pc.pc_data["z"]]
                ).astype(np.float32)

                labels = None
                for f in ["label", "class", "semantic", "labels", "classes", "semantic_id"]:
                    if f in pc.pc_data.dtype.names:
                        labels = pc.pc_data[f].astype(np.int32)
                        break
                
                if labels is None:
                    continue
            except Exception:
                continue

            pose_data = poses_dict[idx_to_ts[pose_num]]
            yield self._transform_and_offset(pts, pose_data, offset), labels, self._get_pos(pose_data, offset)

    def _transform_and_offset(self, points, pose_data, offset):
        pos = pose_data[1:4]
        if offset is not None:
            pos = np.array(pos, dtype=np.float32) - offset
        
        return self.transform_points_to_world(points, pos, pose_data[4:8])

    def _get_pos(self, pose_data, offset):
        pos = np.array(pose_data[1:4], dtype=np.float32)
        if offset is not None:
            pos -= offset
        return pos

    def _build_col_map(self, df: pd.DataFrame) -> dict[str, Any]:
        col_map: dict[str, Any] = {}
        for col in df.columns:
            col_clean = str(col).strip().lower().replace("#", "").replace(" ", "")
            col_map[col_clean] = col
        return col_map

    def _find_timestamp_col(self, col_map: dict[str, Any]) -> Optional[Any]:
        for key, original in col_map.items():
            if key == "t" or "timestamp" in key:
                return original
        return None

    def _parse_pose_row(self, row: pd.Series, col_map: dict[str, Any]) -> Optional[tuple[float, list[float]]]:
        timestamp_col = self._find_timestamp_col(col_map)

        timestamp = None
        pose_index: Optional[int] = None
        x = y = z = qx = qy = qz = qw = None

        if timestamp_col is not None:
            timestamp = row[timestamp_col]
            if "num" in col_map:
                pose_index = int(row[col_map["num"]])

            def _get(name: str):
                return row.get(col_map.get(name), None) if name in col_map else None

            x = _get("x")
            y = _get("y")
            z = _get("z")
            qx = _get("qx")
            qy = _get("qy")
            qz = _get("qz")
            qw = _get("qw")
        else:
            # Positional: [num, t, x, y, z, qx, qy, qz, qw] or [t, x, y, z, qx, qy, qz, qw]
            if len(row) >= 9:
                pose_index = int(row.iloc[0])
                timestamp = row.iloc[1]
                x, y, z = row.iloc[2], row.iloc[3], row.iloc[4]
                qx, qy, qz, qw = row.iloc[5], row.iloc[6], row.iloc[7], row.iloc[8]
            elif len(row) >= 8:
                timestamp = row.iloc[0]
                x, y, z = row.iloc[1], row.iloc[2], row.iloc[3]
                qx, qy, qz, qw = row.iloc[4], row.iloc[5], row.iloc[6], row.iloc[7]
            else:
                return None

        if None in [timestamp, x, y, z, qx, qy, qz, qw]:
            return None

        pose = [pose_index, float(x), float(y), float(z), float(qx), float(qy), float(qz), float(qw)]
        return float(timestamp), pose


# ----------------------------
# Octree strategy
# ----------------------------


def octree_labels_to_colors(labels: np.ndarray) -> np.ndarray:
    label_id_to_color = {label.id: label.color for label in SEMANTIC_LABELS}
    colors = np.zeros((len(labels), 3), dtype=np.float32)
    for i, label_id in enumerate(labels):
        lid = int(label_id)
        if lid in label_id_to_color:
            colors[i] = np.asarray(label_id_to_color[lid], dtype=np.float32) / 255.0
        else:
            colors[i] = [0.5, 0.5, 0.5]
    return colors


def insert_points_into_octree(
    octree, points: np.ndarray, colors: np.ndarray, color_dict: dict | None, resolution: float
) -> int:
    if len(points) == 0:
        return 0

    inserted_count = 0
    error_count = 0
    colors_uint8 = (colors * 255.0).astype(np.uint8)

    for i in range(len(points)):
        point = points[i]
        color = colors_uint8[i]

        try:
            x, y, z = float(point[0]), float(point[1]), float(point[2])
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                error_count += 1
                continue

            MAX_COORD = 1e8
            if (
                x < -MAX_COORD
                or x > MAX_COORD
                or y < -MAX_COORD
                or y > MAX_COORD
                or z < -MAX_COORD
                or z > MAX_COORD
            ):
                error_count += 1
                continue

            r_val, g_val, b_val = int(color[0]), int(color[1]), int(color[2])
            if not (0 <= r_val <= 255 and 0 <= g_val <= 255 and 0 <= b_val <= 255):
                error_count += 1
                continue

            voxel_x = round(x / resolution) * resolution
            voxel_y = round(y / resolution) * resolution
            voxel_z = round(z / resolution) * resolution
            voxel_key = f"{voxel_x:.6f}_{voxel_y:.6f}_{voxel_z:.6f}"

            coord_list = [float(x), float(y), float(z)]
            octree.updateNode(coord_list, True)
            octree.setNodeColor(coord_list, r_val, g_val, b_val)
            inserted_count += 1

            if color_dict is not None:
                if voxel_key not in color_dict:
                    color_dict[voxel_key] = {"counts": {}, "mode": (r_val, g_val, b_val)}
                color_counts = color_dict[voxel_key]["counts"]
                color_tuple = (r_val, g_val, b_val)
                color_counts[color_tuple] = color_counts.get(color_tuple, 0) + 1
                color_dict[voxel_key]["mode"] = max(color_counts, key=color_counts.get)

        except Exception:
            error_count += 1
            if error_count <= 3:
                print(f"WARNING: Error inserting point {i}/{len(points)}")
            continue

    if error_count > 0:
        print(f"WARNING: {error_count}/{len(points)} points failed to insert")
    return inserted_count


def extract_pointcloud_from_octree(
    octree, color_dict: dict | None, coordinate_offset: np.ndarray | None
):
    points = []
    colors = []

    try:
        num_nodes = octree.getNumLeafNodes()
        if num_nodes > 0:
            octree.updateInnerOccupancy()
        else:
            print("WARNING: Octree has 0 leaf nodes, skipping updateInnerOccupancy()")

        if color_dict and len(color_dict) > 0:
            print(f"Extracting points using dictionary keys ({len(color_dict)} entries)...")
            for voxel_key, info in color_dict.items():
                parts = voxel_key.split("_")
                if len(parts) != 3:
                    continue
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                except ValueError:
                    continue

                coord = np.array([x, y, z], dtype=np.float32)
                if coordinate_offset is not None:
                    coord = coord + coordinate_offset.astype(np.float32)
                points.append(coord)

                r, g, b = info["mode"]
                colors.append(
                    np.array([r / 255.0, g / 255.0, b / 255.0], dtype=np.float32)
                )
        else:
            print("WARNING: No color_dict available for extraction")

    except Exception as e:
        print(f"WARNING: Error extracting pointcloud from octree: {e}")
        import traceback

        traceback.print_exc()
        return np.array([]), np.array([])

    if len(points) == 0:
        return np.array([]), np.array([])

    return np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.float32)




@dataclass
class OctreeConfig:
    resolution: float = 0.5
    max_distance: Optional[float] = None
    output_ply: Optional[str] = None
    output_npy: Optional[str] = None


class OctreeMapBuilder(GlobalMapBuilder):
    def __init__(self, *, config: OctreeConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config

    def run(self) -> int:
        pyo, o3d = self._import_libraries()
        if not pyo or not o3d:
            return 1

        octree = pyo.ColorOcTree(self.config.resolution)
        color_dict = {}
        offset = self._calculate_coordinate_offset()

        self._process_sequences(octree, color_dict, offset)

        final_nodes = octree.getNumLeafNodes()
        if final_nodes == 0:
            print("Error: No data accumulated from any sequence!")
            return 1

        octree.updateInnerOccupancy()
        print(f"Final octree statistics: {final_nodes} nodes, {len(color_dict)} colors")

        self._extract_and_save(octree, color_dict, offset, o3d)
        return 0

    def _import_libraries(self):
        try:
            import pyoctomap as pyo
            import open3d as o3d
            return pyo, o3d
        except ImportError as e:
            print(f"ERROR: Failed to import required libraries: {e}")
            return None, None

    def _calculate_coordinate_offset(self):
        if not self.seq_names:
            return None
        first_seq = self.seq_names[0]
        poses_file = os.path.join(self.dataset_path, first_seq, "pose_inW.csv")
        if not os.path.exists(poses_file):
            return None
        
        poses, idx_map = self.load_poses(poses_file)
        if not poses or not idx_map:
            return None
            
        first_idx = min(idx_map.keys())
        first_pose = poses[idx_map[first_idx]]
        return np.array(first_pose[1:4], dtype=np.float32)

    def _process_sequences(self, octree, color_dict, offset):
        for seq in self.seq_names:
            print(f"Processing {seq}")
            for points, labels, sensor_pos in self._yield_scans(seq, offset):
                if len(points) == 0:
                    continue

                colors = octree_labels_to_colors(labels)

                if self.config.max_distance:
                    dists = np.linalg.norm(points - sensor_pos, axis=1)
                    mask = dists <= self.config.max_distance
                    points = points[mask]
                    colors = colors[mask]

                if len(points) > 0:
                    insert_points_into_octree(
                        octree, points, colors, color_dict, self.config.resolution
                    )

    def _extract_and_save(self, octree, color_dict, offset, o3d):
        print("Extracting point cloud from octree...")
        points, colors = extract_pointcloud_from_octree(
            octree, color_dict, offset
        )
        if len(points) == 0:
            print("Error: No points extracted from octree!")
            return

        out_dir = os.path.join(self.dataset_path, "ply")
        os.makedirs(out_dir, exist_ok=True)

        # Save PLY if requested or if no output specified (default)
        if self.config.output_ply or not self.config.output_npy:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            path = self.config.output_ply or os.path.join(
                out_dir, f"inferred_labels_{'_'.join(self.seq_names)}.ply"
            )
            if not path.endswith(".ply"):
                path += ".ply"
            
            o3d.io.write_point_cloud(path, pcd)
            print(f"Successfully saved {len(points)} points to {path}")

        # Save NPY if requested
        if self.config.output_npy:
            path = self.config.output_npy
            if not path.endswith(".npy"):
                path += ".npy"
            
            # Save x,y,z,r,g,b
            data = np.hstack([points, colors])
            np.save(path, data)
            print(f"Successfully saved {len(points)} points to {path}")


# ----------------------------
# Voxel strategy
# ----------------------------


@dataclass
class VoxelConfig:
    voxel_size: float = 1.0
    output_ply: Optional[str] = None
    output_npy: Optional[str] = None


class VoxelMapBuilder(GlobalMapBuilder):
    def __init__(self, *, config: VoxelConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config

    def run(self) -> int:
        print(f"Merging {len(self.seq_names)} sequences into single numpy file")
        all_points, all_labels = [], []

        for i, seq_name in enumerate(self.seq_names):
            print(f"Processing sequence {i + 1}/{len(self.seq_names)}: {seq_name}")
            count = 0
            for points, labels, _ in self._yield_scans(seq_name, None):
                if self.config.voxel_size:
                    points, labels = self._voxelize(points, labels)
                
                if len(points) > 0:
                    all_points.append(points)
                    all_labels.append(labels)
                    count += len(points)
            
            print(f"  Added {count} points from {seq_name}")

        if not all_points:
            print("ERROR: No points accumulated from any sequence!")
            return 1

        merged_points = np.vstack(all_points)
        merged_labels = np.concatenate(all_labels)

        print("Merging all sequences...")
        self._print_label_stats(merged_labels)
        self._save_merged_data(merged_points, merged_labels)

        print(f"DONE! Merged {len(self.seq_names)} sequences into single numpy file")
        return 0

    def _voxelize(self, points, labels):
        voxel_size = self.config.voxel_size
        if voxel_size <= 0:
            return points, labels

        voxel_coords = np.round(points / float(voxel_size)).astype(np.int32)
        voxel_to_labels = defaultdict(list)

        for i, voxel_coord in enumerate(voxel_coords):
            voxel_to_labels[tuple(voxel_coord)].append(int(labels[i]))

        unique_voxels = []
        voxel_labels = []
        for voxel_key, label_list in voxel_to_labels.items():
            voxel_center = np.array(voxel_key, dtype=np.float32) * float(voxel_size)
            unique_voxels.append(voxel_center)

            unique, counts = np.unique(label_list, return_counts=True)
            mode_label = unique[np.argmax(counts)]
            voxel_labels.append(int(mode_label))

        return np.asarray(unique_voxels, dtype=np.float32), np.asarray(
            voxel_labels, dtype=np.int32
        )

    def _print_label_stats(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        print("Label statistics:")
        label_names = {label.id: label.name for label in SEMANTIC_LABELS}
        for lid, count in zip(unique, counts):
            name = label_names.get(int(lid), f"unknown({lid})")
            pct = (count / len(labels)) * 100.0
            print(f"  Label {lid:2d} ({name:20s}): {count:10d} points ({pct:5.2f}%)")

    def _save_merged_data(self, points, labels):
        out_dir = os.path.join(self.dataset_path, "ply")
        os.makedirs(out_dir, exist_ok=True)

        # Save NPY if requested or if no output specified (default)
        if self.config.output_npy or not self.config.output_ply:
            data = np.hstack([points, labels.reshape(-1, 1)])
            out_path = self.config.output_npy or os.path.join(
                out_dir, f"merged_gt_labels_{'_'.join(self.seq_names)}.npy"
            )
            if not out_path.endswith(".npy"):
                out_path += ".npy"
            
            np.save(out_path, data)
            print(f"Successfully saved {len(points)} points to {out_path}")

        # Save PLY if requested
        if self.config.output_ply:
            try:
                import open3d as o3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                
                # Convert labels to colors
                colors = octree_labels_to_colors(labels)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                
                out_path = self.config.output_ply
                if not out_path.endswith(".ply"):
                    out_path += ".ply"
                
                o3d.io.write_point_cloud(out_path, pcd)
                print(f"Successfully saved {len(points)} points to {out_path}")
            except ImportError:
                print("WARNING: open3d not found, cannot save .ply")


def _resolve_seq_names(args) -> list[str]:
    if getattr(args, "seq", None):
        return list(args.seq)
    if getattr(args, "seq_names_csv", None):
        parts = [p.strip() for p in args.seq_names_csv.split(",")]
        parts = [p for p in parts if p]
        if parts:
            return parts
    return list(DEFAULT_SEQ_NAMES)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create a global map for MCD sequences.")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--dataset-path",
        required=True,
        help="Path to dataset directory containing sequence folders",
    )
    common.add_argument(
        "--seq",
        action="append",
        help="Sequence name (repeatable). Default: kth_day_06,kth_day_09,kth_night_05",
    )
    common.add_argument(
        "--seq-names", dest="seq_names_csv", help="Comma-separated sequence names"
    )
    common.add_argument(
        "--max-scans", type=int, default=None, help="Max scans per sequence (default: all)"
    )
    common.add_argument(
        "--downsample-factor",
        type=int,
        default=1,
        help="Process every Nth scan (default: 1)",
    )
    common.add_argument(
        "--output-ply",
        type=str,
        default=None,
        help="Optional output path for .ply (default auto-generated if needed)",
    )
    common.add_argument(
        "--output-npy",
        type=str,
        default=None,
        help="Optional output path for .npy (default auto-generated if needed)",
    )

    sub = p.add_subparsers(dest="command")

    p_oct = sub.add_parser("octree", parents=[common], help="Octree pipeline (default)")
    p_oct.add_argument(
        "--resolution", type=float, default=0.5, help="Octree voxel resolution in meters"
    )
    p_oct.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Max distance (m) from pose to keep points",
    )
    p_oct.set_defaults(builder_kind="octree")

    p_vox = sub.add_parser("voxel", parents=[common], help="Voxel pipeline")
    p_vox.add_argument(
        "--voxel-size",
        type=float,
        default=1.0,
        help="Voxel size in meters for downsampling",
    )
    p_vox.set_defaults(builder_kind="voxel")

    return p


def main(argv=None) -> int:
    parser = build_arg_parser()

    if argv is None:
        argv = sys.argv[1:]

    # Default subcommand: octree.
    if len(argv) == 0:
        parser.print_help()
        return 0
    if argv[0] not in ("octree", "voxel"):
        # If user passed flags first, treat as octree by default
        # (unless they asked for top-level help)
        if "--help" not in argv and "-h" not in argv:
            argv = ["octree", *argv]

    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        # Happens if user asks for help / malformed args
        parser.print_help()
        return 0

    seq_names = _resolve_seq_names(args)
    if args.downsample_factor < 1:
        raise SystemExit("ERROR: --downsample-factor must be >= 1")

    common_kwargs = dict(
        dataset_path=args.dataset_path,
        seq_names=seq_names,
        max_scans=args.max_scans,
        downsample_factor=args.downsample_factor,
    )

    if args.command == "octree":
        builder = OctreeMapBuilder(
            config=OctreeConfig(
                resolution=args.resolution,
                max_distance=args.max_distance,
                output_ply=args.output_ply,
                output_npy=args.output_npy,
            ),
            **common_kwargs,
        )
        return builder.run()

    if args.command == "voxel":
        builder = VoxelMapBuilder(
            config=VoxelConfig(
                voxel_size=args.voxel_size,
                output_ply=args.output_ply,
                output_npy=args.output_npy,
            ),
            **common_kwargs,
        )
        return builder.run()

    raise SystemExit(f"ERROR: Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

