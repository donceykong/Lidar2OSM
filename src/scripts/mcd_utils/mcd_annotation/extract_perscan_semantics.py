#!/usr/bin/env python3
"""
Extract per-scan semantic labels by projecting them onto a GT point cloud.

Example usage:
python extract_per_scan_semantics.py \
    --scan_dir /path/to/lidar_bin/data/ \
    --pose_file /path/to/pose_inW.csv \
    --merged_npy /path/to/merged.npy \
    --output_dir /path/to/output_labels \
    --jobs 16
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# Add src to python path
current_file = Path(__file__).resolve()
src_path = current_file.parent.parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

try:
    from lidar2osm.utils.file_io import read_bin_file
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from lidar2osm.utils.file_io import read_bin_file

# Precompute the inverse transform (lidar->body->world becomes a single multiply)
_LIDAR_TO_BODY = np.linalg.inv(np.array([
    [0.9999135040741837, -0.011166365511073898, -0.006949579221822984, -0.04894521120494695],
    [-0.011356389542502144, -0.9995453006865824, -0.02793249526856565, -0.03126929060348084],
    [-0.006634514801117132, 0.02800900135032654, -0.999585653686922, -0.01755515794222565],
    [0.0, 0.0, 0.0, 1.0]
]))

# Global for worker processes
_gt_points = None
_all_gt_labels = None

def init_worker(gt_points, all_gt_labels):
    global _gt_points, _all_gt_labels
    _gt_points = gt_points
    _all_gt_labels = all_gt_labels

def load_poses(poses_file):
    try:
        df = pd.read_csv(poses_file, comment='#', skipinitialspace=True)
        col_names = [str(col).strip().lower().replace('#', '').replace(' ', '') for col in df.columns]
        
        has_timestamp = any('timestamp' in col or col == 't' for col in col_names)
        has_pose = all(any(c in col for col in col_names) for c in ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        
        if not (has_timestamp and has_pose):
            df = pd.read_csv(poses_file, comment='#', header=None, skipinitialspace=True)
            if len(df.columns) < 8:
                return {}, {}
    except Exception:
        return {}, {}
    
    poses = {}
    index_to_timestamp = {}
    col_map = {str(c).strip().lower().replace('#', '').replace(' ', ''): c for c in df.columns}
    
    for _, row in df.iterrows():
        try:
            timestamp, pose_index = None, None
            
            for k, v in col_map.items():
                if k == 't' or 'timestamp' in k:
                    timestamp = row[v]
                    break
            
            if 'num' in col_map:
                pose_index = int(row[col_map['num']])
            
            if timestamp is not None:
                vals = [row.get(col_map.get(c)) for c in ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']]
                if any(v is None for v in vals): continue
                x, y, z, qx, qy, qz, qw = vals
            elif len(row) >= 9:
                pose_index, timestamp = int(row.iloc[0]), row.iloc[1]
                x, y, z = row.iloc[2], row.iloc[3], row.iloc[4]
                qx, qy, qz, qw = row.iloc[5], row.iloc[6], row.iloc[7], row.iloc[8]
            elif len(row) >= 8:
                timestamp = row.iloc[0]
                x, y, z = row.iloc[1], row.iloc[2], row.iloc[3]
                qx, qy, qz, qw = row.iloc[4], row.iloc[5], row.iloc[6], row.iloc[7]
            else:
                continue

            if None in [timestamp, x, y, z, qx, qy, qz, qw]:
                continue
            
            # Store as numpy array for faster access in transform
            poses[float(timestamp)] = np.array([pose_index, x, y, z, qx, qy, qz, qw], dtype=np.float64)
            
            if pose_index is not None:
                index_to_timestamp[int(pose_index)] = float(timestamp)
        except Exception:
            continue
            
    return poses, index_to_timestamp

def transform_points_to_world(points_xyz, position, quaternion):
    """Transform points: lidar -> body -> world (fused into one operation)."""
    body_to_world = np.eye(4, dtype=np.float64)
    body_to_world[:3, :3] = R.from_quat(quaternion).as_matrix()
    body_to_world[:3, 3] = position
    T = body_to_world @ _LIDAR_TO_BODY
    
    # In-place transformation avoiding extra allocations
    n = len(points_xyz)
    result = np.empty((n, 3), dtype=np.float32)
    result[:] = (T[:3, :3] @ points_xyz.T).T + T[:3, 3]
    return result

def process_single_scan_worker(scan_idx, scan_dir, bin_file, pose_data, output_dir):
    output_file = os.path.join(output_dir, f"{scan_idx + 1:010d}.bin")
    if os.path.exists(output_file):
        return 0

    try:
        points = read_bin_file(os.path.join(scan_dir, bin_file), dtype=np.float32, shape=(-1, 4))
        world_xyz = transform_points_to_world(points[:, :3], pose_data[1:4], pose_data[4:8])
        
        # Vectorized bounding box filter
        b_min, b_max = world_xyz.min(axis=0), world_xyz.max(axis=0)
        mask = ((_gt_points >= b_min) & (_gt_points <= b_max)).all(axis=1)
        
        sub_gt_points = _gt_points[mask]
        if len(sub_gt_points) == 0: return -1
        
        sub_gt_labels = _all_gt_labels[mask]
        
        # Build KDTree with balanced_tree=False for faster construction
        kdtree = cKDTree(sub_gt_points, balanced_tree=False)
        _, indices = kdtree.query(world_xyz, k=1)
        
        sub_gt_labels[indices].astype(np.int32).tofile(output_file)
        return 1
        
    except Exception:
        return -1

def process_all_scans(scan_dir, pose_file, gt_npy_path, output_dir, seq_name, jobs=1):
    if not all(os.path.exists(p) for p in [gt_npy_path, scan_dir, pose_file]):
        raise FileNotFoundError(f"Skipping {seq_name}: Missing required files.")

    poses_dict, index_to_timestamp = load_poses(pose_file)
    if not poses_dict:
        raise ValueError(f"Skipping {seq_name}: No poses loaded.")
    
    bin_files = sorted([f for f in os.listdir(scan_dir) if f.endswith('.bin')])
    if not bin_files:
        raise ValueError(f"Skipping {seq_name}: No .bin files found.")

    gt_data = np.load(gt_npy_path)
    gt_points = np.ascontiguousarray(gt_data[:, :3], dtype=np.float32)
    all_gt_labels = gt_data[:, 3].astype(np.int32)

    os.makedirs(output_dir, exist_ok=True)

    tasks = [(i, scan_dir, f, poses_dict[index_to_timestamp[i + 1]], output_dir)
             for i, f in enumerate(bin_files) if (i + 1) in index_to_timestamp]
    
    success, failed = 0, 0
    with ProcessPoolExecutor(max_workers=jobs, initializer=init_worker, initargs=(gt_points, all_gt_labels)) as executor:
        futures = [executor.submit(process_single_scan_worker, *t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {seq_name}", unit="scan"):
            res = future.result()
            if res == 1: success += 1
            elif res == -1: failed += 1
            
    return success, failed

def main():
    parser = argparse.ArgumentParser(description="Extract semantic labels from GT point clouds per scan.")
    parser.add_argument("--merged_npy", required=True, help="Path to merged .npy file")
    parser.add_argument("--scan_dir", required=True, help="Directory containing .bin files")
    parser.add_argument("--pose_file", required=True, help="Path to pose file")
    parser.add_argument("--output_dir", required=True, help="Directory for output labels")
    parser.add_argument("--jobs", "-j", type=int, default=os.cpu_count(), help="Number of parallel jobs")
    
    args = parser.parse_args()
    
    seq_name = os.path.basename(os.path.dirname(os.path.abspath(args.scan_dir)))
    s, f = process_all_scans(args.scan_dir, args.pose_file, args.merged_npy, args.output_dir, seq_name, args.jobs)
    print(f"Complete. Success: {s}, Failed: {f}")

if __name__ == '__main__':
    main()
