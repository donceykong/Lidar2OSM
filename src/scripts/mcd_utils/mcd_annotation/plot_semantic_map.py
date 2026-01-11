#!/usr/bin/env python3
"""
Plot semantic maps from BIN or PCD files.

Example usage:
  # BIN mode (inferred labels from gt_labels/*.bin)
  python plot_semantic_map.py bin --dataset-path /path/to/MCD --seq kth_day_06

  # PCD mode (GT labels from PCD files)  
  python plot_semantic_map.py pcd --root-path /path/to/MCD/kth_day_06
"""

import argparse
import os
import sys
from collections import namedtuple

import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# Body to LiDAR transformation matrix
BODY_TO_LIDAR_TF = np.array([
    [0.9999135040741837, -0.011166365511073898, -0.006949579221822984, -0.04894521120494695],
    [-0.011356389542502144, -0.9995453006865824, -0.02793249526856565, -0.03126929060348084],
    [-0.006634514801117132, 0.02800900135032654, -0.999585653686922, -0.01755515794222565],
    [0.0, 0.0, 0.0, 1.0]
])

Label = namedtuple("Label", ["name", "id", "color"])

SEM_KITTI_LABELS = [
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
    Label("OSM BUILDING", 45, (0, 0, 255)),
    Label("OSM ROAD", 46, (255, 0, 0)),
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

_SEM_KITTI_COLOR_MAP = {l.name: l.color for l in SEM_KITTI_LABELS}

SEMANTIC_LABELS = [
    Label("barrier", 0, (0, 0, 0)),
    Label("bike", 1, _SEM_KITTI_COLOR_MAP["bicycle"]),
    Label("building", 2, _SEM_KITTI_COLOR_MAP["building"]),
    Label("chair", 3, (0, 0, 0)),
    Label("cliff", 4, (0, 0, 0)),
    Label("container", 5, (0, 0, 0)),
    Label("curb", 6, _SEM_KITTI_COLOR_MAP["sidewalk"]),
    Label("fence", 7, _SEM_KITTI_COLOR_MAP["fence"]),
    Label("hydrant", 8, (0, 0, 0)),
    Label("infosign", 9, (0, 0, 0)),
    Label("lanemarking", 10, _SEM_KITTI_COLOR_MAP["lane-marking"]),
    Label("noise", 11, (0, 0, 0)),
    Label("other", 12, _SEM_KITTI_COLOR_MAP["other-object"]),
    Label("parkinglot", 13, _SEM_KITTI_COLOR_MAP["parking"]),
    Label("pedestrian", 14, _SEM_KITTI_COLOR_MAP["person"]),
    Label("pole", 15, _SEM_KITTI_COLOR_MAP["pole"]),
    Label("road", 16, _SEM_KITTI_COLOR_MAP["road"]),
    Label("shelter", 17, _SEM_KITTI_COLOR_MAP["building"]),
    Label("sidewalk", 18, _SEM_KITTI_COLOR_MAP["sidewalk"]),
    Label("stairs", 19, (0, 0, 0)),
    Label("structure-other", 20, _SEM_KITTI_COLOR_MAP["other-structure"]),
    Label("traffic-cone", 21, (0, 0, 0)),
    Label("traffic-sign", 22, _SEM_KITTI_COLOR_MAP["traffic-sign"]),
    Label("trashbin", 23, (0, 0, 0)),
    Label("treetrunk", 24, _SEM_KITTI_COLOR_MAP["trunk"]),
    Label("vegetation", 25, _SEM_KITTI_COLOR_MAP["vegetation"]),
    Label("vehicle-dynamic", 26, _SEM_KITTI_COLOR_MAP["car"]),
    Label("vehicle-other", 27, _SEM_KITTI_COLOR_MAP["car"]),
    Label("vehicle-static", 28, _SEM_KITTI_COLOR_MAP["car"]),
]

LABEL_ID_TO_COLOR = {l.id: l.color for l in SEMANTIC_LABELS}


def read_bin_file(file_path, dtype, shape=None):
    data = np.fromfile(file_path, dtype=dtype)
    return data.reshape(shape) if shape else data


def labels_to_colors(labels):
    colors = np.zeros((len(labels), 3), dtype=np.float32)
    for i, lid in enumerate(labels):
        color = LABEL_ID_TO_COLOR.get(int(lid), (128, 128, 128))
        colors[i] = np.array(color, dtype=np.float32) / 255.0
    return colors


def load_poses(poses_file):
    """Load poses from CSV file. Returns (poses_dict, index_to_timestamp)."""
    try:
        df = pd.read_csv(poses_file, comment='#', skipinitialspace=True)
        col_names = [str(c).strip().lower().replace('#', '').replace(' ', '') for c in df.columns]
        has_ts = any('timestamp' in c or c == 't' for c in col_names)
        has_pose = all(any(k in c for c in col_names) for k in ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        
        if not (has_ts and has_pose and len(df.columns) >= 8):
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
                if any(v is None for v in vals):
                    continue
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

            poses[float(timestamp)] = [pose_index, float(x), float(y), float(z), 
                                       float(qx), float(qy), float(qz), float(qw)]
            if pose_index is not None:
                index_to_timestamp[pose_index] = float(timestamp)
        except Exception:
            continue

    print(f"Loaded {len(poses)} poses from {os.path.basename(poses_file)}")
    return poses, index_to_timestamp


def transform_points_to_world(points_xyz, position, quaternion):
    body_to_world = np.eye(4)
    body_to_world[:3, :3] = R.from_quat(quaternion).as_matrix()
    body_to_world[:3, 3] = position
    
    lidar_to_body = np.linalg.inv(BODY_TO_LIDAR_TF)
    transform = body_to_world @ lidar_to_body
    
    points_h = np.hstack([points_xyz, np.ones((len(points_xyz), 1), dtype=points_xyz.dtype)])
    return (transform @ points_h.T).T[:, :3]


def process_bin_scans(root_path, poses_dict, index_to_timestamp, sorted_indices, max_distance):
    """Process BIN format scans."""
    data_dir = os.path.join(root_path, "lidar_bin", "data")
    labels_dir = os.path.join(root_path, "gt_labels")
    timestamps_file = os.path.join(root_path, "lidar_bin", "timestamps.txt")
    
    timestamps = np.loadtxt(timestamps_file)
    all_points, all_colors = [], []

    for pose_num in tqdm(sorted_indices, desc="Processing BIN scans", unit="scan"):
        pose_ts = index_to_timestamp[pose_num]
        pose_data = poses_dict[pose_ts]
        position, quaternion = pose_data[1:4], pose_data[4:8]

        bin_file = f"{pose_num:010d}.bin"
        bin_path = os.path.join(data_dir, bin_file)
        label_path = os.path.join(labels_dir, bin_file)

        if not os.path.exists(bin_path) or not os.path.exists(label_path):
            continue

        # Validate timestamps
        if abs(timestamps[pose_num - 1] - pose_ts) > 0.1:
            continue

        try:
            points = read_bin_file(bin_path, dtype=np.float32, shape=(-1, 4))[:, :3]
            labels = read_bin_file(label_path, dtype=np.int32)
            if len(labels) != len(points):
                continue
            colors = labels_to_colors(labels)
        except Exception:
            continue

        world_points = transform_points_to_world(points, position, quaternion)

        if max_distance:
            dists = np.linalg.norm(world_points - position, axis=1)
            mask = dists <= max_distance
            world_points, colors = world_points[mask], colors[mask]

        if len(world_points) > 0:
            all_points.append(world_points)
            all_colors.append(colors)

    return all_points, all_colors


def process_pcd_scans(root_path, poses_dict, index_to_timestamp, sorted_indices):
    """Process PCD format scans."""
    try:
        from pypcd import pypcd
    except ImportError:
        print("ERROR: pypcd not installed. Run: pip install pypcd")
        return [], []

    pcd_dir = os.path.join(root_path, "gt_lidar_labels", "inL_labelled")
    
    # Build PCD file index
    pcd_files = {}
    for f in os.listdir(pcd_dir):
        if f.startswith('cloud_') and f.endswith('.pcd'):
            try:
                idx = int(f.replace('cloud_', '').replace('.pcd', ''))
                pcd_files[idx] = f
            except ValueError:
                continue

    all_points, all_colors = [], []

    for pose_num in tqdm(sorted_indices, desc="Processing PCD scans", unit="scan"):
        if pose_num not in pcd_files:
            continue

        pose_ts = index_to_timestamp[pose_num]
        pose_data = poses_dict[pose_ts]
        position, quaternion = pose_data[1:4], pose_data[4:8]

        pcd_path = os.path.join(pcd_dir, pcd_files[pose_num])
        try:
            pc = pypcd.PointCloud.from_path(pcd_path)
            points = np.column_stack([pc.pc_data['x'], pc.pc_data['y'], pc.pc_data['z']]).astype(np.float32)
            
            if len(points) == 0:
                continue

            # Extract labels
            labels = None
            for field in ['label', 'class', 'semantic', 'labels', 'classes', 'semantic_id']:
                if field in pc.pc_data.dtype.names:
                    labels = pc.pc_data[field].astype(np.int32)
                    break
            
            if labels is not None:
                colors = labels_to_colors(labels)
            else:
                # Fall back to RGB fields
                fields = pc.pc_data.dtype.names
                if 'rgb' in fields:
                    rgb = pc.pc_data['rgb']
                    colors = np.column_stack([
                        ((rgb >> 16) & 0xFF) / 255.0,
                        ((rgb >> 8) & 0xFF) / 255.0,
                        (rgb & 0xFF) / 255.0
                    ]).astype(np.float32)
                else:
                    colors = np.ones((len(points), 3), dtype=np.float32) * 0.5
        except Exception:
            continue

        world_points = transform_points_to_world(points, position, quaternion)
        all_points.append(world_points)
        all_colors.append(colors)

    return all_points, all_colors


def plot_map(input_type, root_path, max_scans=None, downsample_factor=1, voxel_size=None, max_distance=None, visualize=True):
    """Main plotting function."""
    # Determine poses file location
    if input_type == 'bin':
        poses_file = os.path.join(root_path, "pose_inW.csv")
    else:
        poses_file = os.path.join(root_path, "gt_lidar_labels", "pose_inW.csv")

    if not os.path.exists(poses_file):
        print(f"ERROR: Poses file not found: {poses_file}")
        return

    poses_dict, index_to_timestamp = load_poses(poses_file)
    if not poses_dict or not index_to_timestamp:
        print("ERROR: No poses loaded")
        return

    # Build sorted indices with downsampling/limiting
    sorted_indices = sorted(index_to_timestamp.keys())
    if downsample_factor > 1:
        sorted_indices = sorted_indices[::downsample_factor]
    if max_scans:
        sorted_indices = sorted_indices[:max_scans]

    print(f"Processing {len(sorted_indices)} scans...")

    # Process scans based on input type
    if input_type == 'bin':
        all_points, all_colors = process_bin_scans(root_path, poses_dict, index_to_timestamp, sorted_indices, max_distance)
    else:
        all_points, all_colors = process_pcd_scans(root_path, poses_dict, index_to_timestamp, sorted_indices)

    if not all_points:
        print("ERROR: No points accumulated")
        return

    # Voxel downsample each scan
    if voxel_size and voxel_size > 0:
        downsampled_points, downsampled_colors = [], []
        for pts, cols in zip(all_points, all_colors):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols)
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            downsampled_points.append(np.asarray(pcd.points))
            downsampled_colors.append(np.asarray(pcd.colors))
        all_points, all_colors = downsampled_points, downsampled_colors

    # Accumulate
    accumulated_points = np.vstack(all_points)
    accumulated_colors = np.vstack(all_colors)
    print(f"Total points: {len(accumulated_points)}")

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(accumulated_points)
    pcd.colors = o3d.utility.Vector3dVector(accumulated_colors)

    # Save PLY
    seq_name = os.path.basename(os.path.normpath(root_path))
    ply_dir = os.path.join(root_path, "ply")
    os.makedirs(ply_dir, exist_ok=True)
    
    prefix = "inferred_labels" if input_type == 'bin' else "gt_labels"
    ply_path = os.path.join(ply_dir, f"{prefix}_{seq_name}.ply")
    
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved {len(accumulated_points)} points to {ply_path}")

    if visualize:
        print("\nVisualizing... (Q or ESC to quit)")
        o3d.visualization.draw_geometries([pcd])


def main():
    parser = argparse.ArgumentParser(description="Plot semantic maps from BIN or PCD files.")
    subparsers = parser.add_subparsers(dest="input_type", required=True)

    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--max-scans", type=int, help="Max scans to process")
    common.add_argument("--downsample-factor", type=int, default=1, help="Process every Nth scan")
    common.add_argument("--voxel-size", type=float, help="Voxel size for downsampling (meters)")
    common.add_argument("--no-visualize", action="store_true", help="Skip visualization")

    # BIN subcommand
    bin_parser = subparsers.add_parser("bin", parents=[common], help="Process BIN files with GT labels")
    bin_parser.add_argument("--dataset-path", required=True, help="Path to dataset directory")
    bin_parser.add_argument("--seq", required=True, nargs='+', help="Sequence name(s)")
    bin_parser.add_argument("--max-distance", type=float, help="Max distance from pose to keep points")

    # PCD subcommand
    pcd_parser = subparsers.add_parser("pcd", parents=[common], help="Process PCD files with semantic labels")
    pcd_parser.add_argument("--root-path", required=True, help="Path to sequence root directory")

    args = parser.parse_args()

    if args.input_type == 'bin':
        for seq in args.seq:
            root_path = os.path.join(args.dataset_path, seq)
            print(f"\n{'='*60}\nProcessing sequence: {seq}\n{'='*60}")
            plot_map(
                input_type='bin',
                root_path=root_path,
                max_scans=args.max_scans,
                downsample_factor=args.downsample_factor,
                voxel_size=args.voxel_size,
                max_distance=args.max_distance,
                visualize=not args.no_visualize
            )
    else:
        plot_map(
            input_type='pcd',
            root_path=args.root_path,
            max_scans=args.max_scans,
            downsample_factor=args.downsample_factor,
            voxel_size=args.voxel_size,
            visualize=not args.no_visualize
        )


if __name__ == '__main__':
    main()
