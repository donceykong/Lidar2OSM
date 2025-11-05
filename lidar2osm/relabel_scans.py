#!/usr/bin/env python3
"""
Script to relabel LiDAR scans using a global semantic map.
1. Load and visualize global semantic map
2. Load individual scans from robot trajectories
3. Overlay scans on the global map for visualization
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Internal imports
from lidar2osm.utils.file_io import read_bin_file
from lidar2osm.datasets.cu_multi_dataset import labels as sem_kitti_labels
from lidar2osm.core.pointcloud import labels2RGB


def load_global_semantic_map(npy_file):
    """
    Load the global semantic map from numpy file.
    
    Args:
        npy_file: Path to .npy file with shape (N, 5) containing [x, y, z, intensity, semantic_id]
    
    Returns:
        points: (N, 3) array of xyz coordinates in UTM
        intensities: (N,) array of intensities
        labels: (N,) array of semantic labels
    """
    print(f"\nLoading global semantic map from {npy_file}")
    data = np.load(npy_file)
    print(f"Loaded data shape: {data.shape}, dtype: {data.dtype}")
    
    points = data[:, :3]  # x, y, z
    intensities = data[:, 3]  # intensity
    labels = data[:, 4].astype(np.int32)  # semantic_id
    
    print(f"Points: {points.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    print(f"UTM bounds: X=[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
          f"Y=[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
          f"Z=[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    return points, intensities, labels


def plot_semantic_map(points, labels, ax=None, title="Global Semantic Map", alpha_scale=True):
    """
    Plot semantic point cloud with color-coded labels.
    
    Args:
        points: (N, 3) array of xyz coordinates in UTM
        labels: (N,) array of semantic labels
        ax: Matplotlib axis to plot on (creates new figure if None)
        title: Plot title
        alpha_scale: If True, scale alpha by height
    
    Returns:
        fig, ax: Matplotlib figure and axis
    """
    # Get semantic colors
    labels_dict = {label.id: label.color for label in sem_kitti_labels}
    semantic_colors = labels2RGB(labels, labels_dict)
    
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 10))
    else:
        fig = ax.figure
    
    # Calculate alpha based on Z height
    if alpha_scale:
        z_values = points[:, 2]
        normalized_z = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values))
        alpha_values = 0.3 + 0.6 * normalized_z
        alpha_values = np.clip(alpha_values, 0.3, 0.9)
    else:
        alpha_values = 0.7
    
    # Plot points with semantic colors (UTM coordinates: x, y)
    scatter = ax.scatter(points[:, 0], points[:, 1], c=semantic_colors, 
                        s=1.0, alpha=alpha_values, zorder=5)
    
    # Create legend for semantic classes
    unique_labels = np.unique(labels)
    legend_elements = []
    for label_id in unique_labels[:15]:  # Show top 15 classes
        if label_id in labels_dict:
            color = np.array(labels_dict[label_id]) / 255.0
            # Find label name
            label_name = "Unknown"
            for label in sem_kitti_labels:
                if label.id == label_id:
                    label_name = label.name
                    break
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                          markersize=8, label=f'{label_name} ({label_id})')
            )
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(1.02, 1), fontsize=9)
    
    ax.set_xlabel('UTM X (m)', fontsize=12)
    ax.set_ylabel('UTM Y (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    return fig, ax


def load_poses(poses_file):
    """
    Load UTM poses from CSV file.
    
    Args:
        poses_file: Path to CSV file with poses
    
    Returns:
        poses: Dictionary mapping timestamp to [x, y, z, qx, qy, qz, qw]
    """
    import pandas as pd
    
    print(f"\nLoading poses from {poses_file}")
    
    try:
        df = pd.read_csv(poses_file, comment='#', header=None)
        print(f"Successfully read CSV with {len(df)} rows")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {}
    
    poses = {}
    for _, row in df.iterrows():
        try:
            if 'timestamp' in df.columns:
                timestamp = row['timestamp']
                x = row['x']
                y = row['y']
                z = row['z']
                qx = row['qx']
                qy = row['qy']
                qz = row['qz']
                qw = row['qw']
            else:
                if len(row) >= 8:
                    timestamp = row.iloc[0]
                    x = row.iloc[1]
                    y = row.iloc[2]
                    z = row.iloc[3]
                    qx = row.iloc[4]
                    qy = row.iloc[5]
                    qz = row.iloc[6]
                    qw = row.iloc[7]
                else:
                    continue
            
            pose = [float(x), float(y), float(z), float(qx), float(qy), float(qz), float(qw)]
            poses[float(timestamp)] = pose
        except Exception as e:
            continue
    
    print(f"Successfully loaded {len(poses)} poses")
    return poses


def transform_imu_to_lidar(poses):
    """
    Transform poses from IMU frame to LiDAR frame.
    
    Args:
        poses: Dictionary mapping timestamp to [x, y, z, qx, qy, qz, qw]
    
    Returns:
        transformed_poses: Dictionary with transformed poses
    """
    # IMU to LiDAR transformation (from CU-MULTI calibration)
    IMU_TO_LIDAR_T = np.array([-0.058038, 0.015573, 0.049603])
    IMU_TO_LIDAR_Q = [0.0, 0.0, 1.0, 0.0]  # [qx, qy, qz, qw]
    
    imu_to_lidar_rot = R.from_quat(IMU_TO_LIDAR_Q)
    imu_to_lidar_rot_matrix = imu_to_lidar_rot.as_matrix()
    
    transformed_poses = {}
    
    for timestamp, pose in poses.items():
        imu_position = np.array(pose[:3])
        imu_quat = pose[3:7]
        
        imu_rot = R.from_quat(imu_quat)
        imu_rot_matrix = imu_rot.as_matrix()
        
        # Transform position: lidar_pos = imu_pos + imu_rot * imu_to_lidar_translation
        lidar_position = imu_position + imu_rot_matrix @ IMU_TO_LIDAR_T
        
        # Transform orientation: lidar_rot = imu_rot * imu_to_lidar_rot
        lidar_rot_matrix = imu_rot_matrix @ imu_to_lidar_rot_matrix
        lidar_quat = R.from_matrix(lidar_rot_matrix).as_quat()
        
        transformed_pose = np.concatenate([lidar_position, lidar_quat])
        transformed_poses[timestamp] = transformed_pose
    
    return transformed_poses


def transform_points_to_world(points_lidar, pose):
    """
    Transform points from LiDAR frame to world frame using pose.
    
    Args:
        points_lidar: (N, 3) array of points in LiDAR frame
        pose: [x, y, z, qx, qy, qz, qw] pose in world frame
    
    Returns:
        points_world: (N, 3) array of points in world frame
    """
    position = np.array(pose[:3])
    quat = pose[3:7]  # [qx, qy, qz, qw]
    
    # Create transformation matrix
    rotation_matrix = R.from_quat(quat).as_matrix()
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = position
    
    # Transform points
    points_homogeneous = np.hstack([points_lidar, np.ones((points_lidar.shape[0], 1))])
    points_world = (transform_matrix @ points_homogeneous.T).T
    
    return points_world[:, :3]


def relabel_scan_using_nearest_neighbors(scan_points_world, kdtree, global_map_labels, 
                                          max_distance=2.0):
    """
    Relabel scan points using nearest neighbor search in global map.
    
    Args:
        scan_points_world: (N, 3) array of scan points in world frame
        kdtree: Pre-built cKDTree for global map points
        global_map_labels: (M,) array of global map semantic labels
        max_distance: Maximum distance for nearest neighbor matching (meters)
    
    Returns:
        refined_labels: (N,) array of refined semantic labels for scan
        match_distances: (N,) array of distances to nearest neighbors
    """
    # Query nearest neighbors using pre-built tree
    distances, indices = kdtree.query(scan_points_world, k=1)
    
    # Get labels from nearest neighbors
    refined_labels = global_map_labels[indices]
    
    return refined_labels.astype(np.int32), distances


def save_refined_labels(labels, output_file):
    """
    Save refined labels to binary file.
    
    Args:
        labels: (N,) array of semantic labels
        output_file: Path to output file
    """
    # Convert to int32 and save
    labels_int32 = labels.astype(np.int32)
    labels_int32.tofile(output_file)
    print(f"  Saved {len(labels)} labels to {output_file}")


def process_robot_scans(global_map_file, dataset_path, environment, robot, 
                        max_distance=2.0, visualize=False):
    """
    Process all scans for a robot and save refined labels.
    
    Args:
        global_map_file: Path to global semantic map .npy file
        dataset_path: Path to dataset root
        environment: Environment name
        robot: Robot name
        max_distance: Maximum distance for nearest neighbor matching
        visualize: If True, create visualization for first scan
    """
    from tqdm import tqdm
    
    print(f"\n{'='*80}")
    print(f"Processing {robot} in {environment}")
    print(f"{'='*80}")
    
    # Load global semantic map
    map_points, map_intensities, map_labels = load_global_semantic_map(global_map_file)
    
    # Build KD-Tree once for all scans
    from scipy.spatial import cKDTree
    print(f"\nBuilding KD-Tree for {len(map_points)} global map points...")
    kdtree = cKDTree(map_points)
    print("KD-Tree built successfully!")
    
    # Load robot poses
    poses_file = dataset_path / environment / robot / f"{robot}_{environment}_gt_utm_poses.csv"
    if not poses_file.exists():
        print(f"Error: Poses file not found: {poses_file}")
        return
    
    poses = load_poses(poses_file)
    if len(poses) == 0:
        print("No poses loaded!")
        return
    
    # Transform poses from IMU to LiDAR frame
    print("Transforming poses from IMU to LiDAR frame...")
    poses = transform_imu_to_lidar(poses)
    
    # Setup paths
    lidar_path = dataset_path / environment / robot / "lidar_bin/data"
    lidar_files = sorted(list(lidar_path.glob("*.bin")))
    print(f"Found {len(lidar_files)} lidar files")
    
    if len(lidar_files) == 0:
        print("No lidar files found!")
        return
    
    # Create output directory
    output_dir = dataset_path / environment / robot / f"{robot}_{environment}_refined_lidar_labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get timestamps
    timestamps = sorted(list(poses.keys()))
    
    # Determine how many scans to process (minimum of lidar files and poses)
    num_scans_to_process = min(len(lidar_files), len(timestamps))
    
    if len(lidar_files) != len(timestamps):
        print(f"⚠ WARNING: {len(lidar_files)} lidar files but {len(timestamps)} poses")
        print(f"  Will process {num_scans_to_process} scans (the minimum)")
    
    # Statistics
    total_points = 0
    total_close_matches = 0
    
    # Process each scan
    print(f"\nProcessing {num_scans_to_process} scans with pre-built KD-Tree...")
    report_interval = max(1, num_scans_to_process // 10)  # Report every 10%
    
    for frame_idx in tqdm(range(num_scans_to_process), desc=f"Relabeling {robot} scans"):
        
        # Load lidar scan
        scan_file = lidar_files[frame_idx]
        points = read_bin_file(scan_file, dtype=np.float32, shape=(-1, 4))
        points_xyz = points[:, :3]
        
        # Get pose for this frame
        timestamp = timestamps[frame_idx]
        pose = poses[timestamp]
        
        # Transform points to world frame
        world_points = transform_points_to_world(points_xyz, pose)
        
        # Relabel using nearest neighbors (with pre-built KD-Tree)
        refined_labels, distances = relabel_scan_using_nearest_neighbors(
            world_points, kdtree, map_labels, max_distance=max_distance
        )
        
        # Update statistics
        total_points += len(refined_labels)
        total_close_matches += np.sum(distances <= max_distance)
        
        # Periodic progress report
        if (frame_idx + 1) % report_interval == 0:
            current_match_rate = 100 * total_close_matches / total_points if total_points > 0 else 0
            tqdm.write(f"  Progress: {frame_idx + 1}/{num_scans_to_process} scans | "
                      f"Match rate: {current_match_rate:.1f}% within {max_distance}m")
        
        # Save refined labels
        output_file = output_dir / scan_file.name.replace('.bin', '.bin')
        save_refined_labels(refined_labels, output_file)
        
        # Visualize first scan if requested
        if visualize and frame_idx == 0:
            fig, ax = plt.subplots(figsize=(16, 12))
            plot_semantic_map(map_points, map_labels, ax=ax,
                            title=f"Global Map with {robot} Scan 0 Overlay")
            
            # Overlay scan with refined labels
            labels_dict = {label.id: label.color for label in sem_kitti_labels}
            scan_colors = labels2RGB(refined_labels, labels_dict)
            
            ax.scatter(world_points[:, 0], world_points[:, 1], c=scan_colors,
                      s=2.0, alpha=0.8, zorder=10, label=f'{robot} Scan 0 (Refined)')
            
            # Mark robot position
            ax.plot(pose[0], pose[1], 'go', markersize=12, label='Robot Position', zorder=11)
            ax.legend(loc='upper right')
            
            viz_file = output_dir.parent / f"{robot}_scan_overlay_viz.png"
            plt.tight_layout()
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to {viz_file}")
            plt.close()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary for {robot}:")
    print(f"  Scans processed: {num_scans_to_process} / {len(lidar_files)} lidar files")
    print(f"  Total points processed: {total_points}")
    print(f"  Points within {max_distance}m: {total_close_matches} "
          f"({100*total_close_matches/total_points:.1f}%)")
    print(f"  Refined labels saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Relabel LiDAR scans using global semantic map")
    parser.add_argument("--global_map", type=str, default="KL_SEM_MAP_RELABELED.npy",
                       help="Path to global semantic map .npy file")
    parser.add_argument("--dataset_path", type=str,
                       default="/media/donceykong/donceys_data_ssd/datasets/CU_MULTI/data",
                       help="Path to dataset root")
    parser.add_argument("--environment", type=str, default="kittredge_loop",
                       help="Environment name (default: kittredge_loop)")
    parser.add_argument("--robot", type=str, default=None,
                       help="Robot name (default: process all robots)")
    parser.add_argument("--max_distance", type=float, default=2.0,
                       help="Maximum distance for nearest neighbor matching in meters (default: 2.0)")
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualization for first scan of each robot")
    args = parser.parse_args()
    
    global_map_file = Path(args.global_map)
    dataset_path = Path(args.dataset_path)
    
    # Check if global map exists
    if not global_map_file.exists():
        print(f"Error: Global map file not found: {global_map_file}")
        return
    
    # Determine which robots to process
    if args.robot:
        robots = [args.robot]
    else:
        # Process all robots in the environment
        env_path = dataset_path / args.environment
        if not env_path.exists():
            print(f"Error: Environment path not found: {env_path}")
            return
        robots = [d.name for d in env_path.iterdir() if d.is_dir() and d.name.startswith('robot')]
        robots = sorted(robots)
    
    print(f"\nProcessing robots: {', '.join(robots)}")
    
    # Process each robot
    for robot in robots:
        try:
            process_robot_scans(global_map_file, dataset_path, args.environment, robot,
                              max_distance=args.max_distance, visualize=args.visualize)
        except Exception as e:
            print(f"\nError processing {robot}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)


if __name__ == "__main__":
    main()

