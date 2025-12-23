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
                                        k_nearest=100,
                                        max_distance=2.0,
                                        exclude_self=False):
    """
    Relabel scan points using mode label of nearest neighbors within distance threshold.
    
    For each scan point:
    1. Find k_nearest nearest neighbors in the global map
    2. Filter to only neighbors within max_distance
    3. Optionally exclude the point itself (distance = 0) - useful for map self-relabeling
    4. Assign the mode (most common) label from filtered neighbors
    
    Args:
        scan_points_world: (N, 3) array of scan points in world frame
        kdtree: Pre-built cKDTree for global map points
        global_map_labels: (M,) array of global map semantic labels
        k_nearest: Number of nearest neighbors to consider (default: 100)
        max_distance: Maximum distance for nearest neighbor matching in meters (default: 2.0)
        exclude_self: If True, exclude neighbors with distance = 0 (the point itself) (default: False)
    
    Returns:
        refined_labels: (N,) array of refined semantic labels for scan
        match_distances: (N,) array of minimum distances to nearest neighbors within threshold
    """
    from tqdm import tqdm
    num_points = len(scan_points_world)
    refined_labels = np.zeros(num_points, dtype=np.int32)
    match_distances = np.full(num_points, np.inf, dtype=np.float32)
    
    # Query k nearest neighbors for all points at once
    # Returns distances and indices as 2D arrays: (N, k)
    distances_2d, indices_2d = kdtree.query(scan_points_world, k=k_nearest)
    
    # Handle case where kdtree.query returns 1D for k=1
    if distances_2d.ndim == 1:
        distances_2d = distances_2d.reshape(-1, 1)
        indices_2d = indices_2d.reshape(-1, 1)
    
    # Process each scan point
    for i in tqdm(range(num_points), desc="Relabeling scan points"):
        # Get distances and indices for this point
        point_distances = distances_2d[i]
        point_indices = indices_2d[i]
        
        # Filter neighbors within max_distance
        within_threshold = point_distances <= max_distance
        
        # Optionally exclude the point itself (distance = 0)
        if exclude_self:
            within_threshold = within_threshold & (point_distances > 1e-6)  # Exclude distance == 0
        
        if not np.any(within_threshold):
            # No neighbors within threshold - use the closest neighbor's label even if beyond threshold
            # But skip self if exclude_self is True
            if exclude_self and len(point_distances) > 1:
                # Use second closest (first is self)
                refined_labels[i] = global_map_labels[point_indices[1]]
                match_distances[i] = point_distances[1]
            else:
                refined_labels[i] = global_map_labels[point_indices[0]]
                match_distances[i] = point_distances[0]
        else:
            # Get indices of neighbors within threshold
            valid_indices = point_indices[within_threshold]
            valid_distances = point_distances[within_threshold]
            
            # Get labels from valid neighbors
            neighbor_labels = global_map_labels[valid_indices]
            
            # Compute mode (most common label) using numpy
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            mode_label = unique_labels[np.argmax(counts)]
            
            refined_labels[i] = mode_label
            
            # Store minimum distance within threshold
            match_distances[i] = valid_distances.min()
    
    return refined_labels, match_distances


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
    # print(f"  Saved {len(labels)} labels to {output_file}")


def reprocess_map(global_map_file, output_global_map_file, max_distance=2.0):
    """
    Relabel global semantic map using mode label of nearest neighbors.
    
    For each point in the global map:
    1. Find 100 nearest neighbors (excluding itself)
    2. Filter to only neighbors within max_distance
    3. Assign the mode (most common) label from filtered neighbors
    
    This smooths/noise-reduces the labels by considering local spatial consistency.
    
    Args:
        global_map_file: Path to input global semantic map .npy file with format [x, y, z, intensity, semantic_id]
        output_global_map_file: Path to output relabeled map .npy file
        max_distance: Maximum distance for nearest neighbor matching in meters (default: 2.0)
    """
    
    # Load global semantic map
    map_points, map_intensities, map_labels = load_global_semantic_map(global_map_file)
    
    # Build KD-Tree once for all scans
    from scipy.spatial import cKDTree
    print(f"\nBuilding KD-Tree for {len(map_points)} global map points...")
    kdtree = cKDTree(map_points)
    print("KD-Tree built successfully!")
    
    print(f"\nRelabeling global map using nearest neighbors...")
    # Relabel using nearest neighbors (with pre-built KD-Tree)
    # Use exclude_self=True since we're relabeling the map against itself
    refined_labels, distances = relabel_scan_using_nearest_neighbors(
        map_points, kdtree, map_labels, k_nearest=100, max_distance=max_distance, exclude_self=True
    )
    
    # Create output map with format: [x, y, z, intensity, semantic_id]
    output_map = np.column_stack([map_points, map_intensities, refined_labels])
    
    # Save relabeled map
    np.save(output_global_map_file, output_map)
    print(f"Relabeled map saved to {output_global_map_file}")
    print(f"  Points: {len(map_points)}")
    print(f"  Output format: [x, y, z, intensity, semantic_id]")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Relabel LiDAR scans using global semantic map")

    # Dataset path
    parser.add_argument("--dataset_path", type=str, default="/media/donceykong/donceys_data_ssd/datasets/CU_MULTI/data",
                       help="Path to dataset root")

    # Environment name
    parser.add_argument("--environment", type=str, default="kittredge_loop",
                       help="Environment name (default: kittredge_loop)")

    # Input global map file postfix
    # parser.add_argument("--input_global_map_postfix", type=str, default="sem_map_orig",
    #                    help="Input file postfix (default: sem_map_orig)")
    parser.add_argument("--input_global_map_postfix", type=str, default="sem_map_orig_confident",
                       help="Input file postfix (default: sem_map_orig)")

    parser.add_argument("--output_global_map_postfix", type=str, default="knn_smoothed",
                       help="Output file postfix (default: knn_smoothed)")

    parser.add_argument("--max_distance", type=float, default=2.0,
                       help="Maximum distance for nearest neighbor matching in meters (default: 2.0)")


    args = parser.parse_args()

    dataset_path = os.path.join(args.dataset_path)
    env_path = os.path.join(dataset_path, args.environment)
    file_dir = os.path.join(env_path, "additional") # additional folder contains the global semantic map and other files
    input_global_map_file = os.path.join(file_dir, f"{args.environment}_{args.input_global_map_postfix}.npy")
    output_global_map_file = os.path.join(file_dir, f"{args.environment}_{args.input_global_map_postfix}_{args.output_global_map_postfix}.npy")

    # Check if global map exists
    if not os.path.exists(input_global_map_file):
        print(f"Error: Global map file not found: {input_global_map_file}")
        return
    
    try:
        reprocess_map(input_global_map_file, output_global_map_file, max_distance=args.max_distance)
    except Exception as e:
        print(f"\nError processing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)


if __name__ == "__main__":
    main()

