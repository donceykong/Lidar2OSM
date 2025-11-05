#!/usr/bin/env python3
"""
Script to create a global semantic map from LiDAR scans.
Accumulates LiDAR data, applies voxel downsampling, and saves:
- An image visualization of the semantic point cloud
- A numpy file with x, y, z, intensity, semantic_id (all float64)
Option to save in UTM coordinates or convert to lat/lon.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Internal imports
from lidar2osm.utils.file_io import read_bin_file
from lidar2osm.datasets.cu_multi_dataset import labels as sem_kitti_labels
from lidar2osm.core.pointcloud import labels2RGB


def load_poses(poses_file):
    """Load UTM poses from CSV file."""
    import pandas as pd
    
    print(f"\nReading UTM poses CSV file: {poses_file}")
    
    try:
        df = pd.read_csv(poses_file, comment='#')
        print(f"Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
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
            print(f"Error processing row: {e}")
            continue
    
    print(f"Successfully loaded {len(poses)} poses\n")
    return poses


def transform_imu_to_lidar(poses):
    """Transform poses from IMU frame to LiDAR frame."""
    from scipy.spatial.transform import Rotation as R
    
    # IMU to LiDAR transformation
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
        
        lidar_position = imu_position + imu_rot_matrix @ IMU_TO_LIDAR_T
        lidar_rot_matrix = imu_rot_matrix @ imu_to_lidar_rot_matrix
        lidar_quat = R.from_matrix(lidar_rot_matrix).as_quat()
        
        transformed_pose = np.concatenate([lidar_position, lidar_quat])
        transformed_poses[timestamp] = transformed_pose
    
    return transformed_poses


def voxel_downsample(points, intensities, labels=None, voxel_size=1.0):
    """
    Downsample point cloud using voxel centers.
    
    Args:
        points: numpy array of shape (N, 3) containing xyz coordinates
        intensities: numpy array of shape (N,) containing intensity values
        labels: optional numpy array of shape (N,) containing semantic labels
        voxel_size: size of voxel cube in meters (default: 1.0)
    
    Returns:
        downsampled_points: numpy array of shape (M, 3) with voxel centers
        downsampled_intensities: numpy array of shape (M,) with mean intensities
        downsampled_labels: numpy array of shape (M,) with mode labels (if labels provided)
    """
    if len(points) == 0:
        return np.array([]), np.array([]), np.array([]) if labels is not None else (np.array([]), np.array([]))
    
    # Calculate voxel grid dimensions
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    voxel_dims = np.ceil((max_coords - min_coords) / voxel_size).astype(int)
    
    # Assign points to voxels
    voxel_indices = np.floor((points - min_coords) / voxel_size).astype(int)
    voxel_indices = np.clip(voxel_indices, 0, voxel_dims - 1)
    
    # Create unique voxel keys
    voxel_keys = np.array([f"{x}_{y}_{z}" for x, y, z in voxel_indices])
    
    # Find unique voxels and calculate centers
    unique_voxels, inverse_indices = np.unique(voxel_keys, return_inverse=True)
    
    downsampled_points = []
    downsampled_intensities = []
    downsampled_labels = [] if labels is not None else None
    
    for i, voxel_id in enumerate(unique_voxels):
        # Find all points in this voxel
        voxel_mask = inverse_indices == i
        voxel_points = points[voxel_mask]
        voxel_intensities = intensities[voxel_mask]
        
        # Use voxel center (mean of all points in voxel)
        if len(voxel_points) > 0:
            voxel_center = np.mean(voxel_points, axis=0)
            mean_intensity = np.mean(voxel_intensities)
            
            downsampled_points.append(voxel_center)
            downsampled_intensities.append(mean_intensity)
            
            # For labels, use mode (most common label)
            if labels is not None:
                voxel_labels = labels[voxel_mask]
                unique_labels, counts = np.unique(voxel_labels, return_counts=True)
                mode_label = unique_labels[np.argmax(counts)]
                downsampled_labels.append(mode_label)
    
    downsampled_points = np.array(downsampled_points)
    downsampled_intensities = np.array(downsampled_intensities)
    
    if labels is not None:
        downsampled_labels = np.array(downsampled_labels)
        return downsampled_points, downsampled_intensities, downsampled_labels
    else:
        return downsampled_points, downsampled_intensities


def utm_to_latlon(points_utm):
    """Convert UTM coordinates to lat/lon."""
    from pyproj import Transformer
    
    # UTM zone 13N for Colorado
    transformer = Transformer.from_crs("EPSG:32613", "EPSG:4326", always_xy=True)
    
    lons, lats = transformer.transform(points_utm[:, 0], points_utm[:, 1])
    
    # Return as (lat, lon, z)
    latlon_points = np.column_stack([lats, lons, points_utm[:, 2]])
    
    return latlon_points


def accumulate_lidar_scans(dataset_path, environment, robot, poses_dict, 
                           num_scans=500, per_scan_voxel_size=4.0, global_voxel_size=1.0):
    """
    Accumulate LiDAR scans into a global point cloud.
    
    Args:
        dataset_path: Path to dataset
        environment: Environment name
        robot: Robot name
        poses_dict: Dictionary of poses
        num_scans: Number of scans to accumulate
        per_scan_voxel_size: Voxel size for per-scan downsampling
    
    Returns:
        combined_points: (N, 3) array of world coordinates
        combined_intensities: (N,) array of intensities
        combined_labels: (N,) array of semantic labels
    """
    from scipy.spatial.transform import Rotation as R
    
    velodyne_path = Path(dataset_path) / environment / robot / "lidar_bin/data"
    velodyne_files = sorted([f for f in velodyne_path.glob("*.bin")])
    
    # labels_path = Path(dataset_path) / environment / robot / f"{robot}_{environment}_lidar_labels"
    labels_path = Path(dataset_path) / environment / robot / f"{robot}_{environment}_refined_lidar_labels"
    label_files = sorted([f for f in labels_path.glob("*.bin")]) if labels_path.exists() else None
    
    if not label_files:
        print("Warning: No semantic label files found!")
        return None, None, None
    
    print(f"Found {len(velodyne_files)} LiDAR scans and {len(label_files)} label files")
    
    # Sample scans evenly
    total_scans = min(len(velodyne_files), len(label_files))
    sample_count = min(num_scans, total_scans)
    sample_indices = np.linspace(0, total_scans - 1, sample_count, dtype=int)
    print(f"Processing {sample_count} scans...")
    
    all_world_points = []
    all_intensities = []
    all_labels = []
    
    timestamps = list(poses_dict.keys())
    timestamps.sort()
    
    for pose_idx in tqdm(sample_indices, desc="Loading LiDAR scans", unit="scan"):
        if pose_idx >= len(velodyne_files) or pose_idx >= len(timestamps):
            continue
        
        try:
            # Load LiDAR scan
            points = read_bin_file(velodyne_files[pose_idx], dtype=np.float32, shape=(-1, 4))
            points_xyz = points[:, :3]
            intensities = points[:, 3]
            
            # Load semantic labels
            labels = read_bin_file(label_files[pose_idx], dtype=np.int32)
            
            # Ensure same length
            if len(labels) != len(points_xyz):
                min_length = min(len(labels), len(points_xyz))
                labels = labels[:min_length]
                points_xyz = points_xyz[:min_length]
                intensities = intensities[:min_length]
            
            # Get pose for this frame
            timestamp = timestamps[pose_idx]
            pose_data = poses_dict[timestamp]
            
            # Create transformation matrix
            position = pose_data[:3]
            quat = pose_data[3:7]  # [qx, qy, qz, qw]
            
            rotation_matrix = R.from_quat(quat).as_matrix()
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = position
            
            # Transform points to world coordinates
            points_homogeneous = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1))])
            world_points = (transform_matrix @ points_homogeneous.T).T
            world_points_xyz = world_points[:, :3]
            
            # Per-scan voxel downsampling
            if per_scan_voxel_size > 0:
                world_points_xyz, intensities, labels = voxel_downsample(
                    world_points_xyz, intensities, labels, voxel_size=per_scan_voxel_size
                )
            
            # Accumulate
            all_world_points.append(world_points_xyz)
            all_intensities.append(intensities)
            all_labels.append(labels)
            
        except Exception as e:
            print(f"\nError loading scan {pose_idx}: {e}")
            continue
    
    if not all_world_points:
        print("No points accumulated!")
        return None, None, None
    
    # Combine all scans
    combined_points = np.vstack(all_world_points)
    combined_intensities = np.hstack(all_intensities)
    combined_labels = np.hstack(all_labels)
    
    print(f"\nAccumulated {len(combined_points)} points from {len(all_world_points)} scans")
    
    return combined_points, combined_intensities, combined_labels


def plot_semantic_map(points, labels, output_file="global_semantic_map.png", use_latlon=False):
    """
    Plot semantic point cloud and save as image.
    
    Args:
        points: (N, 3) array of coordinates (UTM or lat/lon)
        labels: (N,) array of semantic labels
        output_file: Output filename
        use_latlon: If True, points are in lat/lon format
    """
    # Get semantic colors
    labels_dict = {label.id: label.color for label in sem_kitti_labels}
    semantic_colors = labels2RGB(labels, labels_dict)
    
    print(f"\nCreating semantic map visualization...")
    print(f"Points shape: {points.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    
    plt.figure(figsize=(16, 12))
    
    # Determine which coordinates to use
    if use_latlon:
        x_coords = points[:, 1]  # longitude
        y_coords = points[:, 0]  # latitude
        x_label = 'Longitude'
        y_label = 'Latitude'
    else:
        x_coords = points[:, 0]  # UTM x
        y_coords = points[:, 1]  # UTM y
        x_label = 'UTM X (m)'
        y_label = 'UTM Y (m)'
    
    # Calculate alpha based on Z height
    z_values = points[:, 2]
    normalized_z = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values))
    alpha_values = 0.3 + 0.6 * normalized_z
    alpha_values = np.clip(alpha_values, 0.3, 0.9)
    
    # Plot points with semantic colors
    scatter = plt.scatter(x_coords, y_coords, c=semantic_colors, s=1.0, 
                         alpha=alpha_values, zorder=5)
    
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
        plt.legend(handles=legend_elements, loc='upper left', 
                  bbox_to_anchor=(1.02, 1), fontsize=9)
    
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title('Global Semantic LiDAR Map', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Image saved to {output_file}")
    plt.close()


def save_point_cloud(points, intensities, labels, output_file="global_semantic_map.npy"):
    """
    Save point cloud data to numpy file.
    
    Args:
        points: (N, 3) array of coordinates
        intensities: (N,) array of intensities
        labels: (N,) array of semantic labels
        output_file: Output filename
    """
    # Create structured array with x, y, z, intensity, semantic_id (all float64)
    data = np.zeros((len(points), 5), dtype=np.float64)
    data[:, 0] = points[:, 0]  # x
    data[:, 1] = points[:, 1]  # y
    data[:, 2] = points[:, 2]  # z
    data[:, 3] = intensities    # intensity
    data[:, 4] = labels.astype(np.float64)  # semantic_id
    
    np.save(output_file, data)
    print(f"Point cloud data saved to {output_file}")
    print(f"Shape: {data.shape}, dtype: {data.dtype}")
    print(f"Columns: [x, y, z, intensity, semantic_id]")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create global semantic map from LiDAR scans")

    parser.add_argument("--num_scans", type=int, default=1000, #default=10000,
                       help="Number of scans to accumulate (default: 500)")

    # Per-scan voxel size in meters (default: 4.0)
    parser.add_argument("--per_scan_voxel", type=float, default=0.5,
                       help="Per-scan voxel size in meters (default: 4.0)")

    # Global voxel size in meters (default: 1.0)
    parser.add_argument("--global_voxel", type=float, default=0.5,
                       help="Global voxel size in meters (default: 1.0)")

    # Use lat/lon coordinates (default: keep UTM)
    parser.add_argument("--use_latlon", action="store_true",
                       help="Convert coordinates to lat/lon (default: keep UTM)")

    # Output file prefix (default: KL_SEM_MAP_OG)
    parser.add_argument("--output_prefix", type=str, default="KL_SEM_MAP_OG",
                       help="Output file prefix (default: KL_SEM_MAP_OG)")

    args = parser.parse_args()
    
    # Hardcoded paths
    dataset_path = "/media/donceykong/donceys_data_ssd/datasets/CU_MULTI/data"
    environment = "kittredge_loop"
    robots = ["robot1", "robot2", "robot3", "robot4"]
    
    # Lists to accumulate data from all robots
    all_robots_points = []
    all_robots_intensities = []
    all_robots_labels = []
    
    # Loop through all robots
    for robot in robots:
        print(f"\n{'='*80}")
        print(f"Processing {robot}")
        print(f"{'='*80}")
        
        # Construct file paths
        poses_file = Path(dataset_path) / environment / robot / f"{robot}_{environment}_gt_utm_poses.csv"
        
        if not poses_file.exists():
            print(f"Warning: Poses file not found for {robot}: {poses_file}")
            print(f"Skipping {robot}...")
            continue
        
        # Load poses
        print(f"Loading poses for {robot}...")
        poses = load_poses(poses_file)
        print(f"Loaded {len(poses)} poses for {robot}")
        
        if len(poses) == 0:
            print(f"No poses found for {robot}! Skipping...")
            continue
        
        # Transform poses from IMU to LiDAR frame
        print(f"Transforming poses from IMU to LiDAR frame for {robot}...")
        poses = transform_imu_to_lidar(poses)
        
        # Accumulate LiDAR scans
        print(f"\nAccumulating LiDAR scans for {robot}...")
        robot_points, robot_intensities, robot_labels = accumulate_lidar_scans(
            dataset_path, environment, robot, poses,
            num_scans=args.num_scans,
            per_scan_voxel_size=args.per_scan_voxel,
            global_voxel_size=args.global_voxel
        )
        
        if robot_points is None:
            print(f"Failed to accumulate LiDAR scans for {robot}! Skipping...")
            continue
        
        print(f"Accumulated {len(robot_points)} points from {robot}")
        
        # Add to global lists
        all_robots_points.append(robot_points)
        all_robots_intensities.append(robot_intensities)
        all_robots_labels.append(robot_labels)
    
    # Check if we have any data
    if not all_robots_points:
        print("\nError: No data accumulated from any robot!")
        return
    
    # Combine data from all robots
    print(f"\n{'='*80}")
    print("Combining data from all robots...")
    print(f"{'='*80}")
    combined_points = np.vstack(all_robots_points)
    combined_intensities = np.hstack(all_robots_intensities)
    combined_labels = np.hstack(all_robots_labels)
    print(f"Combined {len(combined_points)} points from {len(all_robots_points)} robots")
    
    # # Final global voxel downsampling across all robots
    # if args.global_voxel > 0:
    #     print(f"\nApplying final global voxel downsampling ({args.global_voxel}m) across all robots...")
    #     combined_points, combined_intensities, combined_labels = voxel_downsample(
    #         combined_points, combined_intensities, combined_labels,
    #         voxel_size=args.global_voxel
    #     )
    #     print(f"Final downsampled to {len(combined_points)} points")
    
    # Convert to lat/lon if requested
    if args.use_latlon:
        print("\nConverting UTM coordinates to lat/lon...")
        combined_points = utm_to_latlon(combined_points)
        coord_type = "latlon"
    else:
        coord_type = "utm"
    
    # Print statistics
    print(f"\nFinal point cloud statistics:")
    print(f"  Total points: {len(combined_points)}")
    print(f"  Coordinate system: {coord_type.upper()}")
    print(f"  X range: [{combined_points[:, 0].min():.6f}, {combined_points[:, 0].max():.6f}]")
    print(f"  Y range: [{combined_points[:, 1].min():.6f}, {combined_points[:, 1].max():.6f}]")
    print(f"  Z range: [{combined_points[:, 2].min():.6f}, {combined_points[:, 2].max():.6f}]")
    print(f"  Intensity range: [{combined_intensities.min():.3f}, {combined_intensities.max():.3f}]")
    print(f"  Unique labels: {np.unique(combined_labels)}")
    
    # Save outputs
    output_image = f"{args.output_prefix}_all_robots_{coord_type}.png"
    output_npy = f"{args.output_prefix}_all_robots_{coord_type}.npy"
    
    plot_semantic_map(combined_points, combined_labels, 
                     output_file=output_image, use_latlon=args.use_latlon)
    
    save_point_cloud(combined_points, combined_intensities, combined_labels,
                    output_file=output_npy)
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"Generated files:")
    print(f"  - {output_image}")
    print(f"  - {output_npy}")
    print(f"\nData from {len(all_robots_points)} robots successfully combined and saved!")


if __name__ == "__main__":
    main()

