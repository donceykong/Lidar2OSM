#!/usr/bin/env python3
"""
Visualize semantic point cloud with initial pose.

This script loads a semantic point cloud from a .npy file and visualizes it
with semantically colored points. It also displays the initial pose from a
CSV file as a coordinate frame.
"""

import os
import sys
import numpy as np
import pandas as pd
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from collections import namedtuple

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# Add dataset_binarize to path for importing create_seq_gt_map_npy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Internal imports
from lidar2osm.core.pointcloud.pointcloud import labels2RGB_tqdm

# Body to LiDAR transformation matrix
BODY_TO_LIDAR_TF = np.array([
    [0.9999135040741837, -0.011166365511073898, -0.006949579221822984, -0.04894521120494695],
    [-0.011356389542502144, -0.9995453006865824, -0.02793249526856565, -0.03126929060348084],
    [-0.006634514801117132, 0.02800900135032654, -0.999585653686922, -0.01755515794222565],
    [0.0, 0.0, 0.0, 1.0]
])

Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        "id",  # An integer ID that is associated with this label.
        "color",  # The color of this label
    ],
)

sem_kitti_labels = [
    # name, id, color
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

# Create a mapping from sem_kitti label names to colors for reuse
_sem_kitti_color_map = {label.name: label.color for label in sem_kitti_labels}

# Semantic labels (converted from dictionary to namedtuple list)
# Colors are reused from sem_kitti_labels where names match
semantic_labels = [
    Label("barrier", 0, (0, 0, 0)),  # Placeholder color, needs assignment
    Label("bike", 1, _sem_kitti_color_map["bicycle"]),  # Matches sem_kitti "bicycle"
    Label("building", 2, _sem_kitti_color_map["building"]),  # Matches sem_kitti "building"
    Label("chair", 3, (0, 0, 0)),  # Placeholder color, needs assignment
    Label("cliff", 4, (0, 0, 0)),  # Placeholder color, needs assignment
    Label("container", 5, (0, 0, 0)),  # Placeholder color, needs assignment
    Label("curb", 6, _sem_kitti_color_map["sidewalk"]),  # Placeholder color, needs assignment
    Label("fence", 7, _sem_kitti_color_map["fence"]),  # Matches sem_kitti "fence"
    Label("hydrant", 8, (0, 0, 0)),  # Placeholder color, needs assignment
    Label("infosign", 9, (0, 0, 0)),  # Placeholder color, needs assignment
    Label("lanemarking", 10, _sem_kitti_color_map["lane-marking"]),  # Matches sem_kitti "lane-marking"
    Label("noise", 11, (255, 0, 0)),  # Placeholder color, needs assignment
    Label("other", 12, _sem_kitti_color_map["other-object"]),  # Matches sem_kitti "other-object"
    Label("parkinglot", 13, _sem_kitti_color_map["parking"]),  # Matches sem_kitti "parking"
    Label("pedestrian", 14, _sem_kitti_color_map["person"]),  # Matches sem_kitti "person"
    Label("pole", 15, _sem_kitti_color_map["pole"]),  # Matches sem_kitti "pole"
    Label("road", 16, _sem_kitti_color_map["road"]),  # Matches sem_kitti "road"
    Label("shelter", 17, _sem_kitti_color_map["building"]),  # Placeholder color, needs assignment
    Label("sidewalk", 18, _sem_kitti_color_map["sidewalk"]),  # Matches sem_kitti "sidewalk"
    Label("stairs", 19, (0, 0, 0)),  # Placeholder color, needs assignment
    Label("structure-other", 20, _sem_kitti_color_map["other-structure"]),  # Matches sem_kitti "other-structure"
    Label("traffic-cone", 21, (0, 0, 0)),  # Placeholder color, needs assignment
    Label("traffic-sign", 22, _sem_kitti_color_map["traffic-sign"]),  # Matches sem_kitti "traffic-sign"
    Label("trashbin", 23, (0, 0, 0)),  # Placeholder color, needs assignment
    Label("treetrunk", 24, _sem_kitti_color_map["trunk"]),  # Matches sem_kitti "trunk"
    Label("vegetation", 25, _sem_kitti_color_map["vegetation"]),  # Matches sem_kitti "vegetation"
    Label("vehicle-dynamic", 26, _sem_kitti_color_map["car"]),  # Matches sem_kitti "car"
    Label("vehicle-other", 27, _sem_kitti_color_map["car"]),  # Matches sem_kitti "other-vehicle"
    Label("vehicle-static", 28, _sem_kitti_color_map["car"]),  # Matches sem_kitti "car"
]

def load_semantic_map(npy_file):
    """
    Load semantic map from numpy file.
    
    Args:
        npy_file: Path to .npy file with columns [x, y, z, intensity, semantic_id] or [x, y, z, semantic_id]
    
    Returns:
        points: (N, 3) array of coordinates [x, y, z]
        labels: (N,) array of semantic labels
    """
    print(f"\nLoading semantic map from {npy_file}")
    data = np.load(npy_file)
    print(f"Loaded data shape: {data.shape}")
    
    points = data[:, :3]  # x, y, z
    
    # Handle different data formats:
    # Format 1: [x, y, z, intensity, semantic_id] - 5 columns
    # Format 2: [x, y, z, semantic_id] - 4 columns
    if data.shape[1] == 5:
        labels = data[:, 4].astype(np.int32)
    elif data.shape[1] == 4:
        labels = data[:, 3].astype(np.int32)
    else:
        print(f"Warning: Unexpected data shape {data.shape}. Assuming [x, y, z, semantic_id] format.")
        labels = data[:, -1].astype(np.int32) if data.shape[1] >= 4 else None
    
    print(f"  Points: {len(points)}")
    print(f"  X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    if labels is not None:
        print(f"  Unique labels: {np.unique(labels)}")
    
    return points, labels


def load_initial_pose(pose_csv_file):
    """
    Load the first pose from a CSV file.
    
    Args:
        pose_csv_file: Path to CSV file with columns [num, t, x, y, z, qx, qy, qz, qw] or similar
    
    Returns:
        pose: Dictionary with 'position' (x, y, z) and 'quaternion' (qx, qy, qz, qw)
        transform_matrix: 4x4 transformation matrix
    """
    print(f"\nLoading initial pose from {pose_csv_file}")
    
    try:
        df = pd.read_csv(pose_csv_file, comment='#')
        print(f"Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None
    
    if len(df) == 0:
        print("Error: CSV file is empty")
        return None, None
    
    # Get the first row (initial pose)
    first_row = df.iloc[0]
    
    try:
        # Check if required columns exist
        required_cols = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        if all(col in df.columns for col in required_cols):
            # Use column names directly
            x = float(first_row['x'])
            y = float(first_row['y'])
            z = float(first_row['z'])
            qx = float(first_row['qx'])
            qy = float(first_row['qy'])
            qz = float(first_row['qz'])
            qw = float(first_row['qw'])
        else:
            # Try positional access (skip first column which might be 'num' or 't')
            # Expected format: [num/t, x, y, z, qx, qy, qz, qw]
            if len(first_row) >= 8:
                x = float(first_row.iloc[1])  # Skip first column (num or t)
                y = float(first_row.iloc[2])
                z = float(first_row.iloc[3])
                qx = float(first_row.iloc[4])
                qy = float(first_row.iloc[5])
                qz = float(first_row.iloc[6])
                qw = float(first_row.iloc[7])
            elif len(first_row) >= 7:
                # Maybe no 'num' column, try starting from index 0
                x = float(first_row.iloc[0])
                y = float(first_row.iloc[1])
                z = float(first_row.iloc[2])
                qx = float(first_row.iloc[3])
                qy = float(first_row.iloc[4])
                qz = float(first_row.iloc[5])
                qw = float(first_row.iloc[6])
            else:
                print(f"Error: CSV row doesn't have enough columns. Found {len(first_row)} columns.")
                return None, None
        
        position = np.array([x, y, z])
        quaternion = np.array([qx, qy, qz, qw])  # scipy uses [x, y, z, w] format
        
        # Convert quaternion to rotation matrix
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        
        # Create 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = position
        
        print(f"  Initial pose position: [{x:.2f}, {y:.2f}, {z:.2f}]")
        print(f"  Initial pose quaternion: [{qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f}]")
        
        return {'position': position, 'quaternion': quaternion}, transform_matrix
        
    except Exception as e:
        print(f"Error processing pose: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def quaternion_pose_to_4x4(trans, quat):
    """Convert quaternion pose to 4x4 transformation matrix."""
    rotation_matrix = R.from_quat(quat).as_matrix()
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = trans
    return transformation_matrix


def visualize_semantic_map_with_pose(npy_file, pose_csv_file):
    """Load and visualize semantic point cloud with initial pose."""
    
    # Load semantic map
    points, labels = load_semantic_map(npy_file)
    
    if labels is None:
        print("Error: No semantic labels found in the file")
        return
    
    # Load initial pose (in body frame)
    pose_dict, body_to_world_tf = load_initial_pose(pose_csv_file)
    
    if body_to_world_tf is None:
        print("Warning: Could not load initial pose. Visualizing point cloud only.")
        transform_matrix = None
    else:
        # Convert from body frame to LiDAR frame
        # The pose from CSV is in body frame (T_body_to_world)
        # We need LiDAR frame pose (T_lidar_to_world)
        # T_lidar_to_world = T_body_to_world @ inv(T_body_to_lidar)
        lidar_to_body_tf = np.linalg.inv(BODY_TO_LIDAR_TF)
        transform_matrix = body_to_world_tf @ lidar_to_body_tf
        print(f"\nBody to world transform (from CSV):")
        print(f"  Position: [{body_to_world_tf[0,3]:.2f}, {body_to_world_tf[1,3]:.2f}, {body_to_world_tf[2,3]:.2f}]")
        print(f"\nLiDAR to world transform (after applying BODY_TO_LIDAR_TF):")
        print(f"  Position: [{transform_matrix[0,3]:.2f}, {transform_matrix[1,3]:.2f}, {transform_matrix[2,3]:.2f}]")
    
    # Downsample BEFORE converting labels to colors (more efficient)
    print("\nDownsampling point cloud before color conversion...")
    # Create temporary point cloud for downsampling
    temp_pcd = o3d.geometry.PointCloud()
    temp_pcd.points = o3d.utility.Vector3dVector(points)
    
    # Calculate appropriate voxel size
    bbox_size = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    voxel_size = 10 #bbox_size * 0.001  # 0.1% of bounding box
    print(f"  Using voxel size: {voxel_size:.3f}")
    
    # Downsample
    downsampled_pcd = temp_pcd.voxel_down_sample(voxel_size=voxel_size)
    downsampled_points = np.asarray(downsampled_pcd.points)
    print(f"  Downsampled from {len(points)} to {len(downsampled_points)} points")
    
    # Find which original points correspond to downsampled points
    # Use KDTree to find nearest neighbors
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, indices = tree.query(downsampled_points, k=1)
    downsampled_labels = labels[indices]
    
    # Now convert semantic labels to RGB colors (only for downsampled points)
    print("Converting semantic labels to RGB colors...")
    labels_dict = {label.id: label.color for label in sem_kitti_labels}
    colors_rgb = labels2RGB_tqdm(downsampled_labels, labels_dict)
    
    # Create Open3D point cloud with downsampled data
    print("Creating Open3D point cloud...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(downsampled_points)
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
    
    # Prepare geometries list
    geometries = [pcd]
    
    # Add coordinate frame for initial pose if available
    if transform_matrix is not None:
        print("\nAdding initial pose coordinate frame...")
        # # Create coordinate frame (size in meters)
        frame_size = max(
            np.linalg.norm(downsampled_points.max(axis=0) - downsampled_points.min(axis=0)) * 0.05,
            5.0  # Minimum size of 5 meters
        )
        
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=frame_size,
            origin=[0, 0, 0]
        )
        coordinate_frame.transform(transform_matrix)
        geometries.append(coordinate_frame)
        
    #     # Also add a sphere at the initial pose position for better visibility
    #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=frame_size * 0.3)
    #     sphere.translate(pose_dict['position'])
    #     sphere.paint_uniform_color([1, 1, 0])  # Yellow
    #     geometries.append(sphere)
    
    print(f"\nVisualizing {len(downsampled_points)} points...")
    if transform_matrix is not None:
        print("  - Yellow sphere: Initial pose position")
        print("  - Coordinate frame: Initial pose orientation (Red=X, Green=Y, Blue=Z)")
    
    # Visualize using Open3D
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Semantic Point Cloud with Initial Pose",
        width=1200,
        height=800,
    )


if __name__ == "__main__":
    # File paths
    npy_file = "/media/donceykong/doncey_ssd_02/datasets/MCD/ply/merged_gt_labels_kth_day_06_kth_day_09_kth_night_05.npy"
    pose_csv_file = "/media/donceykong/doncey_ssd_02/datasets/MCD/kth_day_06/pose_inW.csv"
    
    visualize_semantic_map_with_pose(npy_file, pose_csv_file)

