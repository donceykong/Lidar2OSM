#!/usr/bin/env python3
"""
Plot a single .bin scan on top of its matching environment's GT .ply file.

The GT .ply file is located at: root_path/ply/gt_labels_<seq>.ply
The .bin scan files are located at: root_path/lidar_bin/data/*.bin
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import open3d as o3d
from tqdm import tqdm

# Import dataset_binarize package to set up sys.path for lidar2osm imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lidar2osm.utils.file_io import read_bin_file

# Body to LiDAR transformation matrix
BODY_TO_LIDAR_TF = np.array([
    [0.9999135040741837, -0.011166365511073898, -0.006949579221822984, -0.04894521120494695],
    [-0.011356389542502144, -0.9995453006865824, -0.02793249526856565, -0.03126929060348084],
    [-0.006634514801117132, 0.02800900135032654, -0.999585653686922, -0.01755515794222565],
    [0.0, 0.0, 0.0, 1.0]
])

def load_poses(poses_file):
    """
    Load poses from CSV file.
    
    Args:
        poses_file: Path to CSV file with poses (num, t, x, y, z, qx, qy, qz, qw)
    
    Returns:
        poses: Dictionary mapping timestamp to [index, x, y, z, qx, qy, qz, qw]
        index_to_timestamp: Dictionary mapping index to timestamp
    """
    print(f"\nLoading poses from {poses_file}")
    
    try:
        df = pd.read_csv(poses_file, comment='#', skipinitialspace=True)
        
        # Check if we have the expected columns
        col_names = [str(col).strip().lower().replace('#', '').replace(' ', '') for col in df.columns]
        has_timestamp_col = any('timestamp' in col or col == 't' for col in col_names)
        has_pose_cols = all(any(coord in col for col in col_names) for coord in ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        has_num_col = any('num' in col for col in col_names)
        
        if not (has_timestamp_col and has_pose_cols):
            # Try without header
            df = pd.read_csv(poses_file, comment='#', header=None, skipinitialspace=True)
            if len(df.columns) < 8:
                print(f"ERROR: Only {len(df.columns)} columns found, need at least 8")
                return {}, {}
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {}, {}
    
    poses = {}
    index_to_timestamp = {}
    
    for idx, row in df.iterrows():
        try:
            timestamp = None
            pose_index = None
            x, y, z, qx, qy, qz, qw = None, None, None, None, None, None, None
            
            # Build column map
            col_map = {}
            for col in df.columns:
                col_clean = str(col).strip().lower().replace('#', '').replace(' ', '')
                col_map[col_clean] = col
            
            # Check for timestamp column
            for col in df.columns:
                col_clean = str(col).strip().lower().replace('#', '').replace(' ', '')
                if col_clean == 't' or 'timestamp' in col_clean:
                    timestamp = row[col]
                    break
            
            # Check for index column
            if 'num' in col_map:
                pose_index = int(row[col_map['num']])
            
            # Get pose values
            if timestamp is not None:
                x = row.get(col_map.get('x'), None) if 'x' in col_map else None
                y = row.get(col_map.get('y'), None) if 'y' in col_map else None
                z = row.get(col_map.get('z'), None) if 'z' in col_map else None
                qx = row.get(col_map.get('qx'), None) if 'qx' in col_map else None
                qy = row.get(col_map.get('qy'), None) if 'qy' in col_map else None
                qz = row.get(col_map.get('qz'), None) if 'qz' in col_map else None
                qw = row.get(col_map.get('qw'), None) if 'qw' in col_map else None
            else:
                # Positional indexing
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
                    continue
            
            # Validate all values are present
            if None in [timestamp, x, y, z, qx, qy, qz, qw]:
                continue
            
            # Store pose
            pose = [pose_index, float(x), float(y), float(z), float(qx), float(qy), float(qz), float(qw)]
            poses[float(timestamp)] = pose
            
            if pose_index is not None:
                index_to_timestamp[pose_index] = float(timestamp)
        except Exception as e:
            continue
    
    print(f"Successfully loaded {len(poses)} poses")
    return poses, index_to_timestamp


def transform_points_to_world(points_xyz, position, quaternion, body_to_lidar_tf=None):
    """
    Transform points from lidar frame to world frame using pose.
    
    Args:
        points_xyz: (N, 3) array of points in lidar frame
        position: [x, y, z] translation (body frame position in world)
        quaternion: [qx, qy, qz, qw] rotation quaternion (body frame orientation in world)
        body_to_lidar_tf: Optional 4x4 transformation matrix from body to lidar frame
    
    Returns:
        world_points: (N, 3) array of points in world frame
    """
    # Create rotation matrix from quaternion (body frame orientation)
    body_rotation_matrix = R.from_quat(quaternion).as_matrix()
    
    # Create 4x4 transformation matrix for body frame in world
    body_to_world = np.eye(4)
    body_to_world[:3, :3] = body_rotation_matrix
    body_to_world[:3, 3] = position
    
    # If body_to_lidar transform is provided, compose the transformations
    if body_to_lidar_tf is not None:
        lidar_to_body = np.linalg.inv(body_to_lidar_tf)
        transform_matrix = body_to_world @ lidar_to_body
    else:
        transform_matrix = body_to_world
    
    # Transform points to world coordinates
    points_homogeneous = np.hstack(
        [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)]
    )
    world_points = (transform_matrix @ points_homogeneous.T).T
    world_points_xyz = world_points[:, :3]
    
    return world_points_xyz


def plot_bin_on_gt_ply(root_path, scan_idx=0, merged_npy_path=None, dataset_path=None):
    """
    Load a single .bin scan and plot it on top of the GT point cloud from .npy file.
    
    The GT .npy file is already in world coordinates and needs no transformation.
    The .bin scan is transformed to world coordinates using poses and BODY_TO_LIDAR_TF.
    
    Args:
        root_path: Path to root directory (should contain lidar_bin/data/)
        scan_idx: Index of the scan to load (default: 0, first scan)
        merged_npy_path: Optional path to merged .npy file. If None, will look for merged file in dataset_path/ply/
        dataset_path: Optional path to dataset directory (parent of sequences). Used to find merged .npy if merged_npy_path is None.
    """
    # Extract sequence name from root_path
    seq_name = os.path.basename(os.path.normpath(root_path))
    
    # Determine GT .npy file path
    if merged_npy_path is not None:
        # Use explicitly provided merged .npy path
        gt_npy_path = merged_npy_path
    elif dataset_path is not None:
        # Look for merged .npy file in dataset_path/ply/
        ply_dir = os.path.join(dataset_path, "ply")
        # Look for merged_gt_labels_*.npy files
        merged_files = [f for f in os.listdir(ply_dir) if f.startswith("merged_gt_labels_") and f.endswith(".npy")] if os.path.exists(ply_dir) else []
        if merged_files:
            # Use the first merged file found (or could use most recent)
            gt_npy_path = os.path.join(ply_dir, merged_files[0])
            print(f"Found merged .npy file: {gt_npy_path}")
        else:
            # Fall back to sequence-specific file
            gt_npy_path = os.path.join(root_path, "ply", f"gt_labels_{seq_name}.npy")
            print(f"No merged .npy file found, using sequence-specific: {gt_npy_path}")
    else:
        # Fall back to sequence-specific file
        gt_npy_path = os.path.join(root_path, "ply", f"gt_labels_{seq_name}.npy")
        print(f"Using sequence-specific .npy file: {gt_npy_path}")
    
    bin_data_dir = os.path.join(root_path, "lidar_bin", "data")
    
    # Check if GT .npy file exists
    if not os.path.exists(gt_npy_path):
        print(f"ERROR: GT .npy file not found: {gt_npy_path}")
        return
    
    # Check if bin data directory exists
    if not os.path.exists(bin_data_dir):
        print(f"ERROR: Bin data directory not found: {bin_data_dir}")
        return
    
    # Get all .bin files
    bin_files = sorted([f for f in os.listdir(bin_data_dir) if f.endswith('.bin')])
    
    if len(bin_files) == 0:
        print(f"ERROR: No .bin files found in {bin_data_dir}")
        return
    
    if scan_idx >= len(bin_files):
        print(f"WARNING: scan_idx {scan_idx} >= number of scans {len(bin_files)}, using last scan")
        scan_idx = len(bin_files) - 1
    
    # Get the selected bin file
    bin_file = bin_files[scan_idx]
    bin_path = os.path.join(bin_data_dir, bin_file)
    
    print(f"\n{'='*80}")
    print(f"Plotting bin scan on GT point cloud")
    print(f"{'='*80}")
    print(f"Sequence: {seq_name}")
    print(f"GT .npy file: {gt_npy_path}")
    print(f"Bin scan file: {bin_file} (index {scan_idx})")
    print(f"{'='*80}\n")
    
    # Load GT .npy file (format: [x, y, z, label])
    print(f"Loading GT .npy file: {gt_npy_path}")
    try:
        gt_data = np.load(gt_npy_path)
        if gt_data.shape[1] != 4:
            print(f"ERROR: GT .npy file should have shape (N, 4) with columns [x, y, z, label], got shape {gt_data.shape}")
            return
        gt_points_array = gt_data[:, :3]  # Extract xyz coordinates
        gt_labels = gt_data[:, 3].astype(np.int32)  # Extract labels
    except Exception as e:
        print(f"ERROR: Failed to load GT .npy file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"  Loaded {len(gt_points_array)} points with labels from GT .npy file")
    
    # Load .bin scan file
    print(f"\nLoading .bin scan file: {bin_path}")
    try:
        points = read_bin_file(bin_path, dtype=np.float32, shape=(-1, 4))
        points_xyz = points[:, :3]  # Extract xyz coordinates
        intensities = points[:, 3]  # Extract intensity (for potential color mapping)
    except Exception as e:
        print(f"ERROR: Failed to load .bin file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"  Loaded {len(points_xyz)} points from .bin file")
    
    # Transform bin scan to world coordinates (GT .ply is already in world coordinates)
    poses_file = os.path.join(root_path, "pose_inW.csv")
    if not os.path.exists(poses_file):
        print(f"ERROR: Poses file not found: {poses_file}")
        print(f"  Cannot transform bin scan to world coordinates without poses")
        return
    
    print(f"\nTransforming bin scan to world coordinates...")
    print(f"  Loading poses from: {poses_file}")
    
    # Load poses
    poses_dict, index_to_timestamp = load_poses(poses_file)
    
    if not poses_dict or not index_to_timestamp:
        print(f"ERROR: Could not load poses from {poses_file}")
        return
    
    # Get pose for this scan
    # scan_idx is 0-indexed (array index), but bin files and poses are 1-indexed
    # bin_files[0] = "0000000001.bin", so pose_num = scan_idx + 1
    pose_num = scan_idx + 1
    if pose_num not in index_to_timestamp:
        print(f"ERROR: Pose index {pose_num} not found for scan index {scan_idx}")
        print(f"  Available pose indices: {sorted(index_to_timestamp.keys())[:10]}...")
        return
    
    pose_timestamp = index_to_timestamp[pose_num]
    pose_data = poses_dict[pose_timestamp]
    position = pose_data[1:4]  # [x, y, z]
    quaternion = pose_data[4:8]  # [qx, qy, qz, qw]
    
    # Transform points to world coordinates using BODY_TO_LIDAR_TF
    world_points_xyz = transform_points_to_world(points_xyz, position, quaternion, BODY_TO_LIDAR_TF)
    print(f"  Transformed points to world coordinates using pose {pose_num}")
    
    # Calculate bounding box of transformed bin scan
    bin_min = np.min(world_points_xyz, axis=0)
    bin_max = np.max(world_points_xyz, axis=0)
    print(f"\nBin scan bounds (world coordinates):")
    print(f"  X: [{bin_min[0]:.2f}, {bin_max[0]:.2f}]")
    print(f"  Y: [{bin_min[1]:.2f}, {bin_max[1]:.2f}]")
    print(f"  Z: [{bin_min[2]:.2f}, {bin_max[2]:.2f}]")
    
    # Filter GT points to only include those within bin scan bounds
    print(f"\nFiltering GT points to bin scan bounds...")
    
    # Create mask for points within bounds
    mask = np.all((gt_points_array >= bin_min) & (gt_points_array <= bin_max), axis=1)
    filtered_gt_points = gt_points_array[mask]
    filtered_gt_labels = gt_labels[mask]
    
    if len(filtered_gt_points) == 0:
        print(f"  WARNING: No GT points found within bin scan bounds!")
        return
    
    print(f"  Filtered from {len(gt_points_array)} to {len(filtered_gt_points)} points")
    
    # Get label statistics
    unique_labels, counts = np.unique(filtered_gt_labels, return_counts=True)
    print(f"  Found {len(unique_labels)} unique labels in filtered GT points")
    print(f"  Label distribution: {dict(zip(unique_labels, counts))}")
    
    # Build KD-tree for nearest neighbor search
    print(f"\nBuilding KD-tree for nearest neighbor search...")
    filtered_gt_pcd = o3d.geometry.PointCloud()
    filtered_gt_pcd.points = o3d.utility.Vector3dVector(filtered_gt_points)
    
    # Build KD-tree
    kdtree = o3d.geometry.KDTreeFlann(filtered_gt_pcd)
    
    # Find nearest neighbor for each bin scan point and get its label
    print(f"\nFinding nearest neighbors and extracting labels for {len(world_points_xyz)} bin scan points...")
    bin_labels = np.zeros(len(world_points_xyz), dtype=np.int32)
    
    for i in range(len(world_points_xyz)):
        [_, idx, _] = kdtree.search_knn_vector_3d(world_points_xyz[i], 1)
        bin_labels[i] = gt_labels[idx[0]]
    
    print(f"  Completed nearest neighbor search")
    unique_bin_labels, bin_counts = np.unique(bin_labels, return_counts=True)
    print(f"  Bin scan label distribution: {dict(zip(unique_bin_labels, bin_counts))}")
    
    # Save labels to file
    output_labels_dir = os.path.join(root_path, "gt_labels")
    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)
        print(f"  Created directory: {output_labels_dir}")
    
    # Format scan index as 10-digit zero-padded string (1-indexed to match bin file names)
    # scan_idx is 0-indexed, but bin files are 1-indexed, so output should be scan_idx + 1
    output_filename = f"{scan_idx + 1:010d}.bin"
    output_path = os.path.join(output_labels_dir, output_filename)
    
    print(f"\nSaving labels to: {output_path}")
    bin_labels.tofile(output_path)
    print(f"  Saved {len(bin_labels)} labels as int32 binary file")
    
    # Create Open3D point cloud for the bin scan
    # Color it bright red so it stands out on the GT map
    scan_pcd = o3d.geometry.PointCloud()
    scan_pcd.points = o3d.utility.Vector3dVector(world_points_xyz)
    
    # Color the scan points bright red
    scan_colors = np.ones((len(world_points_xyz), 3), dtype=np.float32) * np.array([1.0, 0.0, 0.0])  # Red
    scan_pcd.colors = o3d.utility.Vector3dVector(scan_colors)
    
    # Convert labels to colors for visualization
    from create_seq_gt_map_npy import labels_to_colors
    filtered_gt_colors = labels_to_colors(filtered_gt_labels)
    filtered_gt_pcd.colors = o3d.utility.Vector3dVector(filtered_gt_colors)
    
    print(f"\n{'='*80}")
    print(f"Point cloud statistics:")
    print(f"  GT (original): {len(gt_points_array)} points")
    print(f"  GT (filtered): {len(filtered_gt_pcd.points)} points")
    print(f"  Bin scan: {len(scan_pcd.points)} points")
    print(f"{'='*80}")
    
    # Visualize both point clouds together
    print(f"\nVisualizing point clouds...")
    print(f"  GT point cloud: colored by semantic labels")
    print(f"  Bin scan: RED")
    print(f"\nControls:")
    print(f"  - Mouse: Rotate view")
    print(f"  - Shift + Mouse: Pan view")
    print(f"  - Mouse wheel: Zoom")
    print(f"  - Q or ESC: Quit")
    print(f"{'='*80}\n")
    
    # Create visualization (use filtered GT point cloud)
    o3d.visualization.draw_geometries([filtered_gt_pcd, scan_pcd])


def process_all_scans(root_path, merged_npy_path=None, dataset_path=None, max_scans=None, seq_name=None):
    """
    Process all scans in the sequence that have viable poses.
    
    Args:
        root_path: Path to root directory (should contain lidar_bin/data/)
        merged_npy_path: Optional path to merged .npy file. If None, will look for merged file in dataset_path/ply/
        dataset_path: Optional path to dataset directory (parent of sequences). Used to find merged .npy if merged_npy_path is None.
        max_scans: Maximum number of scans to process (None for all)
        seq_name: Optional sequence name for progress display (if None, extracted from root_path)
    
    Returns:
        Tuple of (successful_scans, failed_scans)
    """
    
    # Extract sequence name from root_path if not provided
    if seq_name is None:
        seq_name = os.path.basename(os.path.normpath(root_path))
    
    # Determine GT .npy file path
    if merged_npy_path is not None:
        gt_npy_path = merged_npy_path
    elif dataset_path is not None:
        ply_dir = os.path.join(dataset_path, "ply")
        merged_files = [f for f in os.listdir(ply_dir) if f.startswith("merged_gt_labels_") and f.endswith(".npy")] if os.path.exists(ply_dir) else []
        if merged_files:
            gt_npy_path = os.path.join(ply_dir, merged_files[0])
            print(f"Found merged .npy file: {gt_npy_path}")
        else:
            gt_npy_path = os.path.join(root_path, "ply", f"gt_labels_{seq_name}.npy")
            print(f"No merged .npy file found, using sequence-specific: {gt_npy_path}")
    else:
        gt_npy_path = os.path.join(root_path, "ply", f"gt_labels_{seq_name}.npy")
        print(f"Using sequence-specific .npy file: {gt_npy_path}")
    
    bin_data_dir = os.path.join(root_path, "lidar_bin", "data")
    
    # Check if files exist
    if not os.path.exists(gt_npy_path):
        print(f"ERROR: GT .npy file not found: {gt_npy_path}")
        return 0, 0
    
    if not os.path.exists(bin_data_dir):
        print(f"ERROR: Bin data directory not found: {bin_data_dir}")
        return 0, 0
    
    # Load poses
    poses_file = os.path.join(root_path, "pose_inW.csv")
    if not os.path.exists(poses_file):
        print(f"ERROR: Poses file not found: {poses_file}")
        return 0, 0
    
    print(f"\nLoading poses from {poses_file}")
    poses_dict, index_to_timestamp = load_poses(poses_file)
    
    if not poses_dict or not index_to_timestamp:
        print(f"ERROR: Could not load poses from {poses_file}")
        return 0, 0
    
    # Get all .bin files
    bin_files = sorted([f for f in os.listdir(bin_data_dir) if f.endswith('.bin')])
    
    if len(bin_files) == 0:
        print(f"ERROR: No .bin files found in {bin_data_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Processing all scans with viable poses")
    print(f"{'='*80}")
    print(f"Sequence: {seq_name}")
    print(f"GT .npy file: {gt_npy_path}")
    print(f"Total bin files: {len(bin_files)}")
    print(f"Total poses: {len(index_to_timestamp)}")
    print(f"{'='*80}\n")
    
    # Load GT .npy file once (will be reused for all scans)
    print(f"Loading GT .npy file: {gt_npy_path}")
    try:
        gt_data = np.load(gt_npy_path)
        if gt_data.shape[1] != 4:
            print(f"ERROR: GT .npy file should have shape (N, 4) with columns [x, y, z, label], got shape {gt_data.shape}")
            return
        gt_points_array = gt_data[:, :3]  # Extract xyz coordinates
        all_gt_labels = gt_data[:, 3].astype(np.int32)  # Extract labels
    except Exception as e:
        print(f"ERROR: Failed to load GT .npy file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"  Loaded {len(gt_points_array)} points with labels from GT .npy file")
    
    # Create output directory
    output_labels_dir = os.path.join(root_path, "gt_labels")
    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)
        print(f"Created directory: {output_labels_dir}")
    
    # Process each scan
    # scan_idx is 0-indexed (array index), but bin files and poses are 1-indexed
    valid_scan_indices = []
    for scan_idx in range(len(bin_files)):
        pose_num = scan_idx + 1  # Convert 0-indexed scan_idx to 1-indexed pose_num
        if pose_num not in index_to_timestamp:
            continue
        valid_scan_indices.append(scan_idx)
    
    if max_scans is not None:
        valid_scan_indices = valid_scan_indices[:max_scans]
    
    print(f"\nProcessing {len(valid_scan_indices)} scans with viable poses...")
    
    successful_scans = 0
    failed_scans = 0
    
    # Create progress bar with sequence name
    progress_desc = f"Processing {seq_name}" if seq_name else "Processing scans"
    skipped_scans = 0
    
    for scan_idx in tqdm(valid_scan_indices, desc=progress_desc, unit="scan"):
        # Check if label file already exists
        # scan_idx is 0-indexed, but bin files are 1-indexed, so output should be scan_idx + 1
        output_filename = f"{scan_idx + 1:010d}.bin"
        output_path = os.path.join(output_labels_dir, output_filename)
        
        if os.path.exists(output_path):
            skipped_scans += 1
            continue
        
        try:
            # Get pose for this scan
            # Convert 0-indexed scan_idx to 1-indexed pose_num
            pose_num = scan_idx + 1
            pose_timestamp = index_to_timestamp[pose_num]
            pose_data = poses_dict[pose_timestamp]
            position = pose_data[1:4]
            quaternion = pose_data[4:8]
            
            # Load .bin scan file
            bin_file = bin_files[scan_idx]
            bin_path = os.path.join(bin_data_dir, bin_file)
            
            points = read_bin_file(bin_path, dtype=np.float32, shape=(-1, 4))
            points_xyz = points[:, :3]
            
            # Transform to world coordinates
            world_points_xyz = transform_points_to_world(points_xyz, position, quaternion, BODY_TO_LIDAR_TF)
            
            # Calculate bounding box
            bin_min = np.min(world_points_xyz, axis=0)
            bin_max = np.max(world_points_xyz, axis=0)
            
            # Filter GT points to bounding box
            mask = np.all((gt_points_array >= bin_min) & (gt_points_array <= bin_max), axis=1)
            filtered_gt_points = gt_points_array[mask]
            filtered_gt_labels = all_gt_labels[mask]
            
            if len(filtered_gt_points) == 0:
                tqdm.write(f"  Scan {scan_idx}: No GT points in bounding box, skipping")
                failed_scans += 1
                continue
            
            # Validate filtered GT points before building KD-tree
            if len(filtered_gt_points) == 0:
                tqdm.write(f"  Scan {scan_idx}: Empty filtered GT point cloud, skipping")
                failed_scans += 1
                continue
            
            # Validate GT points are finite
            gt_valid_mask = np.all(np.isfinite(filtered_gt_points), axis=1)
            if not np.all(gt_valid_mask):
                tqdm.write(f"  Scan {scan_idx}: Found {np.sum(~gt_valid_mask)} invalid GT points (NaN/Inf), filtering")
                filtered_gt_points = filtered_gt_points[gt_valid_mask]
                filtered_gt_labels = filtered_gt_labels[gt_valid_mask]
                if len(filtered_gt_points) == 0:
                    tqdm.write(f"  Scan {scan_idx}: No valid GT points after filtering, skipping")
                    failed_scans += 1
                    continue
            
            # Build KD-tree using scipy (more stable than Open3D)
            try:
                kdtree = cKDTree(filtered_gt_points)
            except Exception as e:
                tqdm.write(f"  Scan {scan_idx}: Error building KD-tree: {e}, skipping")
                failed_scans += 1
                continue
            
            # Validate points before searching
            valid_points_mask = np.all(np.isfinite(world_points_xyz), axis=1)
            if not np.all(valid_points_mask):
                tqdm.write(f"  Scan {scan_idx}: Found {np.sum(~valid_points_mask)} invalid points (NaN/Inf), filtering")
                world_points_xyz_valid = world_points_xyz[valid_points_mask]
                if len(world_points_xyz_valid) == 0:
                    tqdm.write(f"  Scan {scan_idx}: No valid points after filtering, skipping")
                    failed_scans += 1
                    continue
            else:
                world_points_xyz_valid = world_points_xyz
            
            # Find nearest neighbor for each bin scan point using scipy cKDTree
            # This is much more stable than Open3D's KDTreeFlann
            try:
                distances, indices = kdtree.query(world_points_xyz_valid, k=1)
                # Handle case where query returns scalar for single point
                if distances.ndim == 0:
                    distances = np.array([distances])
                    indices = np.array([indices])
                
                # Assign labels
                bin_labels_valid = filtered_gt_labels[indices].astype(np.int32)
            except Exception as e:
                tqdm.write(f"  Scan {scan_idx}: Error querying KD-tree: {e}, skipping")
                failed_scans += 1
                continue
            
            # Expand bin_labels back to original size if we filtered invalid points
            if not np.all(valid_points_mask):
                bin_labels = np.zeros(len(valid_points_mask), dtype=np.int32)
                bin_labels[valid_points_mask] = bin_labels_valid
            else:
                bin_labels = bin_labels_valid
            
            # Save labels to file
            bin_labels.tofile(output_path)
            
            successful_scans += 1
            
        except Exception as e:
            tqdm.write(f"  Scan {scan_idx}: Error - {e}")
            failed_scans += 1
            continue
    
    print(f"\n{'='*80}")
    print(f"Processing complete for {seq_name}!")
    print(f"  Successful: {successful_scans} scans")
    print(f"  Failed: {failed_scans} scans")
    print(f"  Skipped (already exist): {skipped_scans} scans")
    print(f"  Total: {len(valid_scan_indices)} scans")
    print(f"{'='*80}")
    
    return successful_scans, failed_scans


def process_multiple_sequences(dataset_path, seq_names, merged_npy_path=None, max_scans=None):
    """
    Process multiple sequences to extract semantic labels from GT point cloud.
    
    Args:
        dataset_path: Path to dataset directory containing sequence folders
        seq_names: List of sequence names to process
        merged_npy_path: Optional path to merged .npy file. If None, will look for merged file in dataset_path/ply/
        max_scans: Maximum number of scans to process per sequence (None for all)
    """
    print(f"\n{'='*80}")
    print(f"Processing {len(seq_names)} sequences to extract semantic labels")
    print(f"{'='*80}")
    print(f"Sequences: {seq_names}")
    if merged_npy_path:
        print(f"Merged .npy file: {merged_npy_path}")
    else:
        print(f"Will auto-detect merged .npy file from: {os.path.join(dataset_path, 'ply')}")
    print(f"{'='*80}\n")
    
    total_successful = 0
    total_failed = 0
    
    # Process each sequence
    for seq_idx, seq_name in enumerate(seq_names):
        print(f"\n{'='*80}")
        print(f"Processing sequence {seq_idx + 1}/{len(seq_names)}: {seq_name}")
        print(f"{'='*80}")
        
        root_path = os.path.join(dataset_path, seq_name)
        
        # Process this sequence
        successful, failed = process_all_scans(
            root_path,
            merged_npy_path=merged_npy_path,
            dataset_path=dataset_path,
            max_scans=max_scans,
            seq_name=seq_name
        )
        
        total_successful += successful
        total_failed += failed
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"ALL SEQUENCES PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total sequences processed: {len(seq_names)}")
    print(f"Total successful scans: {total_successful}")
    print(f"Total failed scans: {total_failed}")
    print(f"{'='*80}")


if __name__ == '__main__':
    # Configuration
    dataset_path = "/media/donceykong/doncey_ssd_02/datasets/MCD"
    
    # List of sequences to process
    seq_names = [
        "kth_day_10",
        "kth_day_09",
        "kth_day_06",
        "kth_night_05",
        "kth_night_04",
        "kth_night_01",
    ]
    
    # Explicitly specify merged .npy file path
    merged_npy_path = os.path.join(dataset_path, "ply", "merged_gt_labels_kth_day_06_kth_day_09_kth_night_05.npy")
    
    # Processing parameters
    max_scans = None  # Set to None to process all scans, or specify a number to limit per sequence
    
    # Process all sequences
    process_multiple_sequences(
        dataset_path=dataset_path,
        seq_names=seq_names,
        merged_npy_path=merged_npy_path,
        max_scans=max_scans
    )
    
    # Or process a single sequence:
    # root_path = os.path.join(dataset_path, "kth_night_05")
    # process_all_scans(root_path, merged_npy_path=merged_npy_path, dataset_path=dataset_path, max_scans=max_scans)
    
    # Or process a single scan for visualization:
    # root_path = os.path.join(dataset_path, "kth_night_05")
    # scan_idx = 11  # Index of the scan to visualize (0 = first scan)
    # plot_bin_on_gt_ply(root_path, scan_idx=scan_idx, merged_npy_path=merged_npy_path, dataset_path=dataset_path)

