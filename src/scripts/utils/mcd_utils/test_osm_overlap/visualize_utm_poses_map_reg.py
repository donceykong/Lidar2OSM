#!/usr/bin/env python3
"""
Visualize point cloud map built from world-coordinate poses.

This script loads poses from a CSV file (in world coordinates), loads point clouds and gt_labels
from .bin files, transforms them to world coordinates, and visualizes the
accumulated map with semantic colors after voxel downsampling.
The poses are projected to lat/lon using mercator projection with initial_latlon as origin.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import utilities
from lidar2osm.utils.file_io import read_bin_file

# Import projection utilities
from lidar2osm.core.projection.utils import latlon_to_mercator, mercator_to_latlon, lat_to_scale

# Import semantic labels and color conversion
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from create_seq_gt_map_npy import sem_kitti_labels
from create_seq_gt_map_npy import semantic_labels as sem_kitti_labels
from lidar2osm.core.pointcloud.pointcloud import labels2RGB_tqdm

# Import OSM visualization functions
sys.path.append(os.path.dirname(__file__))
from view_and_select_osm_data import (
    get_osm_buildings_geometries,
    get_osm_road_geometries,
    get_osm_trees_geometries,
    get_osm_grassland_geometries,
    get_osm_water_geometries,
    on_click
)

# Initial position and latlon for kth_day_06
initial_latlon = [59.348268650, 18.073204280]
initial_position = np.array([64.3932532565158, 66.4832330946657, 38.5143341050069])
# [59.34826865, 18.07320428] for kth_day_09

# Body to LiDAR transformation matrix
BODY_TO_LIDAR_TF = np.array([
    [0.9999135040741837, -0.011166365511073898, -0.006949579221822984, -0.04894521120494695],
    [-0.011356389542502144, -0.9995453006865824, -0.02793249526856565, -0.03126929060348084],
    [-0.006634514801117132, 0.02800900135032654, -0.999585653686922, -0.01755515794222565],
    [0.0, 0.0, 0.0, 1.0]
])


def load_poses(pose_csv_file):
    """
    Load poses from CSV file (in world coordinates).
    
    Args:
        pose_csv_file: Path to CSV file with columns [num, t, x, y, z, qx, qy, qz, qw]
                     where x, y, z are world coordinates
    
    Returns:
        poses_dict: Dictionary mapping 'num' (int) to pose data with 'position' and 'quaternion'
        poses_list: List of poses in order (for backward compatibility)
    """
    print(f"\nLoading poses from {pose_csv_file}")
    
    try:
        df = pd.read_csv(pose_csv_file, comment='#')
        print(f"Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {}, []
    
    if len(df) == 0:
        print("Error: CSV file is empty")
        return {}, []
    
    poses_dict = {}
    poses_list = []
    
    for idx, row in df.iterrows():
        try:
            # Get 'num' field
            if 'num' in df.columns:
                num = int(row['num'])
            else:
                # Try first column as num
                num = int(row.iloc[0])
            
            # Check if required columns exist
            required_cols = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
            if all(col in df.columns for col in required_cols):
                x = float(row['x'])
                y = float(row['y'])
                z = float(row['z'])
                qx = float(row['qx'])
                qy = float(row['qy'])
                qz = float(row['qz'])
                qw = float(row['qw'])
            else:
                # Try positional access (skip num and t columns)
                if len(row) >= 8:
                    x = float(row.iloc[2])  # Skip num and t
                    y = float(row.iloc[3])
                    z = float(row.iloc[4])
                    qx = float(row.iloc[5])
                    qy = float(row.iloc[6])
                    qz = float(row.iloc[7])
                    qw = float(row.iloc[8]) if len(row) > 8 else 1.0
                elif len(row) >= 7:
                    x = float(row.iloc[1])  # Skip num
                    y = float(row.iloc[2])
                    z = float(row.iloc[3])
                    qx = float(row.iloc[4])
                    qy = float(row.iloc[5])
                    qz = float(row.iloc[6])
                    qw = float(row.iloc[7]) if len(row) > 7 else 1.0
                else:
                    print(f"Warning: Row {idx} doesn't have enough columns. Skipping.")
                    continue
            
            position = np.array([x, y, z])
            quaternion = np.array([qx, qy, qz, qw])
            
            pose_data = {'position': position, 'quaternion': quaternion}
            poses_dict[num] = pose_data
            poses_list.append(pose_data)
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    print(f"Loaded {len(poses_dict)} poses (num range: {min(poses_dict.keys()) if poses_dict else 'N/A'} to {max(poses_dict.keys()) if poses_dict else 'N/A'})")
    return poses_dict, poses_list


def transform_points_to_world(points_xyz, position, quaternion, body_to_lidar_tf=None):
    """
    Transform points from lidar frame to world frame using pose.
    
    Args:
        points_xyz: (N, 3) array of points in lidar frame
        position: [x, y, z] translation (body frame position in world/UTM)
        quaternion: [qx, qy, qz, qw] rotation quaternion (body frame orientation)
        body_to_lidar_tf: Optional 4x4 transformation matrix from body to lidar frame
    
    Returns:
        world_points: (N, 3) array of points in world/UTM frame
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


def extract_bin_number(bin_file_path):
    """
    Extract the number from a bin file name.
    
    Args:
        bin_file_path: Path to bin file (e.g., "0000000001.bin" or "1.bin")
    
    Returns:
        num: Integer number from the filename, or None if extraction fails
    """
    try:
        # Get filename without extension
        stem = Path(bin_file_path).stem
        # Remove leading zeros and convert to int
        num = int(stem)
        return num
    except (ValueError, AttributeError):
        return None


def build_map_from_poses(sequence_path, pose_csv, 
                         per_scan_voxel_size=4.0, 
                         global_voxel_size=0.5,
                         max_scans=None,
                         frame_skip=1):
    """
    Build a semantic map from point clouds using world-coordinate poses.
    
    Args:
        sequence_path: Path to sequence directory (e.g., /path/to/kth_day_06)
        pose_csv: Path to CSV file with poses in world coordinates
        per_scan_voxel_size: Voxel size for downsampling each scan (meters)
        global_voxel_size: Voxel size for final global map downsampling (meters)
        max_scans: Maximum number of scans to process (None for all)
        frame_skip: Process every Nth frame (default: 1, process all)
    
    Returns:
        points: (N, 3) array of accumulated points in world coordinates
        labels: (N,) array of semantic labels
        initial_pose: Dictionary with 'position' and 'quaternion' of initial pose
        calculated_initial_latlon: [lat, lon] calculated for initial_position (or provided initial_latlon if no offset)
    """
    sequence_path = Path(sequence_path)
    
    # Paths to data directories
    bin_data_dir = sequence_path / "lidar_bin" / "data"
    gt_labels_dir = sequence_path / "gt_labels"
    
    if not bin_data_dir.exists():
        print(f"Error: Bin data directory not found: {bin_data_dir}")
        return None, None, None, None
    
    if not gt_labels_dir.exists():
        print(f"Error: GT labels directory not found: {gt_labels_dir}")
        return None, None, None, None
    
    # Load poses - get dictionary mapping num to pose
    poses_dict, poses_list = load_poses(pose_csv)
    if len(poses_dict) == 0:
        print("Error: No poses loaded!")
        return None, None, None, None
    
    # Get initial pose (pose with smallest num value)
    initial_pose = None
    calculated_initial_latlon = None
    
    if poses_dict:
        initial_num = min(poses_dict.keys())
        initial_pose = poses_dict[initial_num]
        first_pose_position = initial_pose['position'].copy()
        
        print(f"\nInitial pose (num={initial_num}):")
        print(f"  Position (world): [{first_pose_position[0]:.2f}, {first_pose_position[1]:.2f}, {first_pose_position[2]:.2f}]")
        print(f"  Reference initial_position (kth_day_06): [{initial_position[0]:.2f}, {initial_position[1]:.2f}, {initial_position[2]:.2f}]")
        print(f"  Reference initial_latlon (kth_day_06): [{initial_latlon[0]:.8f}, {initial_latlon[1]:.8f}]")
        
        # Calculate offset from reference initial_position to first pose position
        # Since all sequences are in the same coordinate frame, we can use this offset
        position_offset = first_pose_position - initial_position
        
        print(f"\nCalculating initial_latlon for first pose using reference (kth_day_06)...")
        print(f"  Offset from reference: [{position_offset[0]:.2f}, {position_offset[1]:.2f}, {position_offset[2]:.2f}]")
        
        # Compute mercator scale from the reference initial_latlon
        scale = lat_to_scale(initial_latlon[0])
        
        # Convert reference initial_latlon to mercator coordinates
        ref_ox_merc, ref_oy_merc = latlon_to_mercator(initial_latlon[0], initial_latlon[1], scale)
        
        # Apply offset in mercator space (x=east, y=north)
        # The offset is in world coordinates (meters), which directly map to mercator coordinates
        first_pose_merc_x = ref_ox_merc + position_offset[0]  # x is east (longitude/mercator x)
        first_pose_merc_y = ref_oy_merc + position_offset[1]  # y is north (latitude/mercator y)
        
        # Convert mercator coordinates back to lat/lon
        calculated_lat, calculated_lon = mercator_to_latlon(first_pose_merc_x, first_pose_merc_y, scale)
        calculated_initial_latlon = [calculated_lat, calculated_lon]
        
        print(f"  Calculated initial_latlon for first pose: [{calculated_lat:.8f}, {calculated_lon:.8f}]")
    
    # Get all bin files
    all_bin_files = sorted([f for f in bin_data_dir.glob("*.bin")])
    if len(all_bin_files) == 0:
        print(f"Error: No .bin files found in {bin_data_dir}")
        return None, None, None, None
    
    print(f"\nFound {len(all_bin_files)} bin files")
    
    # Extract numbers from bin files and filter to only those with associated poses
    bin_files_with_nums = []
    missing_pose_count = 0
    invalid_name_count = 0
    
    for bin_file in all_bin_files:
        num = extract_bin_number(bin_file)
        if num is None:
            invalid_name_count += 1
            continue
        
        # Only include bin files that have associated poses in the CSV
        if num in poses_dict:
            bin_files_with_nums.append((num, bin_file))
        else:
            missing_pose_count += 1
    
    # Sort by num
    bin_files_with_nums.sort(key=lambda x: x[0])
    
    print(f"\nBin file filtering summary:")
    print(f"  Total bin files: {len(all_bin_files)}")
    print(f"  With valid names and associated poses: {len(bin_files_with_nums)}")
    print(f"  Missing poses (skipped): {missing_pose_count}")
    print(f"  Invalid names (skipped): {invalid_name_count}")
    
    if bin_files_with_nums:
        print(f"  Num range: {bin_files_with_nums[0][0]} to {bin_files_with_nums[-1][0]}")
    else:
        print("Error: No bin files have associated poses!")
        return None, None, None, None
    
    # Limit number of scans if specified (only among those with poses)
    if max_scans is not None:
        bin_files_with_nums = bin_files_with_nums[:max_scans]
    
    # Apply frame skip - only to scans that have associated poses
    bin_files_filtered = [(num, f) for idx, (num, f) in enumerate(bin_files_with_nums) if idx % frame_skip == 0]
    
    print(f"\nProcessing {len(bin_files_filtered)} scans (frame_skip={frame_skip}, from {len(bin_files_with_nums)} scans with poses)")
    print("Note: Only processing scans with associated poses - no interpolation performed.")
    
    # Accumulate points and labels
    all_points = []
    all_labels = []
    processed_count = 0
    
    for num, bin_file in tqdm(bin_files_filtered, desc="Processing scans"):
        # Get pose for this scan - we know it exists since we filtered above
        pose = poses_dict[num]
        processed_count += 1
        
        try:
            # Load point cloud
            points = read_bin_file(str(bin_file), dtype=np.float32, shape=(-1, 4))
            points_xyz = points[:, :3]
            intensities = points[:, 3]
            
            # Load semantic labels
            # Bin files are named 0000000001.bin, 0000000002.bin, etc.
            # Labels are named the same way
            label_file = gt_labels_dir / bin_file.name
            
            if not label_file.exists():
                print(f"Warning: Label file not found: {label_file}. Skipping scan.")
                continue
            
            labels = read_bin_file(str(label_file), dtype=np.int32)
            
            # Ensure same length
            if len(labels) != len(points_xyz):
                min_length = min(len(labels), len(points_xyz))
                labels = labels[:min_length]
                points_xyz = points_xyz[:min_length]
                intensities = intensities[:min_length]
            
            # Filter points by distance (60m threshold) in LiDAR frame before transformation
            # Calculate distance from origin (0,0,0) in LiDAR frame
            distances = np.linalg.norm(points_xyz, axis=1)
            distance_threshold = 50.0  # meters
            
            # Create mask for points within threshold
            valid_mask = distances < distance_threshold
            
            # Apply filter to points, intensities, and labels
            points_xyz = points_xyz[valid_mask]
            intensities = intensities[valid_mask]
            labels = labels[valid_mask]
            
            if len(points_xyz) == 0:
                print(f"Warning: All points filtered out for scan num={num} (distance >= {distance_threshold}m). Skipping.")
                continue
            
            position = pose['position']
            quaternion = pose['quaternion']
            
            # Transform points to world coordinates
            world_points = transform_points_to_world(
                points_xyz, position, quaternion, BODY_TO_LIDAR_TF
            )
            
            # Per-scan voxel downsampling
            if per_scan_voxel_size > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(world_points)
                pcd = pcd.voxel_down_sample(voxel_size=per_scan_voxel_size)
                downsampled_points = np.asarray(pcd.points)
                
                # Find corresponding labels using KDTree
                from scipy.spatial import cKDTree
                tree = cKDTree(world_points)
                _, indices = tree.query(downsampled_points, k=1)
                downsampled_labels = labels[indices]
            else:
                downsampled_points = world_points
                downsampled_labels = labels
            
            # Accumulate
            all_points.append(downsampled_points)
            all_labels.append(downsampled_labels)
            
        except Exception as e:
            print(f"Error processing scan num={num} ({bin_file.name}): {e}")
            continue
    
    if len(all_points) == 0:
        print("Error: No points accumulated!")
        return None, None, None, None
    
    # Combine all points and labels
    print(f"\nSuccessfully processed {processed_count} scans with associated poses")
    print(f"Combining {len(all_points)} scans...")
    combined_points = np.vstack(all_points)
    combined_labels = np.concatenate(all_labels)
    
    print(f"  Total points before global downsampling: {len(combined_points)}")
    
    # Global voxel downsampling
    if global_voxel_size > 0:
        print(f"Applying global voxel downsampling (voxel_size={global_voxel_size:.2f}m)...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        pcd = pcd.voxel_down_sample(voxel_size=global_voxel_size)
        final_points = np.asarray(pcd.points)
        
        # Find corresponding labels
        from scipy.spatial import cKDTree
        tree = cKDTree(combined_points)
        _, indices = tree.query(final_points, k=1)
        final_labels = combined_labels[indices]
        
        print(f"  Total points after global downsampling: {len(final_points)}")
    else:
        final_points = combined_points
        final_labels = combined_labels
    
    return final_points, final_labels, initial_pose, calculated_initial_latlon


def project_points_to_latlon(points, initial_position, origin_latlon):
    """
    Project points from world coordinates to lat-long coordinates using mercator projection.
    
    Args:
        points: (N, 3) array of points in world coordinates
        initial_position: (3,) array of initial position in world coordinates (origin)
        origin_latlon: [lat, lon] of the initial position (origin)
    
    Returns:
        points_latlon: Array of (N, 2) with [lat, lon] for each point
    """
    if len(points) == 0:
        return np.array([])
    
    # Compute mercator scale from latitude
    scale = lat_to_scale(origin_latlon[0])
    
    # Convert origin lat-lon to mercator coordinates
    ox_merc, oy_merc = latlon_to_mercator(origin_latlon[0], origin_latlon[1], scale)
    
    print(f"\nProjecting {len(points)} points to lat-long...")
    print(f"  Initial position (world): [{initial_position[0]:.2f}, {initial_position[1]:.2f}, {initial_position[2]:.2f}]")
    print(f"  Origin (lat-lon): [{origin_latlon[0]:.8f}, {origin_latlon[1]:.8f}]")
    print(f"  Mercator scale: {scale:.6f}")
    
    points_latlon = []
    for i, point in enumerate(points):
        # Compute position relative to initial position
        relative_position = point - initial_position
        
        # Convert relative position (in meters) to mercator coordinates
        # In mercator: x = longitude direction (east), y = latitude direction (north)
        # Assuming world coordinates: x = east, y = north
        merc_x = ox_merc + relative_position[0]  # x is east (longitude/mercator x)
        merc_y = oy_merc + relative_position[1]  # y is north (latitude/mercator y)
        
        # Convert mercator coordinates to lat-lon
        lat, lon = mercator_to_latlon(merc_x, merc_y, scale)
        points_latlon.append([lat, lon])
    
    points_latlon = np.array(points_latlon)
    
    print(f"  Points bounds:")
    print(f"    Latitude: [{points_latlon[:, 0].min():.8f}, {points_latlon[:, 0].max():.8f}]")
    print(f"    Longitude: [{points_latlon[:, 1].min():.8f}, {points_latlon[:, 1].max():.8f}]")
    
    return points_latlon


def visualize_semantic_map(points, labels, show_osm=False, osm_file_path=None, initial_pose=None, initial_latlon=None):
    """
    Visualize semantic point cloud map with Open3D or over OSM.
    
    Args:
        points: (N, 3) array of points in world coordinates
        labels: (N,) array of semantic labels
        show_osm: If True, plot over OSM data (2D matplotlib). If False, use Open3D (3D).
        osm_file_path: Path to OSM file (required if show_osm=True)
        initial_pose: Dictionary with 'position' (world) and 'quaternion' for initial pose
        initial_latlon: [lat, lon] of the initial position (origin) for projection
    """
    if points is None or labels is None:
        print("Error: No points or labels to visualize")
        return
    
    print(f"\nVisualizing semantic map with {len(points)} points...")
    
    # Convert labels to RGB colors
    print("Converting semantic labels to RGB colors...")
    labels_dict = {label.id: label.color for label in sem_kitti_labels}
    colors_rgb = labels2RGB_tqdm(labels, labels_dict)
    
    print(f"  Point cloud bounds (world):")
    print(f"    X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"    Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"    Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    print(f"  Unique labels: {np.unique(labels)}")
    
    if show_osm:
        if osm_file_path is None:
            print("Error: OSM file path required when show_osm=True")
            return
        
        if initial_pose is None:
            print("Error: Initial pose required for projection to lat/lon")
            return
        
        if initial_latlon is None:
            print("Error: initial_latlon required for projection to lat/lon")
            return
        
        # Get initial position from pose
        initial_position = initial_pose['position']
        
        # Project world coordinates to lat-lon using mercator projection
        points_latlon = project_points_to_latlon(points, initial_position, initial_latlon)
        
        # Visualize over OSM
        visualize_map_over_osm(osm_file_path, points_latlon, colors_rgb, labels)
    else:
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
        
        # Visualize
        print("\nOpening visualization window...")
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="Semantic Map from UTM Poses",
            width=1200,
            height=800,
        )


def visualize_map_over_osm(osm_file_path, points_latlon, colors_rgb, labels):
    """
    Visualize point cloud map overlaid on OSM data (2D projection).
    
    Args:
        osm_file_path: Path to OSM XML file
        points_latlon: (N, 2) array of points in [lat, lon] format (2D projection)
        colors_rgb: (N, 3) array of RGB colors in [0, 1] range
        labels: (N,) array of semantic labels
    """
    osm_file_path = Path(osm_file_path)
    
    if not osm_file_path.exists():
        print(f"Error: OSM file not found: {osm_file_path}")
        return
    
    print(f"\nLoading OSM data from: {osm_file_path}")
    print("This may take a moment...")
    
    # Load different OSM features
    all_lons = []
    all_lats = []
    
    # Load and visualize buildings (blue)
    print("Loading buildings...")
    building_polygons = get_osm_buildings_geometries(osm_file_path)
    print(f"  Found {len(building_polygons)} buildings")
    for poly in building_polygons:
        lons, lats = zip(*poly)
        all_lons.extend(lons)
        all_lats.extend(lats)
    
    # Load and visualize roads (gray)
    print("Loading roads...")
    road_lines = get_osm_road_geometries(osm_file_path)
    print(f"  Found {len(road_lines)} road segments")
    for line in road_lines:
        lons, lats = zip(*line)
        all_lons.extend(lons)
        all_lats.extend(lats)
    
    # Load and visualize trees (green)
    print("Loading trees...")
    tree_polygons = get_osm_trees_geometries(osm_file_path)
    print(f"  Found {len(tree_polygons)} trees")
    for poly in tree_polygons:
        lons, lats = zip(*poly)
        all_lons.extend(lons)
        all_lats.extend(lats)
    
    # Load and visualize grassland (light green)
    print("Loading grassland/parks...")
    grass_polygons = get_osm_grassland_geometries(osm_file_path)
    print(f"  Found {len(grass_polygons)} grassland/parks")
    for poly in grass_polygons:
        lons, lats = zip(*poly)
        all_lons.extend(lons)
        all_lats.extend(lats)
    
    # Load and visualize water (cyan)
    print("Loading water features...")
    water_polygons = get_osm_water_geometries(osm_file_path)
    print(f"  Found {len(water_polygons)} water features")
    for poly in water_polygons:
        lons, lats = zip(*poly)
        all_lons.extend(lons)
        all_lats.extend(lats)
    
    # Include point cloud coordinates in bounds calculation
    if len(points_latlon) > 0:
        all_lons.extend(points_latlon[:, 1].tolist())
        all_lats.extend(points_latlon[:, 0].tolist())
    
    if len(all_lons) == 0:
        print("No OSM features found to visualize.")
        return
    
    # Calculate plot bounds
    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)
    lon_margin = (max_lon - min_lon) * 0.05
    lat_margin = (max_lat - min_lat) * 0.05
    
    print(f"\nGeographic bounds:")
    print(f"  Longitude: [{min_lon:.6f}, {max_lon:.6f}]")
    print(f"  Latitude: [{min_lat:.6f}, {max_lat:.6f}]")
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Point Cloud Map over OSM Data - Click anywhere to get lat-long coordinates', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Plot buildings (blue)
    for poly_coords in building_polygons:
        lons, lats = zip(*poly_coords)
        polygon = Polygon(list(zip(lons, lats)), closed=True,
                         facecolor='blue', edgecolor='darkblue',
                         alpha=0.5, linewidth=0.5)
        ax.add_patch(polygon)
    
    # Plot roads (gray)
    for line_coords in road_lines:
        lons, lats = zip(*line_coords)
        ax.plot(lons, lats, color='gray', linewidth=0.8, alpha=0.7)
    
    # Plot trees (green)
    for poly_coords in tree_polygons:
        lons, lats = zip(*poly_coords)
        polygon = Polygon(list(zip(lons, lats)), closed=True,
                         facecolor='green', edgecolor='darkgreen',
                         alpha=0.5, linewidth=0.5)
        ax.add_patch(polygon)
    
    # Plot grassland (light green)
    for poly_coords in grass_polygons:
        lons, lats = zip(*poly_coords)
        polygon = Polygon(list(zip(lons, lats)), closed=True,
                         facecolor='lightgreen', edgecolor='green',
                         alpha=0.5, linewidth=0.5)
        ax.add_patch(polygon)
    
    # Plot water (cyan)
    for poly_coords in water_polygons:
        lons, lats = zip(*poly_coords)
        polygon = Polygon(list(zip(lons, lats)), closed=True,
                         facecolor='cyan', edgecolor='blue',
                         alpha=0.5, linewidth=0.5)
        ax.add_patch(polygon)
    
    # Plot point cloud (scatter plot with semantic colors) - 2D projection
    print(f"\nPlotting point cloud with {len(points_latlon)} points (2D projection)...")
    lons = points_latlon[:, 1]
    lats = points_latlon[:, 0]
    
    # Use semantic colors for points
    ax.scatter(lons, lats, c=colors_rgb, s=0.5, alpha=0.6, 
              label='Point Cloud (Semantic)', zorder=5)
    
    # Set plot limits
    ax.set_xlim(min_lon - lon_margin, max_lon + lon_margin)
    ax.set_ylim(min_lat - lat_margin, max_lat + lat_margin)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.5, label='Buildings'),
        Patch(facecolor='gray', alpha=0.7, label='Roads'),
        Patch(facecolor='green', alpha=0.5, label='Trees'),
        Patch(facecolor='lightgreen', alpha=0.5, label='Grassland/Parks'),
        Patch(facecolor='cyan', alpha=0.5, label='Water'),
        plt.Line2D([0], [0], marker='o', color='orange', 
                  markersize=5, linestyle='None', label='Point Cloud (Semantic)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Connect click event handler
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    print("\n" + "="*60)
    print("Interactive mode enabled!")
    print("Click anywhere on the plot to get lat-long coordinates")
    print("Close the window to exit")
    print("="*60)
    
    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # File paths
    sequence = "kth_night_04"
    sequence_path = f"/media/donceykong/doncey_ssd_02/datasets/MCD/{sequence}"
    pose_csv = f"/media/donceykong/doncey_ssd_02/datasets/MCD/{sequence}/pose_inW.csv"  # Regular poses, not UTM
    osm_file_path = "/media/donceykong/doncey_ssd_02/datasets/MCD/kth.osm"
    
    # Parameters
    per_scan_voxel_size = 4.0  # Voxel size for each scan (meters)
    global_voxel_size = 0      # Voxel size for final map (meters)
    max_scans = None           # None to process all scans
    frame_skip = 10           # Process every Nth frame (1 = all frames)
    
    # Visualization options
    show_osm = True             # If True, plot over OSM. If False, use Open3D 3D visualization
    
    # Build map from world-coordinate poses
    points, labels, initial_pose, calculated_initial_latlon = build_map_from_poses(
        sequence_path, 
        pose_csv,
        per_scan_voxel_size=per_scan_voxel_size,
        global_voxel_size=global_voxel_size,
        max_scans=max_scans,
        frame_skip=frame_skip
    )
    
    # Visualize
    if points is not None and labels is not None:
        # Use calculated initial_latlon if available, otherwise use provided one
        latlon_to_use = calculated_initial_latlon if calculated_initial_latlon is not None else initial_latlon
        visualize_semantic_map(points, labels, show_osm=show_osm, osm_file_path=osm_file_path,
                              initial_pose=initial_pose, initial_latlon=latlon_to_use)
    else:
        print("Error: Failed to build map!")

