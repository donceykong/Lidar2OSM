#!/usr/bin/env python3
"""
Visualize point cloud map built from UTM-based poses.

This script loads UTM poses from a CSV file, loads point clouds and gt_labels
from .bin files, transforms them to world coordinates, and visualizes the
accumulated map with semantic colors after voxel downsampling.
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
from pyproj import Transformer

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import utilities
from lidar2osm.utils.file_io import read_bin_file

# Import semantic labels and color conversion
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from create_seq_gt_map_npy import sem_kitti_labels # semantic_labels as sem_kitti_labels
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

# Body to LiDAR transformation matrix
BODY_TO_LIDAR_TF = np.array([
    [0.9999135040741837, -0.011166365511073898, -0.006949579221822984, -0.04894521120494695],
    [-0.011356389542502144, -0.9995453006865824, -0.02793249526856565, -0.03126929060348084],
    [-0.006634514801117132, 0.02800900135032654, -0.999585653686922, -0.01755515794222565],
    [0.0, 0.0, 0.0, 1.0]
])


def load_utm_poses(pose_csv_file):
    """
    Load UTM poses from CSV file.
    
    Args:
        pose_csv_file: Path to CSV file with columns [num, t, x, y, z, qx, qy, qz, qw]
                     where x, y are UTM coordinates
    
    Returns:
        poses_dict: Dictionary mapping 'num' (int) to pose data with 'position' and 'quaternion'
        poses_list: List of poses in order (for backward compatibility)
    """
    print(f"\nLoading UTM poses from {pose_csv_file}")
    
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


def build_map_from_utm_poses(sequence_path, utm_pose_csv, 
                             per_scan_voxel_size=4.0, 
                             global_voxel_size=0.5,
                             max_scans=None,
                             frame_skip=1):
    """
    Build a semantic map from point clouds using UTM-based poses.
    
    Args:
        sequence_path: Path to sequence directory (e.g., /path/to/kth_day_06)
        utm_pose_csv: Path to CSV file with UTM poses
        per_scan_voxel_size: Voxel size for downsampling each scan (meters)
        global_voxel_size: Voxel size for final global map downsampling (meters)
        max_scans: Maximum number of scans to process (None for all)
        frame_skip: Process every Nth frame (default: 1, process all)
    
    Returns:
        points: (N, 3) array of accumulated points in UTM coordinates
        labels: (N,) array of semantic labels
    """
    sequence_path = Path(sequence_path)
    
    # Paths to data directories
    bin_data_dir = sequence_path / "lidar_bin" / "data"
    gt_labels_dir = sequence_path / "cumulti_osm_style_inferred_labels"
    
    if not bin_data_dir.exists():
        print(f"Error: Bin data directory not found: {bin_data_dir}")
        return None, None, None
    
    if not gt_labels_dir.exists():
        print(f"Error: GT labels directory not found: {gt_labels_dir}")
        return None, None, None
    
    # Load UTM poses - get dictionary mapping num to pose
    poses_dict, poses_list = load_utm_poses(utm_pose_csv)
    if len(poses_dict) == 0:
        print("Error: No poses loaded!")
        return None, None, None
    
    # Get all bin files
    all_bin_files = sorted([f for f in bin_data_dir.glob("*.bin")])
    if len(all_bin_files) == 0:
        print(f"Error: No .bin files found in {bin_data_dir}")
        return None, None, None
    
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
        return None, None, None
    
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
            position = pose['position']
            quaternion = pose['quaternion']
            
            # Transform points to world/UTM coordinates
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
        return None, None, None
    
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
    
    # Get initial pose (pose with smallest num value)
    initial_pose = None
    if poses_dict:
        initial_num = min(poses_dict.keys())
        initial_pose = poses_dict[initial_num]
        print(f"\nInitial pose (num={initial_num}):")
        print(f"  Position (UTM): [{initial_pose['position'][0]:.2f}, {initial_pose['position'][1]:.2f}, {initial_pose['position'][2]:.2f}]")
    
    return final_points, final_labels, initial_pose


def utm_to_latlon(points_utm, utm_zone=34, northern_hemisphere=True, project_2d=False):
    """
    Convert UTM point coordinates to lat/lon.
    
    Args:
        points_utm: (N, 3) array of UTM coordinates [x, y, z]
        utm_zone: UTM zone number (int, e.g., 34) or EPSG code (str, e.g., "EPSG:32634")
                  If integer, will construct EPSG code based on northern_hemisphere.
                  If string, will use directly (northern_hemisphere is ignored).
        northern_hemisphere: Whether in northern hemisphere (default True, only used if utm_zone is int)
        project_2d: If True, return only lat/lon (2D). If False, return [lat, lon, z] (3D).
    
    Returns:
        points_latlon: (N, 2) or (N, 3) array of [lat, lon] or [lat, lon, z]
    """
    print(f"\nConverting {len(points_utm)} points from UTM to lat/lon...")
    
    # Determine source CRS
    if isinstance(utm_zone, str):
        # Direct EPSG code provided
        source_crs = utm_zone
        print(f"  Using EPSG code: {source_crs}")
    elif isinstance(utm_zone, int):
        # Construct EPSG code from zone number
        source_crs = f"EPSG:326{utm_zone:02d}" if northern_hemisphere else f"EPSG:327{utm_zone:02d}"
        print(f"  Using UTM zone {utm_zone} ({'northern' if northern_hemisphere else 'southern'} hemisphere)")
        print(f"  EPSG code: {source_crs}")
    else:
        raise ValueError(f"utm_zone must be int or str, got {type(utm_zone)}: {utm_zone}")
    
    transformer = Transformer.from_crs(
        source_crs,
        "EPSG:4326", 
        always_xy=True
    )
    
    lons, lats = transformer.transform(points_utm[:, 0], points_utm[:, 1])
    
    if project_2d:
        # Project to 2D: only lat/lon
        points_latlon = np.column_stack([lats, lons])
        print(f"  Latitude range: [{lats.min():.6f}, {lats.max():.6f}]")
        print(f"  Longitude range: [{lons.min():.6f}, {lons.max():.6f}]")
        print("  Projected to 2D (Z coordinate discarded)")
    else:
        # Keep 3D: lat/lon/z
        points_latlon = np.column_stack([lats, lons, points_utm[:, 2]])
        print(f"  Latitude range: [{lats.min():.6f}, {lats.max():.6f}]")
        print(f"  Longitude range: [{lons.min():.6f}, {lons.max():.6f}]")
        print(f"  Z range: [{points_utm[:, 2].min():.2f}, {points_utm[:, 2].max():.2f}]")
    
    return points_latlon


def visualize_semantic_map(points, labels, show_osm=False, osm_file_path=None, utm_zone="EPSG:32634", initial_pose=None):
    """
    Visualize semantic point cloud map with Open3D or over OSM.
    
    Args:
        points: (N, 3) array of points in UTM coordinates
        labels: (N,) array of semantic labels
        show_osm: If True, plot over OSM data (2D matplotlib). If False, use Open3D (3D).
        osm_file_path: Path to OSM file (required if show_osm=True)
        utm_zone: UTM zone (int or EPSG string) for conversion
        initial_pose: Dictionary with 'position' (UTM) and 'quaternion' for initial pose (for rotation center)
    """
    if points is None or labels is None:
        print("Error: No points or labels to visualize")
        return
    
    print(f"\nVisualizing semantic map with {len(points)} points...")
    
    # Convert labels to RGB colors
    print("Converting semantic labels to RGB colors...")
    labels_dict = {label.id: label.color for label in sem_kitti_labels}
    colors_rgb = labels2RGB_tqdm(labels, labels_dict)
    
    print(f"  Point cloud bounds (UTM):")
    print(f"    X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"    Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"    Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    print(f"  Unique labels: {np.unique(labels)}")
    
    if show_osm:
        if osm_file_path is None:
            print("Error: OSM file path required when show_osm=True")
            return
        
        # Keep 3D UTM points for transformations (don't project to 2D yet)
        original_points_utm = points.copy()
        
        # Convert initial pose to lat/lon for rotation center
        initial_pose_latlon = None
        if initial_pose is not None:
            initial_pose_utm = initial_pose['position'].reshape(1, 3)
            initial_pose_latlon_3d = utm_to_latlon(initial_pose_utm, utm_zone=utm_zone, project_2d=False)
            initial_pose_latlon = initial_pose_latlon_3d[0, :2]  # Extract lat, lon
            print(f"\nInitial pose for rotation center:")
            print(f"  UTM: [{initial_pose['position'][0]:.2f}, {initial_pose['position'][1]:.2f}, {initial_pose['position'][2]:.2f}]")
            print(f"  Lat/Lon: [{initial_pose_latlon[0]:.8f}, {initial_pose_latlon[1]:.8f}]")
        
        # Convert UTM to lat-lon for visualization (but keep 3D UTM for transformations)
        points_latlon = utm_to_latlon(points, utm_zone=utm_zone, project_2d=True)
        
        # Visualize over OSM
        visualize_map_over_osm(osm_file_path, points_latlon, colors_rgb, labels, 
                               original_points_utm=original_points_utm, utm_zone=utm_zone,
                               initial_pose_latlon=initial_pose_latlon)
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


def rotate_points_2d(points, center, angle_rad):
    """
    Rotate 2D points around a center point.
    
    Args:
        points: (N, 2) array of points [lat, lon] or [x, y]
        center: (2,) array [center_lat, center_lon] or [center_x, center_y]
        angle_rad: Rotation angle in radians (positive = counterclockwise)
    
    Returns:
        rotated_points: (N, 2) array of rotated points
    """
    # Translate to origin
    translated = points - center
    
    # Rotation matrix for 2D
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a],
                                [sin_a, cos_a]])
    
    # Rotate
    rotated = translated @ rotation_matrix.T
    
    # Translate back
    rotated_points = rotated + center
    
    return rotated_points


def visualize_map_over_osm(osm_file_path, points_latlon, colors_rgb, labels, 
                           original_points_utm=None, utm_zone="EPSG:32634", initial_pose_latlon=None):
    """
    Visualize point cloud map overlaid on OSM data (2D projection) with interactive controls.
    
    Args:
        osm_file_path: Path to OSM XML file
        points_latlon: (N, 2) array of points in [lat, lon] format (2D projection)
        colors_rgb: (N, 3) array of RGB colors in [0, 1] range
        labels: (N,) array of semantic labels
        original_points_utm: (N, 3) array of original UTM points (for transformations and saving)
        utm_zone: UTM zone (int or EPSG string) for conversion back to UTM
        initial_pose_latlon: (2,) array [lat, lon] of initial pose position (for rotation center)
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
    
    # Store original points for transformations
    original_points_latlon = points_latlon.copy()
    
    # Use initial pose position as rotation center, or fall back to average if not available
    if initial_pose_latlon is not None:
        rotation_center = initial_pose_latlon.copy()
        print(f"\nUsing initial pose position as rotation center: [{rotation_center[0]:.8f}, {rotation_center[1]:.8f}]")
    else:
        # Fall back to center of point cloud
        center_lat = np.mean(points_latlon[:, 0])
        center_lon = np.mean(points_latlon[:, 1])
        rotation_center = np.array([center_lat, center_lon])
        print(f"\nUsing point cloud center as rotation center: [{rotation_center[0]:.8f}, {rotation_center[1]:.8f}]")
    
    # Transformation state
    lat_offset = 0.0
    lon_offset = 0.0
    rotation_angle_rad = 0.0  # Rotation in radians
    
    # Plot point cloud (scatter plot with semantic colors) - 2D projection
    print(f"\nPlotting point cloud with {len(points_latlon)} points (2D projection)...")
    lons = points_latlon[:, 1]
    lats = points_latlon[:, 0]
    
    # Use semantic colors for points (store reference for updates)
    scatter_plot = ax.scatter(lons, lats, c=colors_rgb, s=0.5, alpha=0.6, 
                             label='Point Cloud (Semantic)', zorder=5)
    
    # Store original bounds for dynamic limit updates
    original_min_lat = min_lat
    original_max_lat = max_lat
    original_min_lon = min_lon
    original_max_lon = max_lon
    
    # Set initial plot limits
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
    
    # Keyboard event handler for interactive transformations
    def on_key(event):
        nonlocal lat_offset, lon_offset, rotation_angle_rad, scatter_plot, original_points_latlon
        nonlocal rotation_center
        
        if original_points_latlon is None or scatter_plot is None:
            return
        
        shift_amount = 0.00001  # Shift amount for translation
        rotation_amount = np.pi / 180.0  # 1 degree per key press
        
        # Handle arrow keys - matplotlib may use different key names
        key = event.key.lower() if hasattr(event.key, 'lower') else str(event.key)
        
        # Handle 's' key to save transformed points to UTM
        if key == 's':
            if original_points_utm is not None:
                # Apply current transformations to get final points in lat/lon
                transformed_points_latlon = apply_transformations(
                    original_points_latlon, lat_offset, lon_offset, rotation_angle_rad, rotation_center
                )
                
                # Convert back to UTM (preserving Z coordinates)
                transformed_points_utm = latlon_to_utm_for_saving(transformed_points_latlon, utm_zone, original_points_utm)
                
                # Save to file
                output_file = "transformed_points_utm.npy"
                np.save(output_file, transformed_points_utm)
                print(f"\nTransformed points saved to UTM: {output_file}")
                print(f"  Current offsets: lat={lat_offset:.9f}, lon={lon_offset:.9f}")
                print(f"  Current rotation: {np.degrees(rotation_angle_rad):.2f} degrees")
                print(f"  Rotation center: [{rotation_center[0]:.8f}, {rotation_center[1]:.8f}]")
                print(f"  Saved {len(transformed_points_utm)} points")
            else:
                print("No original UTM points available for saving!")
            return
        
        # Handle arrow keys for translation
        if key in ['up', 'arrow_up']:
            lat_offset += shift_amount
        elif key in ['down', 'arrow_down']:
            lat_offset -= shift_amount
        elif key in ['right', 'arrow_right']:
            lon_offset += shift_amount
        elif key in ['left', 'arrow_left']:
            lon_offset -= shift_amount
        # Handle rotation keys ('q' = counterclockwise, 'e' = clockwise)
        elif key == 'w':
            rotation_angle_rad += rotation_amount
        elif key == 'e':
            rotation_angle_rad -= rotation_amount
        else:
            return  # Ignore other keys
        
        # Apply transformations
        transformed_points = apply_transformations(
            original_points_latlon, lat_offset, lon_offset, rotation_angle_rad, rotation_center
        )
        
        # Calculate and print current state
        avg_lat = np.mean(transformed_points[:, 0])
        avg_lon = np.mean(transformed_points[:, 1])
        print(f"Transform: lat_offset={lat_offset:.9f}, lon_offset={lon_offset:.9f}, "
              f"rotation={np.degrees(rotation_angle_rad):.2f}°")
        print(f"  Average point position: lat={avg_lat:.9f}, lon={avg_lon:.9f}")
        
        # Update scatter plot
        offsets_xy = transformed_points[:, [1, 0]]  # Swap to [lon, lat] for matplotlib
        scatter_plot.set_offsets(offsets_xy)
        
        # Update plot limits to keep points visible
        new_min_lat = min(original_min_lat, transformed_points[:, 0].min())
        new_max_lat = max(original_max_lat, transformed_points[:, 0].max())
        new_min_lon = min(original_min_lon, transformed_points[:, 1].min())
        new_max_lon = max(original_max_lon, transformed_points[:, 1].max())
        
        new_lat_range = new_max_lat - new_min_lat
        new_lon_range = new_max_lon - new_min_lon
        new_lat_margin = new_lat_range * 0.05 if new_lat_range > 0 else lat_margin
        new_lon_margin = new_lon_range * 0.05 if new_lon_range > 0 else lon_margin
        
        ax.set_xlim(new_min_lon - new_lon_margin, new_max_lon + new_lon_margin)
        ax.set_ylim(new_min_lat - new_lat_margin, new_max_lat + new_lat_margin)
        
        # Force immediate redraw
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    def apply_transformations(points_latlon, lat_off, lon_off, rot_angle, center):
        """
        Apply translation and rotation transformations to points.
        
        Rotation is applied in 3D UTM space around Z-axis through initial pose,
        then projected to 2D lat/lon for visualization.
        """
        # First apply rotation around center in 2D lat/lon plane
        if rot_angle != 0.0:
            points_latlon = rotate_points_2d(points_latlon, center, rot_angle)
        
        # Then apply translation
        transformed = points_latlon.copy()
        transformed[:, 0] += lat_off  # Latitude
        transformed[:, 1] += lon_off  # Longitude
        
        return transformed
    
    def latlon_to_utm_for_saving(points_latlon, utm_zone, original_utm_points=None):
        """Convert lat-lon points back to UTM for saving."""
        # Determine source CRS
        if isinstance(utm_zone, str):
            target_crs = utm_zone
        elif isinstance(utm_zone, int):
            target_crs = f"EPSG:326{utm_zone:02d}"
        else:
            target_crs = "EPSG:32634"  # Default
        
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        
        lons = points_latlon[:, 1]
        lats = points_latlon[:, 0]
        
        # Convert to UTM (returns easting, northing)
        eastings, northings = transformer.transform(lons, lats)
        
        # If we have original UTM points, preserve Z coordinate
        if original_utm_points is not None and len(original_utm_points) == len(points_latlon):
            z_coords = original_utm_points[:, 2]
            return np.column_stack([eastings, northings, z_coords])
        else:
            # No Z coordinate available
            return np.column_stack([eastings, northings])
    
    # Connect keyboard event handler
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Make sure the figure can receive keyboard events
    try:
        fig.canvas.manager.set_window_title('Interactive OSM Visualization - Use arrow keys to shift, q/e to rotate')
    except:
        pass  # Some backends don't support set_window_title
    
    print("\n" + "="*60)
    print("Interactive mode enabled!")
    print("Click anywhere on the plot to get lat-long coordinates")
    print("Arrow keys: Shift point cloud (Up/Down: latitude, Left/Right: longitude)")
    print("W/E keys: Rotate point cloud around initial pose (W: counterclockwise, E: clockwise)")
    print("S key: Save transformed points to UTM as .npy file")
    print("Make sure the plot window has focus to receive keyboard events!")
    print("Close the window to exit")
    print("="*60)
    
    # Show the plot
    plt.tight_layout()
    
    # Make figure active and focused (try to bring window to front)
    try:
        mngr = fig.canvas.manager
        mngr.window.raise_()
        if hasattr(mngr.window, 'activateWindow'):
            mngr.window.activateWindow()
    except:
        pass
    
    plt.show()


if __name__ == "__main__":
    # File paths
    sequence = "kth_day_06"
    sequence_path = f"/media/donceykong/doncey_ssd_02/datasets/MCD/{sequence}"
    utm_pose_csv = f"/media/donceykong/doncey_ssd_02/datasets/MCD/{sequence}/pose_inW_shifted_utm.csv"
    osm_file_path = "/media/donceykong/doncey_ssd_02/datasets/MCD/kth.osm"
    
    # Parameters
    per_scan_voxel_size = 4.0  # Voxel size for each scan (meters)
    global_voxel_size = 0      # Voxel size for final map (meters)
    max_scans = None           # None to process all scans
    frame_skip = 10         # Process every Nth frame (1 = all frames)
    
    # Visualization options
    show_osm = True             # If True, plot over OSM. If False, use Open3D 3D visualization
    
    # Build map from UTM poses
    points, labels, initial_pose = build_map_from_utm_poses(
        sequence_path, 
        utm_pose_csv,
        per_scan_voxel_size=per_scan_voxel_size,
        global_voxel_size=global_voxel_size,
        max_scans=max_scans,
        frame_skip=frame_skip
    )
    
    # UTM zone for conversion
    utm_zone = "EPSG:32634"  # Stockholm area
    
    # Visualize
    if points is not None and labels is not None:
        visualize_semantic_map(points, labels, show_osm=show_osm, osm_file_path=osm_file_path, 
                              utm_zone=utm_zone, initial_pose=initial_pose)
    else:
        print("Error: Failed to build map!")

