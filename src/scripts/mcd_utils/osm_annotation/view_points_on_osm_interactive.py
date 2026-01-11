#!/usr/bin/env python3
"""
Plot robot path and point cloud onto OSM data visualization.

This script loads poses from a sequence (kth_day_06), projects them to lat-long
coordinates using the initial position, and plots the path on top of OSM data.
Additionally, it loads and plots points from an npy file.
"""

import numpy as np
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path
import sys
import os
import open3d as o3d
from scipy.spatial import cKDTree

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import projection utilities
from lidar2osm.core.projection.utils import latlon_to_mercator, mercator_to_latlon, lat_to_scale

# Import semantic labels and color conversion
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from create_seq_gt_map_npy import semantic_labels as sem_kitti_labels
from lidar2osm.core.pointcloud.pointcloud import labels2RGB_tqdm

# Import OSM visualization functions from view_and_select_osm_data
sys.path.append(os.path.dirname(__file__))
from view_and_select_osm_data import (
    get_osm_buildings_geometries,
    get_osm_road_geometries,
    get_osm_trees_geometries,
    get_osm_grassland_geometries,
    get_osm_water_geometries,
    on_click
)

initial_latlon = [59.348268650, 18.073204280] # lat, lon - initial position

def load_poses(pose_csv_file):
    """
    Load all poses from a CSV file.
    
    Args:
        pose_csv_file: Path to CSV file with columns [num, t, x, y, z, qx, qy, qz, qw]
    
    Returns:
        poses: List of dictionaries with 'position' (x, y, z) and 'quaternion' (qx, qy, qz, qw)
        df: Original DataFrame with all columns including 'num' and 't'
    """
    print(f"\nLoading poses from {pose_csv_file}")
    
    try:
        df = pd.read_csv(pose_csv_file, comment='#')
        print(f"Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return [], None
    
    if len(df) == 0:
        print("Error: CSV file is empty")
        return [], None
    
    poses = []
    for idx, row in df.iterrows():
        try:
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
                # Try positional access (skip first columns which might be 'num' or 't')
                if len(row) >= 8:
                    x = float(row.iloc[1])  # Skip first column (num or t)
                    y = float(row.iloc[2])
                    z = float(row.iloc[3])
                    qx = float(row.iloc[4])
                    qy = float(row.iloc[5])
                    qz = float(row.iloc[6])
                    qw = float(row.iloc[7])
                elif len(row) >= 7:
                    x = float(row.iloc[0])
                    y = float(row.iloc[1])
                    z = float(row.iloc[2])
                    qx = float(row.iloc[3])
                    qy = float(row.iloc[4])
                    qz = float(row.iloc[5])
                    qw = float(row.iloc[6])
                else:
                    print(f"Warning: Row {idx} doesn't have enough columns. Skipping.")
                    continue
            
            position = np.array([x, y, z])
            quaternion = np.array([qx, qy, qz, qw])
            
            poses.append({'position': position, 'quaternion': quaternion})
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    print(f"Loaded {len(poses)} poses")
    return poses, df


def project_poses_to_latlon(poses, origin_latlon):
    """
    Project poses from world coordinates to lat-long coordinates.
    
    This function follows the same pattern as convert_pose_to_latlon:
    - Assumes poses are in world coordinates where the first pose is the origin
    - The origin corresponds to origin_latlon
    - World coordinates: x=east (longitude direction), y=north (latitude direction)
    
    Args:
        poses: List of pose dictionaries with 'position' (x, y, z) in world coordinates
        origin_latlon: [lat, lon] of the initial position (origin)
    
    Returns:
        poses_latlon: Array of (N, 2) with [lat, lon] for each pose
    """
    if len(poses) == 0:
        return np.array([])
    
    # Get initial position (first pose) - this is the origin
    initial_position = poses[0]['position']
    
    # Compute mercator scale from latitude
    scale = lat_to_scale(origin_latlon[0])
    
    # Convert origin lat-lon to mercator coordinates
    ox_merc, oy_merc = latlon_to_mercator(origin_latlon[0], origin_latlon[1], scale)
    origin_merc = np.array([ox_merc, oy_merc, 0])
    
    print(f"\nProjecting {len(poses)} poses to lat-long...")
    print(f"  Initial position (world): [{initial_position[0]:.2f}, {initial_position[1]:.2f}, {initial_position[2]:.2f}]")
    print(f"  Origin (lat-lon): [{origin_latlon[0]:.8f}, {origin_latlon[1]:.8f}]")
    print(f"  Origin (mercator): [{ox_merc:.2f}, {oy_merc:.2f}]")
    print(f"  Mercator scale: {scale:.6f}")
    
    poses_latlon = []
    for i, pose in enumerate(poses):
        # Get position in world coordinates
        position = pose['position']
        
        # Compute position relative to initial position (relative to origin)
        relative_position = position - initial_position
        
        # Convert relative position (in meters) to mercator coordinates
        # In mercator: x = longitude direction (east), y = latitude direction (north)
        # Assuming world coordinates: x = east, y = north
        merc_x = ox_merc + relative_position[0]  # x is east (longitude/mercator x)
        merc_y = oy_merc + relative_position[1]  # y is north (latitude/mercator y)
        
        # Convert mercator coordinates to lat-lon
        lat, lon = mercator_to_latlon(merc_x, merc_y, scale)
        poses_latlon.append([lat, lon])
        
        # Debug first and last pose
        if i == 0 or i == len(poses) - 1:
            print(f"  Pose {i}: world=({position[0]:.2f}, {position[1]:.2f}) -> latlon=({lat:.8f}, {lon:.8f})")
    
    poses_latlon = np.array(poses_latlon)
    
    print(f"\n  Path bounds:")
    print(f"    Latitude: [{poses_latlon[:, 0].min():.8f}, {poses_latlon[:, 0].max():.8f}]")
    print(f"    Longitude: [{poses_latlon[:, 1].min():.8f}, {poses_latlon[:, 1].max():.8f}]")
    
    return poses_latlon


def load_points_from_npy(npy_file):
    """
    Load points from a numpy file.
    
    Args:
        npy_file: Path to .npy file with columns [x, y, z, intensity, semantic_id] or [x, y, z, semantic_id]
    
    Returns:
        points: (N, 3) array of coordinates [x, y, z] in world coordinates
        labels: (N,) array of semantic labels (optional)
    """
    print(f"\nLoading points from {npy_file}")
    
    try:
        data = np.load(npy_file)
        print(f"Loaded data shape: {data.shape}")
    except Exception as e:
        print(f"Error loading npy file: {e}")
        return None, None
    
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


def downsample_points(points, labels=None, voxel_size=1.0):
    """
    Downsample points using voxel downsampling, preserving labels.
    
    Args:
        points: (N, 3) array of points
        labels: (N,) array of semantic labels (optional)
        voxel_size: Size of voxel for downsampling (in meters)
    
    Returns:
        downsampled_points: (M, 3) array of downsampled points
        downsampled_labels: (M,) array of downsampled labels (if labels provided)
    """
    if len(points) == 0:
        if labels is not None:
            return points, labels
        return points
    
    print(f"\nDownsampling {len(points)} points with voxel size {voxel_size:.2f}...")
    
    # Create Open3D point cloud for downsampling
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Downsample
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    downsampled_points = np.asarray(downsampled_pcd.points)
    
    print(f"  Downsampled from {len(points)} to {len(downsampled_points)} points")
    
    # If labels are provided, find corresponding labels for downsampled points
    if labels is not None:
        # Use KDTree to find nearest neighbors
        tree = cKDTree(points)
        _, indices = tree.query(downsampled_points, k=1)
        downsampled_labels = labels[indices]
        return downsampled_points, downsampled_labels
    
    return downsampled_points


def project_points_to_latlon(points, initial_position, origin_latlon):
    """
    Project points from world coordinates to lat-long coordinates.
    
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
    
    points_latlon = []
    for i, point in enumerate(points):
        # Compute position relative to initial position
        relative_position = point - initial_position
        
        # Convert relative position (in meters) to mercator coordinates
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


def latlon_to_utm(poses_latlon, utm_zone=34, northern_hemisphere=True):
    """
    Convert lat-lon poses to UTM coordinates.
    
    Args:
        poses_latlon: (N, 2) array with [lat, lon] for each pose
        utm_zone: UTM zone number (default 34 for Stockholm area)
        northern_hemisphere: Whether in northern hemisphere (default True)
    
    Returns:
        poses_utm: (N, 2) array with [easting, northing] for each pose
    """
    from pyproj import Transformer
    
    # Create transformer: EPSG:4326 (WGS84 lat-lon) to UTM
    epsg_code = f"EPSG:326{utm_zone:02d}" if northern_hemisphere else f"EPSG:327{utm_zone:02d}"
    transformer = Transformer.from_crs("EPSG:4326", epsg_code, always_xy=True)
    
    lons = poses_latlon[:, 1]
    lats = poses_latlon[:, 0]
    
    # Convert to UTM (returns easting, northing)
    eastings, northings = transformer.transform(lons, lats)
    
    poses_utm = np.column_stack([eastings, northings])
    
    return poses_utm


def save_poses_to_utm_csv(poses_latlon, original_df, original_poses, output_file, utm_zone=34):
    """
    Save shifted poses to UTM CSV file, preserving 'num' and 't' fields.
    
    Args:
        poses_latlon: (N, 2) array with [lat, lon] for each shifted pose
        original_df: Original DataFrame with 'num' and 't' columns
        original_poses: List of original pose dictionaries with 'position' and 'quaternion'
        output_file: Path to output CSV file
        utm_zone: UTM zone number (default 34 for Stockholm area)
    """
    print(f"\nSaving shifted poses to UTM CSV: {output_file}")
    
    # Convert lat-lon to UTM
    poses_utm_xy = latlon_to_utm(poses_latlon, utm_zone=utm_zone)
    
    # Create new DataFrame with same structure as original
    output_data = []
    
    for i in range(len(poses_latlon)):
        row_data = {}
        
        # Preserve 'num' and 't' fields if they exist in the original DataFrame
        # Check by column name first
        if 'num' in original_df.columns:
            row_data['num'] = original_df.iloc[i]['num']
        # Otherwise try positional access (first column)
        elif len(original_df.columns) >= 9:
            try:
                row_data['num'] = original_df.iloc[i, 0]
            except:
                pass
        
        # Check for 't' column
        if 't' in original_df.columns:
            row_data['t'] = original_df.iloc[i]['t']
        # Otherwise try positional access
        elif 'num' in original_df.columns and len(original_df.columns) >= 9:
            # If 'num' exists, 't' is likely the second column
            try:
                row_data['t'] = original_df.iloc[i, 1]
            except:
                pass
        elif len(original_df.columns) >= 9:
            # If no 'num', 't' might be the first column
            try:
                row_data['t'] = original_df.iloc[i, 0]
            except:
                pass
        
        # UTM coordinates (x, y)
        row_data['x'] = poses_utm_xy[i, 0]  # Easting
        row_data['y'] = poses_utm_xy[i, 1]  # Northing
        
        # Keep original z coordinate and quaternion
        if i < len(original_poses):
            row_data['z'] = original_poses[i]['position'][2]  # z coordinate
            quat = original_poses[i]['quaternion']
            row_data['qx'] = quat[0]
            row_data['qy'] = quat[1]
            row_data['qz'] = quat[2]
            row_data['qw'] = quat[3]
        
        output_data.append(row_data)
    
    # Create DataFrame
    output_df = pd.DataFrame(output_data)
    
    # Reorder columns to match original format: [num, t, x, y, z, qx, qy, qz, qw]
    column_order = []
    if 'num' in output_df.columns:
        column_order.append('num')
    if 't' in output_df.columns:
        column_order.append('t')
    column_order.extend(['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in output_df.columns]
    output_df = output_df[column_order]
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"  Saved {len(output_df)} poses to {output_file}")
    print(f"  UTM zone: {utm_zone}N")
    print(f"  UTM X (Easting) range: [{output_df['x'].min():.2f}, {output_df['x'].max():.2f}]")
    print(f"  UTM Y (Northing) range: [{output_df['y'].min():.2f}, {output_df['y'].max():.2f}]")


def visualize_path_on_osm(osm_file_path, poses_latlon, points_latlon=None, point_colors=None, 
                          original_poses=None, original_df=None, pose_csv_file=None):
    """Visualize OSM data with robot path overlaid."""
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
    
    # Include path coordinates in bounds calculation
    if len(poses_latlon) > 0:
        all_lons.extend(poses_latlon[:, 1].tolist())
        all_lats.extend(poses_latlon[:, 0].tolist())
    
    # Include point cloud coordinates in bounds calculation
    if points_latlon is not None and len(points_latlon) > 0:
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
    ax.set_title('Robot Path and Point Cloud on OSM Data - Click anywhere to get lat-long coordinates', fontsize=14)
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
    
    # Store original poses and points for keyboard adjustment
    original_poses_latlon = poses_latlon.copy() if len(poses_latlon) > 0 else None
    original_points_latlon = points_latlon.copy() if points_latlon is not None else None
    lat_offset = 0.0
    lon_offset = 0.0
    scatter_plot = None
    path_line = None
    start_marker = None
    end_marker = None
    intermediate_markers = None
    # Store original poses and DataFrame for saving
    stored_original_poses = original_poses
    stored_original_df = original_df
    stored_pose_csv_file = pose_csv_file
    
    # Plot robot path (red line with markers)
    if len(poses_latlon) > 0:
        print(f"\nPlotting robot path with {len(poses_latlon)} poses...")
        lons = poses_latlon[:, 1]
        lats = poses_latlon[:, 0]
        
        # Plot path as line (store reference for updates)
        path_line = ax.plot(lons, lats, 'r-', linewidth=2, alpha=0.8, label='Robot Path', zorder=10)[0]
        
        # Plot start point (green marker) - store reference
        start_marker = ax.plot(lons[0], lats[0], 'go', markersize=10, label='Start', zorder=11)[0]
        
        # Plot end point (red marker) - store reference
        end_marker = ax.plot(lons[-1], lats[-1], 'ro', markersize=10, label='End', zorder=11)[0]
        
        # Plot intermediate points (smaller markers, less frequent) - store reference
        if len(poses_latlon) > 2:
            step = max(1, len(poses_latlon) // 50)  # Show every Nth point
            intermediate_markers = ax.plot(lons[::step], lats[::step], 'r.', markersize=3, alpha=0.6, zorder=10)[0]
    
    # Store original bounds for dynamic limit updates
    original_min_lat = min_lat
    original_max_lat = max_lat
    original_min_lon = min_lon
    original_max_lon = max_lon
    
    # Plot point cloud (scatter plot with semantic colors)
    if points_latlon is not None and len(points_latlon) > 0:
        print(f"\nPlotting point cloud with {len(points_latlon)} points...")
        lons = points_latlon[:, 1]
        lats = points_latlon[:, 0]
        
        # Use semantic colors if available, otherwise use orange
        if point_colors is not None and len(point_colors) == len(points_latlon):
            print(f"  Using semantic colors for point cloud")
            # point_colors should be in [0, 1] range
            scatter_plot = ax.scatter(lons, lats, c=point_colors, s=0.5, alpha=0.6, label='Point Cloud (Semantic)', zorder=5)
        else:
            print(f"  Using default orange color for point cloud")
            scatter_plot = ax.scatter(lons, lats, c='orange', s=0.5, alpha=0.6, label='Point Cloud', zorder=5)
    
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
        plt.Line2D([0], [0], color='red', linewidth=2, label='Robot Path'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Start'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='End')
    ]
    
    # Add point cloud to legend if present
    if points_latlon is not None and len(points_latlon) > 0:
        if point_colors is not None:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='orange', 
                                             markersize=5, linestyle='None', label='Point Cloud (Semantic)'))
        else:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='orange', 
                                             markersize=5, linestyle='None', label='Point Cloud'))
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Connect click event handler
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Keyboard event handler for shifting points and poses
    def on_key(event):
        nonlocal lat_offset, lon_offset, scatter_plot, original_points_latlon
        nonlocal original_poses_latlon, path_line, start_marker, end_marker, intermediate_markers
        nonlocal stored_original_poses, stored_original_df, stored_pose_csv_file
        
        if original_points_latlon is None or scatter_plot is None:
            return
        
        shift_amount = 0.00001
        
        # Handle arrow keys - matplotlib may use different key names
        key = event.key.lower() if hasattr(event.key, 'lower') else str(event.key)
        
        # Handle 's' key to save poses
        if key == 's':
            if original_poses_latlon is not None and len(original_poses_latlon) > 0:
                # Calculate shifted poses
                shifted_poses_latlon = original_poses_latlon.copy()
                shifted_poses_latlon[:, 0] += lat_offset  # Latitude
                shifted_poses_latlon[:, 1] += lon_offset  # Longitude
                
                # Generate output filename
                if stored_pose_csv_file:
                    base_name = Path(stored_pose_csv_file).stem
                    output_file = str(Path(stored_pose_csv_file).parent / f"{base_name}_shifted_utm.csv")
                else:
                    output_file = "shifted_poses_utm.csv"
                
                # Save to UTM CSV
                if stored_original_df is not None and stored_original_poses is not None:
                    save_poses_to_utm_csv(shifted_poses_latlon, stored_original_df, stored_original_poses, 
                                         output_file, utm_zone=34)
                    print(f"\nPoses saved! Current offsets: lat={lat_offset:.9f}, lon={lon_offset:.9f}")
                    print(f"Saved to: {output_file}")
                else:
                    print("Error: Missing original DataFrame or poses data!")
            else:
                print("No poses to save!")
            return
        
        # Handle arrow keys
        if key in ['up', 'arrow_up']:
            lat_offset += shift_amount
        elif key in ['down', 'arrow_down']:
            lat_offset -= shift_amount
        elif key in ['right', 'arrow_right']:
            lon_offset += shift_amount
        elif key in ['left', 'arrow_left']:
            lon_offset -= shift_amount
        else:
            # Debug: print unrecognized keys
            # print(f"Unrecognized key: '{key}'")
            return  # Ignore other keys
        
        # Update point positions with offset
        shifted_points = original_points_latlon.copy()
        shifted_points[:, 0] += lat_offset  # Latitude
        shifted_points[:, 1] += lon_offset  # Longitude
        
        # Update pose positions with offset
        shifted_poses = None
        if original_poses_latlon is not None and len(original_poses_latlon) > 0:
            shifted_poses = original_poses_latlon.copy()
            shifted_poses[:, 0] += lat_offset  # Latitude
            shifted_poses[:, 1] += lon_offset  # Longitude
        
        # Calculate and print average position
        avg_lat = np.mean(shifted_points[:, 0])
        avg_lon = np.mean(shifted_points[:, 1])
        print(f"Shifted points: lat_offset={lat_offset:.9f}, lon_offset={lon_offset:.9f}")
        print(f"  Average point position: lat={avg_lat:.9f}, lon={avg_lon:.9f}")
        
        # Print initial shifted pose
        if shifted_poses is not None and len(shifted_poses) > 0:
            initial_pose_lat = shifted_poses[0, 0]
            initial_pose_lon = shifted_poses[0, 1]
            print(f"  Initial shifted pose: lat={initial_pose_lat:.9f}, lon={initial_pose_lon:.9f}")
        
        # Update scatter plot
        # set_offsets expects [x, y] = [lon, lat] format
        offsets_xy = shifted_points[:, [1, 0]]  # Swap to [lon, lat] for matplotlib
        scatter_plot.set_offsets(offsets_xy)
        
        # Update path line and markers
        if shifted_poses is not None and path_line is not None:
            shifted_lons = shifted_poses[:, 1]
            shifted_lats = shifted_poses[:, 0]
            
            # Update path line
            path_line.set_data(shifted_lons, shifted_lats)
            
            # Update start marker
            if start_marker is not None:
                start_marker.set_data([shifted_lons[0]], [shifted_lats[0]])
            
            # Update end marker
            if end_marker is not None:
                end_marker.set_data([shifted_lons[-1]], [shifted_lats[-1]])
            
            # Update intermediate markers
            if intermediate_markers is not None and len(shifted_poses) > 2:
                step = max(1, len(shifted_poses) // 50)
                intermediate_markers.set_data(shifted_lons[::step], shifted_lats[::step])
        
        # Update plot limits to keep points and poses visible
        # Calculate new bounds including shifted points/poses and original OSM bounds
        new_min_lat = min(original_min_lat, shifted_points[:, 0].min())
        new_max_lat = max(original_max_lat, shifted_points[:, 0].max())
        new_min_lon = min(original_min_lon, shifted_points[:, 1].min())
        new_max_lon = max(original_max_lon, shifted_points[:, 1].max())
        
        # Include shifted poses in bounds if available
        if shifted_poses is not None and len(shifted_poses) > 0:
            new_min_lat = min(new_min_lat, shifted_poses[:, 0].min())
            new_max_lat = max(new_max_lat, shifted_poses[:, 0].max())
            new_min_lon = min(new_min_lon, shifted_poses[:, 1].min())
            new_max_lon = max(new_max_lon, shifted_poses[:, 1].max())
        
        # Add margins
        new_lat_range = new_max_lat - new_min_lat
        new_lon_range = new_max_lon - new_min_lon
        new_lat_margin = new_lat_range * 0.05 if new_lat_range > 0 else lat_margin
        new_lon_margin = new_lon_range * 0.05 if new_lon_range > 0 else lon_margin
        
        # Update plot limits
        ax.set_xlim(new_min_lon - new_lon_margin, new_max_lon + new_lon_margin)
        ax.set_ylim(new_min_lat - new_lat_margin, new_max_lat + new_lat_margin)
        
        # Force immediate redraw
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    # Connect keyboard event handler
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Make sure the figure can receive keyboard events
    try:
        fig.canvas.manager.set_window_title('Interactive OSM Visualization - Use arrow keys to shift points')
    except:
        pass  # Some backends don't support set_window_title
    
    print("\n" + "="*60)
    print("Interactive mode enabled!")
    print("Click anywhere on the plot to get lat-long coordinates")
    print("Arrow keys: Shift point cloud and poses (Up/Down: latitude, Left/Right: longitude)")
    print("Press 's' key: Save shifted poses to UTM CSV file")
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
    
    # Show the plot (blocking)
    plt.show()


if __name__ == "__main__":
    # File paths
    sequence = "kth_day_06"
    pose_csv_file = f"/media/donceykong/doncey_ssd_02/datasets/MCD/{sequence}/pose_inW.csv"
    npy_file = "/media/donceykong/doncey_ssd_02/datasets/MCD/ply/merged_gt_labels_kth_day_06_kth_day_09_kth_night_05.npy"
    osm_file_path = "/media/donceykong/doncey_ssd_02/datasets/MCD/kth.osm"
    
    # Load poses and original DataFrame
    poses, original_df = load_poses(pose_csv_file)
    
    if len(poses) == 0:
        print("Error: No poses loaded!")
        sys.exit(1)
    
    if original_df is None:
        print("Error: Could not load original DataFrame!")
        sys.exit(1)
    
    # Get initial position from first pose
    initial_position = poses[0]['position']
    
    # Project poses to lat-long coordinates
    poses_latlon = project_poses_to_latlon(poses, initial_latlon)
    
    # Load and project points from npy file
    points_latlon = None
    point_colors = None
    if os.path.exists(npy_file):
        # Load points and labels
        points, labels = load_points_from_npy(npy_file)
        
        if points is not None and len(points) > 0:
            # Downsample points before projection
            # Use voxel size based on point cloud extent
            bbox_size = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
            voxel_size = min(1.0, bbox_size * 0.01)  # 0.1% of bounding box, minimum 1m
            print(f"\nUsing voxel size: {voxel_size:.2f} meters for downsampling")
            
            # Downsample points and labels together
            if labels is not None:
                downsampled_points, downsampled_labels = downsample_points(points, labels, voxel_size=voxel_size)
                
                # Convert semantic labels to RGB colors
                print("Converting semantic labels to RGB colors...")
                labels_dict = {label.id: label.color for label in sem_kitti_labels}
                point_colors = labels2RGB_tqdm(downsampled_labels, labels_dict)
                print(f"  Generated colors for {len(point_colors)} points")
            else:
                downsampled_points = downsample_points(points, voxel_size=voxel_size)
            
            # Project downsampled points to lat-long coordinates
            points_latlon = project_points_to_latlon(downsampled_points, initial_position, initial_latlon)
        else:
            print("Warning: No points loaded from npy file")
    else:
        print(f"Warning: npy file not found: {npy_file}")
        print("Continuing without point cloud visualization...")
    
    # Visualize path and points on OSM data
    visualize_path_on_osm(osm_file_path, poses_latlon, points_latlon, point_colors,
                          original_poses=poses, original_df=original_df, pose_csv_file=pose_csv_file)
