#!/usr/bin/env python3
"""
Plot robot path onto OSM data visualization.

This script loads poses from a sequence (kth_day_06), projects them to lat-long
coordinates using the initial position, and plots the path on top of OSM data.
"""

import numpy as np
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import projection utilities
from lidar2osm.core.projection.utils import latlon_to_mercator, mercator_to_latlon, lat_to_scale

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

initial_latlon = [59.34825865, 18.07316428]  # lat, lon - initial position


def load_poses(pose_csv_file):
    """
    Load all poses from a CSV file.
    
    Args:
        pose_csv_file: Path to CSV file with columns [num, t, x, y, z, qx, qy, qz, qw]
    
    Returns:
        poses: List of dictionaries with 'position' (x, y, z) and 'quaternion' (qx, qy, qz, qw)
    """
    print(f"\nLoading poses from {pose_csv_file}")
    
    try:
        df = pd.read_csv(pose_csv_file, comment='#')
        print(f"Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []
    
    if len(df) == 0:
        print("Error: CSV file is empty")
        return []
    
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
    return poses


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


def visualize_path_on_osm(osm_file_path, poses_latlon):
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
    ax.set_title('Robot Path on OSM Data - Click anywhere to get lat-long coordinates', fontsize=14)
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
    
    # Plot robot path (red line with markers)
    if len(poses_latlon) > 0:
        print(f"\nPlotting robot path with {len(poses_latlon)} poses...")
        lons = poses_latlon[:, 1]
        lats = poses_latlon[:, 0]
        
        # Plot path as line
        ax.plot(lons, lats, 'r-', linewidth=2, alpha=0.8, label='Robot Path', zorder=10)
        
        # Plot start point (green marker)
        ax.plot(lons[0], lats[0], 'go', markersize=10, label='Start', zorder=11)
        
        # Plot end point (red marker)
        ax.plot(lons[-1], lats[-1], 'ro', markersize=10, label='End', zorder=11)
        
        # Plot intermediate points (smaller markers, less frequent)
        if len(poses_latlon) > 2:
            step = max(1, len(poses_latlon) // 50)  # Show every Nth point
            ax.plot(lons[::step], lats[::step], 'r.', markersize=3, alpha=0.6, zorder=10)
    
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
    sequence = "kth_day_06"
    pose_csv_file = f"/media/donceykong/doncey_ssd_02/datasets/MCD/{sequence}/pose_inW.csv"
    osm_file_path = "/media/donceykong/doncey_ssd_02/datasets/MCD/kth.osm"
    
    # Load poses
    poses = load_poses(pose_csv_file)
    
    if len(poses) == 0:
        print("Error: No poses loaded!")
        sys.exit(1)
    
    # Project poses to lat-long coordinates
    poses_latlon = project_poses_to_latlon(poses, initial_latlon)
    
    # Visualize path on OSM data
    visualize_path_on_osm(osm_file_path, poses_latlon)
