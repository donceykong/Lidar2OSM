#!/usr/bin/env python3
"""
Script to select and plot a single OSM building boundary near robot poses.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Internal imports
from lidar2osm.utils.osm_handler import OSMDataHandler


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


def utm_to_latlon(poses):
    """Convert UTM poses to lat/lon coordinates."""
    from pyproj import Transformer
    
    # UTM zone 13N for Colorado
    transformer = Transformer.from_crs("EPSG:32613", "EPSG:4326", always_xy=True)
    
    positions = []
    timestamps = []
    
    for timestamp, pose in poses.items():
        position = pose[:3]
        positions.append(position)
        timestamps.append(timestamp)
    
    positions = np.array(positions)
    timestamps = np.array(timestamps)
    
    lons, lats = transformer.transform(positions[:, 0], positions[:, 1])
    latlon_positions = np.column_stack([lats, lons, positions[:, 2]])
    
    return latlon_positions, timestamps


def plot_selected_building(osm_handler, poses_latlon, building_idx=0, output_file="selected_building.png"):
    """
    Plot a selected OSM building near the robot poses.
    
    Args:
        osm_handler: OSMDataHandler instance with loaded OSM data
        poses_latlon: Array of poses in lat/lon format
        building_idx: Index of building to plot (0 = first building)
        output_file: Output filename for the plot
    """
    # Get filtered buildings near poses
    filtered_buildings = osm_handler.filter_geometries_by_distance('buildings', poses_latlon, use_centroid=False)
    
    if not filtered_buildings:
        print("No buildings found near poses!")
        return False
    
    print(f"\nFound {len(filtered_buildings)} buildings near poses")
    
    # Select building (default to first one)
    if building_idx >= len(filtered_buildings):
        print(f"Building index {building_idx} out of range. Using building 0.")
        building_idx = 0
    
    selected_building = filtered_buildings[building_idx]
    print(f"\nSelected building {building_idx}:")
    print(f"  Geometry type: {selected_building.geometry.geom_type}")
    
    # Extract building coordinates
    if selected_building.geometry.geom_type == "Polygon":
        coords = list(selected_building.geometry.exterior.coords)
        lons, lats = zip(*coords)
        
        # Calculate building centroid
        centroid = selected_building.geometry.centroid
        print(f"  Centroid: ({centroid.y:.6f}, {centroid.x:.6f})")
        print(f"  Number of vertices: {len(coords)}")
        
        # Find min and max x (longitude) and y (latitude) points
        lons_array = np.array(lons)
        lats_array = np.array(lats)
        
        x_min = lons_array.min()
        x_max = lons_array.max()
        y_min = lats_array.min()
        y_max = lats_array.max()
        
        print(f"  X (Longitude) range: [{x_min:.6f}, {x_max:.6f}]")
        print(f"  Y (Latitude) range: [{y_min:.6f}, {y_max:.6f}]")
        print(f"  Min point (x_min, y_min): ({x_min:.6f}, {y_min:.6f})")
        print(f"  Max point (x_max, y_max): ({x_max:.6f}, {y_max:.6f})")
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Plot the building boundary
        plt.plot(lons, lats, 'b-', linewidth=2, label='Building Boundary')
        plt.fill(lons, lats, color='blue', alpha=0.3)
        
        # Plot building centroid
        plt.plot(centroid.x, centroid.y, 'ro', markersize=10, label='Centroid', zorder=10)
        
        # Plot min and max points
        plt.plot(x_min, y_min, 'mo', markersize=12, label='Min Point (x_min, y_min)', zorder=11)
        plt.plot(x_max, y_max, 'co', markersize=12, label='Max Point (x_max, y_max)', zorder=11)
        
        # Plot robot poses for context
        plt.plot(poses_latlon[:, 1], poses_latlon[:, 0], 'g-', linewidth=1, alpha=0.5, label='Robot Path')
        plt.plot(poses_latlon[0, 1], poses_latlon[0, 0], 'go', markersize=8, label='Start')
        plt.plot(poses_latlon[-1, 1], poses_latlon[-1, 0], 'gs', markersize=8, label='End')
        
        # Add labels for vertices
        for i, (lon, lat) in enumerate(coords[:-1]):  # Skip last vertex (same as first)
            plt.text(lon, lat, str(i), fontsize=8, ha='right', va='bottom')
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'OSM Building {building_idx} Boundary')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {output_file}")
        plt.show()
        
        return True
    else:
        print(f"Building geometry type {selected_building.geometry.geom_type} not supported yet")
        return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Select and plot an OSM building boundary")
    parser.add_argument("--building_idx", type=int, default=0,
                       help="Index of building to plot (default: 0)")
    parser.add_argument("--filter_distance", type=float, default=100.0,
                       help="Distance in meters to filter buildings around poses (default: 100.0)")
    parser.add_argument("--output", type=str, default="selected_building.png",
                       help="Output filename for the plot (default: selected_building.png)")
    args = parser.parse_args()
    
    # Hardcoded paths (same as view_poses_on_osm_semantic.py)
    dataset_path = "/media/donceykong/donceys_data_ssd/datasets/CU_MULTI/data"
    environment = "main_campus"
    robot = "robot1"
    
    # Construct file paths
    poses_file = Path(dataset_path) / environment / robot / f"{robot}_{environment}_gt_utm_poses.csv"
    osm_file = Path(dataset_path) / environment / f"{environment}.osm"
    
    # Check if files exist
    if not poses_file.exists():
        print(f"Error: Poses file not found: {poses_file}")
        return
    
    if not osm_file.exists():
        print(f"Error: OSM file not found: {osm_file}")
        return
    
    # Load poses
    print("Loading poses...")
    poses = load_poses(poses_file)
    print(f"Loaded {len(poses)} poses")
    
    if len(poses) == 0:
        print("No poses found!")
        return
    
    # Transform poses from IMU to LiDAR frame
    print("Transforming poses from IMU to LiDAR frame...")
    poses = transform_imu_to_lidar(poses)
    
    # Convert UTM poses to lat/lon
    print("Converting UTM to lat/lon...")
    poses_latlon, timestamps = utm_to_latlon(poses)
    print(f"Converted {len(poses_latlon)} poses to lat/lon")
    
    # Print statistics
    print(f"\nPose Statistics:")
    print(f"  Latitude range: {poses_latlon[:, 0].min():.6f} to {poses_latlon[:, 0].max():.6f}")
    print(f"  Longitude range: {poses_latlon[:, 1].min():.6f} to {poses_latlon[:, 1].max():.6f}")
    
    # Load OSM data
    print(f"\nLoading OSM data with filter distance: {args.filter_distance}m")
    osm_handler = OSMDataHandler(osm_file, filter_distance=args.filter_distance)
    
    # Only enable buildings semantic
    osm_handler.set_semantics({
        'roads': False,
        'highways': False,
        'buildings': True,
        'trees': False,
        'grassland': False,
        'water': False,
        'parking': False,
        'amenities': False
    })
    
    if not osm_handler.load_osm_data():
        print("Failed to load OSM data")
        return
    
    # Plot selected building
    print(f"\nPlotting building {args.building_idx}...")
    plot_selected_building(osm_handler, poses_latlon, 
                          building_idx=args.building_idx, 
                          output_file=args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

