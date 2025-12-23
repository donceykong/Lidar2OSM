#!/usr/bin/env python3
"""
Script to create offset polygons for OSM buildings and store them in a numpy-compatible format.
Stores "inner" (inward offset), "original", and "outer" (outward offset) polygon points for each building.
Only processes buildings within 100m of robot poses.
"""

import os
import sys
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
from pyproj import Transformer

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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


def create_offset_polygon(polygon, offset_distance):
    """
    Create an offset polygon using shapely's buffer method.
    
    Args:
        polygon: Shapely Polygon object
        offset_distance: Distance in degrees (positive for outer, negative for inner)
    
    Returns:
        Offset polygon or None if invalid
    """
    try:
        offset_poly = polygon.buffer(offset_distance)
        if offset_poly.is_empty or not isinstance(offset_poly, Polygon):
            return None
        return offset_poly
    except Exception as e:
        print(f"  Warning: Could not create offset polygon: {e}")
        return None


def extract_polygon_points(polygon):
    """
    Extract exterior coordinates from a polygon.
    
    Args:
        polygon: Shapely Polygon object
    
    Returns:
        numpy array of shape (N, 2) with [lon, lat] coordinates
    """
    if polygon is None or polygon.is_empty:
        return None
    
    coords = list(polygon.exterior.coords)
    # Remove last point if it's duplicate of first (closed polygon)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    
    # Convert to numpy array: (lon, lat) format
    points = np.array(coords)
    return points


def create_building_offset_data(filtered_buildings, inner_offset_distance=-8e-06, outer_offset_distance=8e-06):
    """
    Create data structure storing inner, original, and outer polygon points for each building.
    
    Args:
        filtered_buildings: List of building geometries from OSM
        inner_offset_distance: Distance for inward offset in degrees (negative value)
        outer_offset_distance: Distance for outward offset in degrees (positive value)
    
    Returns:
        List of dictionaries, each containing:
        {
            'inner': numpy array (N, 2) of [lon, lat] points or None,
            'original': numpy array (M, 2) of [lon, lat] points,
            'outer': numpy array (K, 2) of [lon, lat] points or None
        }
    """
    building_data = []
    
    print(f"\nProcessing {len(filtered_buildings)} buildings...")
    print(f"  Inner offset distance: {inner_offset_distance} degrees")
    print(f"  Outer offset distance: {outer_offset_distance} degrees")
    
    for idx, building in enumerate(filtered_buildings):
        if building.geometry.geom_type != "Polygon":
            print(f"  Building {idx}: Skipping - geometry type {building.geometry.geom_type} not supported")
            continue
        
        original_poly = building.geometry
        
        # Extract original polygon points
        original_points = extract_polygon_points(original_poly)
        if original_points is None:
            print(f"  Building {idx}: Skipping - could not extract original points")
            continue
        
        # Create inner offset (inward)
        inner_poly = create_offset_polygon(original_poly, inner_offset_distance)
        inner_points = extract_polygon_points(inner_poly) if inner_poly is not None else None
        
        # Create outer offset (outward)
        outer_poly = create_offset_polygon(original_poly, outer_offset_distance)
        outer_points = extract_polygon_points(outer_poly) if outer_poly is not None else None
        
        building_dict = {
            'inner': inner_points,
            'original': original_points,
            'outer': outer_points
        }
        
        building_data.append(building_dict)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(filtered_buildings)} buildings...")
    
    print(f"  Successfully processed {len(building_data)} buildings")
    return building_data


def save_building_offsets(building_data, output_file):
    """
    Save building offset data to a .npy file.
    
    The data structure is saved as a numpy array of object dtype, where each element
    is a dictionary-like structure. We use a structured approach that can be loaded back.
    
    Args:
        building_data: List of dictionaries with 'inner', 'original', 'outer' keys
        output_file: Path to output .npy file
    """
    print(f"\nSaving building offset data to {output_file}...")
    
    # Convert to a format that can be saved as .npy
    # We'll save as a numpy array of object dtype containing dictionaries
    # This allows variable-length arrays for each building
    data_array = np.array(building_data, dtype=object)
    
    np.save(output_file, data_array, allow_pickle=True)
    
    print(f"  Saved {len(building_data)} buildings to {output_file}")
    print(f"  File size: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")
    
    # Print statistics
    num_with_inner = sum(1 for b in building_data if b['inner'] is not None)
    num_with_outer = sum(1 for b in building_data if b['outer'] is not None)
    print(f"  Buildings with inner offset: {num_with_inner}/{len(building_data)}")
    print(f"  Buildings with outer offset: {num_with_outer}/{len(building_data)}")


def load_building_offsets(input_file):
    """
    Load building offset data from a .npy file.
    
    Args:
        input_file: Path to input .npy file
    
    Returns:
        List of dictionaries with 'inner', 'original', 'outer' keys
    """
    print(f"\nLoading building offset data from {input_file}...")
    data_array = np.load(input_file, allow_pickle=True)
    building_data = data_array.tolist()
    print(f"  Loaded {len(building_data)} buildings")
    return building_data


def main():
    """Main function."""
    import argparse
    parser = argparse.ArgumentParser(description="Create offset polygons for OSM buildings")

    # Environment name
    parser.add_argument("--environment", type=str, default="kittredge_loop",
                       help="Environment name (default: main_campus)")

    parser.add_argument("--filter_distance", type=float, default=100.0,
                       help="Distance in meters to filter buildings around poses (default: 100.0)")
    
    parser.add_argument("--inner_offset", type=float, default=-8e-06, #MC: default=-15e-06,
                       help="Inner offset distance in degrees (negative for inward, default: -8e-06)")
    
    parser.add_argument("--outer_offset", type=float, default=2e-05,
                       help="Outer offset distance in degrees (positive for outward, default: 8e-06)")
    
    args = parser.parse_args()
    
    # Hardcoded paths (same as in building_offset_example.py)
    dataset_path = "/media/donceykong/donceys_data_ssd/datasets/CU_MULTI/data"
    environment = args.environment
    robots = ["robot1", "robot2", "robot3", "robot4"]  # Can be extended to multiple robots
    
    # File paths
    osm_file = Path(dataset_path) / environment / "additional" / f"{environment}.osm"
    output_file = Path(dataset_path) / environment / "additional" / f"{environment}_building_offsets.npy"
    
    # Check if OSM file exists
    if not osm_file.exists():
        print(f"Error: OSM file not found: {osm_file}")
        return
    
    # Load poses from all robots
    print(f"\n{'='*80}")
    print("Loading Poses from All Robots")
    print(f"{'='*80}")
    
    all_poses_latlon = []
    all_robot_names = []
    
    for robot in robots:
        poses_file = Path(dataset_path) / environment / robot / f"{robot}_{environment}_gt_utm_poses.csv"
        
        if not poses_file.exists():
            print(f"Warning: Poses file not found for {robot}: {poses_file}")
            print(f"Skipping {robot}...")
            continue
        
        print(f"\nLoading poses for {robot}...")
        poses = load_poses(poses_file)
        
        if len(poses) == 0:
            print(f"No poses found for {robot}! Skipping...")
            continue
        
        print(f"Loaded {len(poses)} poses for {robot}")
        
        # Transform poses from IMU to LiDAR frame
        print(f"Transforming {robot} poses from IMU to LiDAR frame...")
        poses = transform_imu_to_lidar(poses)
        
        # Convert UTM poses to lat/lon
        print(f"Converting {robot} poses to lat/lon...")
        robot_poses_latlon, timestamps = utm_to_latlon(poses)
        print(f"Converted {len(robot_poses_latlon)} poses for {robot}")
        
        all_poses_latlon.append(robot_poses_latlon)
        all_robot_names.append(robot)
    
    # Check if we loaded any poses
    if not all_poses_latlon:
        print("\nError: No poses loaded from any robot!")
        return
    
    # Combine all poses
    print(f"\n{'='*60}")
    print(f"Combining poses from {len(all_poses_latlon)} robots...")
    poses_latlon = np.vstack(all_poses_latlon)
    print(f"Total combined poses: {len(poses_latlon)}")
    
    # Print statistics
    print(f"\nCombined Pose Statistics:")
    print(f"  Robots included: {', '.join(all_robot_names)}")
    print(f"  Latitude range: {poses_latlon[:, 0].min():.6f} to {poses_latlon[:, 0].max():.6f}")
    print(f"  Longitude range: {poses_latlon[:, 1].min():.6f} to {poses_latlon[:, 1].max():.6f}")
    
    # Load OSM data
    print(f"\n{'='*80}")
    print("Loading OSM Data")
    print(f"{'='*80}")
    print(f"Filter distance: {args.filter_distance}m")
    osm_handler = OSMDataHandler(osm_file, filter_distance=args.filter_distance)
    
    # Enable buildings semantics
    osm_handler.set_semantics({
        'buildings': True,
        'roads': False,
        'highways': False,
        'trees': False,
        'grassland': False,
        'gardens': False,
        'water': False,
        'parking': False,
        'amenities': False
    })
    
    if not osm_handler.load_osm_data():
        print("Failed to load OSM data")
        return
    
    # Filter buildings within filter_distance of robot poses
    print(f"\n{'='*80}")
    print("Filtering Buildings by Distance")
    print(f"{'='*80}")
    filtered_buildings = osm_handler.filter_geometries_by_distance('buildings', poses_latlon, use_centroid=False)
    
    if not filtered_buildings:
        print("No buildings found within filter distance!")
        return
    
    print(f"Found {len(filtered_buildings)} buildings within {args.filter_distance}m of robot poses")
    
    # Create offset polygons for each building
    print(f"\n{'='*80}")
    print("Creating Building Offset Polygons")
    print(f"{'='*80}")
    building_data = create_building_offset_data(
        filtered_buildings,
        inner_offset_distance=args.inner_offset,
        outer_offset_distance=args.outer_offset
    )
    
    if not building_data:
        print("No building data created!")
        return
    
    # Save to .npy file
    print(f"\n{'='*80}")
    print("Saving Building Offset Data")
    print(f"{'='*80}")
    save_building_offsets(building_data, output_file)
    
    print(f"\n{'='*80}")
    print("SUCCESS!")
    print(f"{'='*80}")
    print(f"Output saved to: {output_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()