#!/usr/bin/env python3
"""
Script to visualize OSM road polylines with their segments and boundary circles.
Helps debug road parsing and visualization.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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


def visualize_roads(osm_handler, poses_latlon, output_file="road_visualization.png", 
                   num_segments=20, proximity_threshold=2.0):
    """
    Visualize OSM roads with their segments and boundary circles.
    
    Args:
        osm_handler: OSMDataHandler instance with loaded OSM data
        poses_latlon: Array of poses in lat/lon format
        output_file: Output filename for the plot
        num_segments: Number of segments to divide each road into
        proximity_threshold: Radius in meters for boundary circles
    """
    from shapely.geometry import LineString, Point
    
    # Get filtered roads near poses
    print(f"\n{'='*80}")
    print("Loading and Filtering Roads")
    print(f"{'='*80}")
    
    try:
        # Note: roads are stored as 'highways' in the OSM handler
        filtered_roads = osm_handler.filter_geometries_by_distance('highways', poses_latlon, use_centroid=False)
    except Exception as e:
        print(f"Error filtering roads: {e}")
        return
    
    if not filtered_roads or len(filtered_roads) == 0:
        print("No roads found near poses!")
        return
    
    print(f"Found {len(filtered_roads)} roads near poses")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # Plot robot poses for context
    print("\nPlotting robot poses...")
    ax.plot(poses_latlon[:, 1], poses_latlon[:, 0], 'g-', linewidth=1.5, 
           alpha=0.6, label='Robot Path', zorder=5)
    ax.plot(poses_latlon[0, 1], poses_latlon[0, 0], 'go', markersize=10, 
           label='Start', zorder=11)
    
    # Convert proximity threshold from meters to degrees (rough approximation)
    proximity_threshold_degrees = proximity_threshold / 111000.0
    
    print(f"\n{'='*80}")
    print("Processing and Visualizing Roads")
    print(f"{'='*80}")
    print(f"Number of segments per road: {num_segments}")
    print(f"Proximity threshold: {proximity_threshold}m ({proximity_threshold_degrees:.6f} degrees)")
    
    total_linestrings = 0
    total_segments = 0
    total_segment_points = 0
    
    # Process each road
    for road_idx, road in enumerate(filtered_roads):
        geom = road.geometry
        
        # Get road tags for debugging
        road_tags = ""
        if hasattr(road, 'tags'):
            road_tags = str(road.tags)
        
        print(f"\nRoad {road_idx + 1}/{len(filtered_roads)}:")
        print(f"  Geometry type: {geom.geom_type}")
        if road_tags:
            print(f"  Tags: {road_tags[:100]}...")  # Print first 100 chars
        
        # Handle different geometry types
        linestrings = []
        if geom.geom_type == 'LineString':
            linestrings = [geom]
        elif geom.geom_type == 'MultiLineString':
            linestrings = list(geom.geoms)
        else:
            print(f"  Skipping - unsupported geometry type: {geom.geom_type}")
            continue
        
        print(f"  Number of linestrings: {len(linestrings)}")
        total_linestrings += len(linestrings)
        
        # Process each linestring
        for ls_idx, linestring in enumerate(linestrings):
            coords = list(linestring.coords)
            
            if len(coords) < 2:
                print(f"  Linestring {ls_idx}: Too few coordinates ({len(coords)}), skipping")
                continue
            
            print(f"  Linestring {ls_idx}: {len(coords)} coordinate points")
            
            # Plot the full road linestring
            lons, lats = zip(*coords)
            ax.plot(lons, lats, 'b-', linewidth=2.0, alpha=0.7, zorder=3)
            
            # Segment each polyline edge (segment between consecutive vertices)
            print(f"  Linestring {ls_idx}: Processing {len(coords)-1} polyline edges")
            
            # Process each polyline edge (between consecutive vertices)
            for edge_idx in range(len(coords) - 1):
                p1 = coords[edge_idx]      # (lon, lat)
                p2 = coords[edge_idx + 1]  # (lon, lat)
                
                # Create a LineString for this edge
                edge_line = LineString([p1, p2])
                edge_length = edge_line.length
                
                if edge_length == 0:
                    continue
                
                # Subdivide this edge into num_segments sub-segments
                # This ensures we handle each polyline edge properly
                edge_segment_points = []
                for i in range(num_segments + 1):
                    fraction = i / num_segments
                    point = edge_line.interpolate(fraction, normalized=True)
                    edge_segment_points.append((point.x, point.y))  # (lon, lat)
                
                # Plot segment points with circles for this edge
                for seg_idx, (lon, lat) in enumerate(edge_segment_points):
                    # Plot segment point
                    ax.plot(lon, lat, 'ro', markersize=4, alpha=0.8, zorder=8)
                    
                    # Draw boundary circle
                    theta = np.linspace(0, 2*np.pi, 50)
                    circle_lons = lon + proximity_threshold_degrees * np.cos(theta)
                    circle_lats = lat + proximity_threshold_degrees * np.sin(theta)
                    ax.plot(circle_lons, circle_lats, 'r:', linewidth=1.0, alpha=0.4, zorder=2)
                    
                    total_segment_points += 1
                
                # Plot sub-segments within this edge
                for i in range(len(edge_segment_points) - 1):
                    p1_lon, p1_lat = edge_segment_points[i]
                    p2_lon, p2_lat = edge_segment_points[i + 1]
                    ax.plot([p1_lon, p2_lon], [p1_lat, p2_lat], 'orange', 
                           linewidth=1.5, alpha=0.6, zorder=4)
                    total_segments += 1
            
            print(f"  Linestring {ls_idx}: Created {len(coords)-1} edges with {num_segments} sub-segments each")
    
    print(f"\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}")
    print(f"Total roads processed: {len(filtered_roads)}")
    print(f"Total linestrings: {total_linestrings}")
    print(f"Total segment points: {total_segment_points}")
    print(f"Total segments: {total_segments}")
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=1.5, label='Robot Path'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Start'),
        Line2D([0], [0], color='blue', linewidth=2.0, label='Road Polylines'),
        Line2D([0], [0], color='orange', linewidth=1.5, label='Road Segments'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label='Segment Points'),
        Line2D([0], [0], color='red', linestyle=':', linewidth=1.0, 
               label=f'Proximity Circles ({proximity_threshold}m)')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'OSM Road Visualization ({len(filtered_roads)} roads, {total_segments} segments)', 
                fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {output_file}")
    plt.show()
    
    print("\nDone!")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize OSM roads with segments and boundary circles")
    parser.add_argument("--filter_distance", type=float, default=150.0,
                       help="Distance in meters to filter roads around poses (default: 150.0)")
    parser.add_argument("--num_segments", type=int, default=20,
                       help="Number of segments to divide each road into (default: 20)")
    parser.add_argument("--proximity_threshold", type=float, default=2.0,
                       help="Radius in meters for boundary circles (default: 2.0)")
    parser.add_argument("--output", type=str, default="road_visualization.png",
                       help="Output filename for the plot (default: road_visualization.png)")
    args = parser.parse_args()
    
    # Hardcoded paths
    dataset_path = "/media/donceykong/donceys_data_ssd/datasets/CU_MULTI/data"
    environment = "main_campus"
    robots = ["robot1", "robot2", "robot3", "robot4"]
    
    # Construct OSM file path
    osm_file = Path(dataset_path) / environment / f"{environment}.osm"
    
    # Check if OSM file exists
    if not osm_file.exists():
        print(f"Error: OSM file not found: {osm_file}")
        return
    
    # Load poses from all robots
    print(f"{'='*80}")
    print("Loading Poses from All Robots")
    print(f"{'='*80}")
    
    all_poses_latlon = []
    
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
        
        # Transform poses from IMU to LiDAR frame
        print(f"Transforming {robot} poses from IMU to LiDAR frame...")
        poses = transform_imu_to_lidar(poses)
        
        # Convert UTM poses to lat/lon
        print(f"Converting {robot} poses to lat/lon...")
        robot_poses_latlon, timestamps = utm_to_latlon(poses)
        print(f"Converted {len(robot_poses_latlon)} poses for {robot}")
        
        all_poses_latlon.append(robot_poses_latlon)
    
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
    print(f"  Latitude range: {poses_latlon[:, 0].min():.6f} to {poses_latlon[:, 0].max():.6f}")
    print(f"  Longitude range: {poses_latlon[:, 1].min():.6f} to {poses_latlon[:, 1].max():.6f}")
    
    # Load OSM data
    print(f"\n{'='*80}")
    print("Loading OSM Data")
    print(f"{'='*80}")
    print(f"Filter distance: {args.filter_distance}m")
    osm_handler = OSMDataHandler(osm_file, filter_distance=args.filter_distance)
    
    # Enable only roads for this visualization
    osm_handler.set_semantics({
        'roads': True,
        'highways': False,
        'buildings': False,
        'trees': False,
        'grassland': False,
        'water': False,
        'parking': False,
        'amenities': False
    })
    
    if not osm_handler.load_osm_data():
        print("Failed to load OSM data")
        return
    
    # Visualize roads
    visualize_roads(osm_handler, poses_latlon, 
                   output_file=args.output,
                   num_segments=args.num_segments,
                   proximity_threshold=args.proximity_threshold)


if __name__ == "__main__":
    main()

