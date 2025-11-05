#!/usr/bin/env python3
"""
Script to select and plot a single OSM building boundary near robot poses.
Modified to load semantic map data and filter points within circular boundary.
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


def load_semantic_map(npy_file):
    """
    Load semantic map from numpy file.
    
    Args:
        npy_file: Path to .npy file with columns [x, y, z, intensity, semantic_id]
    
    Returns:
        points_utm: (N, 3) array of UTM coordinates [x, y, z]
        intensities: (N,) array of intensities
        labels: (N,) array of semantic labels
    """
    print(f"\nLoading semantic map from {npy_file}")
    data = np.load(npy_file)
    print(f"Loaded data shape: {data.shape}")
    print(f"Data columns: [x, y, z, intensity, semantic_id]")
    
    points_utm = data[:, :3]  # x, y, z in UTM
    intensities = data[:, 3]
    labels = data[:, 4].astype(np.int32)
    
    print(f"  Points: {len(points_utm)}")
    print(f"  UTM X range: [{points_utm[:, 0].min():.2f}, {points_utm[:, 0].max():.2f}]")
    print(f"  UTM Y range: [{points_utm[:, 1].min():.2f}, {points_utm[:, 1].max():.2f}]")
    print(f"  UTM Z range: [{points_utm[:, 2].min():.2f}, {points_utm[:, 2].max():.2f}]")
    print(f"  Unique labels: {np.unique(labels)}")
    
    return points_utm, intensities, labels


def utm_points_to_latlon(points_utm):
    """
    Convert UTM point coordinates to lat/lon.
    
    Args:
        points_utm: (N, 3) array of UTM coordinates [x, y, z]
    
    Returns:
        points_latlon: (N, 3) array of [lat, lon, z]
    """
    from pyproj import Transformer
    
    print("\nConverting semantic points from UTM to lat/lon...")
    transformer = Transformer.from_crs("EPSG:32613", "EPSG:4326", always_xy=True)
    
    lons, lats = transformer.transform(points_utm[:, 0], points_utm[:, 1])
    points_latlon = np.column_stack([lats, lons, points_utm[:, 2]])
    
    print(f"  Latitude range: [{lats.min():.6f}, {lats.max():.6f}]")
    print(f"  Longitude range: [{lons.min():.6f}, {lons.max():.6f}]")
    
    return points_latlon


def filter_points_in_circle(points_latlon, center_lon, center_lat, radius_deg):
    """
    Filter points within a circular boundary.
    
    Args:
        points_latlon: (N, 3) array of [lat, lon, z]
        center_lon: Center longitude
        center_lat: Center latitude
        radius_deg: Radius in degrees
    
    Returns:
        mask: Boolean array indicating which points are inside the circle
    """
    # Calculate distance from center for each point
    distances = np.sqrt(
        (points_latlon[:, 1] - center_lon)**2 + 
        (points_latlon[:, 0] - center_lat)**2
    )
    
    mask = distances <= radius_deg
    
    print(f"\nFiltering points within circle:")
    print(f"  Center: ({center_lat:.6f}, {center_lon:.6f})")
    print(f"  Radius: {radius_deg:.6f} degrees")
    print(f"  Points inside: {mask.sum()} / {len(mask)} ({100*mask.sum()/len(mask):.2f}%)")
    
    return mask


def filter_points_in_polygon(points_latlon, polygon_coords):
    """
    Filter points within a polygon boundary using shapely.
    
    Args:
        points_latlon: (N, 3) array of [lat, lon, z]
        polygon_coords: List of (lon, lat) tuples defining the polygon
    
    Returns:
        mask: Boolean array indicating which points are inside the polygon
    """
    from shapely.geometry import Point, Polygon
    
    # Create polygon from coordinates
    polygon = Polygon(polygon_coords)
    
    # Check each point
    mask = np.zeros(len(points_latlon), dtype=bool)
    for i, point in enumerate(points_latlon):
        lat, lon = point[0], point[1]
        p = Point(lon, lat)
        mask[i] = polygon.contains(p)
    
    print(f"\nFiltering points within building polygon:")
    print(f"  Polygon vertices: {len(polygon_coords)}")
    print(f"  Points inside polygon: {mask.sum()} / {len(mask)} ({100*mask.sum()/len(mask):.2f}%)")
    
    return mask


def relabel_building_points_outside_circles(semantic_points, semantic_labels, circle_params,
                                           building_label_id=50, vegetation_label_id=70):
    """
    Relabel points marked as 'building' that fall outside all circular boundaries as 'vegetation'.
    
    Args:
        semantic_points: (N, 3) array of semantic point coordinates [lat, lon, z]
        semantic_labels: (N,) array of semantic labels (will be modified in place)
        circle_params: List of (center_lon, center_lat, radius) tuples
        building_label_id: Label ID for buildings (default: 50)
        vegetation_label_id: Label ID for vegetation (default: 70)
    
    Returns:
        Number of points relabeled
    """
    # Find all points labeled as building
    building_mask = semantic_labels == building_label_id
    building_points = semantic_points[building_mask]
    
    if len(building_points) == 0:
        print("No points labeled as building (ID 50) found for circle filtering")
        return 0
    
    print(f"\nRelabeling building points outside circular boundaries...")
    print(f"  Total points labeled as building: {len(building_points)}")
    
    # Create a mask to track which building points are inside any circle
    inside_any_circle = np.zeros(len(building_points), dtype=bool)
    
    # Check each circle
    for center_lon, center_lat, radius in circle_params:
        # Calculate distance from center for each building point
        distances = np.sqrt(
            (building_points[:, 1] - center_lon)**2 + 
            (building_points[:, 0] - center_lat)**2
        )
        
        # Mark points inside this circle
        inside_this_circle = distances <= radius
        inside_any_circle |= inside_this_circle
    
    # Points that are NOT inside any circle should be relabeled
    points_to_relabel = ~inside_any_circle
    num_relabeled = points_to_relabel.sum()
    
    # Get the original indices of building points in the full semantic_labels array
    building_indices = np.where(building_mask)[0]
    indices_to_relabel = building_indices[points_to_relabel]
    
    # Relabel these points as vegetation
    semantic_labels[indices_to_relabel] = vegetation_label_id
    
    print(f"  Points inside circular boundaries: {inside_any_circle.sum()}")
    print(f"  Points relabeled as vegetation (ID 70): {num_relabeled}")
    
    return num_relabeled


def relabel_building_points_outside_polygons(semantic_points, semantic_labels, filtered_buildings, 
                                             building_label_id=50, vegetation_label_id=70):
    """
    Relabel points marked as 'building' that fall outside all building polygons as 'vegetation'.
    
    Args:
        semantic_points: (N, 3) array of semantic point coordinates [lat, lon, z]
        semantic_labels: (N,) array of semantic labels (will be modified in place)
        filtered_buildings: List of building geometries from OSM
        building_label_id: Label ID for buildings (default: 50)
        vegetation_label_id: Label ID for vegetation (default: 70)
    
    Returns:
        Number of points relabeled
    """
    from shapely.geometry import Point, Polygon
    
    # Find all points labeled as building
    building_mask = semantic_labels == building_label_id
    building_points = semantic_points[building_mask]
    
    if len(building_points) == 0:
        print("No points labeled as building (ID 50) found for polygon filtering")
        return 0
    
    print(f"\nRelabeling building points outside building polygons (but inside circles)...")
    print(f"  Total points labeled as building: {len(building_points)}")
    
    # Create a mask to track which building points are inside any polygon
    inside_any_polygon = np.zeros(len(building_points), dtype=bool)
    
    # Check each building polygon
    for idx, building in enumerate(filtered_buildings):
        if building.geometry.geom_type != "Polygon":
            continue
        
        coords = list(building.geometry.exterior.coords)
        polygon = Polygon(coords)
        
        # Check each building point
        for i, point in enumerate(building_points):
            if inside_any_polygon[i]:
                continue  # Already marked as inside a polygon
            
            lat, lon = point[0], point[1]
            p = Point(lon, lat)
            if polygon.contains(p):
                inside_any_polygon[i] = True
    
    # Points that are NOT inside any polygon should be relabeled
    points_to_relabel = ~inside_any_polygon
    num_relabeled = points_to_relabel.sum()
    
    # Get the original indices of building points in the full semantic_labels array
    building_indices = np.where(building_mask)[0]
    indices_to_relabel = building_indices[points_to_relabel]
    
    # Relabel these points as vegetation
    semantic_labels[indices_to_relabel] = vegetation_label_id
    
    print(f"  Points inside building polygons: {inside_any_polygon.sum()}")
    print(f"  Points relabeled as vegetation (ID 70): {num_relabeled}")
    
    return num_relabeled


def relabel_points_inside_polygons_to_building(semantic_points, semantic_labels, filtered_buildings, 
                                               building_label_id=50):
    """
    Relabel ALL points within building polygons to building (ID 50), regardless of their original semantic ID.
    
    Args:
        semantic_points: (N, 3) array of semantic point coordinates [lat, lon, z]
        semantic_labels: (N,) array of semantic labels (will be modified in place)
        filtered_buildings: List of building geometries from OSM
        building_label_id: Label ID for buildings (default: 50)
    
    Returns:
        Number of points relabeled to building
    """
    from shapely.geometry import Point, Polygon
    
    print(f"\nRelabeling ALL points inside building polygons to building (ID {building_label_id})...")
    print(f"  Total semantic points: {len(semantic_points)}")
    
    # Track which points are inside any polygon
    inside_any_polygon = np.zeros(len(semantic_points), dtype=bool)
    
    # Check each building polygon
    for idx, building in enumerate(filtered_buildings):
        if building.geometry.geom_type != "Polygon":
            continue
        
        coords = list(building.geometry.exterior.coords)
        polygon = Polygon(coords)
        
        print(f"  Processing building {idx+1}/{len(filtered_buildings)}...")
        
        # Check each point (only those not already marked as inside a polygon)
        points_in_this_building = 0
        for i in range(len(semantic_points)):
            if inside_any_polygon[i]:
                continue  # Already marked as inside a polygon
            
            lat, lon = semantic_points[i, 0], semantic_points[i, 1]
            p = Point(lon, lat)
            if polygon.contains(p):
                inside_any_polygon[i] = True
                points_in_this_building += 1
        
        print(f"    Found {points_in_this_building} points inside this building")
    
    # Get indices of all points inside any polygon
    indices_to_relabel = np.where(inside_any_polygon)[0]
    
    # Count how many were already building vs relabeled
    already_building = (semantic_labels[indices_to_relabel] == building_label_id).sum()
    num_relabeled = len(indices_to_relabel) - already_building
    
    # Relabel all points inside polygons to building
    semantic_labels[indices_to_relabel] = building_label_id
    
    print(f"  Total points inside building polygons: {len(indices_to_relabel)}")
    print(f"  Points already labeled as building: {already_building}")
    print(f"  Points relabeled to building (ID {building_label_id}): {num_relabeled}")
    
    return num_relabeled


def create_road_proximity_mask(semantic_points, filtered_roads, proximity_threshold=2.0):
    """
    Create a mask for points within proximity_threshold meters of any road segment.
    
    Uses KD-tree for efficient spatial queries. Roads are segmented into multiple points,
    and a KD-tree is built to quickly find which semantic points are near any road segment.
    
    Args:
        semantic_points: (N, 3) array of semantic point coordinates [lat, lon, z]
        filtered_roads: List of road geometries from OSM
        proximity_threshold: Distance threshold in meters (default: 2.0)
    
    Returns:
        Boolean mask array: True for points near roads (to be protected)
    """
    from shapely.geometry import LineString
    from scipy.spatial import cKDTree
    from tqdm import tqdm
    
    print(f"\n{'='*60}")
    print("Creating Road Proximity Mask (KD-Tree Method)")
    print(f"{'='*60}")
    print(f"  Proximity threshold: {proximity_threshold}m")
    print(f"  Total points to check: {len(semantic_points)}")
    
    if not filtered_roads or len(filtered_roads) == 0:
        print("  No roads found - skipping road masking")
        return np.zeros(len(semantic_points), dtype=bool)
    
    # Initialize mask (False = not near road, True = near road)
    near_road_mask = np.zeros(len(semantic_points), dtype=bool)
    
    # Convert threshold from meters to degrees (approximate)
    # 1 degree ≈ 111 km at equator
    proximity_threshold_degrees = proximity_threshold / 111000.0
    
    # Step 1: Extract all road segment points (lat, lon) from all roads
    print(f"  Extracting segment points from {len(filtered_roads)} roads...")
    road_segment_points = []  # Will store (lon, lat) pairs
    num_segments = 20  # Number of sub-segments per polyline edge
    
    for road in tqdm(filtered_roads, desc="  Processing roads", leave=False):
        geom = road.geometry
        
        # Handle different geometry types
        if geom.geom_type == 'LineString':
            linestrings = [geom]
        elif geom.geom_type == 'MultiLineString':
            linestrings = list(geom.geoms)
        else:
            continue
        
        for linestring in linestrings:
            coords = list(linestring.coords)
            
            if len(coords) < 2:
                continue
            
            # Process each polyline edge (segment between consecutive vertices)
            for edge_idx in range(len(coords) - 1):
                p1 = coords[edge_idx]      # (lon, lat)
                p2 = coords[edge_idx + 1]  # (lon, lat)
                
                # Create a LineString for this edge
                edge_line = LineString([p1, p2])
                edge_length = edge_line.length
                
                if edge_length == 0:
                    continue
                
                # Subdivide this edge into num_segments sub-segments
                for i in range(num_segments + 1):
                    fraction = i / num_segments
                    point = edge_line.interpolate(fraction, normalized=True)
                    # Store as (lon, lat) for consistency with semantic points indexing
                    road_segment_points.append([point.x, point.y])
    
    if len(road_segment_points) == 0:
        print("  No valid road segments found - skipping road masking")
        return np.zeros(len(semantic_points), dtype=bool)
    
    road_segment_points = np.array(road_segment_points)  # Shape: (M, 2) where M is total segment points
    print(f"  Total road segment points extracted: {len(road_segment_points)}")
    
    # Step 2: Build KD-tree from road segment points
    print("  Building KD-tree from road segment points...")
    kdtree = cKDTree(road_segment_points)
    
    # Step 3: Query KD-tree for each semantic point
    print("  Querying KD-tree for semantic points...")
    semantic_points_2d = semantic_points[:, [1, 0]]  # Extract (lon, lat) to match road points
    
    print(f"  Semantic points 2D shape: {semantic_points_2d.shape}")
    print(f"  Road segment points shape: {road_segment_points.shape}")
    print(f"  Proximity threshold (degrees): {proximity_threshold_degrees:.8f}")
    
    # Query all points at once - much faster than looping
    # query_ball_point returns list of indices for all neighbors within radius
    neighbors = kdtree.query_ball_point(semantic_points_2d, r=proximity_threshold_degrees)
    
    # Mark points that have at least one nearby road segment
    # Vectorized approach: check which points have neighbors
    near_road_mask = np.array([len(n) > 0 for n in neighbors], dtype=bool)
    
    num_masked = near_road_mask.sum()
    print(f"  Points near roads (masked): {num_masked} ({100*num_masked/len(semantic_points):.2f}%)")
    
    return near_road_mask


def relabel_points_inside_polygons_to_terrain(semantic_points, semantic_labels, filtered_polygons, 
                                              polygon_type="terrain area",
                                              terrain_label_id=72, building_label_id=50, 
                                              vegetation_label_id=70, max_height_above_ground=2.0,
                                              additional_mask=None):
    """
    Relabel points within terrain polygons (grassland/gardens) to terrain (ID 72), with constraints.
    
    Constraints:
    - Does NOT relabel points already labeled as building (ID 50) or vegetation (ID 70)
    - Does NOT relabel points more than max_height_above_ground meters above the lowest point
    - Does NOT relabel points in additional_mask (e.g., points near roads)
    
    Args:
        semantic_points: (N, 3) array of semantic point coordinates [lat, lon, z]
        semantic_labels: (N,) array of semantic labels (will be modified in place)
        filtered_polygons: List of terrain polygon geometries from OSM (grasslands or gardens)
        polygon_type: String describing the polygon type for logging (default: "terrain area")
        terrain_label_id: Label ID for terrain (default: 72)
        building_label_id: Label ID for buildings to mask out (default: 50)
        vegetation_label_id: Label ID for vegetation to mask out (default: 70)
        max_height_above_ground: Maximum height above lowest point to relabel (default: 2.0 meters)
        additional_mask: Optional boolean mask for additional points to protect (e.g., near roads)
    
    Returns:
        Number of points relabeled to terrain
    """
    from shapely.geometry import Point, Polygon
    
    print(f"\nRelabeling points inside {polygon_type} polygons to terrain (ID {terrain_label_id})...")
    print(f"  Total semantic points: {len(semantic_points)}")
    print(f"  Constraints:")
    print(f"    - Masking out building points (ID {building_label_id})")
    print(f"    - Masking out vegetation points (ID {vegetation_label_id})")
    print(f"    - Only relabeling points within {max_height_above_ground}m of lowest point")
    
    # Create mask for points that should NOT be relabeled
    protected_mask = (semantic_labels == building_label_id) | (semantic_labels == vegetation_label_id)
    
    # Add additional mask (e.g., points near roads)
    if additional_mask is not None:
        protected_mask = protected_mask | additional_mask
        print(f"    - Masking out points from additional mask (e.g., near roads)")
    
    num_protected = protected_mask.sum()
    print(f"  Protected points (building, vegetation, or additional): {num_protected}")
    
    # Track which points are inside any polygon
    inside_any_polygon = np.zeros(len(semantic_points), dtype=bool)
    
    # Check each polygon
    for idx, polygon_feature in enumerate(filtered_polygons):
        if polygon_feature.geometry.geom_type != "Polygon":
            continue
        
        coords = list(polygon_feature.geometry.exterior.coords)
        polygon = Polygon(coords)
        
        print(f"  Processing {polygon_type} {idx+1}/{len(filtered_polygons)}...")
        
        # Check each point (only those not already marked as inside a polygon)
        points_in_this_polygon = 0
        for i in range(len(semantic_points)):
            if inside_any_polygon[i] or protected_mask[i]:
                continue  # Already marked or protected
            
            lat, lon = semantic_points[i, 0], semantic_points[i, 1]
            p = Point(lon, lat)
            if polygon.contains(p):
                inside_any_polygon[i] = True
                points_in_this_polygon += 1
        
        print(f"    Found {points_in_this_polygon} points inside this {polygon_type} (excluding protected)")
    
    # Get indices of all points inside any polygon (excluding protected ones)
    candidate_indices = np.where(inside_any_polygon)[0]
    
    if len(candidate_indices) == 0:
        print(f"  No candidate points found for relabeling")
        return 0
    
    # Apply height constraint: only relabel points within max_height_above_ground of lowest point
    candidate_heights = semantic_points[candidate_indices, 2]
    min_height = candidate_heights.min()
    height_mask = (candidate_heights - min_height) <= 100 #max_height_above_ground
    
    print(f"  Height filtering:")
    print(f"    Minimum height: {min_height:.2f}m")
    print(f"    Maximum allowed height: {min_height + max_height_above_ground:.2f}m")
    print(f"    Points within height threshold: {height_mask.sum()} / {len(candidate_indices)}")
    
    # Get final indices to relabel (inside polygon AND within height threshold)
    indices_to_relabel = candidate_indices[height_mask]
    
    # Count how many were already terrain vs relabeled
    already_terrain = (semantic_labels[indices_to_relabel] == terrain_label_id).sum()
    num_relabeled = len(indices_to_relabel) - already_terrain
    
    # Relabel selected points to terrain
    semantic_labels[indices_to_relabel] = terrain_label_id
    
    print(f"  Total points relabeled to terrain: {len(indices_to_relabel)}")
    print(f"  Points already labeled as terrain: {already_terrain}")
    print(f"  Points newly relabeled to terrain (ID {terrain_label_id}): {num_relabeled}")
    
    return num_relabeled


def save_relabeled_semantic_map(points_utm, intensities, labels, output_file):
    """
    Save relabeled semantic map to .npy file.
    
    Args:
        points_utm: (N, 3) array of UTM coordinates [x, y, z]
        intensities: (N,) array of intensities
        labels: (N,) array of semantic labels
        output_file: Output filename for the .npy file
    """
    print(f"\n{'='*60}")
    print("Saving Relabeled Semantic Map")
    print(f"{'='*60}")
    print(f"  Output file: {output_file}")
    print(f"  Total points: {len(points_utm)}")
    
    # Combine data into single array: [x, y, z, intensity, semantic_id]
    data = np.column_stack([points_utm, intensities, labels])
    
    print(f"  Data shape: {data.shape}")
    print(f"  Columns: [x, y, z, intensity, semantic_id]")
    
    # Save to file
    np.save(output_file, data)
    print(f"  Successfully saved to {output_file}")
    
    # Print label statistics
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\n  Label distribution in saved file:")
    for label_id, count in zip(unique_labels, counts):
        print(f"    ID {label_id}: {count} points ({100*count/len(labels):.1f}%)")


def plot_selected_building(osm_handler, poses_latlon, building_idx=0, output_file="selected_building.png",
                          semantic_points=None, semantic_labels=None, semantic_intensities=None,
                          process_all_buildings=False, relabel_buildings=True, relabel_grasslands=True, 
                          all_robot_poses=None):
    """
    Plot a selected OSM building near the robot poses with optional semantic points.
    
    Args:
        osm_handler: OSMDataHandler instance with loaded OSM data
        poses_latlon: Array of poses in lat/lon format (combined from all robots)
        building_idx: Index of building to plot (0 = first building) - ignored if process_all_buildings=True
        output_file: Output filename for the plot
        semantic_points: Optional (N, 3) array of semantic point coordinates [lat, lon, z]
        semantic_labels: Optional (N,) array of semantic labels (modified in place if relabeling enabled)
        semantic_intensities: Optional (N,) array of intensities
        process_all_buildings: If True, process all buildings instead of just one
        relabel_buildings: If True, relabel points inside/outside building polygons
        relabel_grasslands: If True, relabel points inside grassland polygons to terrain
        all_robot_poses: Optional list of (poses_latlon, robot_name) tuples for plotting individual robot paths
    
    Returns:
        Circle parameters list [(center_lon, center_lat, radius), ...] if successful, None otherwise
    """
    # Get filtered buildings near poses
    filtered_buildings = osm_handler.filter_geometries_by_distance('buildings', poses_latlon, use_centroid=False)
    
    if not filtered_buildings:
        print("No buildings found near poses!")
        return None
    
    print(f"\nFound {len(filtered_buildings)} buildings near poses")
    
    # Determine which buildings to process
    if process_all_buildings:
        buildings_to_process = list(enumerate(filtered_buildings))
        print(f"Processing all {len(buildings_to_process)} buildings")
    else:
        # Select single building (default to first one)
        if building_idx >= len(filtered_buildings):
            print(f"Building index {building_idx} out of range. Using building 0.")
            building_idx = 0
        buildings_to_process = [(building_idx, filtered_buildings[building_idx])]
        print(f"Processing single building {building_idx}")
    
    # Create plot
    plt.figure(figsize=(16, 12))
    
    # Store circle parameters for all buildings
    all_circle_params = []
    
    # Track all filtered points across all buildings
    all_points_in_buildings = []
    all_points_in_circles = []
    all_labels_in_circles = []
    
    # Process each building
    for idx, selected_building in buildings_to_process:
        print(f"\nProcessing building {idx}:")
        print(f"  Geometry type: {selected_building.geometry.geom_type}")
        
        # Extract building coordinates
        if selected_building.geometry.geom_type != "Polygon":
            print(f"  Skipping - geometry type {selected_building.geometry.geom_type} not supported")
            continue
            
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
        
        # Calculate center point and radius
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        radius = np.sqrt((x_max - x_center)**2 + (y_max - y_center)**2)
        
        print(f"  Center point: ({y_center:.6f}, {x_center:.6f})")
        print(f"  Circle radius: {radius:.6f} degrees")
        
        # Store circle parameters
        all_circle_params.append((x_center, y_center, radius))
        
        # Plot building boundary
        plt.plot(lons, lats, 'b-', linewidth=1.5, alpha=0.8, zorder=2)
        plt.fill(lons, lats, color='blue', alpha=0.2, zorder=1)
        
        # Plot building centroid (smaller for multiple buildings)
        marker_size = 6 if process_all_buildings else 10
        plt.plot(centroid.x, centroid.y, 'ro', markersize=marker_size, zorder=10)
        
        # Draw dotted circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = x_center + radius * np.cos(theta)
        circle_y = y_center + radius * np.sin(theta)
        plt.plot(circle_x, circle_y, 'r:', linewidth=1.5, alpha=0.7, zorder=9)
        
        # Filter semantic points if provided
        if semantic_points is not None and semantic_labels is not None:
            print(f"  Filtering semantic points for this building...")
            
            # Step 1: Filter points within circle
            circle_mask = filter_points_in_circle(semantic_points, x_center, y_center, radius)
            circle_filtered_points = semantic_points[circle_mask]
            circle_filtered_labels = semantic_labels[circle_mask]
            
            if len(circle_filtered_points) > 0:
                # Step 2: From circle-filtered points, find which are in the polygon
                polygon_mask = filter_points_in_polygon(circle_filtered_points, coords)
                
                # Separate points into two groups
                points_in_polygon = circle_filtered_points[polygon_mask]
                points_outside_polygon = circle_filtered_points[~polygon_mask]
                labels_outside_polygon = circle_filtered_labels[~polygon_mask]
                
                # Accumulate points for plotting later
                if len(points_in_polygon) > 0:
                    all_points_in_buildings.append(points_in_polygon)
                    print(f"  Found {len(points_in_polygon)} points INSIDE building polygon")
                
                if len(points_outside_polygon) > 0:
                    all_points_in_circles.append(points_outside_polygon)
                    all_labels_in_circles.append(labels_outside_polygon)
                    print(f"  Found {len(points_outside_polygon)} points in circle but OUTSIDE building")
    
    # Process grassland and garden polygons if enabled
    all_grassland_circle_params = []
    all_garden_circle_params = []
    filtered_grasslands = []
    filtered_gardens = []
    
    if relabel_grasslands:
        print(f"\n{'='*60}")
        print("Processing Grassland & Garden Polygons")
        print(f"{'='*60}")
        
        # Get filtered grasslands near poses
        try:
            filtered_grasslands = osm_handler.filter_geometries_by_distance('grassland', poses_latlon, use_centroid=False)
        except:
            filtered_grasslands = []
        
        # Get filtered gardens near poses (leisure=garden)
        try:
            filtered_gardens = osm_handler.filter_geometries_by_distance('gardens', poses_latlon, use_centroid=False)
        except:
            filtered_gardens = []
        
        # Combine both for processing
        all_terrain_features = []
        if filtered_grasslands:
            print(f"\nFound {len(filtered_grasslands)} grassland areas near poses")
            all_terrain_features.extend([(feat, "grassland", "green") for feat in filtered_grasslands])
        else:
            print("No grassland areas found near poses")
        
        if filtered_gardens:
            print(f"Found {len(filtered_gardens)} garden areas near poses")
            all_terrain_features.extend([(feat, "garden", "limegreen") for feat in filtered_gardens])
        else:
            print("No garden areas found near poses")
        
        if all_terrain_features:
            # Process each terrain polygon (grassland or garden)
            for idx, (feature, feature_type, color) in enumerate(all_terrain_features):
                print(f"\nProcessing {feature_type} {idx}:")
                print(f"  Geometry type: {feature.geometry.geom_type}")
                
                # Extract coordinates
                if feature.geometry.geom_type != "Polygon":
                    print(f"  Skipping - geometry type {feature.geometry.geom_type} not supported")
                    continue
                
                coords = list(feature.geometry.exterior.coords)
                lons, lats = zip(*coords)
                
                # Calculate centroid
                centroid = feature.geometry.centroid
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
                
                # Calculate center point and radius
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                radius = np.sqrt((x_max - x_center)**2 + (y_max - y_center)**2)
                
                print(f"  Center point: ({y_center:.6f}, {x_center:.6f})")
                print(f"  Circle radius: {radius:.6f} degrees")
                
                # Store circle parameters
                if feature_type == "grassland":
                    all_grassland_circle_params.append((x_center, y_center, radius))
                else:  # garden
                    all_garden_circle_params.append((x_center, y_center, radius))
                
                # Plot boundary with feature-specific color
                plt.plot(lons, lats, color=color, linewidth=1.5, alpha=0.8, zorder=2)
                plt.fill(lons, lats, color=color, alpha=0.15, zorder=1)
                
                # Plot centroid (smaller marker)
                plt.plot(centroid.x, centroid.y, 'o', color=color, markersize=4, alpha=0.7, zorder=10)
                
                # Draw dotted circle
                theta = np.linspace(0, 2*np.pi, 100)
                circle_x = x_center + radius * np.cos(theta)
                circle_y = y_center + radius * np.sin(theta)
                plt.plot(circle_x, circle_y, ':', color=color, linewidth=1.5, alpha=0.5, zorder=9)
        else:
            print("\nNo grassland or garden areas found near poses!")
    
    # Optionally plot roads for visualization
    if relabel_grasslands:
        print(f"\n{'='*60}")
        print("Processing Roads for Visualization")
        print(f"{'='*60}")
        
        try:
            # Note: roads are stored as 'highways' in the OSM handler
            filtered_roads = osm_handler.filter_geometries_by_distance('highways', poses_latlon, use_centroid=False)
            if filtered_roads and len(filtered_roads) > 0:
                print(f"\nFound {len(filtered_roads)} roads near poses")
                print("Plotting roads (edges only)...")
                
                roads_plotted = 0
                for road in filtered_roads:
                    geom = road.geometry
                    
                    # Handle different geometry types
                    if geom.geom_type == 'LineString':
                        linestrings = [geom]
                    elif geom.geom_type == 'MultiLineString':
                        linestrings = list(geom.geoms)
                    else:
                        continue
                    
                    for linestring in linestrings:
                        coords = list(linestring.coords)
                        if len(coords) >= 2:
                            # Plot the road polyline edges (just the lines, no circles)
                            lons, lats = zip(*coords)
                            plt.plot(lons, lats, 'dimgray', linewidth=1.2, alpha=0.5, zorder=2)
                            roads_plotted += 1
                
                print(f"  Plotted {roads_plotted} road segments (edges only, no bounding circles)")
            else:
                print("No roads found near poses")
        except Exception as e:
            print(f"Warning: Could not plot roads: {e}")
    
    # After processing all buildings and grasslands, apply relabeling if enabled
    if semantic_points is not None and semantic_labels is not None:
        print(f"\n{'='*60}")
        print("Applying OSM-based relabeling...")
        print(f"{'='*60}")
        
        # Building relabeling
        if relabel_buildings and all_circle_params:
            # Step 1: Relabel ALL points inside building polygons to building (ID 50)
            # This ensures everything inside actual OSM building footprints is labeled as building
            relabel_points_inside_polygons_to_building(semantic_points, semantic_labels, filtered_buildings)
            
            # Step 2: Relabel building points outside ALL circular boundaries to vegetation
            # This removes false positive buildings far from any OSM building
            relabel_building_points_outside_circles(semantic_points, semantic_labels, all_circle_params)
            
            # Step 3: Relabel building points inside circles but outside building polygons to vegetation
            # This removes false positive buildings near OSM buildings but not inside the actual footprints
            relabel_building_points_outside_polygons(semantic_points, semantic_labels, filtered_buildings)
        
        # Road masking (before terrain relabeling)
        road_proximity_mask = None
        if relabel_grasslands:
            # Get filtered roads near poses
            try:
                # Note: roads are stored as 'highways' in the OSM handler
                filtered_roads = osm_handler.filter_geometries_by_distance('highways', poses_latlon, use_centroid=False)
                if filtered_roads and len(filtered_roads) > 0:
                    # Create mask for points within 2 meters of road segments
                    road_proximity_mask = create_road_proximity_mask(semantic_points, filtered_roads, 
                                                                     proximity_threshold=2.5)
            except Exception as e:
                print(f"\nWarning: Could not process roads: {e}")
                road_proximity_mask = None
        
        # Grassland and garden relabeling
        if relabel_grasslands:
            # Step 4: Relabel ALL points inside grassland polygons to terrain (ID 72)
            # This ensures everything inside actual OSM grassland areas is labeled as terrain
            # Points near roads are masked out
            if all_grassland_circle_params and filtered_grasslands:
                relabel_points_inside_polygons_to_terrain(semantic_points, semantic_labels, 
                                                         filtered_grasslands, polygon_type="grassland",
                                                         additional_mask=road_proximity_mask)
            
            # Step 5: Relabel ALL points inside garden polygons (leisure=garden) to terrain (ID 72)
            # This ensures everything inside actual OSM garden areas is labeled as terrain
            # Points near roads are masked out
            if all_garden_circle_params and filtered_gardens:
                relabel_points_inside_polygons_to_terrain(semantic_points, semantic_labels, 
                                                         filtered_gardens, polygon_type="garden",
                                                         additional_mask=road_proximity_mask)
    
    # After processing all buildings, plot ALL semantic points to show relabeling effect
    print(f"\n{'='*60}")
    print("Plotting all semantic points...")
    print(f"{'='*60}")
    
    if semantic_points is not None and semantic_labels is not None:
        # Get all semantic colors based on current labels (after relabeling)
        labels_dict = {label.id: label.color for label in sem_kitti_labels}
        all_semantic_colors = labels2RGB(semantic_labels, labels_dict)
        
        # Calculate alpha based on Z height for better visualization
        z_values = semantic_points[:, 2]
        normalized_z = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values))
        alpha_values = 0.2 + 0.6 * normalized_z  # Alpha between 0.2 and 0.8 based on height
        alpha_values = np.clip(alpha_values, 0.2, 0.8)
        
        # Plot all semantic points
        plt.scatter(semantic_points[:, 1], semantic_points[:, 0], 
                   c=all_semantic_colors, s=0.3, alpha=alpha_values, zorder=3,
                   label=f'All Semantic Points ({len(semantic_points)})')
        print(f"Plotted {len(semantic_points)} semantic points with updated labels")
        
        # Print label statistics
        unique_labels, counts = np.unique(semantic_labels, return_counts=True)
        print(f"\nLabel distribution after relabeling:")
        for label_id, count in zip(unique_labels, counts):
            label_name = "Unknown"
            for label in sem_kitti_labels:
                if label.id == label_id:
                    label_name = label.name
                    break
            print(f"  {label_name} (ID {label_id}): {count} points ({100*count/len(semantic_labels):.1f}%)")
    
    # Plot robot poses for context
    if all_robot_poses is not None and len(all_robot_poses) > 0:
        # Plot individual robot paths with different colors
        colors = ['green', 'cyan', 'magenta', 'orange']
        for i, (robot_poses, robot_name) in enumerate(all_robot_poses):
            color = colors[i % len(colors)]
            plt.plot(robot_poses[:, 1], robot_poses[:, 0], '-', color=color, 
                    linewidth=1, alpha=0.6, label=f'{robot_name} Path', zorder=2)
            # Mark start of each robot
            plt.plot(robot_poses[0, 1], robot_poses[0, 0], 'o', color=color, 
                    markersize=6, alpha=0.8, zorder=11)
    else:
        # Plot combined path
        plt.plot(poses_latlon[:, 1], poses_latlon[:, 0], 'g-', linewidth=1, 
                alpha=0.5, label='Combined Robot Paths', zorder=2)
        plt.plot(poses_latlon[0, 1], poses_latlon[0, 0], 'go', markersize=8, 
                label='Start', zorder=11)
        plt.plot(poses_latlon[-1, 1], poses_latlon[-1, 0], 'gs', markersize=8, 
                label='End', zorder=11)
    
    # Finalize plot
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    if process_all_buildings:
        plt.title(f'All {len(all_circle_params)} Buildings with Semantic Points')
    else:
        plt.title(f'Building {building_idx} with Semantic Points')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")
    plt.close()
    
    return all_circle_params if all_circle_params else None


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Select and plot an OSM building boundary with semantic points")
    parser.add_argument("--building_idx", type=int, default=0,
                       help="Index of building to plot (default: 0, ignored if --all_buildings is used)")
    parser.add_argument("--all_buildings", action="store_true",
                       help="Process all buildings within filter distance instead of just one")
    parser.add_argument("--filter_distance", type=float, default=100.0,
                       help="Distance in meters to filter buildings around poses (default: 100.0)")
    parser.add_argument("--output", type=str, default="KL_SEM_MAP_OSM_RELABELED.png",
                       help="Output filename for the plot (default: selected_building_with_semantics.png)")
    parser.add_argument("--semantic_map", type=str, default="KL_SEM_MAP_OG_all_robots_utm.npy",
                       help="Path to semantic map .npy file (default: KL_SEM_MAP_OG_all_robots_utm.npy)")
    parser.add_argument("--relabel_buildings", action="store_true", default=True,
                       help="Relabel building points (ID 50) outside polygons as vegetation (ID 70) (default: True)")
    parser.add_argument("--no_relabel_buildings", dest="relabel_buildings", action="store_false",
                       help="Disable relabeling of building points outside polygons")
    parser.add_argument("--relabel_grasslands", action="store_true", default=True,
                       help="Relabel points inside grassland polygons to terrain (ID 72) (default: True)")
    parser.add_argument("--no_relabel_grasslands", dest="relabel_grasslands", action="store_false",
                       help="Disable relabeling of points inside grassland polygons")
    args = parser.parse_args()
    
    # Hardcoded paths
    dataset_path = "/media/donceykong/donceys_data_ssd/datasets/CU_MULTI/data"
    environment = "kittredge_loop"
    robots = ["robot1"] #, "robot2", "robot3", "robot4"]
    
    # Construct file paths
    osm_file = Path(dataset_path) / environment / f"{environment}.osm"
    semantic_map_file = Path(args.semantic_map)
    
    # Check if OSM file exists
    if not osm_file.exists():
        print(f"Error: OSM file not found: {osm_file}")
        return
    
    # Load semantic map if provided
    semantic_points_latlon = None
    semantic_labels = None
    semantic_intensities = None
    points_utm = None  # Keep UTM coordinates for saving later
    
    if semantic_map_file.exists():
        print(f"\n{'='*80}")
        print("Loading Semantic Map")
        print(f"{'='*80}")
        
        # Load semantic map (in UTM)
        points_utm, semantic_intensities, semantic_labels = load_semantic_map(semantic_map_file)
        
        # Convert to lat/lon
        semantic_points_latlon = utm_points_to_latlon(points_utm)
    else:
        print(f"\nWarning: Semantic map file not found: {semantic_map_file}")
        print("Continuing without semantic points...")
    
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
    
    # Enable buildings, grassland, gardens, and roads semantics
    osm_handler.set_semantics({
        'roads': True,      # Enable roads for masking
        'highways': False,
        'buildings': True,
        'trees': False,
        'grassland': True,  # Enable grassland
        'gardens': True,    # Enable gardens (leisure=garden)
        'water': False,
        'parking': False,
        'amenities': False
    })
    
    if not osm_handler.load_osm_data():
        print("Failed to load OSM data")
        return
    
    # Prepare individual robot poses for visualization
    robot_poses_list = [(robot_poses, robot_name) 
                        for robot_poses, robot_name in zip(all_poses_latlon, all_robot_names)]
    
    # Plot selected building(s) with semantic points
    print(f"\n{'='*80}")
    if args.all_buildings:
        print(f"Plotting All Buildings")
    else:
        print(f"Plotting Building {args.building_idx}")
    print(f"Building relabeling enabled: {args.relabel_buildings}")
    print(f"Grassland relabeling enabled: {args.relabel_grasslands}")
    print(f"{'='*80}")
    result = plot_selected_building(
        osm_handler, poses_latlon, 
        building_idx=args.building_idx,
        output_file=args.output,
        semantic_points=semantic_points_latlon,
        semantic_labels=semantic_labels,
        semantic_intensities=semantic_intensities,
        process_all_buildings=args.all_buildings,
        relabel_buildings=args.relabel_buildings,
        relabel_grasslands=args.relabel_grasslands,
        all_robot_poses=robot_poses_list
    )
    
    if result:
        print(f"\n{'='*80}")
        print("SUCCESS!")
        print(f"{'='*80}")
        print(f"Output saved to: {args.output}")
        
        # Save relabeled semantic map if we have the data and relabeling was performed
        if points_utm is not None and semantic_labels is not None and semantic_intensities is not None:
            if args.relabel_buildings or args.relabel_grasslands:
                output_npy = "MC_SEM_MAP_RELABELED.npy"
                save_relabeled_semantic_map(points_utm, semantic_intensities, semantic_labels, output_npy)
    else:
        print("\nFailed to plot building")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

