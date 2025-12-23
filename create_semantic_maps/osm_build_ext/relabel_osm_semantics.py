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


def load_building_offsets(input_file):
    """
    Load building offset data from a .npy file.
    
    Args:
        input_file: Path to input .npy file
    
    Returns:
        List of dictionaries with 'inner', 'original', 'outer' keys
    """
    print(f"\nLoading building offset data from {input_file}...")
    
    if not Path(input_file).exists():
        print(f"Error: File not found: {input_file}")
        return None
    
    data_array = np.load(input_file, allow_pickle=True)
    building_data = data_array.tolist()
    
    print(f"  Loaded {len(building_data)} buildings")
    
    # Print statistics
    num_with_inner = sum(1 for b in building_data if b['inner'] is not None)
    num_with_outer = sum(1 for b in building_data if b['outer'] is not None)
    num_with_original = sum(1 for b in building_data if b['original'] is not None)
    
    print(f"  Buildings with inner offset: {num_with_inner}/{len(building_data)}")
    print(f"  Buildings with outer offset: {num_with_outer}/{len(building_data)}")
    print(f"  Buildings with original: {num_with_original}/{len(building_data)}")
    
    return building_data

# TODO: make this function more modular via passing in 
def convert_outer_polygons_to_shapely(building_data):
    """
    Convert outer polygon points from building offsets to Shapely Polygon objects.
    
    Args:
        building_data: List of dictionaries with 'inner', 'original', 'outer' keys
    
    Returns:
        List of Shapely Polygon objects (only buildings with valid outer polygons)
    """
    from shapely.geometry import Polygon
    
    outer_polygons = []
    
    for idx, building in enumerate(building_data):
        outer_points = building.get('outer')
        
        if outer_points is None or len(outer_points) < 3:
            print(f"  Building {idx}: Skipping - no valid outer polygon")
            continue
        
        # Convert numpy array of [lon, lat] to list of (lon, lat) tuples
        # Close the polygon by adding first point at the end if needed
        coords = [(float(p[0]), float(p[1])) for p in outer_points]
        
        # Ensure polygon is closed
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        
        try:
            polygon = Polygon(coords)
            if polygon.is_valid and not polygon.is_empty:
                outer_polygons.append(polygon)
            else:
                print(f"  Building {idx}: Skipping - invalid outer polygon")
        except Exception as e:
            print(f"  Building {idx}: Skipping - error creating polygon: {e}")
    
    print(f"  Converted {len(outer_polygons)} valid outer polygons")
    return outer_polygons


def convert_inner_polygons_to_shapely(building_data):
    """
    Convert inner polygon points from building offsets to Shapely Polygon objects.
    Only processes buildings that have valid outer polygons (to match outer_polygons order).
    
    Args:
        building_data: List of dictionaries with 'inner', 'original', 'outer' keys
    
    Returns:
        List of Shapely Polygon objects aligned with outer_polygons
        Returns None for buildings without inner polygons or invalid inner polygons
    """
    from shapely.geometry import Polygon
    
    inner_polygons = []
    
    for idx, building in enumerate(building_data):
        # Only process buildings that have valid outer polygons (same logic as convert_outer_polygons_to_shapely)
        inner_points = building.get('inner')
        if inner_points is None or len(inner_points) < 3:
            continue  # Skip - no valid inner polygon, so won't be in inner_polygons list
        
        # Now process inner polygon for this building
        inner_points = building.get('inner')
        
        if inner_points is None or len(inner_points) < 3:
            inner_polygons.append(None)  # No inner polygon for this building
            continue
        
        # Convert numpy array of [lon, lat] to list of (lon, lat) tuples
        # Close the polygon by adding first point at the end if needed
        coords = [(float(p[0]), float(p[1])) for p in inner_points]
        
        # Ensure polygon is closed
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        
        try:
            polygon = Polygon(coords)
            if polygon.is_valid and not polygon.is_empty:
                inner_polygons.append(polygon)
            else:
                inner_polygons.append(None)
        except Exception as e:
            inner_polygons.append(None)
    
    num_with_inner = sum(1 for p in inner_polygons if p is not None)
    print(f"  Converted {num_with_inner} valid inner polygons (out of {len(inner_polygons)} buildings with valid outer polygons)")
    return inner_polygons


def filter_points_in_rectangle(points, x_min, y_min, x_max, y_max):
    """
    Filter points within a rectangular boundary.
    
    Args:c
        points: (N, 3) array of [lat, lon, z]
        x_min: Minimum longitude
        y_min: Minimum latitude
        x_max: Maximum longitude
        y_max: Maximum latitude
    
    Returns:
        points_in_rectangle: Boolean array indicating which points are inside the rectangle
    """
    points_in_rectangle = (points[:, 1] >= x_min) & (points[:, 1] <= x_max) & (points[:, 0] >= y_min) & (points[:, 0] <= y_max)
    return points_in_rectangle


def get_polygon_bounding_rectangle_params(polygon):
    x_min, y_min, x_max, y_max = polygon.bounds
    return x_min, y_min, x_max, y_max


def get_polygon_bounding_circle_params(polygon):
    coords = list(polygon.exterior.coords)
    lons, lats = zip(*coords)
    lons_array = np.array(lons)
    lats_array = np.array(lats)
    x_min = lons_array.min()
    x_max = lons_array.max()
    y_min = lats_array.min()
    y_max = lats_array.max()
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    # Calculate radius as the maximum distance from center to any corner
    # This ensures the circle fully contains the polygon
    distances = np.sqrt((lons_array - x_center)**2 + (lats_array - y_center)**2)
    radius = distances.max()
    return x_center, y_center, radius


def relabel_points_inside_polygons_to_building(semantic_points, 
                                               semantic_labels, 
                                               outer_building_polygons,
                                               inner_building_polygons=None,
                                               building_label_id=45):
    """
    Relabel points within building polygons to OSM building (ID 45).
    
    Only processes points that are:
    - In outer polygons but NOT in inner polygons (if inner polygons exist)
    
    Args:
        semantic_points: (N, 3) array of semantic point coordinates [lat, lon, z]
        semantic_labels: (N,) array of semantic labels (will be modified in place)
        outer_building_polygons: List of Shapely Polygon objects (outer polygons from building offsets)
        inner_building_polygons: Optional list of Shapely Polygon objects (inner polygons from building offsets)
                       Can contain None for buildings without inner polygons
        building_label_id: Label ID for OSM buildings (default: 45)
    
    Returns:
        Number of points relabeled to building
    """
    import time
    from shapely import contains_xy
    from tqdm import tqdm

    print(f"\nRelabeling points inside building outer polygons (excluding inner) to building (ID {building_label_id})...")
    print(f"  Total semantic points: {len(semantic_points)}")
    print(f"  Number of building polygons: {len(outer_building_polygons)}")
    if inner_building_polygons is not None:
        num_with_inner = sum(1 for p in inner_building_polygons if p is not None)
        print(f"  Number of buildings with inner polygons: {num_with_inner}")
    
    # Filter points to only include allowed labels: unlabeled (0), outlier (1), building (50), other-structure (52)
    # This significantly reduces computation by working with a smaller subset
    allowed_label_ids = [label.id for label in sem_kitti_labels]
    allowed_mask = np.isin(semantic_labels, allowed_label_ids)
    allowed_indices = np.where(allowed_mask)[0]
    
    num_filtered_out = len(semantic_points) - len(allowed_indices)
    print(f"  Initial filtering: Keeping {len(allowed_indices)} points with allowed labels")
    print(f"  Initial filtering: Excluding {num_filtered_out} points with other labels")
    
    if len(allowed_indices) == 0:
        print(f"  No points with allowed labels found - nothing to process")
        return 0
    
    # Work with filtered subset
    filtered_semantic_points = semantic_points[allowed_mask]
    filtered_semantic_labels = semantic_labels[allowed_mask]
    
    # Track which filtered points are in outer polygons but NOT in inner polygons
    in_outer_not_inner_filtered = np.zeros(len(filtered_semantic_points), dtype=bool)

    # Check each building polygon
    for idx, building_poly in enumerate(tqdm(outer_building_polygons, desc="Processing buildings", unit="building")):
        # Step 1: Filter points within rectangle
        x_min, y_min, x_max, y_max = get_polygon_bounding_rectangle_params(building_poly)
        rectangle_mask = filter_points_in_rectangle(filtered_semantic_points, x_min, y_min, x_max, y_max)
        rectangle_filtered_indices = np.where(rectangle_mask)[0]
        rectangle_filtered_points = filtered_semantic_points[rectangle_mask]

        if len(rectangle_filtered_points) == 0:
            continue
        
        # Step 2: From rectangle-filtered points, find which are in the outer polygon
        lats = rectangle_filtered_points[:, 0]
        lons = rectangle_filtered_points[:, 1]

        # Vectorized: returns a boolean array of length M
        inside_outer_mask = contains_xy(building_poly, lons, lats)
        
        # Step 3: Exclude points that are also in inner polygon (if it exists)
        if inner_building_polygons is not None and idx < len(inner_building_polygons) and inner_building_polygons[idx] is not None:
            inner_poly = inner_building_polygons[idx]
            # Check which points are in inner polygon (only check those in outer)
            points_in_outer = rectangle_filtered_points[inside_outer_mask]
            if len(points_in_outer) > 0:
                inner_lats = points_in_outer[:, 0]
                inner_lons = points_in_outer[:, 1]
                inside_inner_mask = contains_xy(inner_poly, inner_lons, inner_lats)
                # Points in outer but NOT in inner
                in_outer_not_inner_local = inside_outer_mask.copy()
                in_outer_not_inner_local[inside_outer_mask] = ~inside_inner_mask
            else:
                in_outer_not_inner_local = inside_outer_mask
        else:
            # No inner polygon, so all points in outer are valid
            in_outer_not_inner_local = inside_outer_mask

        # Map back to filtered indices
        points_in_ring_filtered = rectangle_filtered_indices[in_outer_not_inner_local]
        in_outer_not_inner_filtered[points_in_ring_filtered] = True
    
    # # print the percentage statistics of points inside the building polygons
    # labels_inside_polygons = filtered_semantic_labels[inside_any_polygon_filtered]
    # if len(labels_inside_polygons) > 0:
    #     unique_labels, counts = np.unique(labels_inside_polygons, return_counts=True)
    #     total_points = len(labels_inside_polygons)
        
    #     # Create label name mapping from imported labels
    #     label_names = {label.id: label.name for label in sem_kitti_labels}
        
    #     print(f"\n  Label distribution for points inside building polygons:")
    #     print(f"    Total points: {total_points}")
    #     # Sort by count (descending)
    #     sorted_indices = np.argsort(counts)[::-1]
    #     for idx in sorted_indices:
    #         label_id = unique_labels[idx]
    #         count = counts[idx]
    #         percentage = 100.0 * count / total_points
    #         label_name = label_names.get(label_id, f"unknown-{label_id}")
    #         print(f"    ID {label_id:3d} ({label_name:20s}): {count:8d} points ({percentage:5.2f}%)")
    # else:
    #     print(f"\n  No points found inside building polygons")

    # Get indices of filtered points in outer but not inner (ring)
    filtered_indices_in_ring = np.where(in_outer_not_inner_filtered)[0]
    
    if len(filtered_indices_in_ring) == 0:
        print(f"  No points in outer polygons (excluding inner) - nothing to process")
        return 0
    
    # Map back to original indices in the full array
    indices_to_relabel = allowed_indices[filtered_indices_in_ring]
    
    # Count how many were already building vs relabeled
    already_building = (semantic_labels[indices_to_relabel] == building_label_id).sum()
    num_relabeled = len(indices_to_relabel) - already_building
    
    # Relabel all points in ring (outer - inner) to OSM building (45)
    semantic_labels[indices_to_relabel] = building_label_id
    
    print(f"  Total points in outer polygons (excluding inner): {len(filtered_indices_in_ring)}")
    print(f"  Points already labeled as OSM building: {already_building}")
    print(f"  Points relabeled to OSM building (ID {building_label_id}): {num_relabeled}")
    
    return num_relabeled


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

    print(f"  - OG filtering points within circle:")
    print(f"    - Center: ({center_lat:.6f}, {center_lon:.6f})")
    print(f"    - Radius: {radius_deg:.6f} degrees")
    print(f"    - Points inside: {mask.sum()} / {len(mask)} ({100*mask.sum()/len(mask):.2f}%)")
    
    return mask


# def filter_points_in_polygon(points_latlon, polygon_coords):
#     """
#     Filter points within a polygon boundary using shapely.
    
#     Args:
#         points_latlon: (N, 3) array of [lat, lon, z]
#         polygon_coords: List of (lon, lat) tuples defining the polygon
    
#     Returns:
#         mask: Boolean array indicating which points are inside the polygon
#     """
#     from shapely.geometry import Point, Polygon
    
#     # Create polygon from coordinates
#     polygon = Polygon(polygon_coords)
    
#     # Check each point
#     mask = np.zeros(len(points_latlon), dtype=bool)
#     for i, point in enumerate(points_latlon):
#         lat, lon = point[0], point[1]
#         p = Point(lon, lat)
#         mask[i] = polygon.contains(p)
    
#     print(f"\nFiltering points within building polygon:")
#     print(f"  Polygon vertices: {len(polygon_coords)}")
#     print(f"  Points inside polygon: {mask.sum()} / {len(mask)} ({100*mask.sum()/len(mask):.2f}%)")
    
#     return mask


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


def relabel_building_points_outside_polygons(semantic_points, semantic_labels, outer_building_polygons, 
                                             building_label_id=50, vegetation_label_id=70):
    """
    Relabel points marked as 'building' that fall outside all building outer polygons as 'vegetation'.
    
    Args:
        semantic_points: (N, 3) array of semantic point coordinates [lat, lon, z]
        semantic_labels: (N,) array of semantic labels (will be modified in place)
        outer_building_polygons: List of Shapely Polygon objects (outer polygons from building offsets)
        building_label_id: Label ID for buildings (default: 50)
        vegetation_label_id: Label ID for vegetation (default: 70)
    
    Returns:
        Number of points relabeled
    """
    import time
    from shapely import contains_xy
    
    # Find all points labeled as building
    building_mask = semantic_labels == building_label_id
    building_points = semantic_points[building_mask]
    
    if len(building_points) == 0:
        print("No points labeled as building (ID 50) found for polygon filtering")
        return 0
    
    print(f"\nRelabeling building points outside building outer polygons...")
    print(f"  Total points labeled as building: {len(building_points)}")
    print(f"  Number of building polygons: {len(outer_building_polygons)}")
    
    # Get indices of building points in the full array
    building_indices = np.where(building_mask)[0]
    
    # Extract coordinates for vectorized operations
    # building_points: (M, 3) [lat, lon, z]
    lats = building_points[:, 0]
    lons = building_points[:, 1]
    
    # Create a mask to track which building points are inside any polygon
    inside_any_polygon = np.zeros(len(building_points), dtype=bool)
    
    # Check each building polygon using vectorized contains_xy
    t_start = time.time()
    for idx, building_poly in enumerate(outer_building_polygons):
        # Vectorized check: returns boolean array of length M
        # Only check points not already marked as inside a polygon
        candidate_mask = ~inside_any_polygon
        if not candidate_mask.any():
            continue  # All points already marked as inside
        
        # Vectorized contains_xy check on all candidate points
        inside_this_polygon = contains_xy(building_poly, lons[candidate_mask], lats[candidate_mask])
        
        # Update the mask for points inside this polygon
        inside_any_polygon[candidate_mask] |= inside_this_polygon
    
    print(f"  Polygon checks took {time.time() - t_start:.2f}s")
    
    # Points that are NOT inside any polygon should be relabeled
    points_to_relabel = ~inside_any_polygon
    num_relabeled = points_to_relabel.sum()
    
    # Get the original indices of building points to relabel
    indices_to_relabel = building_indices[points_to_relabel]
    
    # Relabel these points as vegetation
    semantic_labels[indices_to_relabel] = vegetation_label_id
    
    print(f"  Points inside building outer polygons: {inside_any_polygon.sum()}")
    print(f"  Points relabeled as vegetation (ID 70): {num_relabeled}")
    
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


def relabel_points_near_roads_to_road(semantic_points, semantic_labels, road_proximity_mask,
                                      road_label_id=46):
    """
    Relabel points near roads to OSM road label (ID 46), excluding human and vehicle labels.
    
    Excluded labels (will NOT be relabeled):
    - Human labels: person (30), bicyclist (31), motorcyclist (32)
    - Vehicle labels: car (10), bicycle (11), bus (13), motorcycle (15), 
                     on-rails (16), truck (18), other-vehicle (20)
    
    Args:
        semantic_points: (N, 3) array of semantic point coordinates [lat, lon, z]
        semantic_labels: (N,) array of semantic labels (will be modified in place)
        road_proximity_mask: Boolean mask indicating which points are near roads
        road_label_id: Label ID for OSM roads (default: 46)
    
    Returns:
        Number of points relabeled to road
    """
    if road_proximity_mask is None:
        print("  No road proximity mask provided - skipping road relabeling")
        return 0
    
    print(f"\nRelabeling points near roads to OSM road (ID {road_label_id})...")
    print(f"  Total points near roads: {road_proximity_mask.sum()}")
    
    # Get indices of points near roads
    indices_near_roads = np.where(road_proximity_mask)[0]
    
    if len(indices_near_roads) == 0:
        print(f"  No points near roads - nothing to relabel")
        return 0
    
    # Exclude human and vehicle labels
    # Human labels: person (30), bicyclist (31), motorcyclist (32)
    # Vehicle labels: car (10), bicycle (11), bus (13), motorcycle (15), 
    #                 on-rails (16), truck (18), other-vehicle (20)
    # excluded_label_ids = [30, 31, 32, 10, 11, 13, 15, 16, 18, 20]
    excluded_label_ids = [30, 31, 32, 10, 11, 13, 15, 16, 18, 20, 45, 50, 51, 52, 60, 70, 71, 80, 81, 99]
    excluded_mask = np.isin(semantic_labels[indices_near_roads], excluded_label_ids)
    indices_to_relabel = indices_near_roads[~excluded_mask]
    
    num_excluded = excluded_mask.sum()
    if num_excluded > 0:
        print(f"  Excluded {num_excluded} points with human/vehicle labels")
    
    if len(indices_to_relabel) == 0:
        print(f"  No points to relabel after filtering excluded labels")
        return 0
    
    # Count how many were already road vs relabeled
    already_road = (semantic_labels[indices_to_relabel] == road_label_id).sum()
    num_relabeled = len(indices_to_relabel) - already_road
    
    # Relabel points near roads to OSM road (excluding human/vehicle labels)
    semantic_labels[indices_to_relabel] = road_label_id
    
    print(f"  Total points near roads: {len(indices_near_roads)}")
    print(f"  Points excluded (human/vehicle labels): {num_excluded}")
    print(f"  Points already labeled as OSM road: {already_road}")
    print(f"  Points relabeled to OSM road (ID {road_label_id}): {num_relabeled}")
    
    return num_relabeled


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
    from shapely.geometry import Polygon
    from shapely import contains_xy
    import time
    from tqdm import tqdm
    
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
    
    # Extract coordinates for vectorized operations
    lats = semantic_points[:, 0]
    lons = semantic_points[:, 1]
    
    # Check each polygon
    for idx, polygon_feature in enumerate(tqdm(filtered_polygons, desc=f"Processing {polygon_type}", unit="polygon")):
        if polygon_feature.geometry.geom_type != "Polygon":
            continue
        
        polygon = polygon_feature.geometry
        
        print(f"  Processing {polygon_type} {idx+1}/{len(filtered_polygons)}...")
        
        # Filter out protected points and already-marked points using vectorized numpy operations
        candidate_mask = ~(inside_any_polygon | protected_mask)
        
        if not candidate_mask.any():
            print(f"    No candidate points (all protected or already marked)")
            continue
        
        # Get candidate points and their coordinates
        candidate_indices = np.where(candidate_mask)[0]
        candidate_lats = lats[candidate_mask]
        candidate_lons = lons[candidate_mask]
        
        # Vectorized contains_xy check on all candidate points
        t_start = time.time()
        inside_this_polygon = contains_xy(polygon, candidate_lons, candidate_lats)
        
        # Update the mask for points inside this polygon
        inside_any_polygon[candidate_indices[inside_this_polygon]] = True
        points_in_this_polygon = inside_this_polygon.sum()
        
        print(f"    Found {points_in_this_polygon} points inside this {polygon_type} (excluding protected) in {time.time() - t_start:.2f}s")
    
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


# TODO: Remove this function after some testing
def filter_building_circles(building_polygons, 
                            semantic_points=None, 
                            semantic_labels=None, 
                            enable_plotting=False,
                            relabel_plot=None):
    if not building_polygons:
        print("No building polygons provided!")
        return None
    else:
        print(f"\nFound {len(building_polygons)} building polygons")
    
    # Step 1: relabel all points inside all building polygons
    buildings_to_process = list(enumerate(building_polygons))

    building_circle_params = []
    # Process each building
    for idx, polygon in buildings_to_process:
        print(f"\nProcessing building {idx}:")

        # Extract building coordinates from polygon
        coords = list(polygon.exterior.coords)
        lons, lats = zip(*coords)
        
        # Find min and max x (longitude) and y (latitude) points
        lons_array = np.array(lons)
        lats_array = np.array(lats)
        
        x_min = lons_array.min()
        x_max = lons_array.max()
        y_min = lats_array.min()
        y_max = lats_array.max()

        # Calculate center point and radius
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        radius = np.sqrt((x_max - x_center)**2 + (y_max - y_center)**2)

        # Store circle parameters
        building_circle_params.append((x_center, y_center, radius))

    return building_circle_params


def relabel_semantic_map_with_osm(outer_building_polygons, inner_building_polygons, poses_latlon, poses_center, poses_max_point,
                           semantic_points=None, semantic_labels=None, semantic_intensities=None,
                           all_robot_poses=None, osm_handler=None):
    """
    Plot building polygons with optional semantic points.
    
    Args:
        outer_building_polygons: List of Shapely Polygon objects (outer polygons from building offsets)
        inner_building_polygons: List of Shapely Polygon objects (inner polygons from building offsets)
        poses_latlon: Array of poses in lat/lon format (combined from all robots)
        semantic_points: Optional (N, 3) array of semantic point coordinates [lat, lon, z]
        semantic_labels: Optional (N,) array of semantic labels (modified in place if relabeling enabled)
        semantic_intensities: Optional (N,) array of intensities
        all_robot_poses: Optional list of (poses_latlon, robot_name) tuples for plotting individual robot paths
        osm_handler: Optional OSMDataHandler instance (needed for grassland/road processing)
    
    Returns:
        Circle parameters list [(center_lon, center_lat, radius), ...] if successful, None otherwise
    """
    
    # Relabel buildings with OSM data
    building_circle_params = filter_building_circles(outer_building_polygons, semantic_points, semantic_labels)
    
    # Process grassland and garden polygons if enabled
    all_grassland_circle_params = []
    all_garden_circle_params = []
    filtered_grasslands = []
    filtered_gardens = []
    
    if osm_handler is not None:
        print(f"\n{'='*60}")
        print("Processing Grassland & Garden Polygons")
        print(f"{'='*60}")
        
        # Get filtered grasslands near poses
        try:
            print(f"Filtering grassland areas near poses...")
            # filtered_grasslands = osm_handler.filter_geometries_by_distance('grassland', poses_latlon, use_centroid=True)
            filtered_grasslands = osm_handler.filter_geometries_by_pose_center('grassland', poses_center, poses_max_point)
        except:
            print("Error filtering grassland areas near poses: ", e)
        
        # Get filtered gardens near poses (leisure=garden)
        try:
            print(f"Filtering garden areas near poses...")
            # filtered_gardens = osm_handler.filter_geometries_by_distance('gardens', poses_latlon, use_centroid=True)
            filtered_gardens = osm_handler.filter_geometries_by_pose_center('gardens', poses_center, poses_max_point)
        except:
            print("Error filtering garden areas near poses: ", e)
        
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
        else:
            print("\nNo grassland or garden areas found near poses!")

    # After processing all buildings, grasslands, and roads, apply relabeling if enabled
    if semantic_points is not None and semantic_labels is not None:
        print(f"\n{'='*60}")
        print("Applying OSM-based relabeling...")
        print(f"{'='*60}")
        
        # Building relabeling
        print(f"\nRelabeling buildings...")
        # Step 1: Relabel points inside building outer polygons (excluding inner) to OSM building (ID 45)
        # This ensures everything in the ring (outer - inner) is labeled as OSM building
        relabel_points_inside_polygons_to_building(semantic_points, semantic_labels, outer_building_polygons, inner_building_polygons, building_label_id=45)
        
        # Step 2: Relabel building points outside ALL building outer polygons to vegetation
        # This removes false positive buildings that are not inside any actual building polygon
        # Note: We skip the circle check as it's redundant - the polygon check is more precise
        # and already covers all cases (points outside polygons are also outside their bounding circles)
        relabel_building_points_outside_polygons(semantic_points, semantic_labels, outer_building_polygons, building_label_id=45)
        
        # Road relabeling (before terrain relabeling)
        road_proximity_mask = None
        print(f"\nRelabeling roads...")
        # Get filtered roads near poses
        try:
            # Note: roads are stored as 'highways' in the OSM handler
            print(f"Filtering roads near poses...")
            filtered_roads = osm_handler.filter_geometries_by_distance('highways', poses_latlon, use_centroid=False)
            print(f"Found {len(filtered_roads)} roads near poses")
            if filtered_roads and len(filtered_roads) > 0:
                # Create mask for points within 2.5 meters of road segments
                road_proximity_mask = create_road_proximity_mask(semantic_points, 
                                                                 filtered_roads, 
                                                                 proximity_threshold=2.5)
                # Relabel points near roads to OSM road (46), excluding human/vehicle labels
                relabel_points_near_roads_to_road(semantic_points, semantic_labels, road_proximity_mask)
        except Exception as e:
            print(f"\nWarning: Could not process roads: {e}")
            road_proximity_mask = None
        
        # Grassland and garden relabeling
        print(f"\nRelabeling grassland and garden areas...")
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

    # Filter for only OSM building (45) and OSM road (46) points
    keep_mask = None
    print(f"\n{'='*60}")
    print("Filtering to keep only OSM building (45) and OSM road (46) points...")
    print(f"{'='*60}")
    if semantic_points is not None and semantic_labels is not None:
        # # Filter to only keep points with labels 45 (OSM BUILDING) and 46 (OSM ROAD)
        # osm_building_label_id = 45
        # osm_road_label_id = 46
        # keep_mask = (semantic_labels == osm_building_label_id) | (semantic_labels == osm_road_label_id)
        
        # num_before = len(semantic_points)
        # num_after = keep_mask.sum()
        # num_filtered_out = num_before - num_after
        
        # print(f"  Points before filtering: {num_before}")
        # print(f"  Points after filtering (OSM building + OSM road): {num_after}")
        # print(f"  Points filtered out: {num_filtered_out}")
        
        # # Filter all arrays
        # semantic_points = semantic_points[keep_mask]
        # semantic_labels = semantic_labels[keep_mask]
        # if semantic_intensities is not None:
        #     semantic_intensities = semantic_intensities[keep_mask]
        
        # Print label statistics after filtering
        print(f"\n{'='*60}")
        print("Label Statistics After Filtering...")
        print(f"{'='*60}")
        unique_labels, counts = np.unique(semantic_labels, return_counts=True)
        print(f"\nLabel distribution after filtering:")
        for label_id, count in zip(unique_labels, counts):
            label_name = "Unknown"
            for label in sem_kitti_labels:
                if label.id == label_id:
                    label_name = label.name
                    break
            print(f"  {label_name} (ID {label_id}): {count} points ({100*count/len(semantic_labels):.1f}%)")
    
    # Return filtered arrays (only OSM building and OSM road points) and the mask
    return semantic_points, semantic_labels, semantic_intensities, keep_mask


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Select and plot an OSM building boundary with semantic points")

    # Environment name
    parser.add_argument("--environment", type=str, default="main_campus",
                       help="Environment name (default: main_campus)")
    
    parser.add_argument("--filter_distance", type=float, default=100.0,
                       help="Distance in meters to filter buildings around poses (default: 100.0)")

    # Input file postfix
    parser.add_argument("--input_postfix", type=str, default="sem_map_relabeled_utm_knn_smoothed")

    
    args = parser.parse_args()

    
    # Hardcoded paths
    dataset_path = "/media/donceykong/donceys_data_ssd/datasets/CU_MULTI/data"
    robots = ["robot1", "robot2", "robot3", "robot4"]
    
    # Construct file paths
    osm_file = Path(dataset_path) / args.environment / "additional" / f"{args.environment}.osm"
    semantic_map_file = Path(dataset_path) / args.environment / "additional" / f"{args.environment}_{args.input_postfix}.npy"
    building_offsets_file = Path(dataset_path) / args.environment / "additional" /  f"{args.environment}_building_offsets.npy"
    
    # Load building offsets
    print(f"\n{'='*80}")
    print("Loading Building Offsets")
    print(f"{'='*80}")
    building_data = load_building_offsets(building_offsets_file)
    
    if building_data is None:
        print("Error: Could not load building offsets!")
        return
    
    # Convert outer polygons to Shapely Polygon objects
    outer_building_polygons = convert_outer_polygons_to_shapely(building_data)
    
    if not outer_building_polygons:
        print("Error: No valid building outer polygons found!")
        return
    
    # Convert inner polygons to Shapely Polygon objects
    inner_building_polygons = convert_inner_polygons_to_shapely(building_data)
    
    # Check if OSM file exists (needed for grasslands/roads)
    osm_handler = None
    if not osm_file.exists():
        print(f"Warning: OSM file not found: {osm_file}")
        print("Continuing without grassland/road processing...")
    else:
        # Load OSM data for grasslands and roads
        print(f"\n{'='*80}")
        print("Loading OSM Data for Grasslands/Roads")
        print(f"{'='*80}")
        print(f"Filter distance: {args.filter_distance}m")
        osm_handler = OSMDataHandler(osm_file, filter_distance=args.filter_distance)
        
        # Enable grasslands, gardens, and roads semantics
        osm_handler.set_semantics({
            'roads': True,
            'highways': False,
            'buildings': False,  # We're using building offsets instead
            'trees': False,
            'grassland': True,
            'gardens': True,
            'water': False,
            'parking': False,
            'amenities': False
        })
        
        if not osm_handler.load_osm_data():
            print("Warning: Failed to load OSM data. Continuing without grassland/road processing...")
            osm_handler = None
    
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
        poses_file = Path(dataset_path) / args.environment / robot / f"{robot}_{args.environment}_gt_utm_poses.csv"
        
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
    
    # Calculate min and max lat-lon coordinates
    # Min point (lower left): minimum latitude, minimum longitude
    min_lat = poses_latlon[:, 0].min()
    min_lon = poses_latlon[:, 1].min()
    min_point = np.array([min_lat, min_lon])
    
    # Max point (upper right): maximum latitude, maximum longitude
    max_lat = poses_latlon[:, 0].max()
    max_lon = poses_latlon[:, 1].max()
    max_point = np.array([max_lat, max_lon])
    poses_max_point = np.array([max_lat, max_lon])

    print(f"\n{'='*80}")
    print("Calculating Bounding Box and Circle")
    print(f"{'='*80}")
    print(f"  Min point (lower left): lat={min_lat:.6f}, lon={min_lon:.6f}")
    print(f"  Max point (upper right): lat={max_lat:.6f}, lon={max_lon:.6f}")
    
    # Calculate center as midpoint between min and max
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    poses_center = np.array([center_lat, center_lon])
    
    print(f"  Center: lat={center_lat:.6f}, lon={center_lon:.6f}")

    # Prepare individual robot poses for visualization
    robot_poses_list = [(robot_poses, robot_name) 
                        for robot_poses, robot_name in zip(all_poses_latlon, all_robot_names)]
    
    # Plot selected building(s) with semantic points
    print(f"\n{'='*80}")
    print(f"Relabeling All Buildings")
    print(f"{'='*80}")
    # Relabel semantic map and get filtered arrays (only OSM building and OSM road points)
    filtered_semantic_points_latlon, filtered_semantic_labels, filtered_semantic_intensities, keep_mask = relabel_semantic_map_with_osm(
        outer_building_polygons, 
        inner_building_polygons, 
        poses_latlon, 
        poses_center,
        poses_max_point,
        semantic_points=semantic_points_latlon,
        semantic_labels=semantic_labels,
        semantic_intensities=semantic_intensities,
        all_robot_poses=robot_poses_list,
        osm_handler=osm_handler
    )
    
    # Convert filtered semantic points back to UTM for saving
    if filtered_semantic_points_latlon is not None and points_utm is not None:
        # Find which original UTM points correspond to the filtered lat/lon points
        # We need to map back from filtered lat/lon to original UTM
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32613", always_xy=True)
        
        # Convert filtered lat/lon points back to UTM
        filtered_lons = filtered_semantic_points_latlon[:, 1]
        filtered_lats = filtered_semantic_points_latlon[:, 0]
        filtered_x_utm, filtered_y_utm = transformer.transform(filtered_lons, filtered_lats)
        filtered_z = filtered_semantic_points_latlon[:, 2]
        filtered_points_utm = np.column_stack([filtered_x_utm, filtered_y_utm, filtered_z])
    else:
        filtered_points_utm = None
    
    # Save relabeled semantic map to .npy file (only OSM building and OSM road points)
    output_npy = Path(dataset_path) / args.environment / "additional" /  f"{args.environment}_{args.input_postfix}_OSM_FILTERED.npy"

    print(f"\n{'='*80}")
    print(f"Saving Relabeled Semantic Map to {output_npy} file")
    print(f"{'='*80}")
    if filtered_points_utm is not None and filtered_semantic_intensities is not None and filtered_semantic_labels is not None:
        save_relabeled_semantic_map(filtered_points_utm, filtered_semantic_intensities, filtered_semantic_labels, output_npy)
    else:
        print("Warning: Cannot save - filtered arrays are None")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

