#!/usr/bin/env python3
"""
Label Filter Utilities

Functions to filter and relabel point cloud semantic labels based on OSM data.
"""

import numpy as np
import time
from typing import Optional, List, Dict, Any
from shapely.geometry import Point, Polygon, MultiPoint
from shapely.strtree import STRtree
from shapely.prepared import prep
from pyproj import Transformer


# struct to hold building information
class BuildingInfo:
    def __init__(self, 
                min_x, min_y, max_x, max_y, 
                radius, 
                center,
                building_geom):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.radius = radius
        self.center = center
        self.building_geom = building_geom

def filter_building_labels_with_osm(
    points_xyz: np.ndarray,
    labels: np.ndarray,
    osm_handler,
    building_label_id: int = 50,
    vegetation_label_id: int = 70,
    utm_zone: str = "EPSG:32613"
) -> np.ndarray:
    """
    Filter building-labeled points and relabel those not within OSM building polygons as vegetation.
    
    This function takes points labeled as 'building' and checks if they actually fall within
    OSM building polygons. Points labeled as buildings but not within any building polygon
    are relabeled as vegetation.
    
    Args:
        points_xyz: Nx3 array of point coordinates in UTM (world coordinates)
        labels: N array of semantic labels
        osm_handler: OSMDataHandler instance with loaded building geometries
        building_label_id: Label ID for buildings (default: 50)
        vegetation_label_id: Label ID to assign to non-building points (default: 70)
        utm_zone: EPSG code for UTM zone (default: "EPSG:32613" for Colorado)
    
    Returns:
        Modified labels array with filtered building labels
    """
    # Make a copy of labels to avoid modifying the original
    filtered_labels = labels.copy()
    
    # Get building mask (points currently labeled as buildings)
    building_mask = labels == building_label_id
    
    if not building_mask.any():
        print("No building-labeled points found")
        return filtered_labels
    
    num_building_points = building_mask.sum()
    print(f"Processing {num_building_points} building-labeled points")
    
    # Get building geometries from OSM handler
    if not hasattr(osm_handler, 'osm_geometries') or 'buildings' not in osm_handler.osm_geometries:
        print("Warning: No building geometries found in OSM handler")
        # Relabel all building points as vegetation
        filtered_labels[building_mask] = vegetation_label_id
        return filtered_labels
    
    buildings_gdf = osm_handler.osm_geometries['buildings']
    
    if buildings_gdf is None or len(buildings_gdf) == 0:
        print("Warning: Empty building geometries in OSM handler")
        # Relabel all building points as vegetation
        filtered_labels[building_mask] = vegetation_label_id
        return filtered_labels
    
    print(f"Found {len(buildings_gdf)} building polygons in OSM data")
    
    # Extract building points
    building_points_xyz = points_xyz[building_mask]
    
    # Project 2D coordinates (x, y from UTM)
    building_points_2d = building_points_xyz[:, :2]  # Take only x, y
    
    # Convert UTM coordinates to lat/lon for comparison with OSM data
    transformer = Transformer.from_crs(utm_zone, "EPSG:4326", always_xy=True)
    building_points_lon, building_points_lat = transformer.transform(
        building_points_2d[:, 0], 
        building_points_2d[:, 1]
    )
    
    # Create shapely Points for all building-labeled points
    print("Creating point geometries...")
    shapely_points = [Point(lon, lat) for lon, lat in zip(building_points_lon, building_points_lat)]
    
    # Check which points are within building polygons
    print("Checking points against building polygons...")
    points_in_buildings = np.zeros(len(shapely_points), dtype=bool)
    
    for idx, building in buildings_gdf.iterrows():
        geometry = building.geometry
        
        # Handle different geometry types
        if geometry.geom_type == 'Polygon':
            polygons = [geometry]
        elif geometry.geom_type == 'MultiPolygon':
            polygons = list(geometry.geoms)
        else:
            continue
        
        # Check each point against this building's polygon(s)
        for polygon in polygons:
            if not polygon.is_valid:
                continue
            
            for i, point in enumerate(shapely_points):
                if not points_in_buildings[i]:  # Skip if already marked as in building
                    if polygon.contains(point):
                        points_in_buildings[i] = True
    
    # Count points in and out of buildings
    num_in_buildings = points_in_buildings.sum()
    num_outside_buildings = len(points_in_buildings) - num_in_buildings
    
    print(f"Points within building polygons: {num_in_buildings}")
    print(f"Points outside building polygons: {num_outside_buildings}")
    
    # Relabel points outside buildings as vegetation
    # Get original indices of building points
    building_indices = np.where(building_mask)[0]
    outside_building_indices = building_indices[~points_in_buildings]
    
    filtered_labels[outside_building_indices] = vegetation_label_id
    
    print(f"Relabeled {len(outside_building_indices)} points from building to vegetation")
    
    return filtered_labels


def filter_building_labels_with_osm_fast(
    points_xyz: np.ndarray,
    labels: np.ndarray,
    osm_handler,
    poses_latlon: Optional[np.ndarray] = None,
    building_label_id: int = 50,
    vegetation_label_id: int = 70,
    utm_zone: str = "EPSG:32613"
) -> np.ndarray:
    """
    Faster version using spatial indexing (STRtree) and vectorized operations.
    
    This function uses pre-filtered building polygons and spatial indexing 
    for significantly faster point-in-polygon checks.
    
    Args:
        points_xyz: Nx3 array of point coordinates in UTM (world coordinates)
        labels: N array of semantic labels
        osm_handler: OSMDataHandler instance with loaded building geometries
        poses_latlon: Optional array of poses in lat/lon for filtering buildings
        building_label_id: Label ID for buildings (default: 50)
        vegetation_label_id: Label ID to assign to non-building points (default: 70)
        utm_zone: EPSG code for UTM zone (default: "EPSG:32613" for Colorado)
    
    Returns:
        Modified labels array with filtered building labels
    """
    start_time = time.time()
    
    # Make a copy of labels to avoid modifying the original
    filtered_labels = labels.copy()
    
    # Get building mask (points currently labeled as buildings)
    building_mask = labels == building_label_id
    
    if not building_mask.any():
        print("No building-labeled points found")
        return filtered_labels
    
    num_building_points = building_mask.sum()
    print(f"\n=== Building Label Filter (Fast) ===")
    print(f"Processing {num_building_points} building-labeled points")
    
    # Get filtered building geometries using OSM handler's filtering
    if poses_latlon is not None and hasattr(osm_handler, 'filter_geometries_by_distance'):
        print("Filtering building geometries by distance to poses...")
        filtered_buildings = osm_handler.filter_geometries_by_distance(
            'buildings', poses_latlon, use_centroid=False
        )
    else:
        # Fall back to all buildings
        if not hasattr(osm_handler, 'osm_geometries') or 'buildings' not in osm_handler.osm_geometries:
            print("Warning: No building geometries found in OSM handler")
            filtered_labels[building_mask] = vegetation_label_id
            return filtered_labels
        filtered_buildings = osm_handler.osm_geometries['buildings']
    
    if filtered_buildings is None or len(filtered_buildings) == 0:
        print("Warning: No building geometries found")
        # Relabel all building points as vegetation
        filtered_labels[building_mask] = vegetation_label_id
        return filtered_labels
    
    print(f"Using {len(filtered_buildings)} filtered building polygons")
    
    # Extract building points
    building_points_xyz = points_xyz[building_mask]
    
    # Project 2D coordinates (x, y from UTM)
    building_points_2d = building_points_xyz[:, :2]  # Take only x, y
    
    # Convert UTM coordinates to lat/lon for comparison with OSM data (VECTORIZED)
    t1 = time.time()
    print("Converting coordinates to lat/lon...")
    transformer = Transformer.from_crs(utm_zone, "EPSG:4326", always_xy=True)
    building_points_lon, building_points_lat = transformer.transform(
        building_points_2d[:, 0], 
        building_points_2d[:, 1]
    )
    print(f"  Coordinate conversion took {time.time() - t1:.2f}s")
    
    # Create shapely Points (vectorized with numpy arrays)
    t2 = time.time()
    print("Creating point geometries...")
    coords = np.column_stack((building_points_lon, building_points_lat))
    shapely_points = [Point(coord) for coord in coords]
    print(f"  Point creation took {time.time() - t2:.2f}s")
    
    # Extract and prepare building polygons
    t3 = time.time()
    print("Preparing building polygons and spatial index...")
    building_polygons = []
    prepared_polygons = []
    
    for idx, building in filtered_buildings.iterrows():
        geometry = building.geometry
        
        # Handle different geometry types
        if geometry.geom_type == 'Polygon':
            if geometry.is_valid:
                building_polygons.append(geometry)
                prepared_polygons.append(prep(geometry))
        elif geometry.geom_type == 'MultiPolygon':
            for polygon in geometry.geoms:
                if polygon.is_valid:
                    building_polygons.append(polygon)
                    prepared_polygons.append(prep(polygon))
    
    if not building_polygons:
        print("Warning: No valid building polygons found")
        filtered_labels[building_mask] = vegetation_label_id
        return filtered_labels
    
    print(f"Created {len(building_polygons)} valid building polygons")
    print(f"  Polygon preparation took {time.time() - t3:.2f}s")
    
    # Build spatial index (STRtree) for fast spatial queries
    t4 = time.time()
    print("Building spatial index...")
    tree = STRtree(building_polygons)
    print(f"  Spatial index built in {time.time() - t4:.2f}s")
    
    # Check which points are within building polygons using spatial index
    t5 = time.time()
    print("Performing spatial queries...")
    points_in_buildings = np.zeros(len(shapely_points), dtype=bool)
    
    # Process points in batches for better performance and progress reporting
    batch_size = 5000
    num_batches = (len(shapely_points) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(shapely_points))
        batch_points = shapely_points[start_idx:end_idx]
        
        for i, point in enumerate(batch_points):
            global_idx = start_idx + i
            
            # Use spatial index to find candidate polygons (fast bounding box check)
            candidate_indices = tree.query(point)
            
            # Only check actual containment for nearby polygons (prepared geometries are fast)
            for candidate_idx in candidate_indices:
                if prepared_polygons[candidate_idx].contains(point):
                    points_in_buildings[global_idx] = True
                    break  # Stop checking once we find a match
        
        if (batch_idx + 1) % 5 == 0 or batch_idx == num_batches - 1:
            processed = min(end_idx, len(shapely_points))
            pct = 100.0 * processed / len(shapely_points)
            print(f"  Processed {processed}/{len(shapely_points)} points ({pct:.1f}%)...")
    
    # Count points in and out of buildings
    num_in_buildings = points_in_buildings.sum()
    num_outside_buildings = len(points_in_buildings) - num_in_buildings
    
    print(f"Points within building polygons: {num_in_buildings}")
    print(f"Points outside building polygons: {num_outside_buildings}")
    
    # Relabel points outside buildings as vegetation
    # Get original indices of building points
    building_indices = np.where(building_mask)[0]
    outside_building_indices = building_indices[~points_in_buildings]
    
    filtered_labels[outside_building_indices] = vegetation_label_id
    
    print(f"Relabeled {len(outside_building_indices)} points from building to vegetation")
    
    return filtered_labels


import matplotlib.pyplot as plt
def find_min_max_x_y(building_geom):
    """
    Find the minimum and maximum x and y coordinates of a building geometry.
    """
    min_x, min_y, max_x, max_y = building_geom.bounds
    return min_x, min_y, max_x, max_y

def draw_circlular_bounds(building_info_list):
    """
    Draw a circular boundary around a building geometry using matplotlib.
    """
    for building_info in building_info_list:
        fig, ax = plt.subplots()
        ax.add_patch(plt.Circle((building_info.center[0], building_info.center[1]), building_info.radius, color='red', fill=False))
        
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def filter_building_labels_alg2(osm_handler):
    """
    """

    print("\n\nFiltering building labels algorithm 2...")
    print("Finding min and max x and y coordinates of each building...")
    building_info_list = []
    for building in osm_handler.osm_geometries['buildings']:
        min_x, min_y, max_x, max_y = find_min_max_x_y(building)

        radius = np.linalg.norm([max_x - min_x, max_y - min_y]) / 2
        center = np.array([(max_x + min_x) / 2, (max_y + min_y) / 2])

        building_info_list.append(BuildingInfo(min_x, min_y, max_x, max_y, radius, center, building))

    # draw circular bounds around all buildings
    draw_circlular_bounds(building_info_list)
