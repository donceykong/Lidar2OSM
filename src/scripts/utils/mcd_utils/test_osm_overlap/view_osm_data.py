#!/usr/bin/env python3
"""
View OSM data using Open3D visualization.

This script loads OSM data from an XML file and visualizes various features
(buildings, roads, trees, grassland, water, etc.) using Open3D.
"""

import open3d as o3d
import numpy as np
import osmnx as ox
from pathlib import Path


def get_osm_buildings_points(osm_file_path):
    """Extract building polygon points from OSM file."""
    buildings = ox.features_from_xml(osm_file_path, tags={"building": True})
    osm_building_list = []
    for _, building in buildings.iterrows():
        if building.geometry.geom_type == "Polygon":
            exterior_coords = building.geometry.exterior.coords
            for i in range(len(exterior_coords) - 1):
                start_point = [exterior_coords[i][0], exterior_coords[i][1], 0]
                end_point = [exterior_coords[i + 1][0], exterior_coords[i + 1][1], 0]
                osm_building_list.append(start_point)
                osm_building_list.append(end_point)
    return np.array(osm_building_list) if osm_building_list else np.array([]).reshape(0, 3)


def get_osm_road_points(osm_file_path):
    """Extract road linestring points from OSM file."""
    tags = {
        "highway": [
            "motorway", "trunk", "primary", "secondary", "tertiary",
            "unclassified", "residential", "motorway_link", "trunk_link",
            "primary_link", "secondary_link", "tertiary_link",
            "living_street", "service", "pedestrian", "road",
            "cycleway", "foot", "footway", "path"
        ]
    }
    roads = ox.features_from_xml(osm_file_path, tags=tags)
    osm_road_list = []
    for _, road in roads.iterrows():
        if road.geometry.geom_type == "LineString":
            coords = np.array(road.geometry.xy).T
            for i in range(len(coords) - 1):
                start_point = [coords[i][0], coords[i][1], 0]
                end_point = [coords[i + 1][0], coords[i + 1][1], 0]
                osm_road_list.append(start_point)
                osm_road_list.append(end_point)
    return np.array(osm_road_list) if osm_road_list else np.array([]).reshape(0, 3)


def get_osm_trees_points(osm_file_path):
    """Extract tree polygon points from OSM file."""
    trees = ox.features_from_xml(osm_file_path, tags={"natural": "tree"})
    osm_tree_list = []
    for _, tree in trees.iterrows():
        if tree.geometry.geom_type == "Polygon":
            exterior_coords = tree.geometry.exterior.coords
            for i in range(len(exterior_coords) - 1):
                start_point = [exterior_coords[i][0], exterior_coords[i][1], 0]
                end_point = [exterior_coords[i + 1][0], exterior_coords[i + 1][1], 0]
                osm_tree_list.append(start_point)
                osm_tree_list.append(end_point)
    return np.array(osm_tree_list) if osm_tree_list else np.array([]).reshape(0, 3)


def get_osm_grassland_points(osm_file_path):
    """Extract grassland/park polygon points from OSM file."""
    grass = ox.features_from_xml(
        osm_file_path,
        tags={"landuse": ["grass", "recreation_ground"], "leisure": "park"}
    )
    osm_grass_list = []
    for _, g in grass.iterrows():
        if g.geometry.geom_type == "Polygon":
            exterior_coords = g.geometry.exterior.coords
            for i in range(len(exterior_coords) - 1):
                start_point = [exterior_coords[i][0], exterior_coords[i][1], 0]
                end_point = [exterior_coords[i + 1][0], exterior_coords[i + 1][1], 0]
                osm_grass_list.append(start_point)
                osm_grass_list.append(end_point)
    return np.array(osm_grass_list) if osm_grass_list else np.array([]).reshape(0, 3)


def get_osm_water_points(osm_file_path):
    """Extract water polygon points from OSM file."""
    water = ox.features_from_xml(osm_file_path, tags={"natural": "water"})
    osm_water_list = []
    for _, w in water.iterrows():
        if w.geometry.geom_type == "Polygon":
            exterior_coords = w.geometry.exterior.coords
            for i in range(len(exterior_coords) - 1):
                start_point = [exterior_coords[i][0], exterior_coords[i][1], 0]
                end_point = [exterior_coords[i + 1][0], exterior_coords[i + 1][1], 0]
                osm_water_list.append(start_point)
                osm_water_list.append(end_point)
    return np.array(osm_water_list) if osm_water_list else np.array([]).reshape(0, 3)


def convert_polyline_points_to_o3d(polyline_points, rgb_color):
    """Convert polyline points to Open3D LineSet."""
    polyline_pcd = o3d.geometry.LineSet()
    if len(polyline_points) > 0:
        polyline_lines_idx = [[i, i + 1] for i in range(0, len(polyline_points) - 1, 2)]
        polyline_pcd.points = o3d.utility.Vector3dVector(polyline_points)
        polyline_pcd.lines = o3d.utility.Vector2iVector(polyline_lines_idx)
        polyline_pcd.paint_uniform_color(rgb_color)
    return polyline_pcd


def visualize_osm_data(osm_file_path):
    """Load and visualize OSM data using Open3D."""
    osm_file_path = Path(osm_file_path)
    
    if not osm_file_path.exists():
        print(f"Error: OSM file not found: {osm_file_path}")
        return
    
    print(f"Loading OSM data from: {osm_file_path}")
    print("This may take a moment...")
    
    # Load different OSM features
    geometries = []
    
    # Load and visualize buildings (blue)
    print("Loading buildings...")
    osm_building_points = get_osm_buildings_points(osm_file_path)
    if len(osm_building_points) > 0:
        osm_building_pcd = convert_polyline_points_to_o3d(osm_building_points, [0, 0, 1])
        geometries.append(osm_building_pcd)
        print(f"  Found {len(osm_building_points)} building points")
    
    # Load and visualize roads (gray)
    print("Loading roads...")
    osm_road_points = get_osm_road_points(osm_file_path)
    if len(osm_road_points) > 0:
        osm_road_pcd = convert_polyline_points_to_o3d(osm_road_points, [0.5, 0.5, 0.5])
        geometries.append(osm_road_pcd)
        print(f"  Found {len(osm_road_points)} road points")
    
    # Load and visualize trees (green)
    print("Loading trees...")
    osm_tree_points = get_osm_trees_points(osm_file_path)
    if len(osm_tree_points) > 0:
        osm_tree_pcd = convert_polyline_points_to_o3d(osm_tree_points, [0, 1, 0])
        geometries.append(osm_tree_pcd)
        print(f"  Found {len(osm_tree_points)} tree points")
    
    # Load and visualize grassland (light green)
    print("Loading grassland/parks...")
    osm_grass_points = get_osm_grassland_points(osm_file_path)
    if len(osm_grass_points) > 0:
        osm_grass_pcd = convert_polyline_points_to_o3d(osm_grass_points, [0.5, 1, 0.5])
        geometries.append(osm_grass_pcd)
        print(f"  Found {len(osm_grass_points)} grassland points")
    
    # Load and visualize water (cyan)
    print("Loading water features...")
    osm_water_points = get_osm_water_points(osm_file_path)
    if len(osm_water_points) > 0:
        osm_water_pcd = convert_polyline_points_to_o3d(osm_water_points, [0, 1, 1])
        geometries.append(osm_water_pcd)
        print(f"  Found {len(osm_water_points)} water points")
    
    if len(geometries) == 0:
        print("No OSM features found to visualize.")
        return
    
    print(f"\nVisualizing {len(geometries)} feature types...")
    print("Color legend:")
    print("  Blue: Buildings")
    print("  Gray: Roads")
    print("  Green: Trees")
    print("  Light Green: Grassland/Parks")
    print("  Cyan: Water")
    
    # Visualize using Open3D
    o3d.visualization.draw_geometries(
        geometries,
        window_name="OSM Data Visualization",
        width=1200,
        height=800,
    )


if __name__ == "__main__":
    # OSM file path
    osm_file_path = "/media/donceykong/doncey_ssd_02/datasets/MCD/kth.osm"
    
    visualize_osm_data(osm_file_path)

