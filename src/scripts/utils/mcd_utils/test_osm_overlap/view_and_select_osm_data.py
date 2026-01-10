#!/usr/bin/env python3
"""
View OSM data in 2D with interactive click to get lat-long coordinates.

This script loads OSM data from an XML file and visualizes various features
(buildings, roads, trees, grassland, water, etc.) in a 2D matplotlib plot.
Click anywhere on the plot to print the lat-long coordinates of that point.
"""

import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
from pathlib import Path


def get_osm_buildings_geometries(osm_file_path):
    """Extract building polygons from OSM file."""
    buildings = ox.features_from_xml(osm_file_path, tags={"building": True})
    building_polygons = []
    for _, building in buildings.iterrows():
        if building.geometry.geom_type == "Polygon":
            exterior_coords = list(building.geometry.exterior.coords)
            # Remove last point if it's duplicate of first (shapely convention)
            if len(exterior_coords) > 1 and exterior_coords[0] == exterior_coords[-1]:
                exterior_coords = exterior_coords[:-1]
            building_polygons.append(exterior_coords)
    return building_polygons


def get_osm_road_geometries(osm_file_path):
    """Extract road linestrings from OSM file."""
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
    road_lines = []
    for _, road in roads.iterrows():
        if road.geometry.geom_type == "LineString":
            coords = list(road.geometry.coords)
            road_lines.append(coords)
    return road_lines


def get_osm_trees_geometries(osm_file_path):
    """Extract tree polygons from OSM file."""
    trees = ox.features_from_xml(osm_file_path, tags={"natural": "tree"})
    tree_polygons = []
    for _, tree in trees.iterrows():
        if tree.geometry.geom_type == "Polygon":
            exterior_coords = list(tree.geometry.exterior.coords)
            if len(exterior_coords) > 1 and exterior_coords[0] == exterior_coords[-1]:
                exterior_coords = exterior_coords[:-1]
            tree_polygons.append(exterior_coords)
    return tree_polygons


def get_osm_grassland_geometries(osm_file_path):
    """Extract grassland/park polygons from OSM file."""
    grass = ox.features_from_xml(
        osm_file_path,
        tags={"landuse": ["grass", "recreation_ground"], "leisure": "park"}
    )
    grass_polygons = []
    for _, g in grass.iterrows():
        if g.geometry.geom_type == "Polygon":
            exterior_coords = list(g.geometry.exterior.coords)
            if len(exterior_coords) > 1 and exterior_coords[0] == exterior_coords[-1]:
                exterior_coords = exterior_coords[:-1]
            grass_polygons.append(exterior_coords)
    return grass_polygons


def get_osm_water_geometries(osm_file_path):
    """Extract water polygons from OSM file."""
    water = ox.features_from_xml(osm_file_path, tags={"natural": "water"})
    water_polygons = []
    for _, w in water.iterrows():
        if w.geometry.geom_type == "Polygon":
            exterior_coords = list(w.geometry.exterior.coords)
            if len(exterior_coords) > 1 and exterior_coords[0] == exterior_coords[-1]:
                exterior_coords = exterior_coords[:-1]
            water_polygons.append(exterior_coords)
    return water_polygons


def on_click(event):
    """Handle mouse click events to print lat-long coordinates."""
    if event.inaxes is None:
        return
    
    if event.button == 1:  # Left mouse button
        lon = event.xdata  # x-axis is longitude
        lat = event.ydata  # y-axis is latitude
        print(f"Clicked at: Latitude = {lat:.8f}, Longitude = {lon:.8f}")
        print(f"  (Formatted: {lat:.6f}, {lon:.6f})")


def visualize_osm_data(osm_file_path):
    """Load and visualize OSM data in 2D with interactive click."""
    osm_file_path = Path(osm_file_path)
    
    if not osm_file_path.exists():
        print(f"Error: OSM file not found: {osm_file_path}")
        return
    
    print(f"Loading OSM data from: {osm_file_path}")
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
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('OSM Data - Click anywhere to get lat-long coordinates', fontsize=14)
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
        Patch(facecolor='cyan', alpha=0.5, label='Water')
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
    # OSM file path
    osm_file_path = "/media/donceykong/doncey_ssd_02/datasets/MCD/kth.osm"
    
    visualize_osm_data(osm_file_path)
