#!/usr/bin/env python3
"""
Compare OSM Binary and OSM XML files visually.
Loads both formats and displays them in Open3D with toggle controls.

Usage:
    python compare_osm.py --bin map.bin --osm map.osm
"""

import argparse
import struct
import os
import math
import numpy as np
import open3d as o3d
import xml.etree.ElementTree as ET
from collections import defaultdict

# --- Shared Color Scheme ---
OSM_COLORS = {
    'buildings': [0.8, 0.2, 0.2],      # Red
    'roads': [0.3, 0.3, 0.3],          # Dark gray
    'grasslands': [0.6, 0.9, 0.6],     # Light green
    'trees': [0.1, 0.6, 0.1],          # Dark green
    'wood': [0.2, 0.4, 0.2],           # Forest green
    'poles': [0.7, 0.7, 0.0],          # Yellow
    'traffic_signs': [0.0, 0.8, 0.8],  # Cyan
    'barriers': [0.5, 0.5, 0.5],       # Gray
    'default': [0.5, 0.5, 0.5]
}

# Mapping from XML tags to Binary categories
XML_TO_BIN_CAT = {
    'building': 'buildings',
    'highway': 'roads',
    'landuse': 'grasslands', # Simplified mapping
    'natural': 'trees',      # Simplified mapping
    'barrier': 'barriers',
    'amenity': 'buildings'   # Parking often grouped with buildings or handled separately
}

# --- 1. Binary Loader (from visualize_osm.py) ---
def load_osm_bin(bin_file):
    if not os.path.exists(bin_file):
        raise FileNotFoundError(f"OSM bin file not found: {bin_file}")
    
    data = {}
    # Categories expected in the binary file
    categories = ['buildings', 'roads', 'grasslands', 'trees', 'wood', 'poles', 'traffic_signs', 'barriers']

    with open(bin_file, 'rb') as f:
        for cat in categories:
            try:
                num_items_bytes = f.read(4)
                if not num_items_bytes: break
                num_items = struct.unpack('I', num_items_bytes)[0]
                
                items = []
                for _ in range(num_items):
                    n_pts_bytes = f.read(4)
                    if not n_pts_bytes: break
                    n_pts = struct.unpack('I', n_pts_bytes)[0]
                    
                    bytes_data = f.read(n_pts * 2 * 4)
                    if len(bytes_data) < n_pts * 2 * 4: break
                    floats = struct.unpack(f'{n_pts * 2}f', bytes_data)
                    poly_coords = list(zip(floats[0::2], floats[1::2]))
                    items.append(poly_coords)
                
                data[cat] = items
            except struct.error:
                break
    return data

def create_bin_geometries(osm_data, z_offset=0.0):
    geometries = []
    for category, polygons in osm_data.items():
        color = OSM_COLORS.get(category, OSM_COLORS['default'])
        
        for poly_coords in polygons:
            if len(poly_coords) < 2: continue
            
            points_3d = np.array([[x, y, z_offset] for x, y in poly_coords])
            lines = [[i, i + 1] for i in range(len(points_3d) - 1)]
            if len(points_3d) > 2: # Close loop
                lines.append([len(points_3d) - 1, 0])
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points_3d)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color(color)
            geometries.append(line_set)
    return geometries

# --- 2. XML Loader (from visualize_osm_xml.py) ---
def latlon_to_meters(lat, lon, origin_lat, origin_lon):
    R = 6378137.0 
    x = math.radians(lon - origin_lon) * math.cos(math.radians(origin_lat)) * R
    y = math.radians(lat - origin_lat) * R
    return x, y

class OSMLoader:
    def __init__(self, xml_file):
        self.tree = ET.parse(xml_file)
        self.root = self.tree.getroot()
        self.nodes = {}
        self.ways = []
        self.origin_lat = 0
        self.origin_lon = 0
        self._parse()

    def _parse(self):
        # Parse Bounds for Origin
        bounds = self.root.find('bounds')
        if bounds is not None:
            minlat = float(bounds.get('minlat'))
            minlon = float(bounds.get('minlon'))
            maxlat = float(bounds.get('maxlat'))
            maxlon = float(bounds.get('maxlon'))
            self.origin_lat = (minlat + maxlat) / 2.0
            self.origin_lon = (minlon + maxlon) / 2.0
        else:
            first_node = self.root.find('node')
            if first_node is not None:
                self.origin_lat = float(first_node.get('lat'))
                self.origin_lon = float(first_node.get('lon'))
        
        print(f"XML Origin: {self.origin_lat:.6f}, {self.origin_lon:.6f}")

        # Parse Nodes
        for node in self.root.findall('node'):
            nid = node.get('id')
            lat = float(node.get('lat'))
            lon = float(node.get('lon'))
            x, y = latlon_to_meters(lat, lon, self.origin_lat, self.origin_lon)
            self.nodes[nid] = (x, y)

        # Parse Ways
        for way in self.root.findall('way'):
            node_ids = [nd.get('ref') for nd in way.findall('nd')]
            tags = {tag.get('k'): tag.get('v') for tag in way.findall('tag')}
            if node_ids:
                self.ways.append({'nodes': node_ids, 'tags': tags})

    def get_geometries(self, z_offset=0.0):
        geometries = []
        for way in self.ways:
            tags = way['tags']
            node_ids = way['nodes']
            
            # Determine category
            cat = 'default'
            if 'building' in tags: cat = 'buildings'
            elif 'highway' in tags: cat = 'roads'
            elif 'landuse' in tags: cat = 'grasslands'
            elif 'natural' in tags: cat = 'trees'
            elif 'barrier' in tags: cat = 'barriers'
            
            color = OSM_COLORS.get(cat, OSM_COLORS['default'])
            
            # Make XML slightly brighter/different to distinguish from BIN
            # e.g. add tint
            color = [min(1.0, c + 0.2) for c in color] 

            points = []
            for nid in node_ids:
                if nid in self.nodes:
                    x, y = self.nodes[nid]
                    points.append([x, y, z_offset])
            
            if len(points) < 2: continue
            
            lines = [[i, i + 1] for i in range(len(points) - 1)]
            # Close loops for polygons
            is_polygon = cat in ['buildings', 'grasslands', 'trees']
            if len(points) > 2 and is_polygon:
                 lines.append([len(points) - 1, 0])
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color(color)
            geometries.append(line_set)
            
        return geometries

def main():
    parser = argparse.ArgumentParser(description="Compare OSM Binary vs XML")
    parser.add_argument("--bin", type=str, required=True, help="Path to .bin file")
    parser.add_argument("--osm", type=str, required=True, help="Path to .osm file")
    args = parser.parse_args()

    print("Loading Binary Data...")
    bin_data = load_osm_bin(args.bin)
    bin_geoms = create_bin_geometries(bin_data, z_offset=0.0) # Bin at Z=0
    print(f"Loaded {len(bin_geoms)} binary geometries.")

    print("Loading XML Data...")
    xml_loader = OSMLoader(args.osm)
    xml_geoms = xml_loader.get_geometries(z_offset=2.0) # XML at Z=2 (floating above)
    print(f"Loaded {len(xml_geoms)} XML geometries.")

    # Visualization
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Compare OSM Formats (Bin=Bottom, XML=Top)", width=1280, height=720)

    # Add geometries
    for g in bin_geoms: vis.add_geometry(g)
    for g in xml_geoms: vis.add_geometry(g)

    # State tracking
    # We use a mutable object (list) to store boolean state so it can be modified in callbacks
    # state[0] = bin_visible, state[1] = xml_visible
    state = [True, True]

    def toggle_bin(vis):
        state[0] = not state[0]
        print(f"Binary Visible: {state[0]}")
        if state[0]:
            for g in bin_geoms: vis.add_geometry(g, reset_bounding_box=False)
        else:
            for g in bin_geoms: vis.remove_geometry(g, reset_bounding_box=False)
        return False

    def toggle_xml(vis):
        state[1] = not state[1]
        print(f"XML Visible: {state[1]}")
        if state[1]:
            for g in xml_geoms: vis.add_geometry(g, reset_bounding_box=False)
        else:
            for g in xml_geoms: vis.remove_geometry(g, reset_bounding_box=False)
        return False

    print("\nCONTROLS:")
    print("  [B] Toggle Binary Map (Bottom, Darker)")
    print("  [X] Toggle XML Map (Top, Lighter)")
    print("  [Q] Quit")

    vis.register_key_callback(ord('B'), toggle_bin)
    vis.register_key_callback(ord('X'), toggle_xml)

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
