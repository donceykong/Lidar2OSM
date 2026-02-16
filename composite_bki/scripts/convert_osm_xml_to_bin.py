#!/usr/bin/env python3
"""
Convert OSM XML (.osm) to Custom Binary Format (.bin).
Compatible with the C++ BKI loader and visualization scripts.

Usage:
    python convert_osm_xml_to_bin.py --input map.osm --output map.bin
"""

import argparse
import struct
import math
import xml.etree.ElementTree as ET
import numpy as np

# Categories MUST match the order in mcd_config.yaml and the C++ loader
CATEGORIES = [
    'buildings',
    'roads',
    'grasslands',
    'trees',
    'wood',
    'poles',
    'traffic_signs',
    'barriers'
]

def latlon_to_meters(lat, lon, origin_lat, origin_lon):
    """Flat-earth projection matching the visualization/alignment scripts."""
    R = 6378137.0 
    x = math.radians(lon - origin_lon) * math.cos(math.radians(origin_lat)) * R
    y = math.radians(lat - origin_lat) * R
    return x, y

class OSMConverter:
    def __init__(self, xml_file):
        print(f"Parsing {xml_file}...")
        self.tree = ET.parse(xml_file)
        self.root = self.tree.getroot()
        self.nodes = {}  # id -> (x, y)
        self.categorized_geoms = {cat: [] for cat in CATEGORIES}
        
        self.origin_lat = 0
        self.origin_lon = 0
        self._parse_origin()
        self._process_nodes()
        self._process_ways()
        self._process_node_features() # Handle point features like trees/poles

    def _parse_origin(self):
        # Try bounds first (standard OSM)
        bounds = self.root.find('bounds')
        if bounds is not None:
            minlat = float(bounds.get('minlat'))
            minlon = float(bounds.get('minlon'))
            maxlat = float(bounds.get('maxlat'))
            maxlon = float(bounds.get('maxlon'))
            self.origin_lat = (minlat + maxlat) / 2.0
            self.origin_lon = (minlon + maxlon) / 2.0
            print(f"Origin from bounds: {self.origin_lat}, {self.origin_lon}")
        else:
            # Fallback to first node
            node = self.root.find('node')
            if node is not None:
                self.origin_lat = float(node.get('lat'))
                self.origin_lon = float(node.get('lon'))
                print(f"Origin from first node: {self.origin_lat}, {self.origin_lon}")
            else:
                raise ValueError("No bounds or nodes found in XML")

    def _process_nodes(self):
        print("Projecting nodes...")
        for node in self.root.findall('node'):
            nid = node.get('id')
            lat = float(node.get('lat'))
            lon = float(node.get('lon'))
            x, y = latlon_to_meters(lat, lon, self.origin_lat, self.origin_lon)
            self.nodes[nid] = (x, y)

    def _process_node_features(self):
        """Extract point features defined on nodes (trees, poles, signs)."""
        count = 0
        for node in self.root.findall('node'):
            nid = node.get('id')
            if nid not in self.nodes: continue
            
            tags = {tag.get('k'): tag.get('v') for tag in node.findall('tag')}
            if not tags: continue

            cat = None
            if 'natural' in tags and tags['natural'] == 'tree':
                cat = 'trees'
            elif 'man_made' in tags and tags['man_made'] in ['pole', 'mast', 'flagpole']:
                cat = 'poles'
            elif 'highway' in tags and tags['highway'] == 'street_lamp':
                cat = 'poles'
            elif 'power' in tags and tags['power'] == 'pole':
                cat = 'poles'
            elif 'traffic_sign' in tags or ('highway' in tags and tags['highway'] == 'traffic_signals'):
                cat = 'traffic_signs'
            elif 'barrier' in tags:
                cat = 'barriers'
            
            if cat:
                # Add as a single-point polygon
                self.categorized_geoms[cat].append([self.nodes[nid]])
                count += 1
        print(f"Extracted {count} point features from nodes.")

    def _process_ways(self):
        print("Processing ways...")
        count = 0
        for way in self.root.findall('way'):
            node_ids = [nd.get('ref') for nd in way.findall('nd')]
            tags = {tag.get('k'): tag.get('v') for tag in way.findall('tag')}
            
            # Skip ways with missing nodes
            points = []
            for nid in node_ids:
                if nid in self.nodes:
                    points.append(self.nodes[nid])
            
            if len(points) < 2: continue

            cat = None
            # Classification Logic (Order matters for priority)
            if 'building' in tags:
                cat = 'buildings'
            elif 'highway' in tags:
                cat = 'roads'
            elif 'landuse' in tags:
                v = tags['landuse']
                if v in ['forest']: cat = 'wood'
                elif v in ['grass', 'meadow', 'recreation_ground', 'greenfield', 'village_green']: cat = 'grasslands'
            elif 'natural' in tags:
                v = tags['natural']
                if v == 'wood': cat = 'wood'
                elif v == 'tree': cat = 'trees' # Row of trees
                elif v in ['grass', 'scrub', 'heath']: cat = 'grasslands'
            elif 'leisure' in tags:
                v = tags['leisure']
                if v in ['park', 'garden', 'golf_course']: cat = 'grasslands'
            elif 'barrier' in tags:
                cat = 'barriers'
            elif 'man_made' in tags and tags['man_made'] in ['fence', 'wall']:
                cat = 'barriers'
            
            if cat:
                self.categorized_geoms[cat].append(points)
                count += 1
        print(f"Extracted {count} features from ways.")

    def write_binary(self, output_file):
        print(f"Writing to {output_file}...")
        with open(output_file, 'wb') as f:
            total_written = 0
            for cat in CATEGORIES:
                geoms = self.categorized_geoms[cat]
                count = len(geoms)
                print(f"  {cat}: {count}")
                
                # Write number of items in this category
                f.write(struct.pack('I', count))
                
                for points in geoms:
                    n_pts = len(points)
                    f.write(struct.pack('I', n_pts))
                    
                    # Write points as flat float array [x1, y1, x2, y2, ...]
                    flat_coords = []
                    for x, y in points:
                        flat_coords.extend([x, y])
                    
                    f.write(struct.pack(f'{len(flat_coords)}f', *flat_coords))
                
                total_written += count
        print(f"Done. Written {total_written} geometries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input .osm XML file")
    parser.add_argument("--output", required=True, help="Output .bin file")
    args = parser.parse_args()

    converter = OSMConverter(args.input)
    converter.write_binary(args.output)
