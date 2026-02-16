#!/usr/bin/env python3
"""
Visualize OSM XML data directly.
Optimized version: Batches geometry creation to improve performance.
Includes interactive controls for manual alignment with trajectory.

Usage:
    python visualize_osm_xml.py --osm map.osm --pose pose_inW.csv
"""

import argparse
import xml.etree.ElementTree as ET
import numpy as np
import open3d as o3d
import math
import csv
from collections import defaultdict

# --- Color Scheme ---
OSM_COLORS = {
    'building': [0.8, 0.2, 0.2],      # Red
    'highway': [0.3, 0.3, 0.3],       # Dark gray (Roads)
    'landuse': [0.6, 0.9, 0.6],       # Light green
    'natural': [0.1, 0.6, 0.1],       # Dark green
    'barrier': [0.6, 0.3, 0.1],       # Brown
    'amenity': [0.5, 0.5, 1.0],       # Blueish (Parking, etc)
    'default': [0.5, 0.5, 0.5]        # Gray
}

OSM_LABELS = {
    'building': 'Buildings',
    'highway': 'Roads',
    'landuse': 'Landuse (Grass/Forest)',
    'natural': 'Natural',
    'barrier': 'Barriers/Fences',
    'amenity': 'Amenities',
    'default': 'Other'
}

def latlon_to_meters(lat, lon, origin_lat, origin_lon):
    """
    Simple flat-earth projection to convert lat/lon to local x/y meters.
    """
    R = 6378137.0 
    x = math.radians(lon - origin_lon) * math.cos(math.radians(origin_lat)) * R
    y = math.radians(lat - origin_lat) * R
    return x, y

def load_poses(csv_file):
    """
    Load poses from CSV file.
    Format: num,t,x,y,z,qx,qy,qz,qw
    """
    print(f"Loading poses from {csv_file}...")
    positions = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row['x'])
                y = float(row['y'])
                z = float(row['z'])
                positions.append([x, y, z])
            except (ValueError, KeyError):
                continue
                
    if not positions:
        print("Warning: No valid poses found in CSV.")
        return None
        
    print(f"Loaded {len(positions)} poses.")
    return np.array(positions)

def create_thick_lines(points, lines, color, radius=5.0):
    """
    Creates a mesh of cylinders to represent thick lines.
    """
    meshes = []
    points = np.array(points)
    
    for line in lines:
        start = points[line[0]]
        end = points[line[1]]
        vec = end - start
        length = np.linalg.norm(vec)
        
        if length < 0.01: continue
            
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=8)
        cyl.paint_uniform_color(color)
        
        z_axis = np.array([0, 0, 1])
        direction = vec / length
        
        rot_axis = np.cross(z_axis, direction)
        rot_angle = np.arccos(np.dot(z_axis, direction))
        
        if np.linalg.norm(rot_axis) > 0.001:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis / np.linalg.norm(rot_axis) * rot_angle)
            cyl.rotate(R, center=[0, 0, 0])
        elif np.dot(z_axis, direction) < 0:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1, 0, 0]) * np.pi)
            cyl.rotate(R, center=[0, 0, 0])
            
        midpoint = (start + end) / 2
        cyl.translate(midpoint)
        meshes.append(cyl)
        
    if not meshes: return None
    combined = meshes[0]
    for m in meshes[1:]: combined += m
    return combined

class OSMLoader:
    def __init__(self, xml_file):
        print(f"Loading {xml_file}...")
        self.tree = ET.parse(xml_file)
        self.root = self.tree.getroot()
        self.nodes = {}  # id -> (x, y)
        self.ways = []   # list of (tags, node_ids)
        self.bounds = {'minlat': 0, 'minlon': 0, 'maxlat': 0, 'maxlon': 0}
        self.origin_lat = 0
        self.origin_lon = 0
        self._parse()

    def _parse(self):
        # 1. Parse Bounds
        bounds = self.root.find('bounds')
        if bounds is not None:
            self.bounds['minlat'] = float(bounds.get('minlat'))
            self.bounds['minlon'] = float(bounds.get('minlon'))
            self.bounds['maxlat'] = float(bounds.get('maxlat'))
            self.bounds['maxlon'] = float(bounds.get('maxlon'))
            self.origin_lat = (self.bounds['minlat'] + self.bounds['maxlat']) / 2.0
            self.origin_lon = (self.bounds['minlon'] + self.bounds['maxlon']) / 2.0
        else:
            first_node = self.root.find('node')
            if first_node is not None:
                self.origin_lat = float(first_node.get('lat'))
                self.origin_lon = float(first_node.get('lon'))

        print(f"Origin set to: {self.origin_lat:.6f}, {self.origin_lon:.6f}")

        # 2. Parse Nodes
        print("Parsing and projecting nodes...")
        count = 0
        for node in self.root.findall('node'):
            nid = node.get('id')
            lat = float(node.get('lat'))
            lon = float(node.get('lon'))
            x, y = latlon_to_meters(lat, lon, self.origin_lat, self.origin_lon)
            self.nodes[nid] = (x, y)
            count += 1
        print(f"Processed {count} nodes.")

        # 3. Parse Ways
        print("Parsing ways...")
        for way in self.root.findall('way'):
            node_ids = []
            tags = {}
            for nd in way.findall('nd'):
                node_ids.append(nd.get('ref'))
            for tag in way.findall('tag'):
                tags[tag.get('k')] = tag.get('v')
            if node_ids:
                self.ways.append({'nodes': node_ids, 'tags': tags})
        print(f"Processed {len(self.ways)} ways.")

    def get_geometries(self, z_offset=-0.5, thickness=10.0):
        print(f"Batching geometries (Thickness: {thickness}m)...")
        batches = defaultdict(lambda: {'points': [], 'lines': [], 'color': []})
        
        for way in self.ways:
            tags = way['tags']
            node_ids = way['nodes']
            cat = 'default'
            color = OSM_COLORS['default']
            
            if 'building' in tags:
                cat = 'building'; color = OSM_COLORS['building']
            elif 'highway' in tags:
                cat = 'highway'; color = OSM_COLORS['highway']
            elif 'landuse' in tags and tags['landuse'] in ['grass', 'meadow', 'forest', 'commercial', 'residential']:
                cat = 'landuse'; color = OSM_COLORS['landuse']
            elif 'natural' in tags:
                cat = 'natural'; color = OSM_COLORS['natural']
            elif 'barrier' in tags:
                cat = 'barrier'; color = OSM_COLORS['barrier']
            elif 'amenity' in tags:
                cat = 'amenity'; color = OSM_COLORS['amenity']

            points = []
            for nid in node_ids:
                if nid in self.nodes:
                    x, y = self.nodes[nid]
                    points.append([x, y, z_offset])
            
            if len(points) < 2: continue
                
            batch = batches[cat]
            start_idx = len(batch['points'])
            batch['points'].extend(points)
            batch['color'] = color
            
            n_pts = len(points)
            lines = [[start_idx + i, start_idx + i + 1] for i in range(n_pts - 1)]
            is_polygon = cat in ['building', 'landuse', 'amenity', 'natural']
            if is_polygon and n_pts > 2:
                 lines.append([start_idx + n_pts - 1, start_idx])
            
            batch['lines'].extend(lines)

        print("Constructing Open3D objects...")
        geometries = []
        for cat, data in batches.items():
            if not data['points']: continue
            mesh = create_thick_lines(data['points'], data['lines'], data['color'], radius=thickness/2.0)
            if mesh: geometries.append(mesh)
            
        print(f"Created {len(geometries)} merged geometry objects.")
        return geometries

def main():
    parser = argparse.ArgumentParser(description="Visualize OSM XML with Manual Alignment")
    parser.add_argument("--osm", type=str, required=True, help="Path to .osm file")
    parser.add_argument("--pose", type=str, required=False, help="Path to pose CSV file")
    parser.add_argument("--z_offset", type=float, default=0.0, help="Z height for lines")
    parser.add_argument("--thickness", type=float, default=20.0, help="Line thickness in meters")
    args = parser.parse_args()

    # Load Pose Trajectory
    traj_geom = None
    trajectory_z_mean = 0.0
    
    if args.pose:
        poses = load_poses(args.pose)
        if poses is not None:
            # Create trajectory line
            points = poses[:, :3]
            
            # Calculate mean Z to align map
            trajectory_z_mean = np.mean(points[:, 2])
            print(f"Trajectory Mean Z: {trajectory_z_mean:.2f}m")
            
            lines = [[i, i+1] for i in range(len(points)-1)]
            
            # Make trajectory thick and bright orange for visibility
            # Orange RGB: [1.0, 0.5, 0.0]
            traj_geom = create_thick_lines(points, lines, [1.0, 0.5, 0.0], radius=args.thickness) # Same thickness
            print(f"Created trajectory with {len(points)} points")

    # Load OSM (Aligned to Trajectory Z)
    # If no trajectory, use default 0.0 or args.z_offset
    map_z = trajectory_z_mean if args.pose else args.z_offset
    print(f"Aligning OSM Map to Z={map_z:.2f}m")
    
    loader = OSMLoader(args.osm)
    osm_geoms = loader.get_geometries(z_offset=map_z, thickness=args.thickness)

    # Combine for rendering
    geoms = osm_geoms[:]
    if traj_geom:
        geoms.append(traj_geom)
        
    # Coordinate frame (Removed per request)
    # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0, origin=[0, 0, 0])
    # geoms.append(axes)

    # Visualization Setup
    print("Launching visualization...")
    print("Controls:")
    print("  Arrows: Translate OSM Map")
    print("  C/D: Rotate OSM Map")
    print("  +/-: Increase/Decrease Step Size")
    print("  S: Save Aligned Map to XML")
    print("  T: Reset Top-Down View")
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"OSM Align: {args.osm}", width=1280, height=720)
    
    for geo in geoms:
        vis.add_geometry(geo)

    # --- Alignment State ---
    state = {
        'trans_step': 50.0,  # Meters per keypress
        'rot_step': 1.0,     # Degrees per keypress
        'total_trans': np.array([0.0, 0.0, 0.0]),
        'total_rot': 0.0     # Degrees
    }

    def update_map_transform(vis, trans_delta=None, rot_delta=0.0):
        # We need to transform ONLY the OSM geometries, not the trajectory
        # 1. Revert previous transform (Open3D transforms are destructive/accumulative on vertices)
        # However, undoing rotation around an arbitrary center is hard.
        # EASIER STRATEGY: 
        # Apply the incremental delta directly to the meshes.
        
        # Center of rotation (Origin 0,0,0)
        center = np.array([0, 0, 0])
        
        for geo in osm_geoms:
            if rot_delta != 0.0:
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, 0, 1]) * np.radians(rot_delta))
                geo.rotate(R, center=center)
            
            if trans_delta is not None:
                geo.translate(trans_delta)
                
            vis.update_geometry(geo)
            
        # Update State Tracking
        if trans_delta is not None: state['total_trans'] += trans_delta
        state['total_rot'] += rot_delta
        
        print(f"Total Shift: X={state['total_trans'][0]:.1f}m, Y={state['total_trans'][1]:.1f}m | Rot: {state['total_rot']:.1f} deg")
        return True

    def save_transformed_osm(vis):
        """Applies the current transformation to the original Lat/Lon and saves new XML."""
        output_file = args.osm.replace('.osm', '_aligned.osm')
        print(f"\nSaving aligned map to {output_file}...")
        
        # We need to inverse the projection: Meters -> Lat/Lon
        # x = (lon - lon0) * cos(lat0) * R  =>  lon = x / (R * cos(lat0)) + lon0
        # y = (lat - lat0) * R              =>  lat = y / R + lat0
        
        R = 6378137.0
        lat0_rad = math.radians(loader.origin_lat)
        lon0_rad = math.radians(loader.origin_lon)
        cos_lat0 = math.cos(lat0_rad)
        
        # Current Transform
        dx, dy, dz = state['total_trans']
        theta_rad = math.radians(state['total_rot'])
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        
        # Helper to transform a point (x,y)
        def transform_point(x, y):
            # Rotation around origin (0,0)
            x_rot = x * cos_t - y * sin_t
            y_rot = x * sin_t + y * cos_t
            # Translation
            return x_rot + dx, y_rot + dy

        # Apply to all nodes in the XML tree
        count = 0
        for node in loader.tree.findall('node'):
            nid = node.get('id')
            if nid in loader.nodes:
                # Original local coordinates (before visualizer transform)
                orig_x, orig_y = loader.nodes[nid]
                
                # Apply visualizer transform
                new_x, new_y = transform_point(orig_x, orig_y)
                
                # Convert back to Lat/Lon
                new_lat_rad = (new_y / R) + lat0_rad
                new_lon_rad = (new_x / (R * cos_lat0)) + lon0_rad
                
                new_lat = math.degrees(new_lat_rad)
                new_lon = math.degrees(new_lon_rad)
                
                node.set('lat', f"{new_lat:.7f}")
                node.set('lon', f"{new_lon:.7f}")
                count += 1
                
        loader.tree.write(output_file, encoding='UTF-8', xml_declaration=True)
        print(f"Saved {count} aligned nodes.")
        return False

    def change_step_size(vis, direction):
        if direction > 0:
            state['trans_step'] *= 2.0
            state['rot_step'] *= 2.0
            print(f"Step Size INCREASED: Trans={state['trans_step']}m, Rot={state['rot_step']} deg")
        else:
            state['trans_step'] /= 2.0
            state['rot_step'] /= 2.0
            print(f"Step Size DECREASED: Trans={state['trans_step']}m, Rot={state['rot_step']} deg")
        return False

    # Callbacks
    def move_left(vis): return update_map_transform(vis, trans_delta=np.array([-state['trans_step'], 0, 0]))
    def move_right(vis): return update_map_transform(vis, trans_delta=np.array([state['trans_step'], 0, 0]))
    def move_up(vis): return update_map_transform(vis, trans_delta=np.array([0, state['trans_step'], 0]))
    def move_down(vis): return update_map_transform(vis, trans_delta=np.array([0, -state['trans_step'], 0]))
    
    def rot_cw(vis): return update_map_transform(vis, rot_delta=-state['rot_step'])  # Clockwise (negative Z)
    def rot_ccw(vis): return update_map_transform(vis, rot_delta=state['rot_step']) # Counter-Clockwise
    
    def increase_step(vis): return change_step_size(vis, 1)
    def decrease_step(vis): return change_step_size(vis, -1)
    def save_map(vis): return save_transformed_osm(vis)

    # Key Bindings
    # Open3D Key codes: 262=Right, 263=Left, 264=Down, 265=Up (approximate, varies by backend)
    # Using GLFW codes commonly used in Open3D python bindings
    vis.register_key_callback(263, move_left)  # Left
    vis.register_key_callback(262, move_right) # Right
    vis.register_key_callback(265, move_up)    # Up
    vis.register_key_callback(264, move_down)  # Down
    vis.register_key_callback(ord('C'), rot_cw)
    vis.register_key_callback(ord('D'), rot_ccw)
    
    vis.register_key_callback(ord('='), increase_step) # + key (usually)
    vis.register_key_callback(ord('+'), increase_step) # Keypad +
    vis.register_key_callback(ord('-'), decrease_step) # - key
    vis.register_key_callback(ord('S'), save_map)      # S for Save

    # Initial View
    vis.poll_events()
    vis.update_renderer()
    vis.reset_view_point(True)
    
    ctr = vis.get_view_control()
    ctr.set_constant_z_far(1000000)
    ctr.set_constant_z_near(0.1)
    
    # Top Down Reset
    def reset_top_down(vis):
        ctr = vis.get_view_control()
        vis.reset_view_point(True)
        ctr.set_front([0, 0, -1]) 
        ctr.set_lookat([0, 0, 0]) # Look at origin
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.6)
        ctr.set_constant_z_far(1000000)
        return False
    
    vis.register_key_callback(ord('T'), reset_top_down)
    
    reset_top_down(vis) # Apply immediately
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
