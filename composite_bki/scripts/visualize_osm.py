#!/usr/bin/env python3
"""
Visualization script for point clouds and OSM polygons.

Usage:
    python visualize_osm.py --scan data.bin --osm osm.bin [--labels data.label] [--voxel_size 0.2]
"""

import struct
import os
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation as R


# --- OSM Loader (from test_idea.py) ---
def load_osm_bin(bin_file):
    """
    Load OSM geometries from binary file.
    
    Binary format (from create_map_OSM_BEV_GEOM.py):
    - uint32_t: number of buildings
    - For each building: uint32_t point count, then float[2*n_points] (x,y pairs)
    - uint32_t: number of roads
    - For each road: uint32_t point count, then float[2*n_points] (x,y pairs)
    - uint32_t: number of grasslands
    - For each grassland: uint32_t point count, then float[2*n_points] (x,y pairs)
    - uint32_t: number of trees
    - For each tree: uint32_t point count, then float[2*n_points] (x,y pairs)
    - uint32_t: number of wood/forests
    - For each wood: uint32_t point count, then float[2*n_points] (x,y pairs)
    """
    if not os.path.exists(bin_file):
        raise FileNotFoundError(f"OSM bin file not found: {bin_file}")
    
    data = {}
    # NOTE: Only 5 categories in the binary format! (not 8)
    categories = ['buildings', 'roads', 'grasslands', 'trees', 'wood']

    with open(bin_file, 'rb') as f:
        for cat in categories:
            try:
                num_items_bytes = f.read(4)
                if not num_items_bytes:
                    print(f"Warning: Reached end of file at category {cat}")
                    break
                num_items = struct.unpack('I', num_items_bytes)[0]
                
                items = []
                for _ in range(num_items):
                    n_pts_bytes = f.read(4)
                    if not n_pts_bytes:
                        print(f"Warning: Incomplete data for {cat}")
                        break
                    n_pts = struct.unpack('I', n_pts_bytes)[0]
                    
                    bytes_data = f.read(n_pts * 2 * 4)
                    if len(bytes_data) < n_pts * 2 * 4:
                        print(f"Warning: Incomplete coordinate data for {cat}")
                        break
                    floats = struct.unpack(f'{n_pts * 2}f', bytes_data)
                    poly_coords = list(zip(floats[0::2], floats[1::2]))
                    items.append(poly_coords)
                
                data[cat] = items
            except struct.error as e:
                print(f"Warning: Failed to load {cat}: {e}")
                break
    
    # Calculate and store bounds
    all_x = []
    all_y = []
    for items in data.values():
        for coords in items:
            for x, y in coords:
                all_x.append(x)
                all_y.append(y)
    
    if all_x:
        data['world_bounds'] = {
            'x_min': min(all_x),
            'x_max': max(all_x),
            'y_min': min(all_y),
            'y_max': max(all_y)
        }
    
    return data


# --- Color Schemes ---
# NOTE: Only 5 categories exist in the OSM binary format
OSM_COLORS = {
    'buildings': [0.8, 0.2, 0.2],      # Red
    'roads': [0.3, 0.3, 0.3],          # Dark gray
    'grasslands': [0.6, 0.9, 0.6],     # Light green
    'trees': [0.1, 0.6, 0.1],          # Dark green
    'wood': [0.2, 0.4, 0.2],           # Forest green
}

OSM_LABEL_NAMES = {
    'buildings': 'Buildings',
    'roads': 'Roads',
    'grasslands': 'Grasslands',
    'trees': 'Trees',
    'wood': 'Wood/Forest',
}


# MCD Label Colors (from test_idea.py)
MCD_LABEL_COLORS = {
    0: [0.5, 0.5, 0.5],    # barrier - gray
    1: [0.0, 0.0, 1.0],    # bike - blue
    2: [0.8, 0.2, 0.2],    # building - red
    3: [0.6, 0.4, 0.2],    # chair - brown
    4: [0.5, 0.5, 0.4],    # cliff - beige
    5: [0.7, 0.5, 0.3],    # container - tan
    6: [0.8, 0.8, 0.2],    # curb - yellow
    7: [0.6, 0.3, 0.1],    # fence - dark brown
    8: [0.9, 0.1, 0.1],    # hydrant - bright red
    9: [0.2, 0.6, 0.8],    # infosign - light blue
    10: [1.0, 1.0, 1.0],   # lanemarking - white
    11: [0.3, 0.3, 0.3],   # noise - dark gray
    12: [0.6, 0.6, 0.6],   # other - medium gray
    13: [0.4, 0.4, 0.4],   # parkinglot - dark gray
    14: [1.0, 0.5, 0.0],   # pedestrian - orange
    15: [0.7, 0.7, 0.0],   # pole - olive
    16: [0.2, 0.2, 0.2],   # road - very dark gray
    17: [0.5, 0.3, 0.5],   # shelter - purple
    18: [0.7, 0.7, 0.7],   # sidewalk - light gray
    19: [0.6, 0.5, 0.4],   # stairs - tan
    20: [0.5, 0.4, 0.3],   # structure-other - brown-gray
    21: [1.0, 0.3, 0.0],   # traffic-cone - bright orange
    22: [0.0, 0.8, 0.8],   # traffic-sign - cyan
    23: [0.3, 0.5, 0.3],   # trashbin - dark green
    24: [0.4, 0.3, 0.1],   # treetrunk - dark brown
    25: [0.2, 0.8, 0.2],   # vegetation - green
    26: [0.0, 0.5, 1.0],   # vehicle-dynamic - sky blue
    27: [0.3, 0.3, 0.8],   # vehicle-other - dark blue
    28: [0.5, 0.5, 1.0],   # vehicle-static - light blue
}

MCD_LABEL_NAMES = {
    0: 'barrier',
    1: 'bike',
    2: 'building',
    3: 'chair',
    4: 'cliff',
    5: 'container',
    6: 'curb',
    7: 'fence',
    8: 'hydrant',
    9: 'infosign',
    10: 'lanemarking',
    11: 'noise',
    12: 'other',
    13: 'parkinglot',
    14: 'pedestrian',
    15: 'pole',
    16: 'road',
    17: 'shelter',
    18: 'sidewalk',
    19: 'stairs',
    20: 'structure-other',
    21: 'traffic-cone',
    22: 'traffic-sign',
    23: 'trashbin',
    24: 'treetrunk',
    25: 'vegetation',
    26: 'vehicle-dynamic',
    27: 'vehicle-other',
    28: 'vehicle-static',
}

# SemanticKITTI Label Colors
KITTI_LABEL_COLORS = {
    0: [0.0, 0.0, 0.0],       # unlabeled - black
    1: [0.5, 0.5, 0.5],       # outlier - gray
    10: [0.0, 0.0, 1.0],      # car - blue
    11: [0.7, 0.0, 1.0],      # bicycle - purple
    13: [0.0, 0.5, 1.0],      # bus - light blue
    15: [1.0, 0.0, 0.5],      # motorcycle - pink
    16: [0.5, 0.0, 0.5],      # on-rails - dark purple
    18: [0.0, 0.7, 0.7],      # truck - cyan
    20: [0.3, 0.3, 0.7],      # other-vehicle - blue-gray
    30: [1.0, 0.5, 0.0],      # person - orange
    31: [0.8, 0.3, 0.0],      # bicyclist - dark orange
    32: [1.0, 0.2, 0.4],      # motorcyclist - coral
    40: [0.2, 0.2, 0.2],      # road - dark gray
    44: [0.4, 0.4, 0.4],      # parking - gray
    48: [0.7, 0.7, 0.7],      # sidewalk - light gray
    49: [0.5, 0.5, 0.5],      # other-ground - medium gray
    50: [0.8, 0.2, 0.2],      # building - red
    51: [0.6, 0.3, 0.1],      # fence - brown
    52: [0.5, 0.4, 0.3],      # other-structure - tan
    60: [1.0, 1.0, 1.0],      # lane-marking - white
    70: [0.2, 0.8, 0.2],      # vegetation - green
    71: [0.4, 0.3, 0.1],      # trunk - dark brown
    72: [0.3, 0.6, 0.3],      # terrain - olive green
    80: [0.7, 0.7, 0.0],      # pole - yellow
    81: [0.0, 0.8, 0.8],      # traffic-sign - cyan
    99: [0.6, 0.6, 0.6],      # other-object - gray
}

KITTI_LABEL_NAMES = {
    0: 'unlabeled',
    1: 'outlier',
    10: 'car',
    11: 'bicycle',
    13: 'bus',
    15: 'motorcycle',
    16: 'on-rails',
    18: 'truck',
    20: 'other-vehicle',
    30: 'person',
    31: 'bicyclist',
    32: 'motorcyclist',
    40: 'road',
    44: 'parking',
    48: 'sidewalk',
    49: 'other-ground',
    50: 'building',
    51: 'fence',
    52: 'other-structure',
    60: 'lane-marking',
    70: 'vegetation',
    71: 'trunk',
    72: 'terrain',
    80: 'pole',
    81: 'traffic-sign',
    99: 'other-object',
}


def print_color_legend(format_name, label_names, label_colors):
    """
    Print color legend to console using ANSI escape codes for colors.
    """
    print(f"\n=== {format_name} Color Legend ===")
    print(f"{'ID':<12} {'Label':<20} {'Color (RGB)':<20} Sample")
    print("-" * 65)
    
    # Handle string keys (OSM) and int keys (Labels)
    sorted_ids = sorted(label_names.keys(), key=lambda x: (isinstance(x, str), x))
    
    for label_id in sorted_ids:
        if label_id not in label_colors:
            continue
            
        name = label_names[label_id]
        rgb = label_colors[label_id]
        
        # Convert 0-1 float to 0-255 int
        r, g, b = int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
        
        # ANSI escape for truecolor background: \033[48;2;r;g;bm
        # ANSI escape for reset: \033[0m
        # Using two spaces for the color block
        color_block = f"\033[48;2;{r};{g};{b}m    \033[0m"
        
        id_str = str(label_id)
        print(f"{id_str:<12} {name:<20} [{r:>3}, {g:>3}, {b:>3}]   {color_block}")
    print("=" * 65 + "\n")



def load_point_cloud(bin_file):
    """Load point cloud from binary file (x, y, z, intensity)."""
    scan = np.fromfile(bin_file, dtype=np.float32).reshape((-1, 4))
    points = scan[:, :3]
    intensity = scan[:, 3]
    return points, intensity


def load_labels(label_file):
    """Load semantic labels from binary file."""
    labels_raw = np.fromfile(label_file, dtype=np.uint32)
    labels = labels_raw & 0xFFFF  # Lower 16 bits = semantic label
    return labels


def load_poses(pose_file):
    """
    Load poses from CSV file.
    
    CSV format: num,t,x,y,z,qx,qy,qz,qw
    where:
        - num: frame number
        - t: timestamp
        - x,y,z: position in world coordinates
        - qx,qy,qz,qw: orientation quaternion
    
    Args:
        pose_file: Path to CSV file containing poses
    
    Returns:
        Dictionary mapping frame numbers to poses [x,y,z,qx,qy,qz,qw]
    """
    import pandas as pd
    
    if not os.path.exists(pose_file):
        raise FileNotFoundError(f"Pose file not found: {pose_file}")
    
    print(f"Loading poses from {pose_file}...")
    
    # Read CSV file
    df = pd.read_csv(pose_file)
    
    poses = {}
    for _, row in df.iterrows():
        frame_num = int(row['num'])
        x = float(row['x'])
        y = float(row['y'])
        z = float(row['z'])
        qx = float(row['qx'])
        qy = float(row['qy'])
        qz = float(row['qz'])
        qw = float(row['qw'])
        
        poses[frame_num] = np.array([x, y, z, qx, qy, qz, qw])
    
    print(f"Loaded {len(poses)} poses")
    return poses


def transform_points_to_world(points, pose):
    """
    Transform points from sensor frame to world frame using pose.
    
    Args:
        points: (N, 3) array of points in sensor frame
        pose: (7,) array [x, y, z, qx, qy, qz, qw]
    
    Returns:
        (N, 3) array of points in world frame
    """
    position = pose[:3]  # [x, y, z]
    quat = pose[3:7]     # [qx, qy, qz, qw]
    
    # Create 4x4 transformation matrix
    rotation_matrix = R.from_quat(quat).as_matrix()
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = position
    
    # Transform points to world coordinates
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    world_points = (transform_matrix @ points_homogeneous.T).T
    world_points_xyz = world_points[:, :3]
    
    return world_points_xyz


def get_frame_number_from_filename(scan_path):
    """
    Extract frame number from scan filename.
    
    Args:
        scan_path: Path to scan file (e.g., "0000000011.bin")
    
    Returns:
        Frame number as integer
    """
    filename = Path(scan_path).stem  # Get filename without extension
    try:
        return int(filename)
    except ValueError:
        raise ValueError(f"Could not extract frame number from filename: {filename}")


def voxelize_point_cloud(points, colors, voxel_size):
    """Voxelize point cloud using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    return voxel_grid


def create_osm_geometry(osm_data, z_offset=-0.5):
    """
    Create Open3D line sets for OSM polygons.
    
    Args:
        osm_data: Dictionary of OSM categories with polygon coordinates
        z_offset: Z height to place OSM polygons (default: -0.5 to place below points)
    
    Returns:
        List of Open3D LineSet objects
    """
    line_sets = []
    
    for category, polygons in osm_data.items():
        # Skip metadata entries
        if category in ['bounds', 'world_bounds']:
            continue
            
        color = OSM_COLORS.get(category, [0.5, 0.5, 0.5])
        
        for poly_coords in polygons:
            if len(poly_coords) < 2:
                continue
            
            # Convert 2D coordinates to 3D (add z_offset)
            points_3d = np.array([[x, y, z_offset] for x, y in poly_coords])
            
            # Create lines connecting consecutive points
            lines = [[i, i + 1] for i in range(len(points_3d) - 1)]
            
            # Close the polygon if it has more than 2 points
            if len(points_3d) > 2:
                lines.append([len(points_3d) - 1, 0])
            
            # Create LineSet
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points_3d)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
            
            line_sets.append(line_set)
    
    return line_sets


def create_osm_mesh(osm_data, z_offset=-0.5, thickness=0.05):
    """
    Create Open3D triangle meshes for OSM polygons (filled).
    
    Args:
        osm_data: Dictionary of OSM categories with polygon coordinates
        z_offset: Z height to place OSM polygons
        thickness: Thickness of the polygon mesh
    
    Returns:
        List of Open3D TriangleMesh objects
    """
    meshes = []
    
    for category, polygons in osm_data.items():
        # Skip metadata entries
        if category in ['bounds', 'world_bounds']:
            continue
            
        color = OSM_COLORS.get(category, [0.5, 0.5, 0.5])
        
        for poly_coords in polygons:
            if len(poly_coords) < 3:
                continue
            
            try:
                # Create a simple triangulation for the polygon
                points_3d = np.array([[x, y, z_offset] for x, y in poly_coords])
                
                # Simple fan triangulation from first vertex
                triangles = [[0, i, i + 1] for i in range(1, len(points_3d) - 1)]
                
                if len(triangles) == 0:
                    continue
                
                # Create mesh
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(points_3d)
                mesh.triangles = o3d.utility.Vector3iVector(triangles)
                mesh.paint_uniform_color(color)
                mesh.compute_vertex_normals()
                
                meshes.append(mesh)
            except Exception as e:
                print(f"Warning: Failed to create mesh for {category}: {e}")
                continue
    
    return meshes


def get_label_colors(labels, label_format='auto'):
    """
    Get colors for labels based on format.
    
    Args:
        labels: Array of semantic labels
        label_format: 'mcd', 'kitti', or 'auto' (auto-detect)
    
    Returns:
        Array of RGB colors (N, 3)
    """
    colors = np.zeros((len(labels), 3))
    
    # Auto-detect format based on label values
    if label_format == 'auto':
        unique_labels = np.unique(labels)
        max_label = np.max(unique_labels)
        
        # KITTI labels tend to be in ranges like 10, 30, 40, 50, 70, 80
        # MCD labels are typically 0-28
        if max_label > 30 or any(l in [40, 44, 48, 50, 70, 80, 81] for l in unique_labels):
            label_format = 'kitti'
            print("Auto-detected SemanticKITTI label format")
        else:
            label_format = 'mcd'
            print("Auto-detected MCD label format")
    
    color_map = KITTI_LABEL_COLORS if label_format == 'kitti' else MCD_LABEL_COLORS
    
    for i, label in enumerate(labels):
        if label in color_map:
            colors[i] = color_map[label]
        else:
            # Default color for unknown labels (magenta)
            colors[i] = [1.0, 0.0, 1.0]
    
    return colors, label_format


def visualize(scan_path, osm_path, labels_path=None, voxel_size=0.2, 
              use_mesh=False, label_format='auto', z_offset=-0.5, pose_path=None):
    """
    Main visualization function.
    
    Args:
        scan_path: Path to point cloud bin file
        osm_path: Path to OSM bin file
        labels_path: Optional path to labels file
        voxel_size: Voxel size for downsampling (0 = no voxelization)
        use_mesh: If True, render OSM as filled meshes, otherwise as wireframes
        label_format: 'mcd', 'kitti', or 'auto'
        z_offset: Z offset for OSM polygons (if -0.5, auto-calculated from point cloud ground)
        pose_path: Optional path to pose CSV file for transforming points to world frame
    """
    print("Loading point cloud...")
    points, intensity = load_point_cloud(scan_path)
    print(f"Loaded {len(points)} points")
    
    # Apply pose transformation if provided
    if pose_path:
        print("Applying pose transformation...")
        poses = load_poses(pose_path)
        frame_num = get_frame_number_from_filename(scan_path)
        
        if frame_num not in poses:
            print(f"WARNING: Frame {frame_num} not found in pose file!")
            print(f"Available frames: {sorted(poses.keys())[:10]}... (showing first 10)")
            print("Proceeding without transformation.")
        else:
            pose = poses[frame_num]
            print(f"Transforming frame {frame_num} with pose:")
            print(f"  Position: [{pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}]")
            print(f"  Quaternion: [{pose[3]:.4f}, {pose[4]:.4f}, {pose[5]:.4f}, {pose[6]:.4f}]")
            
            points = transform_points_to_world(points, pose)
            print(f"Points transformed to world frame")
            print(f"  X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
            print(f"  Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
            print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # Load labels if provided
    final_label_format = None
    if labels_path:
        print("Loading labels...")
        labels = load_labels(labels_path)
        if len(labels) != len(points):
            print(f"Warning: Label count ({len(labels)}) doesn't match point count ({len(points)})")
            labels = labels[:len(points)]  # Truncate or pad
        colors, final_label_format = get_label_colors(labels, label_format)
        
        # Print legend
        label_names = KITTI_LABEL_NAMES if final_label_format == 'kitti' else MCD_LABEL_NAMES
        label_colors_map = KITTI_LABEL_COLORS if final_label_format == 'kitti' else MCD_LABEL_COLORS
        print_color_legend(final_label_format.upper(), label_names, label_colors_map)
    else:
        # Use intensity for coloring
        print("No labels provided, using intensity for coloring")
        intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
        colors = np.stack([intensity_normalized] * 3, axis=1)  # Grayscale
    
    # Load OSM data
    print("Loading OSM geometries...")
    osm_data = load_osm_bin(osm_path)
    total_osm = sum(len(items) for items in osm_data.values())
    print(f"Loaded {total_osm} OSM geometries")
    
    # Auto-calculate z_offset to align OSM with point cloud ground plane
    # Find the minimum Z value (ground level) in the point cloud
    z_min = points[:, 2].min()
    z_percentile_5 = np.percentile(points[:, 2], 5)  # Use 5th percentile to avoid outliers
    
    # If z_offset is the default value, auto-calculate it
    if z_offset == -0.5:
        # Place OSM slightly below the ground plane (5th percentile)
        auto_z_offset = z_percentile_5 - 0.1
        print(f"Auto-calculated OSM Z-offset: {auto_z_offset:.2f}m")
        print(f"  Point cloud Z range: [{z_min:.2f}m, {points[:, 2].max():.2f}m]")
        print(f"  Using 5th percentile: {z_percentile_5:.2f}m")
        z_offset = auto_z_offset
    else:
        print(f"Using manual Z-offset: {z_offset:.2f}m")
    
    # Create visualizations
    geometries = []
    
    # Add point cloud
    if voxel_size > 0:
        print(f"Voxelizing point cloud with voxel size {voxel_size}...")
        voxel_grid = voxelize_point_cloud(points, colors, voxel_size)
        geometries.append(voxel_grid)
    else:
        print("Creating point cloud...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pcd)
    
    # Add OSM geometries with calculated z_offset
    if use_mesh:
        print("Creating OSM meshes...")
        osm_meshes = create_osm_mesh(osm_data, z_offset=z_offset)
        geometries.extend(osm_meshes)
    else:
        print("Creating OSM wireframes...")
        osm_lines = create_osm_geometry(osm_data, z_offset=z_offset)
        geometries.extend(osm_lines)
    
    # Visualize with custom controls
    print("\nVisualization Controls:")
    print("  - Mouse: Rotate (left), Pan (middle), Zoom (wheel)")
    print("  - Press 'P' to toggle point cloud visibility")
    print("  - Press 'O' to toggle OSM geometries visibility")
    print("  - Press 'H' for help")
    print("  - Press 'Q' to quit")
    print("\nLaunching visualization...")
    
    # Create custom visualizer for interactive controls
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Point Cloud + OSM Visualization", 
                     width=1280, height=720, left=50, top=50)
    
    # Add all geometries and keep track of them
    point_cloud_geo = geometries[0]  # First geometry is always the point cloud
    osm_geometries = geometries[1:]   # Rest are OSM geometries
    
    # Add to visualizer
    vis.add_geometry(point_cloud_geo)
    for osm_geo in osm_geometries:
        vis.add_geometry(osm_geo)
    
    # Track visibility state
    visibility_state = {
        'points_visible': True,
        'osm_visible': True
    }
    
    def toggle_points(vis):
        """Toggle point cloud visibility"""
        visibility_state['points_visible'] = not visibility_state['points_visible']
        
        if visibility_state['points_visible']:
            vis.add_geometry(point_cloud_geo, reset_bounding_box=False)
            print("✓ Point cloud: ON")
        else:
            vis.remove_geometry(point_cloud_geo, reset_bounding_box=False)
            print("✗ Point cloud: OFF")
        return False
    
    def toggle_osm(vis):
        """Toggle OSM geometries visibility"""
        visibility_state['osm_visible'] = not visibility_state['osm_visible']
        
        if visibility_state['osm_visible']:
            for osm_geo in osm_geometries:
                vis.add_geometry(osm_geo, reset_bounding_box=False)
            print("✓ OSM geometries: ON")
        else:
            for osm_geo in osm_geometries:
                vis.remove_geometry(osm_geo, reset_bounding_box=False)
            print("✗ OSM geometries: OFF")
        return False
    
    def show_legend(vis):
        """Show color legend in console"""
        if final_label_format:
            label_names = KITTI_LABEL_NAMES if final_label_format == 'kitti' else MCD_LABEL_NAMES
            label_colors_map = KITTI_LABEL_COLORS if final_label_format == 'kitti' else MCD_LABEL_COLORS
            print_color_legend(final_label_format.upper(), label_names, label_colors_map)
        else:
            print("\nNo semantic labels loaded (intensity mode).\n")
            
        # Also show OSM legend
        print_color_legend("OSM", OSM_LABEL_NAMES, OSM_COLORS)
        return False
    
    def show_help(vis):
        """Display help message"""
        print("\n" + "="*60)
        print("KEYBOARD CONTROLS")
        print("="*60)
        print("  [P] - Toggle point cloud visibility")
        print("  [O] - Toggle OSM geometries visibility")
        print("  [L] - Show color legend in console")
        print("  [H] - Show this help message")
        print("  [Q] - Quit visualization")
        print("\nMOUSE CONTROLS")
        print("  Left button   - Rotate view")
        print("  Middle button - Pan view")
        print("  Scroll wheel  - Zoom in/out")
        print("  Right button  - Zoom")
        print("\nCURRENT STATE:")
        print(f"  Point cloud: {'ON' if visibility_state['points_visible'] else 'OFF'}")
        print(f"  OSM geometries: {'ON' if visibility_state['osm_visible'] else 'OFF'}")
        print("="*60 + "\n")
        return False
    
    # Register keyboard callbacks
    vis.register_key_callback(ord('P'), toggle_points)  # Press 'P' for points
    vis.register_key_callback(ord('O'), toggle_osm)     # Press 'O' for OSM
    vis.register_key_callback(ord('L'), show_legend)    # Press 'L' for Legend
    vis.register_key_callback(ord('H'), show_help)      # Press 'H' for help
    
    # Run visualizer
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize point cloud with OSM polygons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic visualization with intensity coloring (OSM auto-aligned to ground)
  python visualize_osm.py --scan data.bin --osm osm.bin
  
  # With semantic labels
  python visualize_osm.py --scan data.bin --osm osm.bin --labels data.label
  
  # With pose transformation (RECOMMENDED for proper alignment)
  python visualize_osm.py --scan data.bin --osm osm.bin --pose pose_inW.csv
  
  # Complete example with all options
  python visualize_osm.py --scan data.bin --osm osm.bin --labels data.label --pose pose_inW.csv --voxel_size 0.3
  
  # With voxelization
  python visualize_osm.py --scan data.bin --osm osm.bin --labels data.label --voxel_size 0.3
  
  # With filled OSM polygons
  python visualize_osm.py --scan data.bin --osm osm.bin --mesh
  
  # Manual Z-offset (override auto-calculation)
  python visualize_osm.py --scan data.bin --osm osm.bin --z_offset 0.0
  
  # Force KITTI label format
  python visualize_osm.py --scan data.bin --osm osm.bin --labels data.label --format kitti
        """
    )
    
    parser.add_argument('--scan', type=str, required=True,
                        help='Path to point cloud .bin file (x,y,z,intensity)')
    parser.add_argument('--osm', type=str, required=True,
                        help='Path to OSM .bin file')
    parser.add_argument('--labels', type=str, default=None,
                        help='Optional path to .labels file for coloring points')
    parser.add_argument('--voxel_size', type=float, default=0.2,
                        help='Voxel size for downsampling (0 = no voxelization, default: 0.2)')
    parser.add_argument('--mesh', action='store_true',
                        help='Render OSM as filled meshes instead of wireframes')
    parser.add_argument('--format', type=str, choices=['auto', 'mcd', 'kitti'], default='auto',
                        help='Label format: auto, mcd, or kitti (default: auto)')
    parser.add_argument('--z_offset', type=float, default=-0.5,
                        help='Z offset for OSM polygons (default: auto-calculated from point cloud ground plane)')
    parser.add_argument('--pose', type=str, default=None,
                        help='Optional path to pose CSV file for transforming points to world frame')
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.scan).exists():
        print(f"Error: Scan file not found: {args.scan}")
        return
    
    if not Path(args.osm).exists():
        print(f"Error: OSM file not found: {args.osm}")
        return
    
    if args.labels and not Path(args.labels).exists():
        print(f"Error: Labels file not found: {args.labels}")
        return
    
    # Validate pose file if provided
    if args.pose and not Path(args.pose).exists():
        print(f"Error: Pose file not found: {args.pose}")
        return
    
    # Run visualization
    visualize(
        args.scan,
        args.osm,
        args.labels,
        args.voxel_size,
        args.mesh,
        args.format,
        args.z_offset,
        args.pose
    )


if __name__ == "__main__":
    main()
