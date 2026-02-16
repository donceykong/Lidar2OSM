#!/usr/bin/env python3
"""
Visualize a voxel map created by build_voxel_map.py

This script loads a saved voxel map and visualizes it using Open3D.
"""

import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
from build_voxel_map import VoxelMap


# MCD Label Colors (from visualize_osm.py)
MCD_LABEL_COLORS = {
    0: [0.5, 0.5, 0.5],    # barrier
    1: [0.0, 0.0, 1.0],    # bike
    2: [0.8, 0.2, 0.2],    # building
    7: [0.6, 0.3, 0.1],    # fence
    13: [0.4, 0.4, 0.4],   # parkinglot
    14: [1.0, 0.5, 0.0],   # pedestrian
    15: [0.7, 0.7, 0.0],   # pole
    16: [0.2, 0.2, 0.2],   # road
    18: [0.7, 0.7, 0.7],   # sidewalk
    22: [0.0, 0.8, 0.8],   # traffic-sign
    24: [0.4, 0.3, 0.1],   # treetrunk
    25: [0.2, 0.8, 0.2],   # vegetation
    26: [0.0, 0.5, 1.0],   # vehicle-dynamic
}

# KITTI Label Colors (from visualize_osm.py)
KITTI_LABEL_COLORS = {
    0: [0.0, 0.0, 0.0],    # unlabeled
    10: [0.0, 0.0, 1.0],   # car
    30: [1.0, 0.5, 0.0],   # person
    40: [0.2, 0.2, 0.2],   # road
    44: [0.4, 0.4, 0.4],   # parking
    48: [0.7, 0.7, 0.7],   # sidewalk
    50: [0.8, 0.2, 0.2],   # building
    51: [0.6, 0.3, 0.1],   # fence
    70: [0.2, 0.8, 0.2],   # vegetation
    71: [0.4, 0.3, 0.1],   # trunk
    80: [0.7, 0.7, 0.0],   # pole
    81: [0.0, 0.8, 0.8],   # traffic-sign
}


def get_label_colors(labels, label_format='auto'):
    """
    Get colors for labels.
    
    Args:
        labels: Array of semantic labels
        label_format: 'mcd', 'kitti', or 'auto'
    
    Returns:
        (N, 3) array of RGB colors
    """
    colors = np.zeros((len(labels), 3))
    
    # Auto-detect format
    if label_format == 'auto':
        unique_labels = np.unique(labels)
        max_label = np.max(unique_labels)
        
        if max_label > 30 or any(l in [40, 44, 48, 50, 70, 80, 81] for l in unique_labels):
            label_format = 'kitti'
            print("Auto-detected KITTI label format")
        else:
            label_format = 'mcd'
            print("Auto-detected MCD label format")
    
    color_map = KITTI_LABEL_COLORS if label_format == 'kitti' else MCD_LABEL_COLORS
    
    for i, label in enumerate(labels):
        if label in color_map:
            colors[i] = color_map[label]
        else:
            # Default color (magenta) for unknown labels
            colors[i] = [1.0, 0.0, 1.0]
    
    return colors


def visualize_voxel_map(voxel_map_file, color_by='labels', label_format='auto', 
                       min_points=1, point_size=1.0):
    """
    Visualize a voxel map.
    
    Args:
        voxel_map_file: Path to saved voxel map (.pkl)
        color_by: 'labels', 'intensity', or 'height'
        label_format: 'mcd', 'kitti', or 'auto'
        min_points: Minimum points per voxel to display
        point_size: Size of points in visualization
    """
    print(f"Loading voxel map from {voxel_map_file}...")
    voxel_map = VoxelMap.load(voxel_map_file)
    
    # Get statistics
    stats = voxel_map.get_statistics()
    print("\nVoxel Map Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Get voxel centers
    print(f"\nExtracting voxel centers (min_points={min_points})...")
    centers = voxel_map.get_occupied_voxel_centers(min_points=min_points)
    print(f"Found {len(centers)} occupied voxels")
    
    if len(centers) == 0:
        print("ERROR: No occupied voxels found!")
        return
    
    # Determine coloring
    if color_by == 'labels':
        print("Coloring by semantic labels...")
        labels = voxel_map.get_voxel_labels(min_points=min_points)
        
        if len(labels) == 0 or np.all(labels == 0):
            print("WARNING: No labels found, falling back to intensity coloring")
            color_by = 'intensity'
        else:
            unique_labels = np.unique(labels)
            print(f"Unique labels: {sorted(unique_labels.tolist())}")
            colors = get_label_colors(labels, label_format)
    
    if color_by == 'intensity':
        print("Coloring by intensity...")
        intensities = voxel_map.get_voxel_intensities(min_points=min_points)
        
        if len(intensities) > 0 and not np.all(intensities == 0):
            # Normalize intensities
            int_min, int_max = intensities.min(), intensities.max()
            normalized = (intensities - int_min) / (int_max - int_min + 1e-8)
            colors = np.stack([normalized] * 3, axis=1)  # Grayscale
            print(f"Intensity range: [{int_min:.2f}, {int_max:.2f}]")
        else:
            print("WARNING: No intensity data, falling back to height coloring")
            color_by = 'height'
    
    if color_by == 'height':
        print("Coloring by height (Z coordinate)...")
        z_values = centers[:, 2]
        z_min, z_max = z_values.min(), z_values.max()
        normalized_z = (z_values - z_min) / (z_max - z_min + 1e-8)
        
        # Use a colormap-like coloring (blue -> green -> red)
        colors = np.zeros((len(centers), 3))
        colors[:, 2] = 1.0 - normalized_z  # Blue decreases with height
        colors[:, 1] = 1.0 - np.abs(normalized_z - 0.5) * 2  # Green peaks at mid-height
        colors[:, 0] = normalized_z  # Red increases with height
        
        print(f"Height range: [{z_min:.2f}, {z_max:.2f}]")
    
    # Create point cloud
    print("Creating visualization...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Print coordinate ranges
    print(f"\nCoordinate ranges:")
    print(f"  X: [{centers[:, 0].min():.2f}, {centers[:, 0].max():.2f}]")
    print(f"  Y: [{centers[:, 1].min():.2f}, {centers[:, 1].max():.2f}]")
    print(f"  Z: [{centers[:, 2].min():.2f}, {centers[:, 2].max():.2f}]")
    
    # Visualize
    print("\nVisualization Controls:")
    print("  - Mouse: Rotate (left), Pan (middle), Zoom (wheel)")
    print("  - Press 'H' for help")
    print("  - Press 'Q' to quit")
    print("\nLaunching visualization...")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Voxel Map - {Path(voxel_map_file).name}", 
                      width=1280, height=720)
    vis.add_geometry(pcd)
    
    # Set point size
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a voxel map created by build_voxel_map.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize with semantic labels
  python visualize_voxel_map.py --input voxel_map.pkl --color labels
  
  # Visualize with intensity
  python visualize_voxel_map.py --input voxel_map.pkl --color intensity
  
  # Visualize with height coloring
  python visualize_voxel_map.py --input voxel_map.pkl --color height
  
  # Filter voxels with minimum points
  python visualize_voxel_map.py --input voxel_map.pkl --min_points 5
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                        help='Path to voxel map file (.pkl)')
    parser.add_argument('--color', type=str, choices=['labels', 'intensity', 'height'],
                        default='labels',
                        help='Coloring scheme (default: labels)')
    parser.add_argument('--format', type=str, choices=['auto', 'mcd', 'kitti'],
                        default='auto',
                        help='Label format for coloring (default: auto)')
    parser.add_argument('--min_points', type=int, default=1,
                        help='Minimum points per voxel to display (default: 1)')
    parser.add_argument('--point_size', type=float, default=2.0,
                        help='Point size for visualization (default: 2.0)')
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.input).exists():
        print(f"ERROR: Voxel map file not found: {args.input}")
        return 1
    
    # Visualize
    visualize_voxel_map(
        voxel_map_file=args.input,
        color_by=args.color,
        label_format=args.format,
        min_points=args.min_points,
        point_size=args.point_size
    )
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
