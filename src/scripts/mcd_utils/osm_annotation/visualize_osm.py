#!/usr/bin/env python3
"""
Unified script to visualize OSM data, robot path, and point clouds.
Supports interactive clicking for coords and arrow keys for shifting data.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
import sys
import os
import numpy as np
from pathlib import Path
from collections import namedtuple

# Import utilities
from utils import PoseUtils, PointCloudUtils, ProjectionUtils, OSMUtils

# Import semantic label utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from view_initial_pose import semantic_labels as sem_kitti_labels
from lidar2osm.core.pointcloud.pointcloud import labels2RGB_tqdm

initial_latlon_default = [59.348268650, 18.073204280]

def on_click(event):
    if event.inaxes and event.button == 1:
        print(f"Clicked: Lat={event.ydata:.8f}, Lon={event.xdata:.8f}")

def visualize(osm_file_path, output_pose_path, origin_latlon, poses_latlon=None, points_latlon=None, point_colors=None, 
              original_poses=None, original_df=None):
    
    if not Path(osm_file_path).exists():
        print(f"Error: OSM file not found: {osm_file_path}")
        return

    print(f"Loading OSM data from {osm_file_path}...")
    
    # Load Geometries
    buildings = OSMUtils.get_buildings(osm_file_path)
    roads = OSMUtils.get_roads(osm_file_path)
    trees = OSMUtils.get_trees(osm_file_path)
    grass = OSMUtils.get_grassland(osm_file_path)
    water = OSMUtils.get_water(osm_file_path)
    
    # Collect all coords for bounds
    all_lons, all_lats = [], []
    for feats in [buildings, trees, grass, water]:
        for poly in feats:
            l, t = zip(*poly)
            all_lons.extend(l); all_lats.extend(t)
    for line in roads:
        l, t = zip(*line)
        all_lons.extend(l); all_lats.extend(t)

    if poses_latlon is not None and len(poses_latlon) > 0:
        all_lons.extend(poses_latlon[:, 1])
        all_lats.extend(poses_latlon[:, 0])
        
    if points_latlon is not None and len(points_latlon) > 0:
        all_lons.extend(points_latlon[:, 1])
        all_lats.extend(points_latlon[:, 0])

    if not all_lons:
        print("Nothing to visualize.")
        return

    # Setup Plot
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_aspect('equal')
    ax.set_title('OSM Visualization (Interactive: Arrows to shift, S to save)', fontsize=14)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)
    
    # Plot Features
    for poly in buildings: ax.add_patch(Polygon(poly, closed=True, facecolor='blue', alpha=0.5))
    for line in roads: ax.plot(*zip(*line), color='gray', linewidth=0.8, alpha=0.7)
    for poly in trees: ax.add_patch(Polygon(poly, closed=True, facecolor='green', alpha=0.5))
    for poly in grass: ax.add_patch(Polygon(poly, closed=True, facecolor='lightgreen', alpha=0.5))
    for poly in water: ax.add_patch(Polygon(poly, closed=True, facecolor='cyan', alpha=0.5))

    # Plot Path
    path_line, start_marker, end_marker = None, None, None
    if poses_latlon is not None and len(poses_latlon) > 0:
        lons, lats = poses_latlon[:, 1], poses_latlon[:, 0]
        path_line = ax.plot(lons, lats, 'r-', linewidth=2, alpha=0.8, label='Robot Path', zorder=10)[0]
        start_marker = ax.plot(lons[0], lats[0], 'go', markersize=10, label='Start')[0]
        end_marker = ax.plot(lons[-1], lats[-1], 'ro', markersize=10, label='End')[0]

    # Plot Points
    scatter_plot = None
    if points_latlon is not None and len(points_latlon) > 0:
        c = point_colors if point_colors is not None else 'orange'
        scatter_plot = ax.scatter(points_latlon[:, 1], points_latlon[:, 0], c=c, s=0.5, alpha=0.6, zorder=5)

    # State for interactivity
    state = {'lat_off': 0.0, 'lon_off': 0.0}
    orig_points = points_latlon.copy() if points_latlon is not None else None
    orig_poses = poses_latlon.copy() if poses_latlon is not None else None

    def on_key(event):
        key = event.key
        step = 0.00001
        
        if key == 's' and original_poses is not None:
            PoseUtils.save_shifted_poses_csv(
                state['lat_off'], state['lon_off'], 
                original_df, original_poses, origin_latlon, output_pose_path
            )
            return

        if key in ['up', 'arrow_up']: state['lat_off'] += step
        elif key in ['down', 'arrow_down']: state['lat_off'] -= step
        elif key in ['right', 'arrow_right']: state['lon_off'] += step
        elif key in ['left', 'arrow_left']: state['lon_off'] -= step
        else: return

        print(f"Offset: Lat={state['lat_off']:.6f}, Lon={state['lon_off']:.6f}")

        # Update Points
        if scatter_plot and orig_points is not None:
            new_xy = orig_points.copy()
            new_xy[:, 0] += state['lat_off']
            new_xy[:, 1] += state['lon_off']
            scatter_plot.set_offsets(new_xy[:, [1, 0]]) # Swap for XY

        # Update Path
        if path_line and orig_poses is not None:
            p = orig_poses.copy()
            p[:, 0] += state['lat_off']
            p[:, 1] += state['lon_off']
            path_line.set_data(p[:, 1], p[:, 0])
            start_marker.set_data([p[0, 1]], [p[0, 0]])
            end_marker.set_data([p[-1, 1]], [p[-1, 0]])
        
        fig.canvas.draw_idle()

    # Final Setup
    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)
    margin_x = (max_lon - min_lon) * 0.05
    margin_y = (max_lat - min_lat) * 0.05
    ax.set_xlim(min_lon - margin_x, max_lon + margin_x)
    ax.set_ylim(min_lat - margin_y, max_lat + margin_y)
    
    # Legend
    legend_elements = [
        Patch(facecolor='blue', alpha=0.5, label='Buildings'),
        Patch(facecolor='gray', alpha=0.7, label='Roads'),
        Patch(facecolor='green', alpha=0.5, label='Trees'),
        Patch(facecolor='lightgreen', alpha=0.5, label='Grassland'),
        Patch(facecolor='cyan', alpha=0.5, label='Water'),
    ]
    if poses_latlon is not None:
        legend_elements.extend([
            plt.Line2D([0], [0], color='red', linewidth=2, label='Robot Path'),
            plt.Line2D([0], [0], marker='o', color='g', label='Start'),
            plt.Line2D([0], [0], marker='o', color='r', label='End'),
        ])
    if points_latlon is not None:
         legend_elements.append(plt.Line2D([0], [0], marker='o', color='orange', linestyle='None', label='Point Cloud'))

    ax.legend(handles=legend_elements, loc='upper right')
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print("\nControls:")
    print("  Click: Print coordinates")
    print("  Arrow Keys: Shift data")
    print(f"  S: Save shifted poses to {output_pose_path}")
    
    plt.tight_layout()
    plt.show()

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize OSM data, robot path, and point clouds.")
    parser.add_argument("--pose", type=str, required=False, help="Path to pose CSV file")
    parser.add_argument("--npy", type=str, required=False, help="Path to point cloud NPY file")
    parser.add_argument("--osm", type=str, required=True, help="Path to OSM XML file")
    parser.add_argument("--output-pose", type=str, required=True, help="Path to save shifted poses CSV")
    parser.add_argument("--origin-latlon", type=float, nargs=2, default=initial_latlon_default,
                       metavar=('LAT', 'LON'), help="Origin lat/lon for projection (default: KTH)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    origin_latlon = args.origin_latlon
    
    # Load Data
    poses, df = PoseUtils.load_poses(args.pose)
    poses_latlon = ProjectionUtils.project_poses_to_latlon(poses, origin_latlon) if poses else None
    
    points_latlon, colors = None, None
    if args.npy and Path(args.npy).exists():
        pts, lbls = PointCloudUtils.load_from_npy(args.npy)
        if pts is not None:
            # Downsample logic
            bbox = np.linalg.norm(pts.max(0) - pts.min(0))
            voxel_size = min(1.0, bbox * 0.01)
            pts_ds, lbls_ds = PointCloudUtils.downsample(pts, lbls, voxel_size)
            
            # Colorize
            if lbls_ds is not None and sem_kitti_labels:
                ld = {l.id: l.color for l in sem_kitti_labels}
                colors = labels2RGB_tqdm(lbls_ds, ld)
            
            if poses:
                points_latlon = ProjectionUtils.project_points_to_latlon(pts_ds, poses[0]['position'], origin_latlon)
            else:
                print("Warning: No poses found, cannot project points without initial position.")

    visualize(args.osm, args.output_pose, origin_latlon, poses_latlon, points_latlon, colors, poses, df)
