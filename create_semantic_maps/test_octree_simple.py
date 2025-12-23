#!/usr/bin/env python3
"""
Simple test script to check if pyoctomap works with UTM coordinates.
This helps isolate whether the segfault is from octree initialization or coordinate handling.
"""

import os
import sys
import numpy as np
import pyoctomap as pyo
from pathlib import Path
import random
# Try to import Open3D for visualization
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


def visualize_color_octree(tree, title="ColorOcTree Visualization"):
    """
    Visualize ColorOcTree with actual RGB colors from nodes.
    
    Args:
        tree: ColorOcTree instance
        title: Title for the visualization window
    """
    if not OPEN3D_AVAILABLE:
        return
    
    # Extract colored points from the tree
    colored_points = []
    colored_rgb = []
    
    # Get bounding box to sample points
    bbx_min = tree.getBBXMin()
    bbx_max = tree.getBBXMax()
    # # If bounding box is not set, use a reasonable default based on known structures
    # if bbx_min[0] == bbx_max[0] and bbx_min[1] == bbx_max[1] and bbx_min[2] == bbx_max[2]:
    #     # Use a default bounding box that covers all our structures
    #     bbx_min = np.array([-3.0, -4.0, -0.5])
    #     bbx_max = np.array([7.5, 7.0, 3.5])

    resolution = tree.getResolution()
    
    # Sample points in the bounding box
    x_range = np.arange(bbx_min[0], bbx_max[0] + resolution, resolution)
    y_range = np.arange(bbx_min[1], bbx_max[1] + resolution, resolution)
    z_range = np.arange(bbx_min[2], bbx_max[2] + resolution, resolution)
    
    # Sample every point in the bounding box
    for x in x_range:
        for y in y_range:
            for z in z_range:
                coord = np.array([x, y, z])
                search_node = tree.search(coord)
                if search_node:
                    # Check if node is occupied
                    if tree.isNodeOccupied(search_node):
                        # Get color from ColorOcTreeNode
                        if isinstance(search_node, pyo.ColorOcTreeNode):
                            color = search_node.getColor()
                            # Convert RGB from 0-255 to 0.0-1.0 for Open3D
                            rgb_normalized = [c / 255.0 for c in color]
                            colored_points.append(coord)
                            colored_rgb.append(rgb_normalized)
    
    if not colored_points:
        return

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(colored_points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colored_rgb))
    
    # Save as ply
    o3d.io.write_point_cloud(f"{title}.ply", pcd)

        
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import functions from main script
from lidar2osm.utils.file_io import read_bin_file
from scipy.spatial.transform import Rotation as R
from lidar2osm.datasets.cu_multi_dataset import labels as sem_kitti_labels
from lidar2osm.core.pointcloud import labels2RGB

print("Testing pyoctomap ColorOcTree with UTM coordinates...")

# Test 4: Load and process a single real scan
print("\n4. Testing with a single REAL scan from dataset...")
print("   (This will help isolate the segfault issue)")

# Dataset paths (adjust these to match your setup)
dataset_path = "/media/donceykong/donceys_data_ssd/datasets/CU_MULTI/data"
environment = "kittredge_loop"
robot = "robot1"

velodyne_path = Path(dataset_path) / environment / robot / "lidar_bin/data"
labels_path = Path(dataset_path) / environment / robot / f"{robot}_{environment}_lidar_labels"
poses_file = Path(dataset_path) / environment / robot / f"{robot}_{environment}_gt_utm_poses.csv"

try:
    if not velodyne_path.exists():
        print(f"   ⚠ SKIP: Velodyne path not found: {velodyne_path}")
    elif not labels_path.exists():
        print(f"   ⚠ SKIP: Labels path not found: {labels_path}")
    elif not poses_file.exists():
        print(f"   ⚠ SKIP: Poses file not found: {poses_file}")
    else:
        # Load poses using same logic as main script
        import pandas as pd
        df = pd.read_csv(poses_file, comment='#')
        print(f"   ✓ Loaded CSV: {len(df)} rows, columns: {list(df.columns)}")
        
        poses = {}
        for _, row in df.iterrows():
            try:
                if 'timestamp' in df.columns:
                    timestamp = float(row['timestamp'])
                    x = float(row['x'])
                    y = float(row['y'])
                    z = float(row['z'])
                    qx = float(row['qx'])
                    qy = float(row['qy'])
                    qz = float(row['qz'])
                    qw = float(row['qw'])
                else:
                    # Try positional access if no column names
                    if len(row) >= 8:
                        timestamp = float(row.iloc[0])
                        x = float(row.iloc[1])
                        y = float(row.iloc[2])
                        z = float(row.iloc[3])
                        qx = float(row.iloc[4])
                        qy = float(row.iloc[5])
                        qz = float(row.iloc[6])
                        qw = float(row.iloc[7])
                    else:
                        continue
                
                poses[timestamp] = [x, y, z, qx, qy, qz, qw]
            except Exception as e:
                continue
        
        timestamps = sorted(poses.keys())
        print(f"   ✓ Loaded {len(poses)} poses")
        
        # Get first scan files
        velodyne_files = sorted(list(velodyne_path.glob("*.bin")))
        label_files = sorted(list(labels_path.glob("*.bin")))
        
        if len(velodyne_files) == 0 or len(label_files) == 0:
            print(f"   ⚠ SKIP: No scan files found (velodyne: {len(velodyne_files)}, labels: {len(label_files)})")
        elif len(timestamps) == 0:
            print(f"   ⚠ SKIP: No timestamps found in poses file")
            print(f"   CSV has {len(df)} rows but couldn't parse any poses")
        else:
            print(f"   Loading first scan from {robot}...")
            print(f"   Found {len(velodyne_files)} scans, using first one")
            
            # Load first scan
            points = read_bin_file(velodyne_files[0], dtype=np.float32, shape=(-1, 4))
            points_xyz = points[:, :3]
            intensities = points[:, 3]
            labels = read_bin_file(label_files[0], dtype=np.int32)
            
            print(f"   ✓ Loaded {len(points_xyz)} points")
            
            # Store ORIGINAL first pose position for shifting back later
            first_timestamp = timestamps[0]
            original_first_pose_position = np.array(poses[first_timestamp][:3])  # Store original position
            print(f"   Original first pose position: [{original_first_pose_position[0]:.2f}, {original_first_pose_position[1]:.2f}, {original_first_pose_position[2]:.2f}]")
            
            # Shift all poses to be relative to first pose (translation only, keep rotations)
            poses = {timestamp: [x - original_first_pose_position[0], 
                                 y - original_first_pose_position[1], 
                                 z - original_first_pose_position[2], 
                                 qx, qy, qz, qw] 
                    for timestamp, [x, y, z, qx, qy, qz, qw] in poses.items()}
            
            # Get shifted pose for transformation
            first_pose = poses[first_timestamp]
            position = poses[first_timestamp][:3]  # Should be near [0, 0, 0] now
            quaternion = poses[first_timestamp][3:7]
            rotation_matrix = R.from_quat(quaternion).as_matrix()
            
            print(f"   Shifted first pose position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")

            # Transform to world coordinates
            print(f"   Transforming points to world frame...")
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = position
            
            points_homogeneous = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)])
            world_points = (transform_matrix @ points_homogeneous.T).T
            world_points_xyz = world_points[:, :3]
            
            print(f"   ✓ Transformed {len(world_points_xyz)} points")
            print(f"   Point range: x=[{world_points_xyz[:, 0].min():.2f}, {world_points_xyz[:, 0].max():.2f}], "
                  f"y=[{world_points_xyz[:, 1].min():.2f}, {world_points_xyz[:, 1].max():.2f}], "
                  f"z=[{world_points_xyz[:, 2].min():.2f}, {world_points_xyz[:, 2].max():.2f}]")
            
            # Convert labels to RGB colors
            print(f"\n   Converting labels to RGB colors...")
            labels_dict = {label.id: label.color for label in sem_kitti_labels}
            label_colors = labels2RGB(labels, labels_dict)  # Returns RGB in [0, 1] range
            # Convert to [0, 255] range for octree
            label_colors_uint8 = (label_colors * 255).astype(np.uint8)
            print(f"   ✓ Converted {len(label_colors_uint8)} labels to colors")
            print(f"   Unique labels: {np.unique(labels)}")
            
            # Prepare points and colors for insertion
            test_points = world_points_xyz
            test_labels = labels
            test_colors = label_colors_uint8
            
            # Try inserting all points with their label colors
            print(f"\n   Testing octree insertion with {len(test_points)} points...")
            
            octree4 = pyo.ColorOcTree(0.1)
            
            inserted_count = 0
            for i in range(len(test_points)):
                coord = [float(test_points[i, 0]), float(test_points[i, 1]), float(test_points[i, 2])]
                try:
                    octree4.updateNode(coord, True)
                    # Use color from label (RGB values in 0-255 range)
                    r, g, b = int(test_colors[i, 0]), int(test_colors[i, 1]), int(test_colors[i, 2])
                    octree4.setNodeColor(coord, r, g, b)
                    inserted_count += 1
                except Exception as e:
                    print(f"   ✗ Failed to insert point {i}: {e}")
                    break
            
            # We update inner occupancy after each scan
            octree4.updateInnerOccupancy()

            print(f"   Calling getNumLeafNodes()... (this might segfault)")
            nodes4 = octree4.getNumLeafNodes()

            print(f"   ✓ Inserted {inserted_count}/{len(test_points)} points, octree has {nodes4} leaf nodes")

            # Visualize the octree
            print(f"   Visualizing the octree...")
            visualize_color_octree(octree4)

except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")
