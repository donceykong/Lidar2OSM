#!/usr/bin/env python3
"""
Script to create a global semantic map from LiDAR scans using pyoctomap ColorOcTree.
Accumulates LiDAR data using octree-based spatial indexing, and saves:
- An image visualization of the semantic point cloud
- A numpy file with x, y, z, intensity, semantic_id (all float64)
Option to save in UTM coordinates or convert to lat/lon.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import pyoctomap
import pyoctomap as pyo
from collections import namedtuple

# Internal imports
from lidar2osm.utils.file_io import read_bin_file
from lidar2osm.core.pointcloud import labels2RGB


Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        "id",  # An integer ID that is associated with this label.
        "color",  # The color of this label
        "KEEP",   # Keep if True, Ignore if false
    ],
)

sem_kitti_labels = [
    # name, id, color, keep
    Label("unlabeled", 0, (0, 0, 0), True),
    Label("outlier", 1, (0, 0, 0), True),
    Label("car", 10, (0, 0, 142), True),
    Label("bicycle", 11, (119, 11, 32), False),
    Label("bus", 13, (250, 80, 100), False),
    Label("motorcycle", 15, (0, 0, 230), False),
    Label("on-rails", 16, (255, 0, 0), False),
    Label("truck", 18, (0, 0, 70), False),
    Label("other-vehicle", 20, (51, 0, 51), False),
    Label("person", 30, (220, 20, 60), False),
    Label("bicyclist", 31, (200, 40, 255), False),
    Label("motorcyclist", 32, (90, 30, 150), False),
    Label("road", 40, (128, 64, 128), True),
    Label("parking", 44, (250, 170, 160), True),
    Label("OSM BUILDING", 45, (0, 0, 255), True),   # OSM
    Label("OSM ROAD", 46, (255, 0, 0), True),       # OSM
    Label("sidewalk", 48, (244, 35, 232), True),
    Label("other-ground", 49, (81, 0, 81), True),
    Label("building", 50, (0, 100, 0), True),
    Label("fence", 51, (190, 153, 153), True),
    Label("other-structure", 52, (0, 150, 255), True),
    Label("lane-marking", 60, (170, 255, 150), True),
    Label("vegetation", 70, (107, 142, 35), True),
    Label("trunk", 71, (0, 60, 135), True),
    Label("terrain", 72, (152, 251, 152), True),
    Label("pole", 80, (153, 153, 153), True),
    Label("traffic-sign", 81, (0, 0, 255), True),
    Label("other-object", 99, (255, 255, 50), True),
]


# Set of label IDs that should be kept when filtering
KEEP_LABEL_IDS = {label.id for label in sem_kitti_labels if label.KEEP}


# Import voxel downsampling from other script
try:
    from create_global_sem_map import voxel_downsample_CENTER as voxel_downsample_scan
except ImportError:
    # Fallback: define it here if import fails
    def voxel_downsample_scan(points, intensities, labels=None, voxel_size=1.0):
        """Simple voxel downsampling - placeholder if import fails."""
        # For now, just return as-is - we'll add proper implementation if needed
        if labels is not None:
            return points, intensities, labels
        return points, intensities


def filter_points_height(points, intensities, labels, pose_data):
    """Filter/remap points based on height relative to the current pose.
    
    Rules (relative to pose z):
    - Drop vegetation points that are below 0.5 m.
    - Drop car points that are above 2.0 m.
    - Relabel all remaining points below 3.0 m to outlier (id=1).
    """
    if points is None or labels is None or len(points) == 0:
        return points, intensities, labels
    
    pose_z = pose_data[2] if pose_data is not None and len(pose_data) >= 3 else 0.0
    rel_z = points[:, 2] - pose_z
    
    VEG_ID = 70
    CAR_ID = 10
    OUTLIER_ID = 1
    
    # Build keep mask for vegetation and car height constraints
    keep_mask = np.ones(len(points), dtype=bool)
    keep_mask &= ~((labels == VEG_ID) & (rel_z < 0.5))
    keep_mask &= ~((labels == CAR_ID) & (rel_z > 2.0))
    
    filtered_points = points[keep_mask]
    filtered_labels = labels[keep_mask]
    filtered_intensities = intensities[keep_mask] if intensities is not None else intensities
    
    if len(filtered_points) == 0:
        return filtered_points, filtered_intensities, filtered_labels
    
    # Relabel points below 3.0 m to outlier
    rel_z_filtered = pose_z - filtered_points[:, 2]

    for rel_z in rel_z_filtered:
        if rel_z > 3.0:
            print(f"Relabeling point {rel_z} to outlier. pose z: {pose_z}")

    relabel_mask = rel_z_filtered > 3.0
    if np.any(relabel_mask):
        filtered_labels = filtered_labels.copy()
        filtered_labels[relabel_mask] = OUTLIER_ID
    
    return filtered_points, filtered_intensities, filtered_labels


def filter_points(points, intensities, labels):
    """
    Remove points whose semantic label is marked with KEEP=False.
    
    Args:
        points: (N, 3) array of xyz coordinates.
        intensities: (N,) array of intensity values (optional).
        labels: (N,) array of semantic label IDs.
    
    Returns:
        Tuple of filtered (points, intensities, labels) with only KEEP=True labels.
    """
    # Nothing to filter if labels are missing
    if labels is None or points is None:
        return points, intensities, labels
    
    # Ensure equal lengths before masking
    target_len = min(
        len(points),
        len(labels),
        len(intensities) if intensities is not None else len(points),
    )
    if len(points) != target_len or len(labels) != target_len or (
        intensities is not None and len(intensities) != target_len
    ):
        print("Points and intensities or labels are not the same size!")
        # points = points[:target_len]
        # labels = labels[:target_len]
        # if intensities is not None:
        #     intensities = intensities[:target_len]
    
    # Mask points based on KEEP flag
    keep_mask = np.isin(labels, list(KEEP_LABEL_IDS))
    
    filtered_points = points[keep_mask]
    filtered_labels = labels[keep_mask]
    filtered_intensities = intensities[keep_mask] if intensities is not None else intensities
    
    return filtered_points, filtered_intensities, filtered_labels


def load_poses(poses_file):
    """Load UTM poses from CSV file."""
    import pandas as pd
    
    print(f"\nReading UTM poses CSV file: {poses_file}")
    
    try:
        df = pd.read_csv(poses_file, comment='#')
        print(f"Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {}
    
    poses = {}
    for _, row in df.iterrows():
        try:
            if 'timestamp' in df.columns:
                timestamp = row['timestamp']
                x = row['x']
                y = row['y']
                z = row['z']
                qx = row['qx']
                qy = row['qy']
                qz = row['qz']
                qw = row['qw']
            else:
                if len(row) >= 8:
                    timestamp = row.iloc[0]
                    x = row.iloc[1]
                    y = row.iloc[2]
                    z = row.iloc[3]
                    qx = row.iloc[4]
                    qy = row.iloc[5]
                    qz = row.iloc[6]
                    qw = row.iloc[7]
                else:
                    continue
            
            pose = [float(x), float(y), float(z), float(qx), float(qy), float(qz), float(qw)]
            poses[float(timestamp)] = pose
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    print(f"Successfully loaded {len(poses)} poses\n")
    return poses


def transform_imu_to_lidar(poses):
    """Transform poses from IMU frame to LiDAR frame."""
    from scipy.spatial.transform import Rotation as R
    
    # IMU to LiDAR transformation
    IMU_TO_LIDAR_T = np.array([-0.058038, 0.015573, 0.049603])
    IMU_TO_LIDAR_Q = [0.0, 0.0, 1.0, 0.0]  # [qx, qy, qz, qw]
    
    imu_to_lidar_rot = R.from_quat(IMU_TO_LIDAR_Q)
    imu_to_lidar_rot_matrix = imu_to_lidar_rot.as_matrix()
    
    transformed_poses = {}
    
    for timestamp, pose in poses.items():
        imu_position = np.array(pose[:3])
        imu_quat = pose[3:7]
        
        imu_rot = R.from_quat(imu_quat)
        imu_rot_matrix = imu_rot.as_matrix()
        
        lidar_position = imu_position + imu_rot_matrix @ IMU_TO_LIDAR_T
        lidar_rot_matrix = imu_rot_matrix @ imu_to_lidar_rot_matrix
        lidar_quat = R.from_matrix(lidar_rot_matrix).as_quat()
        
        transformed_pose = np.concatenate([lidar_position, lidar_quat])
        transformed_poses[timestamp] = transformed_pose
    
    return transformed_poses


def labels_to_rgb_colors(labels):
    """
    Convert semantic labels to RGB colors [0, 1].
    
    Args:
        labels: (N,) array of semantic label IDs
    
    Returns:
        colors: (N, 3) array of RGB colors [0, 1]
    """
    labels_dict = {label.id: label.color for label in sem_kitti_labels}
    colors = labels2RGB(labels, labels_dict)
    # labels2RGB returns colors in [0, 255], convert to [0, 1]
    return colors / 255.0


def insert_points_into_octree(octree, points, colors, intensities=None, labels=None, 
                               intensity_dict=None, label_dict=None, color_dict=None, resolution=0.5):
    """
    Insert points with RGB colors into pyoctomap ColorOcTree.
    
    PRIMARY STORAGE: Octree stores point coordinates and colors.
    METADATA ONLY: Dictionaries store intensities and labels (metadata octree can't store).
    
    Strategy:
    - Octree is the source of truth for spatial data (coordinates, colors)
    - Only store metadata in dictionaries AFTER octree insertion succeeds
    - Use integer tuple voxel keys for efficient deduplication
    - Only store unique voxels (deduplicate before dictionary storage)
    
    Args:
        octree: pyoctomap.ColorOcTree instance (PRIMARY storage)
        points: numpy array of shape (N, 3) containing xyz coordinates
        colors: numpy array of shape (N, 3) containing RGB colors [0, 1] (stored in octree)
        intensities: optional (N,) array of intensity values (stored in dict if provided)
        labels: optional (N,) array of semantic labels (stored in dict if provided)
        intensity_dict: Dictionary for intensity metadata {voxel_key: {'sum': float, 'count': int, 'mean': float}}
        label_dict: Dictionary for label metadata {voxel_key: {'counts': dict, 'mode': int}}
        color_dict: DEPRECATED - colors stored in octree. Kept for backward compatibility only.
        resolution: Octree resolution in meters
    
    Returns:
        Number of points successfully inserted into octree
    """
    if len(points) == 0:
        return 0
    
    inserted_count = 0
    error_count = 0
    octree_insert_failed_count = 0
    
    # Vectorized operation: convert colors from [0, 1] to [0, 255] integers
    colors_uint8 = (colors * 255.0).astype(np.uint8)
    
    # Track which voxels have been successfully inserted into octree
    # This prevents storing metadata for failed octree insertions
    voxel_octree_status = {}  # {voxel_key: bool} - True if successfully in octree
    
    for i in range(len(points)):
        point = points[i]
        color = colors_uint8[i]
        
        try:
            # Extract coordinates as floats and ensure they're Python floats, not numpy types
            x = float(point[0])
            y = float(point[1])
            z = float(point[2])
            # Create coordinate list explicitly as Python list (not numpy array)
            coord = [x, y, z]
            
            # Round to voxel resolution for consistent key generation
            # Keep string format for compatibility with extraction function
            voxel_x = round(x / resolution) * resolution
            voxel_y = round(y / resolution) * resolution
            voxel_z = round(z / resolution) * resolution
            voxel_key = f"{voxel_x:.6f}_{voxel_y:.6f}_{voxel_z:.6f}"
            
            # Check for invalid coordinates
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                error_count += 1
                if error_count <= 3:
                    print(f"WARNING: Invalid coordinates: ({x}, {y}, {z})")
                continue
            
            # Check for extremely large coordinates that might cause segfaults
            # pyoctomap might have issues with very large UTM coordinates
            # Typical UTM coordinates are in range [-2e6, 2e7] for x and [0, 1e7] for y
            # But we'll allow up to 1e8 to be safe
            MAX_COORD = 1e8  # Reasonable upper bound
            MIN_COORD = -1e8  # Reasonable lower bound
            if x < MIN_COORD or x > MAX_COORD or y < MIN_COORD or y > MAX_COORD or z < MIN_COORD or z > MAX_COORD:
                error_count += 1
                if error_count <= 3:
                    print(f"WARNING: Coordinate out of reasonable range: ({x:.2f}, {y:.2f}, {z:.2f})")
                continue
            
            # Validate color values are in valid range [0-255]
            r_val = int(color[0])
            g_val = int(color[1])
            b_val = int(color[2])
            if not (0 <= r_val <= 255 and 0 <= g_val <= 255 and 0 <= b_val <= 255):
                error_count += 1
                if error_count <= 3:
                    print(f"WARNING: Invalid color values: ({r_val}, {g_val}, {b_val})")
                continue
            
            # Insert point into octree using updateNode
            # ColorOcTree updateNode takes coordinates as [x, y, z] and occupancy boolean
            # This will automatically expand the tree bounds if needed
            # ONLY store in dictionaries AFTER successful octree insertion
            octree_insert_succeeded = False
            
            try:
                # Ensure coord is a proper Python list with explicit float conversion
                coord_list = [float(x), float(y), float(z)]
                
                # Call updateNode - this may segfault with large coordinates
                octree.updateNode(coord_list, True)
                
                # Set color directly in the octree node
                octree.setNodeColor(coord_list, r_val, g_val, b_val)
                
                # Only mark as successful if no exception was raised
                octree_insert_succeeded = True
                inserted_count += 1
                
                # Track that this voxel is successfully in octree
                voxel_octree_status[voxel_key] = True
                
            except (ValueError, TypeError, RuntimeError) as update_error:
                # If updateNode fails with a Python exception, skip this point
                octree_insert_failed_count += 1
                error_count += 1
                if error_count <= 10:
                    print(f"WARNING: updateNode failed for point ({x:.2f}, {y:.2f}, {z:.2f}): {update_error}")
                continue
            except Exception as unexpected_error:
                # Catch any other unexpected errors (but segfaults can't be caught)
                octree_insert_failed_count += 1
                error_count += 1
                if error_count <= 10:
                    print(f"WARNING: Unexpected error inserting point ({x:.2f}, {y:.2f}, {z:.2f}): {unexpected_error}")
                continue
            
            # ONLY store metadata in dictionaries if octree insertion succeeded
            # This ensures dictionaries only contain data for points actually in octree
            if not octree_insert_succeeded:
                continue
            
            # Store intensity metadata (octree can't store this)
            if intensities is not None and intensity_dict is not None:
                if voxel_key not in intensity_dict:
                    intensity_dict[voxel_key] = {'sum': 0.0, 'count': 0, 'mean': 0.0}
                intensity_dict[voxel_key]['sum'] += intensities[i]
                intensity_dict[voxel_key]['count'] += 1
                intensity_dict[voxel_key]['mean'] = intensity_dict[voxel_key]['sum'] / intensity_dict[voxel_key]['count']
            
            # Store label metadata (octree can't store this)
            if labels is not None and label_dict is not None:
                if voxel_key not in label_dict:
                    label_dict[voxel_key] = {'counts': {}, 'mode': labels[i]}
                label_counts = label_dict[voxel_key]['counts']
                label_id = int(labels[i])
                if label_id in label_counts:
                    label_counts[label_id] += 1
                else:
                    label_counts[label_id] = 1
                # Update mode (most frequent label)
                label_dict[voxel_key]['mode'] = max(label_counts, key=label_counts.get)
            
            # Note: color_dict is deprecated - colors are stored in octree
            # Keeping for backward compatibility only (can be removed if extraction uses octree)
            if color_dict is not None:
                if voxel_key not in color_dict:
                    color_dict[voxel_key] = {'counts': {}, 'mode': (r_val, g_val, b_val)}
                color_counts = color_dict[voxel_key]['counts']
                color_tuple = (r_val, g_val, b_val)
                if color_tuple in color_counts:
                    color_counts[color_tuple] += 1
                else:
                    color_counts[color_tuple] = 1
                color_dict[voxel_key]['mode'] = max(color_counts, key=color_counts.get)
                
        except Exception as e:
            error_count += 1
            if error_count <= 10:  # Print first 10 errors for debugging
                print(f"WARNING: Error inserting point {i}/{len(points)}: {e}")
                if error_count <= 3:  # Full traceback for first 3
                    import traceback
                    traceback.print_exc()
            continue
    
    if error_count > 0:
        print(f"WARNING: {error_count}/{len(points)} points failed to insert")
        if error_count == len(points):
            print(f"ERROR: ALL points failed to insert! This suggests a fundamental issue.")
            print(f"  First point was: {points[0] if len(points) > 0 else 'N/A'}")
            print(f"  Octree resolution: {resolution}")
    
    if octree_insert_failed_count > 0:
        print(f"WARNING: {octree_insert_failed_count} octree insertions failed (exceptions caught)")
        if octree_insert_failed_count == len(points):
            print(f"ERROR: ALL octree insertions failed! Octree may not be working properly.")
            print(f"  This could cause segfaults or memory issues.")
            print(f"  First point coordinates: {points[0] if len(points) > 0 else 'N/A'}")
    
    return inserted_count


def extract_pointcloud_from_octree(octree, intensity_dict=None, label_dict=None, color_dict=None, 
                                   resolution=0.5, coordinate_offset=None):
    """
    Extract pointcloud with RGB colors, intensities, and labels from pyoctomap ColorOcTree.
    
    Args:
        octree: pyoctomap.ColorOcTree instance (may have _color_dict attached)
        intensity_dict: Optional dictionary mapping voxel keys to intensity info
        label_dict: Optional dictionary mapping voxel keys to label info
        color_dict: Optional dictionary mapping voxel keys to color info
        resolution: Octree resolution for voxel key generation
        coordinate_offset: Optional offset to add back to coordinates (to restore original UTM coordinates)
    
    Returns:
        points: numpy array of shape (N, 3) containing xyz coordinates (in original frame if offset provided)
        colors: numpy array of shape (N, 3) containing RGB colors [0, 1]
        intensities: numpy array of shape (N,) containing intensities (or zeros if not provided)
        labels: numpy array of shape (N,) containing labels (or zeros if not provided)
    """
    points = []
    colors = []
    intensities = []
    labels = []
    
    try:
        # First, ensure inner occupancy is updated (only if we have nodes)
        # CRITICAL: updateInnerOccupancy() segfaults when there are 0 leaf nodes!
        num_nodes = octree.getNumLeafNodes()
        if num_nodes > 0:
            octree.updateInnerOccupancy()
        else:
            print("WARNING: Octree has 0 leaf nodes, skipping updateInnerOccupancy()")
        
        # Manual tree traversal since begin_leafs() doesn't exist in this pyoctomap version
        # We'll traverse the tree recursively starting from the root
        def traverse_node(node, depth=0, max_depth=20):
            """Recursively traverse octree nodes to find leaf nodes."""
            if node is None or depth > max_depth:
                return
            
            # Check if this is a leaf node (no children)
            if not octree.nodeHasChildren(node):
                # This is a leaf node, check if occupied
                if octree.isNodeOccupied(node):
                    # Get coordinate from node
                    # We need to reconstruct the coordinate from the node's key
                    # For now, we'll use search with a coordinate - but we need the coord
                    # Actually, we can't get coord from node directly without the key
                    # So we'll need a different approach
                    pass
            else:
                # Has children, recurse
                for i in range(8):
                    try:
                        child = octree.getNodeChild(node, i)
                        traverse_node(child, depth + 1, max_depth)
                    except:
                        continue
        
        # Alternative: Use bounding box to search for nodes
        # Get bounding box of the octree
        try:
            bbox_min = octree.getBBXMin()
            bbox_max = octree.getBBXMax()
            print(f"Octree bounding box: min={bbox_min}, max={bbox_max}")
        except:
            # If we can't get bbox, try to search in a grid pattern
            # But this is inefficient - let's try a different approach
            pass
        
        # Since begin_leafs() doesn't exist, we'll use a workaround:
        # Search through the dictionary keys and verify nodes exist
        # This is not ideal but works if we have the voxel keys
        
        # Actually, the best approach is to use the intensity/label dictionaries
        # which have the voxel keys, and then search for those coordinates
        if intensity_dict is not None and len(intensity_dict) > 0:
            print(f"Extracting points using dictionary keys ({len(intensity_dict)} entries)...")
            for voxel_key in intensity_dict.keys():
                # Parse voxel key back to coordinates
                parts = voxel_key.split('_')
                if len(parts) == 3:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        z = float(parts[2])
                        coord = [x, y, z]
                        
                        # Since updateNode isn't creating nodes (search fails with bounds errors),
                        # we'll extract points directly from dictionaries without querying octree
                        # This avoids the "out of OcTree bounds" errors
                        
                        # Apply coordinate offset to shift back to original frame
                        coord = np.array([x, y, z])
                        if coordinate_offset is not None:
                            coord = coord + coordinate_offset
                        points.append(coord)
                        
                        # Get color from dictionary if available (stored during insertion)
                        # Otherwise use default gray
                        if color_dict is not None and voxel_key in color_dict:
                            r, g, b = color_dict[voxel_key]['mode']
                            rgb = np.array([r / 255.0, g / 255.0, b / 255.0])
                            colors.append(rgb)
                        else:
                            # No color stored, use default gray
                            colors.append(np.array([0.5, 0.5, 0.5]))
                        
                        # Get intensity and label from dictionaries
                        if intensity_dict is not None and voxel_key in intensity_dict:
                            intensities.append(intensity_dict[voxel_key]['mean'])
                        else:
                            intensities.append(0.0)
                        
                        if label_dict is not None and voxel_key in label_dict:
                            labels.append(label_dict[voxel_key]['mode'])
                        else:
                            labels.append(0)
                    except ValueError:
                        # Invalid key format, skip
                        continue
        else:
            # No dictionary, can't extract - this shouldn't happen
            print("WARNING: No intensity_dict available for extraction")
    
    except Exception as e:
        print(f"WARNING: Error extracting pointcloud from octree: {e}")
        import traceback
        traceback.print_exc()
        # Return empty arrays if extraction fails
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    if len(points) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)
    intensities = np.array(intensities, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    return points, colors, intensities, labels


def utm_to_latlon(points_utm):
    """Convert UTM coordinates to lat/lon."""
    from pyproj import Transformer
    
    # UTM zone 13N for Colorado
    transformer = Transformer.from_crs("EPSG:32613", "EPSG:4326", always_xy=True)
    
    lons, lats = transformer.transform(points_utm[:, 0], points_utm[:, 1])
    
    # Return as (lat, lon, z)
    latlon_points = np.column_stack([lats, lons, points_utm[:, 2]])
    
    return latlon_points


def save_point_cloud(points, intensities, labels, output_file="global_semantic_map.npy", file_dir="."):
    """
    Save point cloud data to numpy file.
    
    Args:
        points: (N, 3) array of coordinates
        intensities: (N,) array of intensities
        labels: (N,) array of semantic labels
        output_file: Output filename
    """
    # Create structured array with x, y, z, intensity, semantic_id (all float64)
    data = np.zeros((len(points), 5), dtype=np.float64)
    data[:, 0] = points[:, 0]  # x
    data[:, 1] = points[:, 1]  # y
    data[:, 2] = points[:, 2]  # z
    data[:, 3] = intensities    # intensity
    data[:, 4] = labels.astype(np.float64)  # semantic_id
    
    # Save to numpy file
    file_path = os.path.join(file_dir, output_file)
    np.save(file_path, data)
    print(f"Point cloud data saved to {output_file}")
    print(f"Shape: {data.shape}, dtype: {data.dtype}")
    print(f"Columns: [x, y, z, intensity, semantic_id]")


def accumulate_lidar_scans(dataset_path, 
                           environment, 
                           robot, 
                           poses_dict,
                           octree,
                           intensity_dict,
                           label_dict,
                           color_dict,
                           num_scans=500, 
                           per_scan_voxel_size=4.0,
                           global_voxel_size=0.5,
                           min_distance=3.0,
                           max_distance=60.0):
    """
    Accumulate LiDAR scans into a global ColorOcTree for a single robot.
    
    Args:
        dataset_path: Path to dataset
        environment: Environment name
        robot: Robot name
        poses_dict: Dictionary of poses {timestamp -> [x, y, z, qx, qy, qz, qw]}
        octree: pyoctomap.ColorOcTree instance (modified in place)
        intensity_dict: Dictionary to store intensities per voxel (modified in place)
        label_dict: Dictionary to store labels per voxel (modified in place)
        num_scans: Number of scans to accumulate
        per_scan_voxel_size: Voxel size for per-scan downsampling (not used with octree)
        global_voxel_size: Octree resolution in meters
        min_distance: Minimum distance for filtering points
        max_distance: Maximum distance for filtering points
    
    Returns:
        Number of scans processed
    """
    from scipy.spatial.transform import Rotation as R

    velodyne_path = Path(dataset_path) / environment / robot / "lidar_bin/data"
    velodyne_files = sorted([f for f in velodyne_path.glob("*.bin")])

    # labels_path = Path(dataset_path) / environment / robot / f"{robot}_{environment}_lidar_labels"
    labels_path = Path(dataset_path) / environment / robot / f"{robot}_{environment}_lidar_labels_confidence"
    label_conf_path = Path(labels_path) / f"confidence_scores"
    # labels_path = Path(dataset_path) / environment / robot / f"{robot}_{environment}_new_inferred_lidar_labels"

    label_files = sorted([f for f in labels_path.glob("*.bin")]) if labels_path.exists() else None
    label_conf_files = sorted([f for f in label_conf_path.glob("*.bin")]) if label_conf_path.exists() else None

    if not label_files:
        print(f"Warning: No semantic label files found in {labels_path}!")
        print(f"  Path exists: {labels_path.exists()}")
        return 0
    
    print(f"Found {len(velodyne_files)} LiDAR scans and {len(label_files)} label files and {len(label_conf_files)} label conf files")
    
    total_scans = min(len(velodyne_files), len(label_files))
    sample_count = min(num_scans, total_scans)
    sample_indices = np.linspace(0, total_scans - 1, sample_count, dtype=int)
    print(f"Processing {sample_count} scans (from {total_scans} total)...")
    
    timestamps = sorted(poses_dict.keys())
    
    # Track initial octree size
    initial_octree_size = octree.getNumLeafNodes()
    
    scan_count = 0
    points_inserted_total = 0
    
    for pose_idx in tqdm(sample_indices, desc=f"Loading {robot} scans", unit="scan"):
        if pose_idx >= len(velodyne_files) or pose_idx >= len(timestamps):
            continue

        try:
            # Load LiDAR scan
            points = read_bin_file(velodyne_files[pose_idx], dtype=np.float32, shape=(-1, 4))
            points_xyz = points[:, :3]
            intensities = points[:, 3]
            
            # Load semantic labels (must be loaded before filtering)
            labels = read_bin_file(label_files[pose_idx], dtype=np.int32)
            label_confs = read_bin_file(label_conf_files[pose_idx], dtype=np.float16)
                                   
            # Ensure same length before filtering
            if len(labels) != len(points_xyz) or len(labels) != len(label_confs):
                raise ValueError(
                    f"Length mismatch: labels={len(labels)} points_xyz={len(points_xyz)}, label_confs={len(label_confs)}"
                )
                # min_length = min(len(labels), len(points_xyz))
                # labels = labels[:min_length]
                # points_xyz = points_xyz[:min_length]
                # intensities = intensities[:min_length]
            
            # Only keep points with a confidence higher than 0.9
            high_conf_mask = label_confs > 0.99
            points_xyz = points_xyz[high_conf_mask]
            labels = labels[high_conf_mask]
            intensities = intensities[high_conf_mask]

            # # Remove all points beyond max_distance
            # mask = np.linalg.norm(points_xyz, axis=1) <= max_distance
            # points_xyz = points_xyz[mask]
            # labels = labels[mask]
            # intensities = intensities[mask]
            
            # # Remove all points closer than min_distance
            # mask = np.linalg.norm(points_xyz, axis=1) >= min_distance
            # points_xyz = points_xyz[mask]
            # labels = labels[mask]
            # intensities = intensities[mask]
            
            # Get pose for this frame
            timestamp = timestamps[pose_idx]
            pose_data = poses_dict[timestamp]
            
            position = pose_data[:3]
            quat = pose_data[3:7]  # [qx, qy, qz, qw]
            
            rotation_matrix = R.from_quat(quat).as_matrix()
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = position
            
            # Transform points to world coordinates
            # Since poses are already shifted relative to first pose, 
            # the transformed points will naturally be in shifted coordinate frame
            points_homogeneous = np.hstack(
                [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)]
            )
            world_points = (transform_matrix @ points_homogeneous.T).T
            world_points_xyz = world_points[:, :3]
            
            # Validate transformed points before insertion
            if not np.all(np.isfinite(world_points_xyz)):
                print(f"WARNING: Scan {pose_idx} has non-finite transformed points, skipping")
                continue
            
            # Filter points, intensities, and labels that are False for "Keep"
            world_points_xyz, intensities, labels = filter_points(world_points_xyz, intensities, labels)
            
            # Filter and relabel points based on height in Z relative to current pose
            world_points_xyz, intensities, labels = filter_points_height(world_points_xyz, intensities, labels, pose_data)

            # Per-scan voxel downsampling to reduce points before octree insertion
            if per_scan_voxel_size > 0:
                # Use per_scan_voxel_size for initial downsampling (coarser)
                # Then octree will further downsample to global_voxel_size (finer)
                world_points_xyz, intensities, labels = voxel_downsample_scan(
                    world_points_xyz, intensities, labels, voxel_size=per_scan_voxel_size
                )
            
            # Convert labels to RGB colors (after downsampling)
            colors = labels_to_rgb_colors(labels)
            
            # Insert points into octree (octree will further downsample to global_voxel_size)
            inserted = insert_points_into_octree(
                octree, world_points_xyz, colors, 
                intensities=intensities, labels=labels,
                intensity_dict=intensity_dict,
                label_dict=label_dict,
                color_dict=color_dict,
                resolution=global_voxel_size
            )
            points_inserted_total += inserted
            
            # CRITICAL FIX: Only call updateInnerOccupancy() if we have leaf nodes
            # updateInnerOccupancy() segfaults when there are 0 leaf nodes!
            current_nodes = octree.getNumLeafNodes()
            
            if current_nodes > 0:
                # Update inner occupancy only if we have nodes
                # This propagates occupancy information from leaves to root
                octree.updateInnerOccupancy()
                # Re-check after update
                current_nodes = octree.getNumLeafNodes()
            else:
                # No nodes created - skip updateInnerOccupancy() to avoid segfault
                if scan_count == 0:
                    print(f"    WARNING: Octree has 0 leaf nodes after first scan!")
                    print(f"    Large UTM coordinates may be causing insertions to fail silently.")
                    print(f"    Consider using coordinate shifting to start at (0,0,0)")
            
            if inserted == 0 and len(world_points_xyz) > 0:
                print(f"\nWARNING: Scan {pose_idx} had {len(world_points_xyz)} points but 0 were inserted!")
                print(f"  First point: {world_points_xyz[0]}")
                print(f"  Point range: x=[{world_points_xyz[:, 0].min():.2f}, {world_points_xyz[:, 0].max():.2f}], "
                      f"y=[{world_points_xyz[:, 1].min():.2f}, {world_points_xyz[:, 1].max():.2f}], "
                      f"z=[{world_points_xyz[:, 2].min():.2f}, {world_points_xyz[:, 2].max():.2f}]")
            
            scan_count += 1

        except Exception as e:
            print(f"\nError loading scan {pose_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final update of inner occupancy to ensure all nodes are properly created
    octree.updateInnerOccupancy()
    
    # Print statistics
    final_octree_size = octree.getNumLeafNodes()
    new_nodes = final_octree_size - initial_octree_size
    print(f"\nProcessed {scan_count} scans from {robot}")
    print(f"  Points inserted: {points_inserted_total}")
    print(f"  Octree nodes before: {initial_octree_size}, after: {final_octree_size} (+{new_nodes})")
    
    if scan_count > 0 and points_inserted_total == 0:
        print(f"\nWARNING: Processed {scan_count} scans but inserted 0 points!")
        print(f"  This suggests all insertions failed silently.")
    elif scan_count > 0 and points_inserted_total > 0 and final_octree_size == 0:
        print(f"\nWARNING: Inserted {points_inserted_total} points but octree has 0 leaf nodes!")
        print(f"  This suggests updateNode is not creating nodes. Trying alternative approach...")
    
    return scan_count


def accumulate_all_robots(dataset_path, 
                          environment, 
                          robots, 
                          num_scans=500, 
                          per_scan_voxel_size=1.0, 
                          global_voxel_size=0.01,
                          min_distance=3.0,
                          max_distance=60.0):
    """
    Accumulate LiDAR scans from all robots into a global ColorOcTree.
    
    Args:
        dataset_path: Path to dataset
        environment: Environment name
        robots: List of robot names
        num_scans: Number of scans to accumulate per robot
        per_scan_voxel_size: Not used (kept for compatibility)
        global_voxel_size: Octree resolution in meters
        min_distance: Minimum distance for filtering points
        max_distance: Maximum distance for filtering points
    
    Returns:
        octree: pyoctomap.ColorOcTree instance with all accumulated points
        intensity_dict: Dictionary mapping voxel keys to intensity info
        label_dict: Dictionary mapping voxel keys to label info
        color_dict: Dictionary mapping voxel keys to color info
    """
    # Initialize ColorOcTree with specified resolution
    octree = pyo.ColorOcTree(global_voxel_size)
    print(f"\nInitialized ColorOcTree with resolution: {global_voxel_size}m")
    print("Note: Octree will be initialized with the first actual point from the first scan (not [0,0,0])")
    
    # Initialize dictionaries to store intensities, labels, and colors per voxel
    intensity_dict = {}
    label_dict = {}
    color_dict = {}
    
    # Calculate coordinate offset once from first robot's first pose
    # This shifts all points to start near (0,0,0) for better octree behavior
    coordinate_offset = None
    if len(robots) > 0:
        first_robot = robots[0]
        poses_file = Path(dataset_path) / environment / first_robot / f"{first_robot}_{environment}_gt_utm_poses.csv"
        if poses_file.exists():
            print(f"\nCalculating coordinate offset from {first_robot}'s first pose...")
            poses = load_poses(poses_file)
            if len(poses) > 0:
                poses = transform_imu_to_lidar(poses)
                timestamps = sorted(poses.keys())
                if len(timestamps) > 0:
                    first_pose = poses[timestamps[0]]
                    coordinate_offset = np.array(first_pose[:3])
                    print(f"  Coordinate offset: [{coordinate_offset[0]:.2f}, {coordinate_offset[1]:.2f}, {coordinate_offset[2]:.2f}]")
                    print(f"  All points will be shifted by this offset to start near (0,0,0)")
    
    # Loop through all robots
    for robot in robots:
        print(f"\n{'='*80}")
        print(f"Processing {robot}")
        print(f"{'='*80}")
        
        poses_file = Path(dataset_path) / environment / robot / f"{robot}_{environment}_gt_utm_poses.csv"
        
        if not poses_file.exists():
            print(f"Warning: Poses file not found for {robot}: {poses_file}")
            print(f"Skipping {robot}...")
            continue
        
        print(f"Loading poses for {robot}...")
        poses = load_poses(poses_file)
        print(f"Loaded {len(poses)} poses for {robot}")
        
        if len(poses) == 0:
            print(f"No poses found for {robot}! Skipping...")
            continue
        
        print(f"Transforming poses from IMU to LiDAR frame for {robot}...")
        poses = transform_imu_to_lidar(poses)
        
        # Shift all poses to be relative to the global first pose (coordinate_offset)
        # This shifts coordinates near (0,0,0) for better octree behavior
        # All robots use the SAME offset so they're in the same coordinate frame
        if coordinate_offset is not None:
            print(f"Shifting poses to be relative to global first pose...")
            timestamps = sorted(poses.keys())
            if len(timestamps) > 0:
                # Shift all poses: subtract the global coordinate_offset from all positions
                # Keep rotations unchanged
                shifted_poses = {}
                for timestamp, pose in poses.items():
                    position = np.array(pose[:3])
                    quat = pose[3:7]
                    shifted_position = position - coordinate_offset  # Use global offset
                    shifted_poses[timestamp] = [shifted_position[0], shifted_position[1], shifted_position[2], 
                                               quat[0], quat[1], quat[2], quat[3]]
                poses = shifted_poses
                first_pose_after_shift = poses[timestamps[0]][:3]
                print(f"  Shifted {len(poses)} poses using global offset")
                print(f"  First pose position after shift: [{first_pose_after_shift[0]:.2f}, {first_pose_after_shift[1]:.2f}, {first_pose_after_shift[2]:.2f}]")
        
        print(f"\nAccumulating LiDAR scans for {robot} into octree...")
        scan_count = accumulate_lidar_scans(
            dataset_path,
            environment,
            robot,
            poses,
            octree,
            intensity_dict,
            label_dict,
            color_dict,
            num_scans=num_scans,
            per_scan_voxel_size=per_scan_voxel_size,
            global_voxel_size=global_voxel_size,
            min_distance=min_distance,
            max_distance=max_distance,
        )
        
        if scan_count == 0:
            print(f"Warning: No scans processed for {robot}")
    
    # Final update of inner occupancy for all robots
    # CRITICAL: Only call if we have leaf nodes (prevents segfault)
    final_nodes = octree.getNumLeafNodes()
    if final_nodes > 0:
        octree.updateInnerOccupancy()
        final_nodes = octree.getNumLeafNodes()  # Re-check after update
    else:
        print(f"\n  ⚠ WARNING: Octree has 0 leaf nodes - skipping updateInnerOccupancy() to prevent segfault")
        print(f"    This suggests coordinate shifting is needed or octree insertion is failing")
    
    final_nodes = octree.getNumLeafNodes()

    print(f"\n{'='*80}")
    print(f"Final octree statistics:")
    print(f"  Total leaf nodes: {final_nodes}")
    print(f"  Octree resolution: {global_voxel_size}m")
    print(f"  Intensity dictionary entries: {len(intensity_dict)}")
    print(f"  Label dictionary entries: {len(label_dict)}")
    # Debug: If we have dictionary entries but no nodes, updateNode is failing
    if len(intensity_dict) > 0 and final_nodes <= 1:
        print(f"\n  WARNING: {len(intensity_dict)} dictionary entries but only {final_nodes} leaf nodes!")
        print(f"  This suggests updateNode() is silently failing for large UTM coordinates.")
        print(f"  The octree may have size limits or need different initialization.")
    print(f"{'='*80}")
    
    return octree, intensity_dict, label_dict, color_dict, coordinate_offset


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create global semantic map from LiDAR scans")

    # Environment name (default: main_campus)
    parser.add_argument("--environment", type=str, default="kittredge_loop",
                       help="Environment name (default: main_campus)")

    # Number of scans to accumulate (default: 2000)
    parser.add_argument("--num_scans", type=int, default=2000,
                       help="Number of scans to accumulate (default: 4000)")

    # Per-scan voxel size in meters (default: 4.0)
    parser.add_argument("--per_scan_voxel", type=float, default=1.0,
                       help="Per-scan voxel size in meters (default: 1.0)")

    # Global voxel size in meters (default: 1.0)
    parser.add_argument("--global_voxel", type=float, default=0.1,
                       help="Global voxel size in meters (default: 0.1)")

    # Min distance for filtering points (default: 0.1)
    parser.add_argument("--min_distance", type=float, default=3.0,
                       help="Min distance for filtering points (default: 3.0)")

    # Max distance for filtering points (default: 10.0)
    parser.add_argument("--max_distance", type=float, default=60.0,
                       help="Max distance for filtering points (default: 60.0)")

    # Use lat/lon coordinates (default: keep UTM)
    parser.add_argument("--use_latlon", action="store_true",
                       help="Convert coordinates to lat/lon (default: keep UTM)")

    # Output file prefix (example: KL_SEM_MAP_OG)
    parser.add_argument("--output_postfix", type=str, default="sem_map_orig_confident",
                       help="Output file prefix (default: sem_map_orig)")

    # Save image (default: False)
    parser.add_argument("--save_image", action="store_true",
                       help="Save image (default: False)")

    args = parser.parse_args()
    
    # Hardcoded paths
    dataset_path = "/media/donceykong/donceys_data_ssd/datasets/CU_MULTI/data"
    environment = args.environment
    robots = ["robot1", "robot2", "robot3", "robot4"]
    file_dir = os.path.join(dataset_path, environment, "additional")

    # Accumulate all robots into octree
    octree, intensity_dict, label_dict, color_dict, coordinate_offset = accumulate_all_robots(
        dataset_path,
        environment,
        robots,
        num_scans=args.num_scans,
        per_scan_voxel_size=args.per_scan_voxel,
        global_voxel_size=args.global_voxel,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
    )

    # Check if we have any data
    if octree.getNumLeafNodes() == 0:
        print("\nError: No data accumulated from any robot!")
        return
    
    # Extract point cloud from octree
    print(f"\n{'='*80}")
    print("Extracting point cloud from octree...")
    if coordinate_offset is not None:
        print(f"Shifting coordinates back to original UTM frame (offset: [{coordinate_offset[0]:.2f}, {coordinate_offset[1]:.2f}, {coordinate_offset[2]:.2f}])")
    print(f"{'='*80}")
    
    combined_points, combined_colors, combined_intensities, combined_labels = extract_pointcloud_from_octree(
        octree, 
        intensity_dict=intensity_dict,
        label_dict=label_dict,
        color_dict=color_dict,
        resolution=args.global_voxel,
        coordinate_offset=coordinate_offset
    )
    
    if len(combined_points) == 0:
        print("\nError: No points extracted from octree!")
        return
    
    # Convert to lat/lon if requested
    if args.use_latlon:
        print("\nConverting UTM coordinates to lat/lon...")
        combined_points = utm_to_latlon(combined_points)
        coord_type = "latlon"
    else:
        coord_type = "utm"
    
    # Print statistics
    print(f"\nFinal point cloud statistics:")
    print(f"  Total points: {len(combined_points)}")
    print(f"  Coordinate system: {coord_type.upper()}")
    print(f"  X range: [{combined_points[:, 0].min():.6f}, {combined_points[:, 0].max():.6f}]")
    print(f"  Y range: [{combined_points[:, 1].min():.6f}, {combined_points[:, 1].max():.6f}]")
    print(f"  Z range: [{combined_points[:, 2].min():.6f}, {combined_points[:, 2].max():.6f}]")
    print(f"  Intensity range: [{combined_intensities.min():.3f}, {combined_intensities.max():.3f}]")
    print(f"  Unique labels: {np.unique(combined_labels)}")
    
    output_npy = f"{environment}_{args.output_postfix}.npy"
    save_point_cloud(combined_points, combined_intensities, combined_labels,
                    file_dir=file_dir, output_file=output_npy)
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"Generated files:")
    print(f"  - {output_npy}")
    print(f"\nData from {len(robots)} robots successfully combined and saved!")


if __name__ == "__main__":
    main()

