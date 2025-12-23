#!/usr/bin/env python3
"""
Script to create a global voxel grid from multiple robot lidar scans.
Uses relative poses to accumulate all scans in a common coordinate frame (each robot starts at 0,0,0).
Each scan is downsampled with 4m voxel grid before accumulation.
Publishes global map every 10 scans as PointCloud2.
"""

import os
import sys
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
# import open3d as o3d  # Not using Open3D anymore

# Import pyoctomap
import pyoctomap as pyo

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.serialization import deserialize_message, serialize_message
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    from tf2_msgs.msg import TFMessage
    import rosbag2_py
    from rosidl_runtime_py.utilities import get_message
    from builtin_interfaces.msg import Time
except ImportError as e:
    print(f"Error: Missing ROS2 dependencies: {e}")
    print("Please install: pip install rosbag2-py rclpy")
    sys.exit(1)


def extract_pose_from_tf(transform):
    """
    Extract position and quaternion from TF transform.
    
    Args:
        transform: geometry_msgs/msg/TransformStamped
    
    Returns:
        position: [x, y, z]
        quaternion: [qx, qy, qz, qw]
    """
    t = transform.transform.translation
    r = transform.transform.rotation
    position = np.array([t.x, t.y, t.z])
    quaternion = np.array([r.x, r.y, r.z, r.w])
    return position, quaternion


def load_poses_from_bag(poses_bag_path, robot_name=None):
    """
    Load relative poses from ROS2 bag containing TF messages.
    Auto-detects the correct TF frame pair (usually world -> {robot}_os_sensor).
    
    Args:
        poses_bag_path: Path to ROS2 bag with TF messages
        robot_name: Robot name for frame detection (e.g., "robot1")
    
    Returns:
        poses: Dictionary mapping timestamp (in nanoseconds) to [x, y, z, qx, qy, qz, qw]
    """
    print(f"Loading poses from: {poses_bag_path}")
    
    storage_options = rosbag2_py.StorageOptions(uri=str(poses_bag_path), storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    # First pass: detect frame IDs
    frame_pairs = set()
    msg_type = get_message('tf2_msgs/msg/TFMessage')
    scan_count = 0
    
    while reader.has_next() and scan_count < 10:
        (topic, data, timestamp) = reader.read_next()
        
        if topic == '/tf':
            tf_msg = deserialize_message(data, msg_type)
            for transform in tf_msg.transforms:
                # print(f"\nTransform: {transform.header.frame_id} -> {transform.child_frame_id}")
                frame_pairs.add((transform.header.frame_id, transform.child_frame_id))
            scan_count += 1
    
    # Determine which frame pair to use
    source_frame = None
    target_frame = None
    
    # Look for world -> sensor transform
    for parent, child in frame_pairs:
        if parent == 'world' or parent == 'odom':
            # print(f"\nUsing TF: {parent} -> {child}")
            source_frame = parent
            target_frame = child
            break
    
    if source_frame is None and len(frame_pairs) > 0:
        # Just use the first pair found
        source_frame, target_frame = list(frame_pairs)[0]
    
    if source_frame is None:
        print(f"ERROR: No TF frames found in bag")
        return {}
    
    print(f"Using TF: {source_frame} -> {target_frame}")
    
    # Reset reader to beginning
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    # Second pass: extract poses
    poses = {}
    
    while reader.has_next():
        (topic, data, timestamp) = reader.read_next()
        
        if topic == '/tf':
            tf_msg = deserialize_message(data, msg_type)
            
            for transform in tf_msg.transforms:
                # Look for the transform from source to target frame
                if (transform.header.frame_id == source_frame and 
                    transform.child_frame_id == target_frame):
                    
                    position, quaternion = extract_pose_from_tf(transform)
                    
                    # Use header timestamp as key
                    stamp = transform.header.stamp
                    timestamp_ns = stamp.sec * int(1e9) + stamp.nanosec
                    
                    poses[timestamp_ns] = np.concatenate([position, quaternion])
    
    print(f"Loaded {len(poses)} poses from bag")
    return poses


def transform_points(points, position, quaternion):
    """
    Transform points using position and quaternion.
    
    Args:
        points: Nx3 array of points
        position: [x, y, z]
        quaternion: [qx, qy, qz, qw]
    
    Returns:
        transformed_points: Nx3 array of transformed points
    """
    # Create rotation matrix from quaternion
    rotation = R.from_quat(quaternion).as_matrix()
    
    # Apply rotation and translation
    transformed = (rotation @ points.T).T + position
    
    return transformed


def unpack_rgb_from_uint32(rgb_uint32):
    """
    Unpack RGB from uint32 to separate channels.
    
    Args:
        rgb_uint32: N array of RGB packed as uint32
    
    Returns:
        rgb_float: Nx3 array of RGB values normalized to [0, 1]
    """
    # Convert to uint32 if needed (handle different types)
    rgb_uint32 = rgb_uint32.astype(np.uint32) if rgb_uint32.dtype != np.uint32 else rgb_uint32
    
    r = ((rgb_uint32 >> 16) & 0xFF).astype(np.float32) / 255.0
    g = ((rgb_uint32 >> 8) & 0xFF).astype(np.float32) / 255.0
    b = (rgb_uint32 & 0xFF).astype(np.float32) / 255.0
    
    return np.stack([r, g, b], axis=1)


def parse_pointcloud2_with_rgb(msg):
    """
    Parse PointCloud2 message with RGB field.
    
    Args:
        msg: sensor_msgs/msg/PointCloud2
    
    Returns:
        points: Nx3 array of xyz coordinates
        colors: Nx3 array of RGB colors [0, 1]
    """
    try:
        # Map datatype codes
        datatype_map = {
            1: ('i1', 1), 2: ('u1', 1), 3: ('i2', 2), 4: ('u2', 2),
            5: ('i4', 4), 6: ('u4', 4), 7: ('f4', 4), 8: ('f8', 8),
        }
        
        # Build dtype with offsets
        dtype_list = []
        last_offset = 0
        
        for field in msg.fields:
            if field.offset > last_offset:
                dtype_list.append((f'_pad{last_offset}', 'u1', field.offset - last_offset))
            
            np_dtype, dtype_size = datatype_map.get(field.datatype, ('u1', 1))
            field_size = dtype_size * field.count
            
            if field.count == 1:
                dtype_list.append((field.name, np_dtype))
            else:
                dtype_list.append((field.name, np_dtype, field.count))
            
            last_offset = field.offset + field_size
        
        if msg.point_step > last_offset:
            dtype_list.append((f'_pad{last_offset}', 'u1', msg.point_step - last_offset))
        
        # Parse data
        cloud_array = np.frombuffer(msg.data, dtype=np.dtype(dtype_list))
        
        # Extract xyz
        points = np.zeros((len(cloud_array), 3), dtype=np.float32)
        points[:, 0] = cloud_array['x']
        points[:, 1] = cloud_array['y']
        points[:, 2] = cloud_array['z']
        
        # Check for invalid values
        if np.any(~np.isfinite(points)):
            print(f"WARNING: Found NaN/Inf in points, filtering...")
            valid_mask = np.all(np.isfinite(points), axis=1)
            points = points[valid_mask]
            cloud_array = cloud_array[valid_mask]
        
        # Extract RGB
        if 'rgb' in cloud_array.dtype.names:
            rgb_uint32 = cloud_array['rgb']
            colors = unpack_rgb_from_uint32(rgb_uint32)
        else:
            colors = np.ones((len(points), 3), dtype=np.float32) * 0.5  # Gray if no RGB
        
        # Ensure colors are valid
        colors = np.clip(colors, 0.0, 1.0)
        
        return points, colors
    
    except Exception as e:
        print(f"ERROR in parse_pointcloud2_with_rgb: {e}")
        print(f"  Fields: {[f.name for f in msg.fields]}")
        print(f"  Point step: {msg.point_step}")
        print(f"  Data size: {len(msg.data)}")
        raise


def find_nearest_pose(timestamp_ns, poses):
    """
    Find the nearest pose to a given timestamp.
    
    Args:
        timestamp_ns: Timestamp in nanoseconds
        poses: Dictionary of timestamp -> pose
    
    Returns:
        pose: [x, y, z, qx, qy, qz, qw] or None if not found
    """
    if len(poses) == 0:
        return None
    
    timestamps = np.array(list(poses.keys()))
    idx = np.argmin(np.abs(timestamps - timestamp_ns))
    nearest_timestamp = timestamps[idx]
    
    # Check if within 100ms
    time_diff = abs(timestamp_ns - nearest_timestamp) / 1e9

    # Error out if the time difference is greater than 100ms
    if time_diff > 0.1:
        print(f"ERROR: Time difference is greater than 100ms: {time_diff}")
        raise ValueError(f"Time difference is greater than 100ms: {time_diff}")
    else:
        return poses[nearest_timestamp]


def rebuild_color_dict_from_octree(octree, color_dict, resolution, verbose=True):
    """
    Rebuild color dictionary from octree nodes to ensure perfect 1:1 synchronization.
    Uses mode color (most frequent) for each voxel.
    This function guarantees len(color_dict) == octree.getNumLeafNodes()
    
    Args:
        octree: pyoctomap.OcTree instance
        color_dict: Dictionary mapping voxel keys to color info
        resolution: Octree resolution in meters
        verbose: If True, print verification messages (default: True)
    
    Returns:
        cleaned_color_dict: Dictionary with exactly one entry per octree node (1:1 match)
    """
    cleaned_color_dict = {}
    
    # Build a lookup structure for faster color matching
    # Create a list of (voxel_coords, color) tuples from color_dict
    color_lookup = []
    for voxel_key, color_info in color_dict.items():
        parts = voxel_key.split('_')
        if len(parts) == 3:
            try:
                coords = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
                # Extract mode color from color_info
                if isinstance(color_info, dict) and 'mode_color' in color_info:
                    color = color_info['mode_color']
                else:
                    color = np.array(color_info)  # Legacy format
                color_lookup.append((coords, color, voxel_key))  # Keep original key for exact matching
            except (ValueError, TypeError):
                continue
    
    if len(color_lookup) == 0:
        # No colors to match, use defaults
        for leaf_it in octree.begin_leafs():
            if octree.isNodeOccupied(leaf_it):
                coord = leaf_it.getCoordinate()
                x, y, z = coord[0], coord[1], coord[2]
                # Use exact octree node coordinates as key
                voxel_key = f"{x:.6f}_{y:.6f}_{z:.6f}"
                cleaned_color_dict[voxel_key] = {'mode_color': np.array([0.5, 0.5, 0.5])}
        return cleaned_color_dict
    
    # Convert to numpy for efficient matching
    color_coords = np.array([c[0] for c in color_lookup])
    color_values = [c[1] for c in color_lookup]
    color_keys = [c[2] for c in color_lookup]  # Original keys for exact matching
    
    # Iterate through ALL octree nodes - this is the source of truth
    node_count = 0
    for leaf_it in octree.begin_leafs():
        if octree.isNodeOccupied(leaf_it):
            node_count += 1
            coord = leaf_it.getCoordinate()
            x, y, z = coord[0], coord[1], coord[2]
            
            # Use EXACT octree node coordinates as key (not rounded)
            # This ensures 1:1 mapping - each octree node gets exactly one color entry
            voxel_key = f"{x:.6f}_{y:.6f}_{z:.6f}"
            
            # Round to voxel resolution for color matching
            voxel_x = round(x / resolution) * resolution
            voxel_y = round(y / resolution) * resolution
            voxel_z = round(z / resolution) * resolution
            voxel_coord = np.array([voxel_x, voxel_y, voxel_z])
            
            # Try to find matching color from original color_dict
            # First try exact voxel key match (in case color_dict already uses exact keys)
            if voxel_key in color_dict:
                color_info = color_dict[voxel_key]
                if isinstance(color_info, dict) and 'mode_color' in color_info:
                    cleaned_color_dict[voxel_key] = {'mode_color': color_info['mode_color'].copy()}
                else:
                    cleaned_color_dict[voxel_key] = {'mode_color': np.array(color_info)}
            else:
                # Try rounded key match (in case color_dict uses rounded keys from insertion)
                rounded_key = f"{voxel_x:.6f}_{voxel_y:.6f}_{voxel_z:.6f}"
                if rounded_key in color_dict:
                    color_info = color_dict[rounded_key]
                    if isinstance(color_info, dict) and 'mode_color' in color_info:
                        cleaned_color_dict[voxel_key] = {'mode_color': color_info['mode_color'].copy()}
                    else:
                        cleaned_color_dict[voxel_key] = {'mode_color': np.array(color_info)}
                else:
                    # Find nearest color by distance (within resolution tolerance)
                    if len(color_coords) > 0:
                        # Use exact coordinates for distance calculation (more accurate)
                        distances = np.linalg.norm(color_coords - np.array([x, y, z]), axis=1)
                        nearest_idx = np.argmin(distances)
                        nearest_dist = distances[nearest_idx]
                        
                        # Use a more generous tolerance - resolution itself (not resolution/2)
                        # This handles cases where coordinates differ slightly due to rounding
                        if nearest_dist < resolution:
                            cleaned_color_dict[voxel_key] = {'mode_color': color_values[nearest_idx].copy()}
                        else:
                            # Default gray - this shouldn't happen often if colors were inserted correctly
                            cleaned_color_dict[voxel_key] = {'mode_color': np.array([0.5, 0.5, 0.5])}
                    else:
                        cleaned_color_dict[voxel_key] = {'mode_color': np.array([0.5, 0.5, 0.5])}
    
    # Verify 1:1 match
    expected_nodes = octree.getNumLeafNodes()
    actual_colors = len(cleaned_color_dict)
    if actual_colors != expected_nodes:
        print(f"  ERROR: Rebuild mismatch: {actual_colors} colors != {expected_nodes} nodes")
    elif verbose:
        print(f"  Verified 1:1 match: {actual_colors} colors == {expected_nodes} nodes")
    
    return cleaned_color_dict


def extract_pointcloud_from_octree(octree, color_dict, resolution=0.1):
    """
    Extract pointcloud with RGB colors from pyoctomap OcTree using a color dictionary.
    Uses mode color (most frequent) for each voxel.
    Uses exact octree node coordinates as keys for matching.
    
    Args:
        octree: pyoctomap.OcTree instance
        color_dict: Dictionary mapping voxel keys to color info
                   Format: {voxel_key: {'mode_color': np.array, ...}}
                   Keys are exact octree node coordinates: "x_y_z"
        resolution: octree resolution in meters (for consistent key generation)
    
    Returns:
        points: numpy array of shape (N, 3) containing xyz coordinates
        colors: numpy array of shape (N, 3) containing RGB colors [0, 1]
    """
    points = []
    colors = []
    
    try:
        # Iterate through all leaf nodes using begin_leafs()
        for leaf_it in octree.begin_leafs():
            # Check if node is occupied using isNodeOccupied(iterator)
            if octree.isNodeOccupied(leaf_it):
                # Get point coordinates using getCoordinate()
                coord = leaf_it.getCoordinate()
                x, y, z = coord[0], coord[1], coord[2]
                point = np.array([x, y, z])
                points.append(point)
                
                # Use EXACT octree node coordinates as key (same as rebuild function)
                voxel_key = f"{x:.6f}_{y:.6f}_{z:.6f}"
                
                # Get color from dictionary - use mode color
                if voxel_key in color_dict:
                    if isinstance(color_dict[voxel_key], dict) and 'mode_color' in color_dict[voxel_key]:
                        rgb = color_dict[voxel_key]['mode_color']
                    else:
                        # Legacy format - treat as direct color array
                        rgb = np.array(color_dict[voxel_key])
                else:
                    # Default gray color if not found
                    rgb = np.array([0.5, 0.5, 0.5])
                colors.append(rgb)
    except Exception as e:
        print(f"WARNING: Error extracting pointcloud from octree: {e}")
        import traceback
        traceback.print_exc()
        # Return empty arrays if extraction fails
        return np.array([]), np.array([])
    
    if len(points) == 0:
        return np.array([]), np.array([])
    
    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)
    
    return points, colors


def quantize_color_to_semantic(color):
    """
    Quantize RGB color to nearest semantic color value.
    Semantic colors are integers [0-255] divided by 255.0, so we round to nearest 1/255.
    
    Args:
        color: numpy array of shape (3,) with RGB values [0, 1]
    
    Returns:
        quantized_color: numpy array with color quantized to nearest 1/255 increment
    """
    # Quantize to nearest 1/255 (semantic colors are integers / 255.0)
    quantized = np.round(color * 255.0) / 255.0
    # Clamp to [0, 1]
    quantized = np.clip(quantized, 0.0, 1.0)
    return quantized


def insert_points_into_octree(octree, points, colors, color_dict, resolution=0.1):
    """
    Insert points with RGB colors into pyoctomap OcTree.
    Colors are stored in a separate dictionary keyed by voxel coordinates.
    Uses mode (most frequent color) instead of averaging.
    
    Args:
        octree: pyoctomap.OcTree instance
        points: numpy array of shape (N, 3) containing xyz coordinates
        colors: numpy array of shape (N, 3) containing RGB colors [0, 1]
        color_dict: Dictionary to store colors (modified in-place)
                   Format: {voxel_key: {'color_counts': dict, 'mode_color': np.array}}
        resolution: octree resolution in meters (default: 0.1)
    """
    if len(points) == 0:
        return 0
    
    inserted_count = 0
    error_count = 0
    
    for i in range(len(points)):
        point = points[i]
        color = colors[i]
        
        try:
            # Extract coordinates as floats
            x, y, z = float(point[0]), float(point[1]), float(point[2])
            
            # Insert point into octree using updateNode
            # pyoctomap updateNode takes coordinates as [x, y, z] and occupancy boolean
            # Note: updateNode will update existing nodes or create new ones
            octree.updateNode([x, y, z], True)
            
            # Verify insertion by checking if node exists
            # search method also takes coordinates as [x, y, z] and returns OcTreeNode or None
            node = octree.search([x, y, z])
            
            if node is not None:
                # Round to voxel resolution to create consistent keys
                # Use the exact coordinates we're inserting
                voxel_x = round(x / resolution) * resolution
                voxel_y = round(y / resolution) * resolution
                voxel_z = round(z / resolution) * resolution
                voxel_key = f"{voxel_x:.6f}_{voxel_y:.6f}_{voxel_z:.6f}"
                
                # Check if this voxel_key already exists in color_dict
                was_already_in_dict = voxel_key in color_dict
                
                # Only count as new insertion if this voxel_key wasn't in our dictionary
                if not was_already_in_dict:
                    inserted_count += 1
                
                # Store/update color in dictionary using mode (most frequent)
                if was_already_in_dict:
                    # Existing voxel - track color counts and update mode
                    # Handle both new format (exact coordinates) and legacy format (rounded keys)
                    color_info = color_dict[voxel_key]
                    
                    # If it's the new format with mode_color, convert to color_counts format
                    if isinstance(color_info, dict) and 'mode_color' in color_info and 'color_counts' not in color_info:
                        # Convert from mode_color-only format to color_counts format for tracking
                        mode_color = color_info['mode_color']
                        quantized_mode = tuple(np.round(quantize_color_to_semantic(np.array(mode_color)), decimals=3))
                        color_dict[voxel_key] = {
                            'color_counts': {quantized_mode: 1},
                            'mode_color': mode_color.copy()
                        }
                    elif not isinstance(color_info, dict):
                        # Convert legacy format (direct color array)
                        old_color = color_info
                        quantized_old = quantize_color_to_semantic(np.array(old_color))
                        color_dict[voxel_key] = {
                            'color_counts': {tuple(quantized_old): 1},
                            'mode_color': quantized_old
                        }
                    # Now color_dict[voxel_key] is guaranteed to have color_counts
                    
                    # Quantize color to nearest semantic color value
                    quantized_color = quantize_color_to_semantic(color)
                    color_tuple = tuple(quantized_color)
                    
                    # Add this color to counts
                    if color_tuple in color_dict[voxel_key]['color_counts']:
                        color_dict[voxel_key]['color_counts'][color_tuple] += 1
                    else:
                        color_dict[voxel_key]['color_counts'][color_tuple] = 1
                    
                    # Update mode color (most frequent)
                    mode_tuple = max(color_dict[voxel_key]['color_counts'], 
                                   key=color_dict[voxel_key]['color_counts'].get)
                    color_dict[voxel_key]['mode_color'] = np.array(mode_tuple)
                else:
                    # New voxel - initialize with quantized color
                    quantized_color = quantize_color_to_semantic(color)
                    color_tuple = tuple(quantized_color)
                    color_dict[voxel_key] = {
                        'color_counts': {color_tuple: 1},
                        'mode_color': quantized_color.copy()
                    }
            else:
                error_count += 1
                if error_count <= 3:
                    print(f"WARNING: Point ({x:.2f}, {y:.2f}, {z:.2f}) insertion failed - node not found after insertion")
                continue
                
        except Exception as e:
            error_count += 1
            if error_count <= 5:  # Only print first few errors
                print(f"WARNING: Error inserting point {i}: {e}")
                import traceback
                traceback.print_exc()
            continue
    
    if error_count > 5:
        print(f"WARNING: {error_count} points failed to insert (showing first 5 errors)")
    
    return inserted_count


def create_pointcloud2_message(points, colors, frame_id="map", timestamp_sec=0):
    """
    Create PointCloud2 message with RGB colors.
    
    Args:
        points: Nx3 array of xyz
        colors: Nx3 array of RGB [0, 1]
        frame_id: Frame ID for the point cloud
        timestamp_sec: Timestamp in seconds
    
    Returns:
        msg: sensor_msgs/msg/PointCloud2
    """
    msg = PointCloud2()
    
    # Header
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.header.stamp = Time()
    msg.header.stamp.sec = int(timestamp_sec)
    msg.header.stamp.nanosec = int((timestamp_sec - int(timestamp_sec)) * 1e9)
    
    msg.height = 1
    msg.width = len(points)
    msg.is_bigendian = False
    msg.is_dense = False
    
    # Fields
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
    ]
    msg.fields = fields
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    
    # Pack data
    data = np.zeros(len(points), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('rgb', np.uint32)
    ])
    
    data['x'] = points[:, 0]
    data['y'] = points[:, 1]
    data['z'] = points[:, 2]
    
    # Pack RGB to uint32
    r = (colors[:, 0] * 255).astype(np.uint32)
    g = (colors[:, 1] * 255).astype(np.uint32)
    b = (colors[:, 2] * 255).astype(np.uint32)
    data['rgb'] = (r << 16) | (g << 8) | b
    
    msg.data = data.tobytes()
    
    return msg


def process_robot_incremental(dataset_path, environment, robot, output_writer, batch_size=10, max_scans=None, octree_resolution=0.1):
    """
    Process a single robot and incrementally accumulate/publish its scans using pyoctomap.
    Copies ALL lidar scans and odometry messages to output bag.
    Each scan is inserted into an OcTree octree for efficient spatial representation.
    RGB colors are maintained in a separate dictionary keyed by voxel coordinates.
    Publishes global map every batch_size scans.
    
    Args:
        dataset_path: Path to dataset root
        environment: Environment name
        robot: Robot name
        output_writer: ROS2 bag writer for output bag
        batch_size: Number of scans to accumulate before publishing global map (default: 10)
        max_scans: Maximum number of scans to process (default: None = all)
        octree_resolution: Resolution of the octree in meters (default: 0.1)
    
    Returns:
        None
    """
    print(f"\n{'='*80}")
    print(f"Processing {robot} in {environment}")
    print(f"Octree resolution: {octree_resolution}m, Publish every: {batch_size} scans")
    if max_scans:
        print(f"Max scans: {max_scans} (testing mode)")
    print(f"{'='*80}")
    
    robot_dir = dataset_path / environment / robot
    
    # Paths
    lidar_bag_path = robot_dir / f"{robot}_{environment}_lidar_rgb"
    poses_bag_path = robot_dir / f"{robot}_{environment}_gt_rel_poses"
    
    if not lidar_bag_path.exists():
        print(f"ERROR: Lidar bag not found: {lidar_bag_path}")
        return None, None
    
    if not poses_bag_path.exists():
        print(f"ERROR: Poses bag not found: {poses_bag_path}")
        return None, None
    
    # Load poses
    poses = load_poses_from_bag(poses_bag_path, robot_name=robot)
    if len(poses) == 0:
        print("ERROR: No poses loaded")
        return None, None
    
    # Initialize OcTree for spatial indexing (without built-in color support)
    octree = pyo.OcTree(octree_resolution)
    
    # Dictionary to store RGB colors keyed by voxel coordinates
    color_dict = {}
    
    # Copy ALL odometry/TF messages from poses bag to output bag
    if output_writer is not None:
        print("Copying odometry/TF messages to output bag...")
        odom_storage_opts = rosbag2_py.StorageOptions(uri=str(poses_bag_path), storage_id='sqlite3')
        odom_converter_opts = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        
        odom_reader = rosbag2_py.SequentialReader()
        odom_reader.open(odom_storage_opts, odom_converter_opts)
        
        # Create topics in output bag for TF and other odom messages
        odom_topics = odom_reader.get_all_topics_and_types()
        for topic in odom_topics:
            try:
                topic_info = rosbag2_py.TopicMetadata(
                    name=topic.name,
                    type=topic.type,
                    serialization_format='cdr'
                )
                output_writer.create_topic(topic_info)
            except:
                pass  # Topic might already exist, that's okay
        
        # Copy all odom messages
        odom_count = 0
        while odom_reader.has_next():
            (topic, data, timestamp) = odom_reader.read_next()
            output_writer.write(topic, data, timestamp)
            odom_count += 1
        
        print(f"Copied {odom_count} odometry/TF messages\n")
    
    # Open bag reader
    storage_options = rosbag2_py.StorageOptions(uri=str(lidar_bag_path), storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    # Find PointCloud2 topic and create topics in output bag
    topic_types = reader.get_all_topics_and_types()
    pointcloud_topic = None
    
    # Create ALL topics from lidar bag in the output bag
    if output_writer is not None:
        for topic in topic_types:
            try:
                topic_info = rosbag2_py.TopicMetadata(
                    name=topic.name,
                    type=topic.type,
                    serialization_format='cdr'
                )
                output_writer.create_topic(topic_info)
            except:
                pass  # Topic might already exist
    
    # Find the PointCloud2 topic
    for topic in topic_types:
        if topic.type == 'sensor_msgs/msg/PointCloud2' and 'points' in topic.name:
            pointcloud_topic = topic.name
            break
    
    if not pointcloud_topic:
        print("ERROR: No PointCloud2 topic found")
        return None, None
    
    print(f"Streaming from topic: {pointcloud_topic}")
    
    # Get total message count for progress bar
    metadata = reader.get_metadata()
    total_messages = sum([t.message_count for t in metadata.topics_with_message_count 
                          if t.topic_metadata.name == pointcloud_topic])
    
    # Limit to max_scans if specified
    if max_scans and max_scans < total_messages:
        total_messages = max_scans
        print(f"Processing {total_messages} scans (limited by --max_scans)...\n")
    else:
        print(f"Processing {total_messages} scans (streaming one at a time)...\n")
    
    processed_count = 0
    publish_count = 0
    msg_type = get_message('sensor_msgs/msg/PointCloud2')
    
    # Process messages and insert into octree
    with tqdm(total=total_messages, desc=f"Processing {robot}") as pbar:
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            
            # Copy ALL messages (lidar scans, IMU, etc.) to output bag
            if output_writer is not None:
                output_writer.write(topic, data, timestamp)
            
            if topic != pointcloud_topic:
                continue
            
            # Deserialize pointcloud message
            msg = deserialize_message(data, msg_type)
            
            # Find nearest pose
            pose = find_nearest_pose(timestamp, poses)
            if pose is None:
                pbar.update(1)
                continue
            
            try:
                # Parse point cloud
                points, colors = parse_pointcloud2_with_rgb(msg)
                
                # Skip if no points
                if len(points) == 0:
                    pbar.update(1)
                    continue
                
                # Transform to world frame
                position = pose[:3]
                quaternion = pose[3:]
                transformed_points = transform_points(points, position, quaternion)
                
                # Validate transformed points
                if not np.all(np.isfinite(transformed_points)):
                    print(f"WARNING: Invalid transformed points, skipping scan")
                    pbar.update(1)
                    continue
                
                # Insert points into octree (octree handles downsampling automatically)
                inserted = insert_points_into_octree(octree, transformed_points, colors, color_dict, resolution=octree_resolution)
                
                # Always rebuild color_dict after every scan to ensure perfect 1:1 match
                # Insertion uses rounded keys which can create mismatches, so rebuild from octree (source of truth)
                color_dict = rebuild_color_dict_from_octree(octree, color_dict, octree_resolution, verbose=False)
                
                # Print the number of points inserted into the octree
                if processed_count % 10 == 0:  # Print every 10 scans to avoid spam
                    # print(f"Scan {processed_count}: Inserted {inserted}/{len(transformed_points)} points, Octree nodes: {octree.getNumLeafNodes()}, Colors: {len(color_dict)}")
                    if len(color_dict) != octree.getNumLeafNodes():
                        print(f"  ERROR: Mismatch after rebuild: {len(color_dict)} colors != {octree.getNumLeafNodes()} nodes")
                
            except Exception as e:
                print(f"\nERROR processing scan {processed_count}: {e}")
                import traceback
                traceback.print_exc()
                pbar.update(1)
                continue
            
            processed_count += 1
            pbar.update(1)
            
            # Check if we've reached max_scans limit
            if max_scans and processed_count >= max_scans:
                tqdm.write(f"\nReached max_scans limit ({max_scans}), stopping...")
                break
            
            # Every batch_size scans: publish global map from octree
            if processed_count % batch_size == 0:
                # Always rebuild color_dict from octree to ensure perfect 1:1 match
                # This is the source of truth - only octree nodes get colors
                # print(f"  Rebuilding color_dict from octree: {len(color_dict)} entries -> {octree.getNumLeafNodes()} nodes")
                # color_dict = rebuild_color_dict_from_octree(octree, color_dict, octree_resolution, verbose=True)
                # print(f"  Rebuilt color_dict: {len(color_dict)} entries (1:1 match with octree)")
                
                # Extract pointcloud from octree with colors
                all_points, all_colors = extract_pointcloud_from_octree(octree, color_dict, resolution=octree_resolution)
                
                if len(all_points) > 0:
                    # Convert timestamp to seconds and add 1/30 second offset
                    # This ensures the global map is published slightly after the last lidar scan
                    timestamp_sec = (timestamp / 1e9) + (1.0 / 30.0)
                    
                    # Create message from octree
                    pc_msg = create_pointcloud2_message(
                        all_points, 
                        all_colors,
                        frame_id="world", 
                        timestamp_sec=timestamp_sec
                    )
                    
                    # Write to output bag with adjusted timestamp (in nanoseconds)
                    if output_writer is not None:
                        adjusted_timestamp_ns = int(timestamp_sec * 1e9)
                        output_writer.write(f'/robot{robot}/map', serialize_message(pc_msg), adjusted_timestamp_ns)
                        publish_count += 1
    
    # Final rebuild to ensure perfect 1:1 match
    print(f"\nFinal rebuild: {len(color_dict)} entries -> {octree.getNumLeafNodes()} nodes")
    color_dict = rebuild_color_dict_from_octree(octree, color_dict, octree_resolution, verbose=True)
    print(f"Final color_dict: {len(color_dict)} entries (should match {octree.getNumLeafNodes()} nodes exactly)")
    
    # Final publish from octree
    all_points, all_colors = extract_pointcloud_from_octree(octree, color_dict, resolution=octree_resolution)
    
    if len(all_points) > 0 and output_writer is not None:
        timestamp_sec = timestamp / 1e9 if processed_count > 0 else 0.0
        pc_msg = create_pointcloud2_message(
            all_points, 
            all_colors,
            frame_id="world", 
            timestamp_sec=timestamp_sec
        )
        output_writer.write(f'/robot{robot}/map', serialize_message(pc_msg), timestamp)
        publish_count += 1
    
    print(f"\n{'='*60}")
    print(f"Completed {robot}:")
    print(f"  Scans processed: {processed_count}")
    print(f"  Map published: {publish_count} times")
    print(f"  Final octree size: {octree.getNumLeafNodes()} nodes")
    print(f"{'='*60}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create global voxel grid from multi-robot lidar scans (incremental with publishing)"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/media/donceykong/donceys_data_ssd/datasets/CU_MULTI/data",
        help="Path to dataset root"
    )

    parser.add_argument(
        "--environment",
        type=str,
        default="kittredge_loop",
        help="Environment name"
    )

    parser.add_argument(
        "--robots",
        type=str,
        nargs='+',
        default=None,
        help="Robot names to process (default: all robots)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Number of scans to process before publishing global map (default: 20 scans, ~1.0 Hz publish rate)"
    )

    parser.add_argument(
        "--max_scans",
        type=int,
        default=None,
        help="Maximum number of scans to process (default: all scans)"
    )

    parser.add_argument(
        "--output_postfix",
        type=str,
        default="sem_map",
        help="Output file postfix for saving point cloud (default: sem_map)"
    )

    parser.add_argument(
        "--octree_resolution",
        type=float,
        default=0.5     ,
        help="Resolution of the octree in meters (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    
    # Determine which robots to process
    if args.robots:
        robots = args.robots
    else:
        env_path = os.path.join(dataset_path, args.environment)
        if not os.path.exists(env_path):
            print(f"ERROR: Environment path not found: {env_path}")
            return
        robots = [d for d in os.listdir(env_path) 
                  if os.path.isdir(os.path.join(env_path, d)) and d.startswith('robot')]
        robots = sorted(robots)
    
    print(f"\n{'='*80}")
    print("Incremental Global Octree Map Creation (pyoctomap)")
    print(f"{'='*80}")
    print(f"Environment: {args.environment}")
    print(f"Robots: {', '.join(robots)}")
    print(f"Octree resolution: {args.octree_resolution}m")
    print(f"Batch size: {args.batch_size} scans (~{args.batch_size/10:.1f} Hz publish rate)")
    print(f"{'='*80}\n")
    
    try:
        for robot in robots:
            # Create output bag for THIS ROBOT
            robot_dir = os.path.join(dataset_path, args.environment, robot)
            
            if args.output_postfix:
                output_path = os.path.join(robot_dir, f"{robot}_{args.environment}_{args.output_postfix}")
            
            # Remove existing bag if it exists
            if os.path.exists(output_path):
                print(f"\nRemoving existing bag: {output_path}")
                os.remove(output_path)
            
            print(f"Creating output bag: {output_path}")
            
            storage_options = rosbag2_py.StorageOptions(
                uri=output_path,
                storage_id='sqlite3'
            )
            converter_options = rosbag2_py.ConverterOptions(
                input_serialization_format='cdr',
                output_serialization_format='cdr'
            )
            
            writer = rosbag2_py.SequentialWriter()
            writer.open(storage_options, converter_options)
            
            # Create global_map topic
            topic_info = rosbag2_py.TopicMetadata(
                name=f'/robot{robot}/map',
                type='sensor_msgs/msg/PointCloud2',
                serialization_format='cdr'
            )
            writer.create_topic(topic_info)
            
            # Process this robot
            process_robot_incremental(
                dataset_path, args.environment, robot, writer, args.batch_size, args.max_scans, args.octree_resolution
            )
            
            # Close this robot's bag
            del writer
            
        print(f"\n{'='*80}")
        print("SUCCESS! All robots processed.")
        print(f"{'='*80}")
        print(f"Processed {len(robots)} robot(s)")
        print(f"Output bags created in each robot's directory:")
        for robot in robots:
            robot_dir = os.path.join(dataset_path, args.environment, robot)
            output_bag = os.path.join(robot_dir, f"{robot}_{args.environment}_{args.output_postfix}")
            if os.path.exists(output_bag):
                print(f"  - {output_bag}")

        # print(f"\nTo view:")
        # print(f"  ros2 bag play <robot_bag_path>")
        # print(f"  # Open RViz2 and add PointCloud2 on /global_map")
        # print(f"{'='*80}")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

