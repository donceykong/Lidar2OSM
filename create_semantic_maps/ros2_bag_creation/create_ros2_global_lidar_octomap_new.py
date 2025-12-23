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


def voxel_downsample_points_colors(points, colors, voxel_size=1.0):
    """
    Downsample point cloud using voxel centers (numpy-based, no Open3D).
    
    Args:
        points: numpy array of shape (N, 3) containing xyz coordinates
        colors: numpy array of shape (N, 3) containing RGB colors [0, 1]
        voxel_size: size of voxel cube in meters
    
    Returns:
        downsampled_points: numpy array of shape (M, 3) with voxel centers
        downsampled_colors: numpy array of shape (M, 3) with mean colors
    """
    if len(points) == 0:
        return points, colors
    
    # Calculate voxel grid dimensions
    min_coords = np.min(points, axis=0)
    voxel_indices = np.floor((points - min_coords) / voxel_size).astype(np.int64)
    
    # Create unique voxel keys
    voxel_keys = np.array([f"{x}_{y}_{z}" for x, y, z in voxel_indices])
    
    # Find unique voxels and calculate centers
    unique_voxels, inverse_indices = np.unique(voxel_keys, return_inverse=True)
    
    downsampled_points = []
    downsampled_colors = []
    
    for i in range(len(unique_voxels)):
        # Find all points in this voxel
        voxel_mask = inverse_indices == i
        voxel_points = points[voxel_mask]
        voxel_colors = colors[voxel_mask]
        
        # Use voxel center (mean of all points in voxel)
        if len(voxel_points) > 0:
            voxel_center = np.mean(voxel_points, axis=0)
            mean_color = np.mean(voxel_colors, axis=0)
            
            downsampled_points.append(voxel_center)
            downsampled_colors.append(mean_color)
    
    downsampled_points = np.array(downsampled_points, dtype=np.float32)
    downsampled_colors = np.array(downsampled_colors, dtype=np.float32)
    
    return downsampled_points, downsampled_colors


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
        # Find field offsets by name
        field_dict = {field.name: (field.offset, field.datatype, field.count) 
                     for field in msg.fields}
        
        # Map datatype codes to numpy dtypes and sizes
        datatype_map = {
            1: (np.int8, 1), 2: (np.uint8, 1), 3: (np.int16, 2), 4: (np.uint16, 2),
            5: (np.int32, 4), 6: (np.uint32, 4), 7: (np.float32, 4), 8: (np.float64, 8),
        }
        
        # Get number of points
        num_points = msg.width * msg.height
        
        # Extract xyz points directly from byte data (vectorized)
        points = np.zeros((num_points, 3), dtype=np.float32)
        
        if 'x' in field_dict and 'y' in field_dict and 'z' in field_dict:
            x_offset, x_dtype, x_count = field_dict['x']
            y_offset, y_dtype, y_count = field_dict['y']
            z_offset, z_dtype, z_count = field_dict['z']
            
            x_np_dtype, x_size = datatype_map.get(x_dtype, (np.float32, 4))
            y_np_dtype, y_size = datatype_map.get(y_dtype, (np.float32, 4))
            z_np_dtype, z_size = datatype_map.get(z_dtype, (np.float32, 4))
            
            # Reshape data to (num_points, point_step) for efficient slicing
            data_array = np.frombuffer(msg.data, dtype=np.uint8).reshape(num_points, msg.point_step)
            
            # Extract x, y, z fields using vectorized byte slicing
            for i in range(num_points):
                points[i, 0] = np.frombuffer(data_array[i, x_offset:x_offset+x_size].tobytes(), 
                                            dtype=x_np_dtype)[0]
                points[i, 1] = np.frombuffer(data_array[i, y_offset:y_offset+y_size].tobytes(), 
                                            dtype=y_np_dtype)[0]
                points[i, 2] = np.frombuffer(data_array[i, z_offset:z_offset+z_size].tobytes(), 
                                            dtype=z_np_dtype)[0]
        else:
            raise ValueError("Missing x, y, or z fields in point cloud")
        
        # Check for invalid values
        if np.any(~np.isfinite(points)):
            print(f"WARNING: Found NaN/Inf in points, filtering...")
            valid_mask = np.all(np.isfinite(points), axis=1)
            points = points[valid_mask]
        else:
            valid_mask = np.ones(num_points, dtype=bool)
        
        # Extract RGB directly from byte data (vectorized)
        colors = np.zeros((len(points), 3), dtype=np.float32)
        
        # Use the same data_array from above for RGB extraction
        data_array = np.frombuffer(msg.data, dtype=np.uint8).reshape(num_points, msg.point_step)
        
        # Store RGB extraction metadata for error reporting
        rgb_extraction_info = {"method": None, "offset": None, "dtype": None, "raw_values": None}
        
        if 'rgb' in field_dict:
            rgb_offset, rgb_dtype, rgb_count = field_dict['rgb']
            rgb_extraction_info["method"] = "packed_rgb_uint32"
            rgb_extraction_info["offset"] = rgb_offset
            rgb_extraction_info["dtype"] = rgb_dtype
            
            # RGB should be UINT32 (datatype 6) packed as R, G, B, A (or R, G, B)
            if rgb_dtype == 6:  # UINT32
                # Extract RGB bytes for all valid points at once
                valid_indices = np.where(valid_mask)[0]
                
                # Extract RGB uint32 values using proper byte interpretation
                # PointCloud2 RGB is typically stored as little-endian uint32
                # Memory layout: bytes [B, G, R, A] interpreted as 0xAABBGGRR (little-endian)
                rgb_byte_slices = data_array[valid_indices, rgb_offset:rgb_offset+4]
                
                # Convert to contiguous byte buffer for frombuffer
                # Each point's RGB is 4 consecutive bytes
                # We need to ensure bytes are contiguous, so use tobytes() or reshape
                rgb_bytes_contiguous = rgb_byte_slices.tobytes()  # This ensures contiguous byte order
                
                # Convert to uint32 array using little-endian interpretation
                # frombuffer reads every 4 bytes as a little-endian uint32
                rgb_uint32 = np.frombuffer(rgb_bytes_contiguous, dtype='<u4')  # '<u4' = little-endian uint32
                
                rgb_extraction_info["raw_values"] = rgb_uint32[:10] if len(rgb_uint32) > 10 else rgb_uint32
                
                colors = unpack_rgb_from_uint32(rgb_uint32)
            else:
                raise ValueError(
                    f"RGB field has unexpected datatype {rgb_dtype} (expected 6 for UINT32). "
                    f"Fields: {[f.name for f in msg.fields]}, "
                    f"RGB field info: offset={rgb_offset}, datatype={rgb_dtype}, count={rgb_count}"
                )
        else:
            # Check for separate r, g, b fields
            if 'r' in field_dict and 'g' in field_dict and 'b' in field_dict:
                r_offset, r_dtype, r_count = field_dict['r']
                g_offset, g_dtype, g_count = field_dict['g']
                b_offset, b_dtype, b_count = field_dict['b']
                
                rgb_extraction_info["method"] = "separate_rgb_fields"
                rgb_extraction_info["offset"] = f"r={r_offset}, g={g_offset}, b={b_offset}"
                rgb_extraction_info["dtype"] = f"r={r_dtype}, g={g_dtype}, b={b_dtype}"
                
                r_np_dtype, r_size = datatype_map.get(r_dtype, (np.uint8, 1))
                g_np_dtype, g_size = datatype_map.get(g_dtype, (np.uint8, 1))
                b_np_dtype, b_size = datatype_map.get(b_dtype, (np.uint8, 1))
                
                valid_indices = np.where(valid_mask)[0]
                
                # Extract r, g, b for all valid points at once
                r_vals = np.frombuffer(data_array[valid_indices, r_offset:r_offset+r_size].tobytes(), 
                                      dtype=r_np_dtype)
                g_vals = np.frombuffer(data_array[valid_indices, g_offset:g_offset+g_size].tobytes(), 
                                      dtype=g_np_dtype)
                b_vals = np.frombuffer(data_array[valid_indices, b_offset:b_offset+b_size].tobytes(), 
                                      dtype=b_np_dtype)
                
                rgb_extraction_info["raw_values"] = {
                    "r": r_vals[:10] if len(r_vals) > 10 else r_vals,
                    "g": g_vals[:10] if len(g_vals) > 10 else g_vals,
                    "b": b_vals[:10] if len(b_vals) > 10 else b_vals
                }
                
                colors[:, 0] = r_vals.astype(np.float32) / 255.0
                colors[:, 1] = g_vals.astype(np.float32) / 255.0
                colors[:, 2] = b_vals.astype(np.float32) / 255.0
            else:
                raise ValueError(
                    f"No RGB color field found in point cloud. "
                    f"Available fields: {[f.name for f in msg.fields]}. "
                    f"Expected either 'rgb' (UINT32) or separate 'r', 'g', 'b' fields."
                )
        
        # Validate that colors are not all the same (indicates RGB extraction error)
        if len(colors) > 0:
            # Check if all colors are identical
            if len(colors) > 1:
                first_color = colors[0]
                all_same = np.allclose(colors, first_color, atol=1e-6)
                
                if all_same:
                    raw_vals_str = str(rgb_extraction_info["raw_values"]) if rgb_extraction_info["raw_values"] is not None else "N/A"
                    raise ValueError(
                        f"ERROR: All colors in point cloud are identical: {first_color}. "
                        f"This indicates an RGB extraction error.\n"
                        f"  Number of points: {len(colors)}\n"
                        f"  RGB extraction method: {rgb_extraction_info['method']}\n"
                        f"  RGB offset: {rgb_extraction_info['offset']}\n"
                        f"  RGB datatype: {rgb_extraction_info['dtype']}\n"
                        f"  Point step: {msg.point_step}\n"
                        f"  Fields: {[f.name for f in msg.fields]}\n"
                        f"  First 10 raw RGB values: {raw_vals_str}\n"
                        f"This is likely caused by incorrect byte offset or endianness in RGB extraction."
                    )
            
            # # Also check for suspiciously low unique color count
            # unique_colors = len(np.unique(colors.reshape(-1, 3), axis=0))
            # if unique_colors == 1 and len(colors) > 1:
            #     raise ValueError(
            #         f"ERROR: All {len(colors)} points have the exact same color: {colors[0]}. "
            #         f"This indicates RGB extraction is reading the same value for all points."
            #     )
            # elif unique_colors < 10 and len(colors) > 100:
            #     # If we have many points but very few unique colors, warn
            #     print(f"WARNING: Only {unique_colors} unique colors found in {len(colors)} points")
            #     print(f"  First 5 colors: {colors[:5]}")
            #     print(f"  RGB field info: {field_dict.get('rgb', 'NOT FOUND')}")
        
        # Ensure colors are valid
        colors = np.clip(colors, 0.0, 1.0)
        
        return points, colors
    
    except Exception as e:
        print(f"ERROR in parse_pointcloud2_with_rgb: {e}")
        print(f"  Fields: {[f.name for f in msg.fields]}")
        print(f"  Point step: {msg.point_step}")
        print(f"  Data size: {len(msg.data)}")
        import traceback
        traceback.print_exc()
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


def extract_pointcloud_from_octree(octree, voxel_dict=None, resolution=0.5):
    """
    Extract pointcloud with RGB colors from pyoctomap ColorOcTree.
    
    Uses dictionary keys to know which coordinates to extract, then gets colors
    directly from ColorOcTreeNode objects using node.getColor().
    
    Based on create_global_sem_map_octree.py approach.
    
    Args:
        octree: pyoctomap.ColorOcTree instance
        voxel_dict: Dictionary mapping voxel keys to metadata (used to know which coords to extract)
                   If None, will try to use bounding box search (less efficient)
        resolution: Octree resolution for voxel key parsing
    
    Returns:
        points: numpy array of shape (N, 3) containing xyz coordinates
        colors: numpy array of shape (N, 3) containing RGB colors [0, 1]
    """
    points = []
    colors = []
    
    try:
        # First, ensure inner occupancy is updated (only if we have nodes)
        num_nodes = octree.getNumLeafNodes()
        if num_nodes > 0:
            octree.updateInnerOccupancy()
        else:
            print("WARNING: Octree has 0 leaf nodes, skipping updateInnerOccupancy()")
            return np.array([]), np.array([])
        
        # Use dictionary keys to know which coordinates to extract (same as reference file)
        if voxel_dict is not None and len(voxel_dict) > 0:
            for voxel_key in voxel_dict.keys():
                # Parse voxel key back to coordinates
                parts = voxel_key.split('_')
                if len(parts) == 3:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        z = float(parts[2])
                        coord = [x, y, z]
                        
                        # Get the node and then get color from the node object
                        node = octree.search(coord)
                        if node is None:
                            continue  # Skip if node doesn't exist
                        
                        if not octree.isNodeOccupied(node):
                            continue  # Skip if node is not occupied
                        
                        # Get color from ColorOcTreeNode using getColor() method
                        try:
                            r, g, b = node.getColor()  # Call on node, not on tree
                            rgb = np.array([r / 255.0, g / 255.0, b / 255.0])
                        except Exception as e:
                            raise RuntimeError(
                                f"Failed to get color from octree node at coordinate {coord}. "
                                f"This should never happen if colors were properly inserted. Error: {e}"
                            )
                        
                        points.append(np.array([x, y, z]))
                        colors.append(rgb)
                    except ValueError:
                        # Invalid key format, skip
                        continue
        else:
            # No dictionary available - this shouldn't happen in normal operation
            # but provide fallback using bounding box search
            print("WARNING: No voxel_dict available, using bounding box search (inefficient)")
            try:
                bbox_min = octree.getBBXMin()
                bbox_max = octree.getBBXMax()
                resolution = octree.getResolution()
                
                # Search at octree resolution
                x_coords = np.arange(bbox_min[0], bbox_max[0] + resolution, resolution)
                y_coords = np.arange(bbox_min[1], bbox_max[1] + resolution, resolution)
                z_coords = np.arange(bbox_min[2], bbox_max[2] + resolution, resolution)
                
                # Limit to reasonable size
                max_points_per_dim = 100
                if len(x_coords) > max_points_per_dim:
                    x_coords = np.linspace(bbox_min[0], bbox_max[0], max_points_per_dim)
                if len(y_coords) > max_points_per_dim:
                    y_coords = np.linspace(bbox_min[1], bbox_max[1], max_points_per_dim)
                if len(z_coords) > max_points_per_dim:
                    z_coords = np.linspace(bbox_min[2], bbox_max[2], max_points_per_dim)
                
                for x in x_coords:
                    for y in y_coords:
                        for z in z_coords:
                            coord = [float(x), float(y), float(z)]
                            node = octree.search(coord)
                            if node is not None and octree.isNodeOccupied(node):
                                points.append(np.array(coord))
                                try:
                                    # Get color from ColorOcTreeNode using getColor() method
                                    r, g, b = node.getColor()  # Call on node, not on tree
                                    rgb = np.array([r / 255.0, g / 255.0, b / 255.0])
                                except Exception as e:
                                    raise RuntimeError(
                                        f"Failed to get color from octree node at coordinate {coord} "
                                        f"during bounding box search. This should never happen if colors "
                                        f"were properly inserted. Error: {e}"
                                    )
                                colors.append(rgb)
            except Exception as bbox_error:
                print(f"WARNING: Could not extract using bounding box search: {bbox_error}")
                return np.array([]), np.array([])
                
    except Exception as e:
        print(f"WARNING: Error extracting pointcloud from octree: {e}")
        import traceback
        traceback.print_exc()
        return np.array([]), np.array([])
    
    if len(points) == 0:
        return np.array([]), np.array([])
    
    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)
    
    return points, colors


def insert_points_into_octree(octree, points, colors, voxel_dict=None, resolution=0.5):
    """
    Insert points with RGB colors into pyoctomap ColorOcTree.
    Colors are stored directly in the octree nodes.
    Also maintains a voxel_dict to track which voxels exist (for efficient extraction).
    
    Based on create_global_sem_map_octree.py implementation.
    
    Args:
        octree: pyoctomap.ColorOcTree instance
        points: numpy array of shape (N, 3) containing xyz coordinates
        colors: numpy array of shape (N, 3) containing RGB colors [0, 1]
        voxel_dict: Optional dictionary to track voxel keys (for extraction)
        resolution: Octree resolution for voxel key generation
    
    Returns:
        Number of points inserted
    """
    if len(points) == 0:
        return 0
    
    inserted_count = 0
    error_count = 0
    
    # Vectorized operation: convert colors from [0, 1] to [0, 255] integers
    colors_uint8 = (colors * 255.0).astype(np.uint8)
    
    for i in range(len(points)):
        point = points[i]
        color = colors_uint8[i]
        
        try:
            # Extract coordinates as floats and ensure they're Python floats
            x = float(point[0])
            y = float(point[1])
            z = float(point[2])
            
            # Check for invalid coordinates
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                error_count += 1
                if error_count <= 3:
                    print(f"WARNING: Invalid coordinates: ({x}, {y}, {z})")
                continue
            
            # Check for extremely large coordinates that might cause segfaults
            MAX_COORD = 1e8
            MIN_COORD = -1e8
            if x < MIN_COORD or x > MAX_COORD or y < MIN_COORD or y > MAX_COORD or z < MIN_COORD or z > MAX_COORD:
                error_count += 1
                if error_count <= 3:
                    print(f"WARNING: Coordinate out of reasonable range: ({x:.2f}, {y:.2f}, {z:.2f})")
                continue
            
            # Validate color values
            r_val = int(color[0])
            g_val = int(color[1])
            b_val = int(color[2])
            if not (0 <= r_val <= 255 and 0 <= g_val <= 255 and 0 <= b_val <= 255):
                error_count += 1
                if error_count <= 3:
                    print(f"WARNING: Invalid color values: ({r_val}, {g_val}, {b_val})")
                continue
            
            # Round to voxel resolution for consistent key generation
            voxel_x = round(x / resolution) * resolution
            voxel_y = round(y / resolution) * resolution
            voxel_z = round(z / resolution) * resolution
            voxel_key = f"{voxel_x:.6f}_{voxel_y:.6f}_{voxel_z:.6f}"
            
            # Create coordinate list explicitly as Python list
            coord_list = [float(x), float(y), float(z)]
            
            # Insert point into octree using updateNode
            # ColorOcTree updateNode takes coordinates as [x, y, z] and occupancy boolean
            octree.updateNode(coord_list, True)
            
            # Set color directly in the octree node
            # setNodeColor takes coordinates and RGB values [0-255]
            octree.setNodeColor(coord_list, r_val, g_val, b_val)
            
            # Track voxel in dictionary for efficient extraction
            if voxel_dict is not None:
                voxel_dict[voxel_key] = True  # Just track existence
            
            inserted_count += 1
                
        except (ValueError, TypeError, RuntimeError) as e:
            error_count += 1
            if error_count <= 5:
                print(f"WARNING: Error inserting point {i}: {e}")
            continue
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"WARNING: Unexpected error inserting point {i}: {e}")
            continue
    
    if error_count > 5:
        print(f"WARNING: {error_count} points failed to insert (showing first 5 errors)")
    
    return inserted_count


def set_node_color_from_children_mode(octree, coord):
    """
    Set a voxel's color using the mode (most frequent) color of its children.
    
    This function collects colors from all occupied child nodes, finds the most
    frequent color, and sets it on the parent node. This is useful for updating
    parent node colors when children have been modified or when collapsing nodes.
    
    Based on pyoctomap ColorOcTree API:
    - getNodeChild(node, idx) gets child at index 0-7
    - ColorOcTreeNode.getColor() returns (r, g, b) tuple [0-255]
    - setNodeColor(coord, r, g, b) sets color on node at coordinate
    
    Args:
        octree: pyoctomap.ColorOcTree instance
        coord: Coordinate [x, y, z] of the parent node
    
    Returns:
        tuple (r, g, b) of the mode color set [0-255], or None if:
        - Node not found
        - Node has no children
        - No occupied children found
    """
    from collections import Counter
    
    # Get the node from coordinate
    node = octree.search(coord)
    if node is None:
        return None
    
    # Check if node has children
    if not octree.nodeHasChildren(node):
        return None
    
    # Collect colors from all occupied children
    child_colors = []
    
    # Octree has 8 children (indices 0-7)
    for child_idx in range(8):
        try:
            child_node = octree.getNodeChild(node, child_idx)
            
            # Check if child exists and is occupied
            if child_node is not None and octree.isNodeOccupied(child_node):
                # Get color from child node
                # ColorOcTreeNode.getColor() returns (r, g, b) tuple [0-255]
                r, g, b = child_node.getColor()
                child_colors.append((r, g, b))
        except Exception:
            # Child doesn't exist or error accessing it, skip
            continue
    
    if len(child_colors) == 0:
        return None
    
    # Find mode (most frequent) color
    # Counter automatically counts occurrences of each color tuple
    color_counts = Counter(child_colors)
    mode_color = color_counts.most_common(1)[0][0]
    
    # Set the mode color on the parent node
    r, g, b = mode_color
    octree.setNodeColor(coord, r, g, b)
    
    return mode_color


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


def process_robot_incremental(dataset_path, environment, robot, output_writer, batch_size=10, max_scans=None, 
                              octree_resolution=0.1, coordinate_offset=None, per_scan_voxel_size=1.0, max_distance=60.0):
    """
    Process a single robot and incrementally accumulate/publish its scans using pyoctomap.
    Copies ALL lidar scans and odometry messages to output bag.
    Each scan is inserted into a ColorOcTree octree for efficient spatial representation with built-in color support.
    Publishes global map every batch_size scans.
    
    Based on create_global_sem_map_octree.py implementation.
    
    Args:
        dataset_path: Path to dataset root
        environment: Environment name
        robot: Robot name
        output_writer: ROS2 bag writer for output bag
        batch_size: Number of scans to accumulate before publishing global map (default: 10)
        max_scans: Maximum number of scans to process (default: None = all)
        octree_resolution: Resolution of the octree in meters (default: 0.1)
        coordinate_offset: Optional offset to subtract from poses (for coordinate shifting)
        per_scan_voxel_size: Voxel size for per-scan downsampling before octree insertion (default: 1.0)
    
    Returns:
        None
    """
    print(f"\n{'='*80}")
    print(f"Processing {robot} in {environment}")
    print(f"Octree resolution: {octree_resolution}m, Publish every: {batch_size} scans")
    if max_scans:
        print(f"Max scans: {max_scans} (testing mode)")
    if coordinate_offset is not None:
        print(f"Coordinate offset: [{coordinate_offset[0]:.2f}, {coordinate_offset[1]:.2f}, {coordinate_offset[2]:.2f}]")
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
    
    # Shift poses if coordinate_offset is provided
    # This shifts coordinates near (0,0,0) for better octree behavior
    if coordinate_offset is not None:
        print(f"Shifting poses to be relative to coordinate offset...")
        shifted_poses = {}
        for timestamp, pose in poses.items():
            position = np.array(pose[:3])
            quat = pose[3:]
            shifted_position = position - coordinate_offset
            shifted_poses[timestamp] = np.concatenate([shifted_position, quat])
        poses = shifted_poses
        print(f"  Shifted {len(poses)} poses")
    
    # Initialize ColorOcTree for spatial indexing with built-in color support
    octree = pyo.ColorOcTree(octree_resolution)
    print(f"Initialized ColorOcTree with resolution: {octree_resolution}m")
    
    # Dictionary to track voxels for efficient extraction (same as reference file)
    voxel_dict = {}
    
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
                
                # Remove all points beyond max_distance
                mask = np.linalg.norm(points, axis=1) <= max_distance
                points = points[mask]
                colors = colors[mask]

                # Skip if no points
                if len(points) == 0:
                    pbar.update(1)
                    continue
                
                # Transform to world frame
                # Since poses are already shifted relative to coordinate_offset,
                # the transformed points will naturally be in shifted coordinate frame
                position = pose[:3]
                quaternion = pose[3:]
                transformed_points = transform_points(points, position, quaternion)
                
                # Validate transformed points
                if not np.all(np.isfinite(transformed_points)):
                    print(f"WARNING: Invalid transformed points, skipping scan")
                    pbar.update(1)
                    continue
                
                # Per-scan voxel downsampling to reduce points before octree insertion
                # This reduces 100k+ points to ~1k-10k unique voxels per scan
                if per_scan_voxel_size > 0:
                    transformed_points, colors = voxel_downsample_points_colors(
                        transformed_points, colors, voxel_size=per_scan_voxel_size
                    )
                
                # Insert points into octree (octree will further downsample to octree_resolution)
                inserted = insert_points_into_octree(
                    octree, transformed_points, colors, 
                    voxel_dict=voxel_dict, resolution=octree_resolution
                )
                
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
                    if processed_count == 0:
                        print(f"  ⚠ WARNING: Octree has 0 leaf nodes after first scan!")
                        print(f"    Large coordinates may be causing insertions to fail silently.")
                        print(f"    Consider using coordinate shifting to start at (0,0,0)")
                
                # Print the number of points inserted into the octree
                if processed_count % 10 == 0:  # Print every 10 scans to avoid spam
                    pass  # Can add logging here if needed
                
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
                # Extract pointcloud from octree with colors (using dictionary for efficient extraction)
                all_points, all_colors = extract_pointcloud_from_octree(
                    octree, voxel_dict=voxel_dict, resolution=octree_resolution
                )
                
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
                        output_writer.write(f'/{robot}/map', serialize_message(pc_msg), adjusted_timestamp_ns)
                        publish_count += 1
    
    # Final update of inner occupancy for this robot
    # CRITICAL: Only call if we have leaf nodes (prevents segfault)
    final_nodes = octree.getNumLeafNodes()
    if final_nodes > 0:
        octree.updateInnerOccupancy()
        final_nodes = octree.getNumLeafNodes()  # Re-check after update
    else:
        print(f"\n  ⚠ WARNING: Octree has 0 leaf nodes - skipping updateInnerOccupancy() to prevent segfault")
    
    # Final publish from octree
    all_points, all_colors = extract_pointcloud_from_octree(
        octree, voxel_dict=voxel_dict, resolution=octree_resolution
    )
    
    if len(all_points) > 0 and output_writer is not None:
        timestamp_sec = timestamp / 1e9 if processed_count > 0 else 0.0
        pc_msg = create_pointcloud2_message(
            all_points, 
            all_colors,
            frame_id="world", 
            timestamp_sec=timestamp_sec
        )
        output_writer.write(f'/{robot}/map', serialize_message(pc_msg), timestamp)
        publish_count += 1
    
    print(f"\n{'='*60}")
    print(f"Completed {robot}:")
    print(f"  Scans processed: {processed_count}")
    print(f"  Map published: {publish_count} times")
    print(f"  Final octree size: {final_nodes} nodes")
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
        default="main_campus",
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
        help="Output file path for saving point cloud (default: auto-generated)"
    )

    parser.add_argument(
        "--octree_resolution",
        type=float,
        default=0.5, # 0.5m resolution is good for global maps
        help="Resolution of the octree in meters (default: 0.5)"
    )
    
    parser.add_argument(
        "--per_scan_voxel",
        type=float,
        default=1.0,
        help="Per-scan voxel size in meters for downsampling before octree insertion (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    
    # Determine which robots to process
    if args.robots:
        robots = args.robots
    else:
        env_path = dataset_path / args.environment
        if not env_path.exists():
            print(f"ERROR: Environment path not found: {env_path}")
            return
        robots = [d.name for d in env_path.iterdir() 
                  if d.is_dir() and d.name.startswith('robot')]
        robots = sorted(robots)
    
    # Calculate coordinate offset once from first robot's first pose
    # This shifts all points to start near (0,0,0) for better octree behavior
    coordinate_offset = None
    if len(robots) > 0:
        first_robot = robots[0]
        poses_bag_path = dataset_path / args.environment / first_robot / f"{first_robot}_{args.environment}_gt_rel_poses"
        if poses_bag_path.exists():
            print(f"\nCalculating coordinate offset from {first_robot}'s first pose...")
            first_robot_poses = load_poses_from_bag(poses_bag_path, robot_name=first_robot)
            if len(first_robot_poses) > 0:
                # Get first pose (sorted by timestamp)
                timestamps = sorted(first_robot_poses.keys())
                if len(timestamps) > 0:
                    first_pose = first_robot_poses[timestamps[0]]
                    coordinate_offset = np.array(first_pose[:3])
                    print(f"  Coordinate offset: [{coordinate_offset[0]:.2f}, {coordinate_offset[1]:.2f}, {coordinate_offset[2]:.2f}]")
                    print(f"  All points will be shifted by this offset to start near (0,0,0)")
    
    print(f"\n{'='*80}")
    print("Incremental Global Octree Map Creation (pyoctomap)")
    print(f"{'='*80}")
    print(f"Environment: {args.environment}")
    print(f"Robots: {', '.join(robots)}")
    print(f"Octree resolution: {args.octree_resolution}m")
    print(f"Per-scan voxel size: {args.per_scan_voxel}m")
    print(f"Batch size: {args.batch_size} scans (~{args.batch_size/10:.1f} Hz publish rate)")
    print(f"{'='*80}\n")
    
    try:
        for robot in robots:
            # Create output bag for THIS ROBOT
            robot_dir = os.path.join(dataset_path, args.environment, robot)
            
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
            
            # Create map topic
            topic_info = rosbag2_py.TopicMetadata(
                name=f'/{robot}/map',
                type='sensor_msgs/msg/PointCloud2',
                serialization_format='cdr'
            )
            writer.create_topic(topic_info)
            
            # Process this robot
            process_robot_incremental(
                dataset_path, args.environment, robot, writer, 
                batch_size=args.batch_size, 
                max_scans=args.max_scans, 
                octree_resolution=args.octree_resolution,
                coordinate_offset=coordinate_offset,
                per_scan_voxel_size=args.per_scan_voxel
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

