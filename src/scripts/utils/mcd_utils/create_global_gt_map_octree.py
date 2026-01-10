#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from tqdm import tqdm
from collections import namedtuple
import pyoctomap as pyo

# Import dataset_binarize package to set up sys.path for lidar2osm imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lidar2osm.utils.file_io import read_bin_file

# Body to LiDAR transformation matrix
BODY_TO_LIDAR_TF = np.array([
    [0.9999135040741837, -0.011166365511073898, -0.006949579221822984, -0.04894521120494695],
    [-0.011356389542502144, -0.9995453006865824, -0.02793249526856565, -0.03126929060348084],
    [-0.006634514801117132, 0.02800900135032654, -0.999585653686922, -0.01755515794222565],
    [0.0, 0.0, 0.0, 1.0]
])

def read_bin_file(file_path, dtype, shape=None):
    """
    Reads a .bin file and reshapes the data according to the provided shape.

    Args:
        file_path (str): The path to the .bin file.
        dtype (data-type): The data type of the file content (e.g., np.float32, np.int16).
        shape (tuple, optional): The desired shape of the output array. If None, the data is returned as a 1D array.

    Returns:
        np.ndarray: The data read from the .bin file, reshaped according to the provided shape.
    """
    data = np.fromfile(file_path, dtype=dtype)
    if shape:
        return data.reshape(shape)
    return data


Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        "id",  # An integer ID that is associated with this label.
        "color",  # The color of this label
    ],
)

sem_kitti_labels = [
    # name, id, color
    Label("unlabeled", 0, (0, 0, 0)),
    Label("outlier", 1, (0, 0, 0)),
    Label("car", 10, (0, 0, 142)),
    Label("bicycle", 11, (119, 11, 32)),
    Label("bus", 13, (250, 80, 100)),
    Label("motorcycle", 15, (0, 0, 230)),
    Label("on-rails", 16, (255, 0, 0)),
    Label("truck", 18, (0, 0, 70)),
    Label("other-vehicle", 20, (51, 0, 51)),
    Label("person", 30, (220, 20, 60)),
    Label("bicyclist", 31, (200, 40, 255)),
    Label("motorcyclist", 32, (90, 30, 150)),
    Label("road", 40, (128, 64, 128)),
    Label("parking", 44, (250, 170, 160)),
    Label("OSM BUILDING", 45, (0, 0, 255)),   # OSM
    Label("OSM ROAD", 46, (255, 0, 0)),       # OSM
    Label("sidewalk", 48, (244, 35, 232)),
    Label("other-ground", 49, (81, 0, 81)),
    Label("building", 50, (0, 100, 0)),
    Label("fence", 51, (190, 153, 153)),
    Label("other-structure", 52, (0, 150, 255)),
    Label("lane-marking", 60, (170, 255, 150)),
    Label("vegetation", 70, (107, 142, 35)),
    Label("trunk", 71, (0, 60, 135)),
    Label("terrain", 72, (152, 251, 152)),
    Label("pole", 80, (153, 153, 153)),
    Label("traffic-sign", 81, (0, 0, 255)),
    Label("other-object", 99, (255, 255, 50)),
]

def labels_to_colors(labels):
    """
    Convert semantic label IDs to RGB colors using sem_kitti_labels.
    
    Args:
        labels: (N,) array of semantic label IDs
    
    Returns:
        colors: (N, 3) array of RGB colors in [0, 1] range
    """
    # Create a mapping from label ID to color (RGB tuple in 0-255 range)
    label_id_to_color = {label.id: label.color for label in sem_kitti_labels}
    
    # Map labels to colors
    colors = np.zeros((len(labels), 3), dtype=np.float32)
    for i, label_id in enumerate(labels):
        label_id_int = int(label_id)
        if label_id_int in label_id_to_color:
            # Convert from (0-255) range to (0-1) range
            color_255 = label_id_to_color[label_id_int]
            colors[i] = np.array(color_255, dtype=np.float32) / 255.0
        else:
            # Unknown label, use gray
            colors[i] = [0.5, 0.5, 0.5]
    
    return colors

def insert_points_into_octree(octree, points, colors, color_dict=None, resolution=0.5):
    """
    Insert points with RGB colors into pyoctomap ColorOcTree.
    
    Args:
        octree: pyoctomap.ColorOcTree instance (PRIMARY storage)
        points: numpy array of shape (N, 3) containing xyz coordinates
        colors: numpy array of shape (N, 3) containing RGB colors [0, 1] (stored in octree)
        color_dict: Dictionary for color metadata {voxel_key: {'counts': dict, 'mode': tuple}}
        resolution: Octree resolution in meters
    
    Returns:
        Number of points successfully inserted into octree
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
            # Extract coordinates as floats and ensure they're Python floats, not numpy types
            x = float(point[0])
            y = float(point[1])
            z = float(point[2])
            
            # Round to voxel resolution for consistent key generation
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
            
            # Check for extremely large coordinates
            MAX_COORD = 1e8
            MIN_COORD = -1e8
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
            try:
                coord_list = [float(x), float(y), float(z)]
                
                # Call updateNode - this creates/updates the node
                octree.updateNode(coord_list, True)
                
                # Set color directly in the octree node
                octree.setNodeColor(coord_list, r_val, g_val, b_val)
                
                inserted_count += 1
                
                # Store color in dictionary for extraction
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
                
            except (ValueError, TypeError, RuntimeError) as update_error:
                error_count += 1
                if error_count <= 10:
                    print(f"WARNING: updateNode failed for point ({x:.2f}, {y:.2f}, {z:.2f}): {update_error}")
                continue
            except Exception as unexpected_error:
                error_count += 1
                if error_count <= 10:
                    print(f"WARNING: Unexpected error inserting point ({x:.2f}, {y:.2f}, {z:.2f}): {unexpected_error}")
                continue
                
        except Exception as e:
            error_count += 1
            if error_count <= 10:
                print(f"WARNING: Error inserting point {i}/{len(points)}: {e}")
            continue
    
    if error_count > 0:
        print(f"WARNING: {error_count}/{len(points)} points failed to insert")
    
    return inserted_count


def extract_pointcloud_from_octree(octree, color_dict=None, resolution=0.5, coordinate_offset=None):
    """
    Extract pointcloud with RGB colors from pyoctomap ColorOcTree.
    
    Args:
        octree: pyoctomap.ColorOcTree instance
        color_dict: Dictionary mapping voxel keys to color info
        resolution: Octree resolution for voxel key generation
        coordinate_offset: Optional offset to add back to coordinates
    
    Returns:
        points: numpy array of shape (N, 3) containing xyz coordinates
        colors: numpy array of shape (N, 3) containing RGB colors [0, 1]
    """
    points = []
    colors = []
    
    try:
        # Update inner occupancy only if we have nodes
        num_nodes = octree.getNumLeafNodes()
        if num_nodes > 0:
            octree.updateInnerOccupancy()
        else:
            print("WARNING: Octree has 0 leaf nodes, skipping updateInnerOccupancy()")
        
        # Extract points using color_dict keys
        if color_dict is not None and len(color_dict) > 0:
            print(f"Extracting points using dictionary keys ({len(color_dict)} entries)...")
            for voxel_key in color_dict.keys():
                # Parse voxel key back to coordinates
                parts = voxel_key.split('_')
                if len(parts) == 3:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        z = float(parts[2])
                        
                        # Apply coordinate offset if provided
                        coord = np.array([x, y, z])
                        if coordinate_offset is not None:
                            coord = coord + coordinate_offset
                        points.append(coord)
                        
                        # Get color from dictionary
                        if voxel_key in color_dict:
                            r, g, b = color_dict[voxel_key]['mode']
                            rgb = np.array([r / 255.0, g / 255.0, b / 255.0])
                            colors.append(rgb)
                        else:
                            # Default gray if not in dict
                            colors.append(np.array([0.5, 0.5, 0.5]))
                    except ValueError:
                        continue
        else:
            print("WARNING: No color_dict available for extraction")
    
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

def load_poses(poses_file):
    """
    Load poses from CSV file.
    
    Args:
        poses_file: Path to CSV file with poses (num, t, x, y, z, qx, qy, qz, qw)
    
    Returns:
        poses: Dictionary mapping timestamp to [index, x, y, z, qx, qy, qz, qw]
        index_to_timestamp: Dictionary mapping index to timestamp
    """
    print(f"\nLoading poses from {poses_file}")
    
    # First, inspect the CSV file to understand its format
    try:
        # Read first few lines to inspect format
        with open(poses_file, 'r') as f:
            first_lines = [f.readline().strip() for _ in range(5)]
        
        print(f"\nCSV file inspection:")
        print(f"  First 3 lines of file:")
        for i, line in enumerate(first_lines[:3]):
            print(f"    Line {i+1}: {line[:100]}")  # Print first 100 chars
        
        # Try different reading strategies
        df = None
        
        # Strategy 1: Try with header, handling comment lines
        try:
            df = pd.read_csv(poses_file, comment='#', skipinitialspace=True)
            print(f"\n  Attempted to read with header (comment='#')")
            print(f"  Columns found: {list(df.columns)}")
            print(f"  Number of columns: {len(df.columns)}")
            
            # Check if we have the expected columns (handle various naming conventions)
            col_names = [str(col).strip().lower().replace('#', '').replace(' ', '') for col in df.columns]
            has_timestamp_col = any('timestamp' in col or col == 't' for col in col_names)
            has_pose_cols = all(any(coord in col for col in col_names) for coord in ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
            
            # Check for num column (optional)
            has_num_col = any('num' in col for col in col_names)
            
            if has_timestamp_col and has_pose_cols and (len(df.columns) >= 8 or (has_num_col and len(df.columns) >= 9)):
                print(f"  ✓ Format detected: Has header with column names")
                if has_num_col:
                    print(f"    Note: Found 'num' column, will be ignored")
            elif len(df.columns) == 8 or len(df.columns) == 9:
                print(f"  ⚠ Format: {len(df.columns)} columns but unclear header format, trying positional")
                df = None  # Will try positional
            else:
                print(f"  ⚠ Format: Unexpected column count ({len(df.columns)}), trying positional")
                df = None
        except Exception as e:
            print(f"  Failed to read with header: {e}")
            df = None
        
        # Strategy 2: Try without header (positional)
        if df is None:
            try:
                df = pd.read_csv(poses_file, comment='#', header=None, skipinitialspace=True)
                print(f"\n  Attempted to read without header (positional)")
                print(f"  Number of columns: {len(df.columns)}")
                if len(df.columns) >= 8:
                    print(f"  ✓ Format detected: No header, using positional indexing")
                else:
                    print(f"  ✗ Error: Only {len(df.columns)} columns found, need at least 8")
                    return {}
            except Exception as e:
                print(f"  Failed to read without header: {e}")
                return {}
        
        print(f"\n  Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
        print(f"  First few data rows:")
        print(df.head(3))
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    poses = {}
    index_to_timestamp = {}
    
    for idx, row in df.iterrows():
        try:
            # Try to get timestamp and index - handle various column name formats
            timestamp = None
            pose_index = None
            x, y, z, qx, qy, qz, qw = None, None, None, None, None, None, None
            
            # Build column map for pose values
            col_map = {}
            for col in df.columns:
                col_clean = str(col).strip().lower().replace('#', '').replace(' ', '')
                col_map[col_clean] = col
            
            # Check for timestamp column (handle various naming: 't', 'timestamp', etc.)
            for col in df.columns:
                col_clean = str(col).strip().lower().replace('#', '').replace(' ', '')
                # Check for 't' (single letter) or 'timestamp'
                if col_clean == 't' or 'timestamp' in col_clean:
                    timestamp = row[col]
                    break
            
            # Check for index column ('num')
            if 'num' in col_map:
                pose_index = int(row[col_map['num']])
            
            # Get pose values by column name
            if timestamp is not None:
                # We have a timestamp column, get pose values by name
                x = row.get(col_map.get('x'), None) if 'x' in col_map else None
                y = row.get(col_map.get('y'), None) if 'y' in col_map else None
                z = row.get(col_map.get('z'), None) if 'z' in col_map else None
                qx = row.get(col_map.get('qx'), None) if 'qx' in col_map else None
                qy = row.get(col_map.get('qy'), None) if 'qy' in col_map else None
                qz = row.get(col_map.get('qz'), None) if 'qz' in col_map else None
                qw = row.get(col_map.get('qw'), None) if 'qw' in col_map else None
            else:
                # No timestamp column found, use positional indexing
                # Format: [num, t, x, y, z, qx, qy, qz, qw] or [t, x, y, z, qx, qy, qz, qw]
                if len(row) >= 9:
                    # Has num column: first column is index
                    pose_index = int(row.iloc[0])
                    timestamp = row.iloc[1]
                    x = row.iloc[2]
                    y = row.iloc[3]
                    z = row.iloc[4]
                    qx = row.iloc[5]
                    qy = row.iloc[6]
                    qz = row.iloc[7]
                    qw = row.iloc[8]
                elif len(row) >= 8:
                    # No num column
                    timestamp = row.iloc[0]
                    x = row.iloc[1]
                    y = row.iloc[2]
                    z = row.iloc[3]
                    qx = row.iloc[4]
                    qy = row.iloc[5]
                    qz = row.iloc[6]
                    qw = row.iloc[7]
                else:
                    if idx < 5:
                        print(f"  Row {idx} has only {len(row)} columns, need at least 8, skipping")
                    continue
            
            # Validate all values are present
            if None in [timestamp, x, y, z, qx, qy, qz, qw]:
                if idx < 5:
                    print(f"  Row {idx} missing values, skipping")
                continue
            
            # Store pose with index: [index, x, y, z, qx, qy, qz, qw]
            pose = [pose_index, float(x), float(y), float(z), float(qx), float(qy), float(qz), float(qw)]
            poses[float(timestamp)] = pose
            
            # Store index to timestamp mapping
            if pose_index is not None:
                index_to_timestamp[pose_index] = float(timestamp)
        except Exception as e:
            if idx < 5:  # Only print first few errors
                print(f"  Error processing row {idx}: {e}")
                print(f"  Row data: {row.tolist()[:9] if len(row) >= 9 else row.tolist()}")
            continue
    
    print(f"\nSuccessfully loaded {len(poses)} poses")
    if len(poses) > 0:
        sample_ts = list(poses.keys())[0]
        sample_pose = poses[sample_ts]
        print(f"  Sample pose (timestamp {sample_ts}, index {sample_pose[0]}): position=[{sample_pose[1]:.2f}, {sample_pose[2]:.2f}, {sample_pose[3]:.2f}], quat=[{sample_pose[4]:.3f}, {sample_pose[5]:.3f}, {sample_pose[6]:.3f}, {sample_pose[7]:.3f}]")
    return poses, index_to_timestamp


def find_closest_pose(timestamp, poses_dict, exact_match_threshold=0.001):
    """
    Find the closest pose timestamp to the given timestamp.
    
    Args:
        timestamp: Target timestamp
        poses_dict: Dictionary mapping timestamp to pose
        exact_match_threshold: Time difference threshold in seconds to consider an exact match (default: 0.001s = 1ms)
    
    Returns:
        Tuple of (closest timestamp key, time_difference), or (None, None) if poses_dict is empty
        Returns None for timestamp if no exact match found (time_diff > threshold)
    """
    if not poses_dict:
        return None, None
    
    pose_timestamps = np.array(list(poses_dict.keys()))
    time_diffs = np.abs(pose_timestamps - timestamp)
    closest_idx = np.argmin(time_diffs)
    closest_ts = pose_timestamps[closest_idx]
    time_diff = time_diffs[closest_idx]
    
    # Check if it's an exact match
    if time_diff > exact_match_threshold:
        # Warn but return None to indicate no exact match
        print(f"  WARNING: No exact pose match for timestamp {timestamp:.6f}. "
              f"Closest pose at {closest_ts:.6f} (difference: {time_diff:.6f}s). Skipping scan.")
        return None, time_diff
    
    return closest_ts, time_diff


def transform_points_to_world(points_xyz, position, quaternion, body_to_lidar_tf=None):
    """
    Transform points from lidar frame to world frame using pose.
    
    Args:
        points_xyz: (N, 3) array of points in lidar frame
        position: [x, y, z] translation (body frame position in world)
        quaternion: [qx, qy, qz, qw] rotation quaternion (body frame orientation in world)
        body_to_lidar_tf: Optional 4x4 transformation matrix from body to lidar frame
    
    Returns:
        world_points: (N, 3) array of points in world frame
    """
    # Create rotation matrix from quaternion (body frame orientation)
    body_rotation_matrix = R.from_quat(quaternion).as_matrix()
    
    # Create 4x4 transformation matrix for body frame in world
    body_to_world = np.eye(4)
    body_to_world[:3, :3] = body_rotation_matrix
    body_to_world[:3, 3] = position
    
    # If body_to_lidar transform is provided, compose the transformations
    # world_to_lidar = world_to_body * body_to_lidar
    # So: lidar_to_world = (body_to_lidar)^-1 * body_to_world
    if body_to_lidar_tf is not None:
        # Transform from body to lidar, then from body to world
        # T_lidar_to_world = T_body_to_world * T_lidar_to_body
        # T_lidar_to_body = inv(T_body_to_lidar)
        lidar_to_body = np.linalg.inv(body_to_lidar_tf)
        transform_matrix = body_to_world @ lidar_to_body
    else:
        transform_matrix = body_to_world
    
    # Transform points to world coordinates
    points_homogeneous = np.hstack(
        [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)]
    )
    world_points = (transform_matrix @ points_homogeneous.T).T
    world_points_xyz = world_points[:, :3]
    
    return world_points_xyz


def accumulate_sequence_scans(dataset_path, 
                              seq_name, 
                              poses_dict,
                              index_to_timestamp,
                              octree,
                              color_dict,
                              max_scans=None,
                              downsample_factor=1,
                              resolution=0.5,
                              max_distance=None):
    """
    Accumulate LiDAR scans from a single sequence into a global ColorOcTree.
    
    Args:
        dataset_path: Path to dataset directory
        seq_name: Name of the sequence
        poses_dict: Dictionary mapping timestamp to pose data
        index_to_timestamp: Dictionary mapping index to timestamp
        octree: pyoctomap.ColorOcTree instance (modified in place)
        color_dict: Dictionary to store colors per voxel (modified in place)
        max_scans: Maximum number of scans to process (None for all)
        downsample_factor: Process every Nth scan (1 = all scans)
        resolution: Octree resolution in meters
        max_distance: Maximum distance in meters from pose position to keep points (None to keep all)
    
    Returns:
        Number of scans processed
    """
    root_path = os.path.join(dataset_path, seq_name)
    data_dir = os.path.join(root_path, "lidar_bin/data")
    timestamps_file = os.path.join(root_path, "lidar_bin/timestamps.txt")
    labels_dir = os.path.join(root_path, "cumulti_inferred_labels")
    
    # Check if files exist
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        return 0
    
    if not os.path.exists(timestamps_file):
        print(f"ERROR: Timestamps file not found: {timestamps_file}")
        return 0
    
    if not os.path.exists(labels_dir):
        print(f"ERROR: Labels directory not found: {labels_dir}")
        return 0
    
    # Load timestamps
    timestamps = np.loadtxt(timestamps_file)
    
    # Get all bin files
    bin_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.bin')])
    
    if len(bin_files) != len(timestamps):
        print(f"WARNING: Number of bin files ({len(bin_files)}) != number of timestamps ({len(timestamps)})")
        min_count = min(len(bin_files), len(timestamps))
        bin_files = bin_files[:min_count]
        timestamps = timestamps[:min_count]
    
    # Process poses in order by their index (num value)
    sorted_pose_indices = sorted(index_to_timestamp.keys())
    
    # Apply downsampling to pose indices (process every Nth pose)
    if downsample_factor > 1:
        sorted_pose_indices = sorted_pose_indices[::downsample_factor]
    
    # Limit number of poses if specified
    if max_scans:
        sorted_pose_indices = sorted_pose_indices[:max_scans]
    
    # Track initial octree size
    initial_octree_size = octree.getNumLeafNodes()
    
    scan_count = 0
    points_inserted_total = 0
    
    # Process poses with progress bar
    for pose_num in tqdm(sorted_pose_indices, desc=f"Processing {seq_name}", unit="scan"):
        # Get the corresponding scan index (pose is 1-indexed, scan is 0-indexed)
        scan_idx = pose_num - 1
        
        # Check if scan index is valid
        if scan_idx < 0 or scan_idx >= len(bin_files):
            continue
        
        # Get pose data
        pose_timestamp = index_to_timestamp[pose_num]
        pose_data = poses_dict[pose_timestamp]
        position = pose_data[1:4]  # [x, y, z]
        quaternion = pose_data[4:8]  # [qx, qy, qz, qw]
        
        # Get corresponding scan
        bin_file = bin_files[scan_idx]
        scan_timestamp = timestamps[scan_idx]
        
        # Validate timestamps match (within threshold)
        time_diff = abs(scan_timestamp - pose_timestamp)
        if time_diff > 0.1:  # 100ms threshold
            tqdm.write(f"  WARNING: Timestamp mismatch at scan {scan_idx}: diff={time_diff:.6f}s, skipping")
            continue
        
        # Load point cloud
        bin_path = os.path.join(data_dir, bin_file)
        try:
            points = read_bin_file(bin_path, dtype=np.float32, shape=(-1, 4))
            points_xyz = points[:, :3]
        except Exception as e:
            tqdm.write(f"  Error loading {bin_file}: {e}")
            continue
        
        # Load corresponding label file
        label_file = bin_file
        label_path = os.path.join(labels_dir, label_file)
        
        if not os.path.exists(label_path):
            continue
        
        try:
            # Load labels (assuming int32 format)
            labels = read_bin_file(label_path, dtype=np.int32, shape=(-1))
            
            # Validate that labels match points
            if len(labels) != len(points_xyz):
                continue
            
            # Convert labels to colors
            colors = labels_to_colors(labels)
        except Exception as e:
            tqdm.write(f"  Error loading labels from {label_file}: {e}")
            continue
        
        # Transform to world coordinates (apply body-to-lidar transformation)
        world_points = transform_points_to_world(points_xyz, position, quaternion, BODY_TO_LIDAR_TF)
        
        # Filter points by distance from pose position if max_distance is specified
        if max_distance is not None and max_distance > 0:
            # Calculate distance from pose position for each point
            distances = np.linalg.norm(world_points - position, axis=1)
            # Keep points within max_distance
            mask = distances <= max_distance
            world_points = world_points[mask]
            colors = colors[mask]
            
            if len(world_points) == 0:
                continue
        
        # Insert points into octree
        inserted = insert_points_into_octree(
            octree, world_points, colors,
            color_dict=color_dict,
            resolution=resolution
        )
        points_inserted_total += inserted
        
        # Update inner occupancy only if we have nodes
        current_nodes = octree.getNumLeafNodes()
        if current_nodes > 0:
            octree.updateInnerOccupancy()
        else:
            if scan_count == 0:
                tqdm.write(f"    WARNING: Octree has 0 leaf nodes after first scan!")
        
        scan_count += 1
    
    # Final update of inner occupancy
    final_octree_size = octree.getNumLeafNodes()
    new_nodes = final_octree_size - initial_octree_size
    print(f"\nProcessed {scan_count} scans from {seq_name}")
    print(f"  Points inserted: {points_inserted_total}")
    print(f"  Octree nodes before: {initial_octree_size}, after: {final_octree_size} (+{new_nodes})")
    
    return scan_count


def accumulate_all_sequences(dataset_path, 
                             seq_names, 
                             max_scans=None,
                             downsample_factor=1,
                             resolution=0.5,
                             max_distance=None):
    """
    Accumulate LiDAR scans from all sequences into a global ColorOcTree.
    
    Args:
        dataset_path: Path to dataset directory
        seq_names: List of sequence names
        max_scans: Maximum number of scans to process per sequence (None for all)
        downsample_factor: Process every Nth scan (1 = all scans)
        resolution: Octree resolution in meters
        max_distance: Maximum distance in meters from pose position to keep points (None to keep all)
    
    Returns:
        octree: pyoctomap.ColorOcTree instance with all accumulated points
        color_dict: Dictionary mapping voxel keys to color info
        coordinate_offset: Coordinate offset used (or None)
    """
    # Initialize ColorOcTree with specified resolution
    octree = pyo.ColorOcTree(resolution)
    print(f"\nInitialized ColorOcTree with resolution: {resolution}m")
    
    # Initialize dictionary to store colors per voxel
    color_dict = {}
    
    # Calculate coordinate offset once from first sequence's first pose
    coordinate_offset = None
    if len(seq_names) > 0:
        first_seq = seq_names[0]
        poses_file = os.path.join(dataset_path, first_seq, "pose_inW.csv")
        if os.path.exists(poses_file):
            print(f"\nCalculating coordinate offset from {first_seq}'s first pose...")
            poses_dict, index_to_timestamp = load_poses(poses_file)
            if len(poses_dict) > 0 and len(index_to_timestamp) > 0:
                first_idx = min(index_to_timestamp.keys())
                first_timestamp = index_to_timestamp[first_idx]
                first_pose = poses_dict[first_timestamp]
                coordinate_offset = np.array(first_pose[1:4])  # [x, y, z]
                print(f"  Coordinate offset: [{coordinate_offset[0]:.2f}, {coordinate_offset[1]:.2f}, {coordinate_offset[2]:.2f}]")
    
    # Loop through all sequences
    for seq_name in seq_names:
        print(f"\n{'='*80}")
        print(f"Processing {seq_name}")
        print(f"{'='*80}")
        
        poses_file = os.path.join(dataset_path, seq_name, "pose_inW.csv")
        
        if not os.path.exists(poses_file):
            print(f"Warning: Poses file not found for {seq_name}: {poses_file}")
            print(f"Skipping {seq_name}...")
            continue
        
        print(f"Loading poses for {seq_name}...")
        poses_dict, index_to_timestamp = load_poses(poses_file)
        
        if len(poses_dict) == 0:
            print(f"No poses found for {seq_name}! Skipping...")
            continue
        
        if not index_to_timestamp:
            print(f"No index column found in poses CSV for {seq_name}. Skipping...")
            continue
        
        # Shift all poses to be relative to the global first pose (coordinate_offset)
        if coordinate_offset is not None:
            print(f"Shifting poses to be relative to global first pose...")
            shifted_poses_dict = {}
            for timestamp, pose_data in poses_dict.items():
                position = np.array(pose_data[1:4])
                shifted_position = position - coordinate_offset
                shifted_pose = [pose_data[0], shifted_position[0], shifted_position[1], shifted_position[2]] + pose_data[4:8]
                shifted_poses_dict[timestamp] = shifted_pose
            poses_dict = shifted_poses_dict
        
        print(f"\nAccumulating LiDAR scans for {seq_name} into octree...")
        scan_count = accumulate_sequence_scans(
            dataset_path,
            seq_name,
            poses_dict,
            index_to_timestamp,
            octree,
            color_dict,
            max_scans=max_scans,
            downsample_factor=downsample_factor,
            resolution=resolution,
            max_distance=max_distance,
        )
        
        if scan_count == 0:
            print(f"Warning: No scans processed for {seq_name}")
    
    # Final update of inner occupancy for all sequences
    final_nodes = octree.getNumLeafNodes()
    if final_nodes > 0:
        octree.updateInnerOccupancy()
        final_nodes = octree.getNumLeafNodes()
    else:
        print(f"\n  ⚠ WARNING: Octree has 0 leaf nodes - skipping updateInnerOccupancy() to prevent segfault")
    
    print(f"\n{'='*80}")
    print(f"Final octree statistics:")
    print(f"  Total leaf nodes: {final_nodes}")
    print(f"  Octree resolution: {resolution}m")
    print(f"  Color dictionary entries: {len(color_dict)}")
    print(f"{'='*80}")
    
    return octree, color_dict, coordinate_offset


if __name__ == '__main__':
    # Update these paths to match your setup
    dataset_path = "/media/donceykong/doncey_ssd_02/datasets/MCD"
    seq_names = ["kth_day_06", "kth_day_09", "kth_night_05"]

    # Optional: limit number of scans for faster processing
    max_scans = 10000  # Set to None to process all scans
    
    # Optional: downsample scans (process every Nth scan)
    downsample_factor = 5  # Set to 1 to process all scans, 2 for every other scan, etc.
    
    # Octree resolution (in meters)
    resolution = 0.5  # 0.5m voxels
    
    # Maximum distance from pose position to keep points (in meters)
    max_distance = 60.0  # e.g., 100.0 to keep points within 100m from pose position
    
    # Accumulate all sequences into octree
    print(f"\n{'='*80}")
    print(f"Accumulating {len(seq_names)} sequences into octree")
    print(f"{'='*80}")
    
    octree, color_dict, coordinate_offset = accumulate_all_sequences(
        dataset_path,
        seq_names,
        max_scans=max_scans,
        downsample_factor=downsample_factor,
        resolution=resolution,
        max_distance=max_distance,
    )
    
    # Check if we have any data
    if octree.getNumLeafNodes() == 0:
        print("\nError: No data accumulated from any sequence!")
        sys.exit(1)
    
    # Extract point cloud from octree
    print(f"\n{'='*80}")
    print("Extracting point cloud from octree...")
    if coordinate_offset is not None:
        print(f"Shifting coordinates back to original frame (offset: [{coordinate_offset[0]:.2f}, {coordinate_offset[1]:.2f}, {coordinate_offset[2]:.2f}])")
    print(f"{'='*80}")
    
    points, colors = extract_pointcloud_from_octree(
        octree,
        color_dict=color_dict,
        resolution=resolution,
        coordinate_offset=coordinate_offset
    )
    
    if len(points) == 0:
        print("\nError: No points extracted from octree!")
        sys.exit(1)
    
    # Print statistics
    print(f"\nFinal point cloud statistics:")
    print(f"  Total points: {len(points)}")
    print(f"  X range: [{points[:, 0].min():.6f}, {points[:, 0].max():.6f}]")
    print(f"  Y range: [{points[:, 1].min():.6f}, {points[:, 1].max():.6f}]")
    print(f"  Z range: [{points[:, 2].min():.6f}, {points[:, 2].max():.6f}]")
    
    # Create Open3D point cloud with semantic colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save to PLY file
    # Create output directory in dataset_path
    ply_dir = os.path.join(dataset_path, "ply")
    if not os.path.exists(ply_dir):
        os.makedirs(ply_dir)
        print(f"Created directory: {ply_dir}")
    
    # Create filename from sequence names
    seq_name_str = "_".join(seq_names)
    ply_filename = f"inferred_labels_{seq_name_str}.ply"
    ply_path = os.path.join(ply_dir, ply_filename)
    
    # Save point cloud
    print(f"\nSaving point cloud to: {ply_path}")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Successfully saved {len(points)} points to {ply_path}")
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"Generated file:")
    print(f"  - {ply_path}")
    print(f"\nData from {len(seq_names)} sequences successfully combined and saved!")

