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

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.serialization import deserialize_message
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

# Internal imports
from lidar2osm.datasets.cu_multi_dataset import labels as sem_kitti_labels


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


def voxel_downsample(points, colors, voxel_size):
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
    max_coords = np.max(points, axis=0)
    voxel_dims = np.ceil((max_coords - min_coords) / voxel_size).astype(int)
    
    # Assign points to voxels
    voxel_indices = np.floor((points - min_coords) / voxel_size).astype(int)
    voxel_indices = np.clip(voxel_indices, 0, voxel_dims - 1)
    
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


def process_robot_incremental(dataset_path, environment, robot, output_writer, batch_size=10, max_scans=None):
    """
    Process a single robot and incrementally accumulate/publish its scans.
    Copies ALL lidar scans and odometry messages to output bag.
    Each scan is downsampled with 4m voxel size before accumulation.
    Publishes global map every batch_size scans (no additional downsampling).
    
    Args:
        dataset_path: Path to dataset root
        environment: Environment name
        robot: Robot name
        publisher: ROS2 publisher for PointCloud2
        output_writer: ROS2 bag writer for output bag
        batch_size: Number of scans to accumulate before publishing global map (default: 10)
        max_scans: Maximum number of scans to process (default: None = all)
    
    Returns:
        final_points: Final accumulated points
        final_colors: Final accumulated colors
    """
    print(f"\n{'='*80}")
    print(f"Processing {robot} in {environment}")
    print(f"Per-scan voxel: 4.0m, Publish every: {batch_size} scans")
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
    
    # Open bag reader (DON'T load all messages!)
    storage_options = rosbag2_py.StorageOptions(uri=str(lidar_bag_path), storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    # Find PointCloud2 topic and create topics in output bag
    topic_types = reader.get_all_topics_and_types()
    print(f"\n\n\nTopic types: {topic_types}\n\n\n")
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

                # frame_id = topic.header.frame_id
                # print(f"\n\n\nPointCloud2 topic frame_id: {frame_id}\n\n\n")

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
    
    # Initialize global accumulated map
    # Accumulated points and colors (simple numpy arrays, no Open3D)
    accumulated_points = []
    accumulated_colors = []
    
    processed_count = 0
    publish_count = 0
    msg_type = get_message('sensor_msgs/msg/PointCloud2')
    
    # STREAM messages one at a time (DON'T store in list!)
    with tqdm(total=total_messages, desc=f"Processing {robot}") as pbar:
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            
            # Copy ALL messages (lidar scans, IMU, etc.) to output bag
            if output_writer is not None:
                output_writer.write(topic, data, timestamp)
            
            if topic != pointcloud_topic:
                continue
            
            # Deserialize ONE message at a time
            msg = deserialize_message(data, msg_type)
            
            # Find nearest pose
            pose = find_nearest_pose(timestamp, poses)
            if pose is None:
                pbar.update(1)
                continue
            
            try:
                # Parse ONE point cloud at a time
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
                
                # Voxel downsample THIS SINGLE SCAN (4m) - numpy based, no Open3D
                scan_points, scan_colors = voxel_downsample(transformed_points, colors, voxel_size=4.0)
                
            except Exception as e:
                print(f"\nERROR processing scan {processed_count}: {e}")
                import traceback
                traceback.print_exc()
                pbar.update(1)
                continue
            
            # Add to global map
            accumulated_points.append(scan_points)
            accumulated_colors.append(scan_colors)
            
            processed_count += 1
            pbar.update(1)
            
            # Check if we've reached max_scans limit
            if max_scans and processed_count >= max_scans:
                tqdm.write(f"\nReached max_scans limit ({max_scans}), stopping...")
                break
            
            # Every batch_size scans: publish global map (no additional downsampling)
            if processed_count % batch_size == 0:
                # Combine all accumulated points
                if accumulated_points:
                    all_points = np.vstack(accumulated_points)
                    all_colors = np.vstack(accumulated_colors)

                    # Convert timestamp to seconds
                    timestamp_sec = timestamp / 1e9

                    print(f"\n Time: {timestamp}\n")
                    print(f"\n Timestamp in seconds: {timestamp_sec}\n")
                    print(f"\n Processed count: {processed_count}\n")
                    print(f"\n Batch size: {batch_size}\n")
                    
                    # Create message (no additional downsampling)
                    pc_msg = create_pointcloud2_message(
                        all_points, 
                        all_colors,
                        frame_id="world", 
                        timestamp_sec=timestamp_sec
                    )
                    
                    # Also write to output bag
                    if output_writer is not None:
                        from rclpy.serialization import serialize_message
                        output_writer.write(f'/robot{robot}/global_map', serialize_message(pc_msg), timestamp)
    
    print(f"\n{'='*60}")
    print(f"Completed {robot}:")
    print(f"  Scans processed: {processed_count}")
    print(f"  Map published: {publish_count} times")
    # print(f"  Final points: {len(final_points)}")
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
        help="Number of scans to process before publishing global map (default: 10)"
    )
    parser.add_argument(
        "--max_scans",
        type=int,
        default=None,
        help="Maximum number of scans to process (default: all scans)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for saving point cloud (default: auto-generated)"
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
    
    print(f"\n{'='*80}")
    print("Incremental Global Voxel Map Creation")
    print(f"{'='*80}")
    print(f"Environment: {args.environment}")
    print(f"Robots: {', '.join(robots)}")
    print(f"Per-scan voxel: 4.0m")
    print(f"Batch size: {args.batch_size} scans (~{args.batch_size/10:.1f} Hz publish rate)")
    print(f"{'='*80}\n")
    
    try:
        for robot in robots:
            # Create output bag for THIS ROBOT
            robot_dir = dataset_path / args.environment / robot
            
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = robot_dir / f"{robot}_{args.environment}_with_global_map"
            
            # Remove existing bag if it exists
            if output_path.exists():
                import shutil
                print(f"\nRemoving existing bag: {output_path}")
                shutil.rmtree(output_path)
            
            print(f"Creating output bag: {output_path}")
            
            storage_options = rosbag2_py.StorageOptions(
                uri=str(output_path),
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
                name=f'/robot{robot}/global_map',
                type='sensor_msgs/msg/PointCloud2',
                serialization_format='cdr'
            )
            writer.create_topic(topic_info)
            
            # Process this robot
            process_robot_incremental(
                dataset_path, args.environment, robot, writer, args.batch_size, args.max_scans
            )
            
            # Close this robot's bag
            del writer
            
        print(f"\n{'='*80}")
        print("SUCCESS! All robots processed.")
        print(f"{'='*80}")
        print(f"Processed {len(robots)} robot(s)")
        print(f"Output bags created in each robot's directory:")
        for robot in robots:
            robot_dir = dataset_path / args.environment / robot
            output_bag = robot_dir / f"{robot}_{args.environment}_with_global_map"
            if output_bag.exists():
                print(f"  - {output_bag}")
        print(f"\nTo view:")
        print(f"  ros2 bag play <robot_bag_path>")
        print(f"  # Open RViz2 and add PointCloud2 on /global_map")
        print(f"{'='*80}")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

