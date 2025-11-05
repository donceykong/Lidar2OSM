#!/usr/bin/env python3
"""
Script to create ROS2 lidar bags with semantic RGB information.
Reads existing ROS2 bags and adds RGB channels based on semantic labels.

The output bag will contain PointCloud2 messages with fields: x, y, z, intensity, rgb
"""

import os
import sys
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ROS2 imports
try:
    import rclpy
    from rclpy.serialization import deserialize_message, serialize_message
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    import rosbag2_py
    from rosidl_runtime_py.utilities import get_message
except ImportError as e:
    print(f"Error: Missing ROS2 dependencies: {e}")
    print("Please install: pip install rosbag2-py rclpy")
    sys.exit(1)

# Internal imports
from lidar2osm.utils.file_io import read_bin_file
from lidar2osm.datasets.cu_multi_dataset import labels as sem_kitti_labels


def labels2RGB_uint32(label_ids, labels_dict):
    """
    Convert semantic labels to RGB colors packed as uint32 for ROS PointCloud2.
    
    Args:
        label_ids (np array of int): The semantic labels.
        labels_dict (dict): Dictionary containing semantic label IDs and RGB values.
    
    Returns:
        rgb_array (np array of uint32): RGB values packed as uint32.
    """
    rgb_array = np.zeros(len(label_ids), dtype=np.uint32)
    
    for idx, label_id in enumerate(label_ids):
        if label_id in labels_dict:
            color = labels_dict.get(label_id, [0, 0, 0])  # Default color is black
            # Pack RGB as uint32: 0x00RRGGBB
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            rgb_array[idx] = (r << 16) | (g << 8) | b
        else:
            rgb_array[idx] = 0  # Black for unknown labels
    
    return rgb_array


def convert_pointcloud2_to_array(msg):
    """
    Convert ROS2 PointCloud2 message to numpy array.
    
    Args:
        msg: sensor_msgs/msg/PointCloud2 message
    
    Returns:
        points: Nx4 numpy array with x, y, z, intensity
    """
    # Map datatype codes to numpy dtypes
    datatype_map = {
        PointField.INT8: ('i1', 1),
        PointField.UINT8: ('u1', 1),
        PointField.INT16: ('i2', 2),
        PointField.UINT16: ('u2', 2),
        PointField.INT32: ('i4', 4),
        PointField.UINT32: ('u4', 4),
        PointField.FLOAT32: ('f4', 4),
        PointField.FLOAT64: ('f8', 8),
    }
    
    # Build dtype with proper offsets
    dtype_list = []
    last_offset = 0
    
    for field in msg.fields:
        # Add padding if necessary
        if field.offset > last_offset:
            padding_size = field.offset - last_offset
            dtype_list.append((f'_pad{last_offset}', f'u1', padding_size))
        
        # Get field dtype and size
        np_dtype, dtype_size = datatype_map.get(field.datatype, ('u1', 1))
        field_size = dtype_size * field.count
        
        if field.count == 1:
            dtype_list.append((field.name, np_dtype))
        else:
            dtype_list.append((field.name, np_dtype, field.count))
        
        last_offset = field.offset + field_size
    
    # Add final padding to reach point_step
    if msg.point_step > last_offset:
        padding_size = msg.point_step - last_offset
        dtype_list.append((f'_pad{last_offset}', 'u1', padding_size))
    
    # Create structured array from binary data
    try:
        cloud_array = np.frombuffer(msg.data, dtype=np.dtype(dtype_list))
    except ValueError as e:
        print(f"Error parsing point cloud: {e}")
        print(f"Point step: {msg.point_step}, Last offset: {last_offset}")
        print(f"Data size: {len(msg.data)}, Expected: {msg.point_step * msg.width * msg.height}")
        raise
    
    # Extract x, y, z, intensity
    num_points = msg.width * msg.height
    points = np.zeros((num_points, 4), dtype=np.float32)
    points[:, 0] = cloud_array['x']
    points[:, 1] = cloud_array['y']
    points[:, 2] = cloud_array['z']
    
    # Try to get intensity if it exists
    if 'intensity' in cloud_array.dtype.names:
        points[:, 3] = cloud_array['intensity']
    elif 'i' in cloud_array.dtype.names:
        points[:, 3] = cloud_array['i']
    
    return points


def create_pointcloud2_with_rgb(header, points, intensities, rgb_values):
    """
    Create ROS2 PointCloud2 message with x, y, z, intensity, and rgb fields.
    
    Args:
        header: ROS2 message header
        points: Nx3 numpy array with x, y, z coordinates
        intensities: N numpy array with intensity values
        rgb_values: N numpy array with RGB values packed as uint32
    
    Returns:
        msg: sensor_msgs/msg/PointCloud2 message
    """
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(points)
    msg.is_bigendian = False
    msg.is_dense = False
    
    # Define fields: x, y, z, intensity, rgb
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=16, datatype=PointField.UINT32, count=1),
    ]
    msg.fields = fields
    msg.point_step = 20  # 4 bytes * 4 fields + 4 bytes for rgb
    msg.row_step = msg.point_step * msg.width
    
    # Pack data
    data = np.zeros(len(points), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32),
        ('rgb', np.uint32)
    ])
    
    data['x'] = points[:, 0]
    data['y'] = points[:, 1]
    data['z'] = points[:, 2]
    data['intensity'] = intensities
    data['rgb'] = rgb_values
    
    msg.data = data.tobytes()
    
    return msg


def process_bag(input_bag_path, output_bag_path, labels_dir, robot_name, environment):
    """
    Process ROS2 bag and add RGB channel based on semantic labels.
    
    Args:
        input_bag_path: Path to input ROS2 bag directory
        output_bag_path: Path to output ROS2 bag directory
        labels_dir: Path to directory containing semantic label files
        robot_name: Name of the robot (e.g., "robot1")
        environment: Environment name (e.g., "main_campus")
    """
    print(f"\n{'='*80}")
    print(f"Processing ROS2 bag for {robot_name} in {environment}")
    print(f"Input bag: {input_bag_path}")
    print(f"Output bag: {output_bag_path}")
    print(f"Labels dir: {labels_dir}")
    print(f"{'='*80}\n")
    
    # Create labels dictionary
    labels_dict = {label.id: label.color for label in sem_kitti_labels}
    
    # Get all label files
    label_files = sorted(list(Path(labels_dir).glob("*.bin")))
    print(f"Found {len(label_files)} label files")
    
    if len(label_files) == 0:
        print(f"ERROR: No label files found in {labels_dir}")
        return
    
    # Setup storage options for reading
    storage_options = rosbag2_py.StorageOptions(
        uri=str(input_bag_path),
        storage_id='sqlite3'
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    # Open input bag for reading
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    # Get topic types
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}
    
    # Find the point cloud topic
    pointcloud_topic = None
    for topic_name in type_map.keys():
        if 'points' in topic_name and type_map[topic_name] == 'sensor_msgs/msg/PointCloud2':
            pointcloud_topic = topic_name
            break
    
    if pointcloud_topic is None:
        print("ERROR: No PointCloud2 topic found in bag")
        return
    
    print(f"Found PointCloud2 topic: {pointcloud_topic}")
    
    # Setup storage options for writing
    output_storage_options = rosbag2_py.StorageOptions(
        uri=str(output_bag_path),
        storage_id='sqlite3'
    )
    
    # Open output bag for writing
    writer = rosbag2_py.SequentialWriter()
    writer.open(output_storage_options, converter_options)
    
    # Create topic in output bag
    topic_info = rosbag2_py.TopicMetadata(
        name=pointcloud_topic,
        type='sensor_msgs/msg/PointCloud2',
        serialization_format='cdr'
    )
    writer.create_topic(topic_info)
    
    # Also copy other topics (like IMU, metadata)
    for topic in topic_types:
        if topic.name != pointcloud_topic:
            other_topic_info = rosbag2_py.TopicMetadata(
                name=topic.name,
                type=topic.type,
                serialization_format='cdr'
            )
            writer.create_topic(other_topic_info)
    
    # Process messages
    frame_idx = 0
    message_count = 0
    pointcloud_count = 0
    
    print("\nProcessing messages...")
    
    # Get total message count for progress bar
    metadata = reader.get_metadata()
    total_messages = sum([topic.message_count for topic in metadata.topics_with_message_count])
    
    with tqdm(total=total_messages, desc="Processing messages") as pbar:
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            
            if topic == pointcloud_topic:
                # Deserialize PointCloud2 message
                msg_type = get_message('sensor_msgs/msg/PointCloud2')
                msg = deserialize_message(data, msg_type)
                
                # Check if we have a corresponding label file
                if frame_idx < len(label_files):
                    # Convert PointCloud2 to numpy array
                    points = convert_pointcloud2_to_array(msg)
                    points_xyz = points[:, :3]
                    intensities = points[:, 3]
                    
                    # Load semantic labels
                    label_file = label_files[frame_idx]
                    try:
                        labels = read_bin_file(str(label_file), dtype=np.int32)
                    except Exception as e:
                        print(f"\nERROR: Frame {frame_idx}: Failed to read label file {label_file}: {e}")
                        frame_idx += 1
                        pbar.update(1)
                        continue
                    
                    # Check if point count matches
                    if len(labels) != len(points):
                        print(f"\nWARNING: Frame {frame_idx}: Point count mismatch: "
                              f"{len(points)} points vs {len(labels)} labels. Skipping frame.")
                        frame_idx += 1
                        pbar.update(1)
                        continue
                    
                    # Convert labels to RGB
                    rgb_values = labels2RGB_uint32(labels, labels_dict)
                    
                    # Create new PointCloud2 message with RGB
                    new_msg = create_pointcloud2_with_rgb(msg.header, points_xyz, intensities, rgb_values)
                    
                    # Serialize and write to output bag
                    writer.write(pointcloud_topic, serialize_message(new_msg), timestamp)
                    
                    pointcloud_count += 1
                    frame_idx += 1
                else:
                    print(f"\nWARNING: No label file for frame {frame_idx}")
            else:
                # Copy other messages as-is
                writer.write(topic, data, timestamp)
            
            message_count += 1
            pbar.update(1)
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total messages processed: {message_count}")
    print(f"  PointCloud2 messages with RGB: {pointcloud_count}")
    print(f"  Output bag: {output_bag_path}")
    print(f"{'='*60}\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create ROS2 lidar bags with semantic RGB information"
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
        help="Environment name (default: main_campus)"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        help="Robot name (default: process all robots)"
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default=None,
        help="Custom path to labels directory (overrides default)"
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    
    # Determine which robots to process
    if args.robot:
        robots = [args.robot]
    else:
        # Process all robots in the environment
        env_path = dataset_path / args.environment
        if not env_path.exists():
            print(f"ERROR: Environment path not found: {env_path}")
            return
        robots = [d.name for d in env_path.iterdir() if d.is_dir() and d.name.startswith('robot')]
        robots = sorted(robots)
    
    print(f"\nProcessing robots: {', '.join(robots)}")
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Process each robot
        for robot in robots:
            try:
                robot_dir = dataset_path / args.environment / robot
                
                # Input bag path
                input_bag_path = robot_dir / f"{robot}_{args.environment}_lidar"
                if not input_bag_path.exists():
                    print(f"\nERROR: Input bag not found: {input_bag_path}")
                    continue
                
                # Labels directory
                labels_dir = robot_dir / f"{robot}_{args.environment}_refined_lidar_labels"
                
                if not labels_dir.exists():
                    print(f"\nERROR: Labels directory not found: {labels_dir}")
                    continue
                
                # Output bag path
                output_bag_path = robot_dir / f"{robot}_{args.environment}_lidar_rgb"
                
                # Remove output bag if it exists
                if output_bag_path.exists():
                    import shutil
                    print(f"\nRemoving existing output bag: {output_bag_path}")
                    shutil.rmtree(output_bag_path)
                
                # Process bag
                process_bag(
                    input_bag_path=input_bag_path,
                    output_bag_path=output_bag_path,
                    labels_dir=labels_dir,
                    robot_name=robot,
                    environment=args.environment
                )
                
            except Exception as e:
                print(f"\nERROR processing {robot}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    finally:
        # Shutdown ROS2
        rclpy.shutdown()
    
    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)


if __name__ == "__main__":
    main()

