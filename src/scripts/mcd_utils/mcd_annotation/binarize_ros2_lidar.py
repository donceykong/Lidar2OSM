#!/usr/bin/env python3
import os
import sys
import numpy as np
from decimal import Decimal
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs_py import point_cloud2 as pc2


def pointcloud_msg_to_numpy(msg, datatype=np.float32):
    fields = ['x', 'y', 'z', 'intensity']
    points = pc2.read_points(msg, field_names=fields, skip_nans=False)

    # Convert generator to plain ndarray (N, 4)
    pointcloud_numpy = np.array([ [p[0], p[1], p[2], p[3]] for p in points ], dtype=datatype)

    return pointcloud_numpy


def save_ouster_pointcloud(lidar_pc_bin_path, pc2_ros2_msg, lidar_pc_number):
    pointcloud = pointcloud_msg_to_numpy(pc2_ros2_msg)
    filename = f"{lidar_pc_bin_path}/unsorted_lidar_pointcloud_{lidar_pc_number:010d}.bin"
    pointcloud.tofile(filename)


def read_bag(bag_path, bin_dir, topic_name):
    lidar_ts_index_dict = {}

    # Counter for messages on the target topic (1-indexed, represents index in bag file for this topic)
    # This is due to how the GT poses are associated with the lidar scans
    lidar_pc_number = 1

    # Make dirs
    lidar_bin_dir = os.path.join(bin_dir)
    lidar_data_dir = os.path.join(lidar_bin_dir, "data")
    os.makedirs(lidar_data_dir, exist_ok=True)
    lidar_ts_path = os.path.join(lidar_bin_dir, "timestamps.txt")

    # Open the bag
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id="")
    converter_options = ConverterOptions("cdr", "cdr")
    reader.open(storage_options, converter_options)

    # Get topic type
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_type = get_message(type_map[topic_name])

    while reader.has_next():
        topic, data, bag_ns = reader.read_next()
        if topic != topic_name:
            continue

        msg = deserialize_message(data, msg_type)
        sec = msg.header.stamp.sec
        nsec = msg.header.stamp.nanosec
        header_sec = Decimal(sec) + Decimal(nsec) / Decimal(1e9)

        # Use the sequential index in the bag file for this topic
        # (increments only for messages matching the target topic)
        lidar_ts_index_dict[header_sec] = lidar_pc_number
        save_ouster_pointcloud(lidar_data_dir, msg, lidar_pc_number)
        lidar_pc_number += 1

    # Save sorted timestamps
    timestamps_np = np.array(sorted(lidar_ts_index_dict.keys()))
    np.savetxt(lidar_ts_path, timestamps_np, fmt='%s')

    # Rename files in order
    for idx, timestamp in enumerate(timestamps_np, start=1):
        orig_index = lidar_ts_index_dict[timestamp]
        orig_file = f"{lidar_data_dir}/unsorted_lidar_pointcloud_{orig_index:010d}.bin"
        new_file = f"{lidar_data_dir}/{idx:010d}.bin"
        os.rename(orig_file, new_file)
        if orig_index != idx:
            print(f"    - Orig index: {orig_index}, new index: {idx}")

    total_messages = lidar_pc_number - 1  # -1 because the counter is 1-indexed
    print(f" - Total messages on {topic_name}: {total_messages}")


if __name__ == '__main__':
    root_path = "/media/donceykong/doncey_ssd_02/datasets/MCD"
    sequences = ["kth_day_09", "kth_day_06", "kth_night_05", "kth_night_04", "kth_night_01"]

    for seq in sequences:
        bag_path = os.path.join(root_path, seq, "bags", f"{seq}_os1_64")
        bin_dir = os.path.join(root_path, seq, "lidar_bin")
        topic_name = "/os_cloud_node/points"

        read_bag(bag_path, bin_dir, topic_name)
