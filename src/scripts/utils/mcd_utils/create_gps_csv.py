#!/usr/bin/env python3
import os
import csv
from decimal import Decimal
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def read_gps_bag(bag_path, output_csv_path, topic_name):
    """
    Read GPS data from a ROS2 bag file and save to CSV.
    
    Args:
        bag_path: Path to the ROS2 bag file
        output_csv_path: Path to output CSV file
        topic_name: Name of the GPS topic (e.g., '/gps/fix' or '/navsat/fix')
    """
    gps_data = []
    
    # Open the bag
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id="")
    converter_options = ConverterOptions("cdr", "cdr")
    reader.open(storage_options, converter_options)
    
    # Get topic type
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    
    if topic_name not in type_map:
        print(f"ERROR: Topic '{topic_name}' not found in bag file.")
        print(f"Available topics: {list(type_map.keys())}")
        return
    
    msg_type = get_message(type_map[topic_name])
    
    print(f"Reading GPS data from topic: {topic_name}")
    
    while reader.has_next():
        topic, data, bag_ns = reader.read_next()
        if topic != topic_name:
            continue
        
        msg = deserialize_message(data, msg_type)
        
        # Extract timestamp
        sec = msg.header.stamp.sec
        nsec = msg.header.stamp.nanosec
        timestamp = Decimal(sec) + Decimal(nsec) / Decimal(1e9)
        
        # Extract latitude and longitude
        latitude = msg.latitude
        longitude = msg.longitude
        
        gps_data.append({
            'timestamp': float(timestamp),
            'lat': latitude,
            'lon': longitude
        })
    
    # Sort by timestamp
    gps_data.sort(key=lambda x: x['timestamp'])
    
    # Write to CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'lat', 'lon']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in gps_data:
            writer.writerow(row)
    
    print(f" - Total GPS messages: {len(gps_data)}")
    print(f" - Saved to: {output_csv_path}")


if __name__ == '__main__':
    root_path = "/media/donceykong/doncey_ssd_02/datasets/MCD/kth_day_06"
    bag_path = os.path.join(root_path, "kth_day_06_os1_64")  # Update with your GPS bag path
    output_csv_path = os.path.join(root_path, "gps_data.csv")
    topic_name = "/gps/fix"  # Common GPS topic names: /gps/fix, /navsat/fix, /fix
    
    read_gps_bag(bag_path, output_csv_path, topic_name)

