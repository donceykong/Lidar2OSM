import os

import numpy as np

# Define the label names and corresponding IDs
labels = {
    0: "unlabeled",
    1: "ego vehicle",
    2: "rectification border",
    3: "out of roi",
    4: "static",
    5: "dynamic",
    6: "ground",
    7: "road",
    8: "sidewalk",
    9: "parking",
    10: "rail track",
    11: "building",
    12: "wall",
    13: "fence",
    14: "guard rail",
    15: "bridge",
    16: "tunnel",
    17: "pole",
    18: "polegroup",
    19: "traffic light",
    20: "traffic sign",
    21: "vegetation",
    22: "terrain",
    23: "sky",
    24: "person",
    25: "rider",
    26: "car",
    27: "truck",
    28: "bus",
    29: "caravan",
    30: "trailer",
    31: "train",
    32: "motorcycle",
    33: "bicycle",
    34: "garage",
    35: "gate",
    36: "stop",
    37: "smallpole",
    38: "lamp",
    39: "trash bin",
    40: "vending machine",
    41: "box",
    42: "unknown construction",
    43: "unknown vehicle",
    44: "unknown object",
    100: "OSM BUILDING",
    101: "OSM ROAD",
}


# Function to load the label file
def load_label_file(file_path):
    return np.fromfile(file_path, dtype=np.int32)


# Function to calculate the stats
def calculate_stats(labels_array):
    total_points = len(labels_array)
    stats = {}

    for label_id in labels.keys():
        num_points = np.sum(labels_array == label_id)
        percent_points = num_points / total_points
        stats[label_id] = percent_points

    return stats


# Path to the label file
label_file_path = "/home/donceykong/Desktop/datasets/KITTI-360/data_3d_semantics/2013_05_28_drive_0000_sync/osm_labels/0000000072.bin"

# Load the label file
labels_array = load_label_file(label_file_path)

# Calculate the stats
stats = calculate_stats(labels_array)

# Save the stats to a file in the desired format
with open("label_stats.txt", "w") as f:
    for label_id, percent_points in stats.items():
        f.write(f"  {label_id}: {percent_points}\n")
