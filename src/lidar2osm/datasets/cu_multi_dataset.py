#!/usr/bin/env python3

# External Imports
import os
import numpy as np
from collections import namedtuple
import re
from scipy.spatial.transform import Rotation as R

# For poses class
from typing import Dict, List, Tuple
import csv
from dataclasses import dataclass

# Internal Imports
from lidar2osm.core.projection import convert_ego_poses_to_latlon
from lidar2osm.utils.file_io import read_bin_file


@dataclass
class Pose:
    t: float
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float

class CU_MULTI_DATASET:
    def __init__(self, root_path=None):
        self.root_path = root_path

        # Used in data extraction. Only points with these ids will be used for the
        # associated extraction. First label is the OSM label. 
        # Following labels are SEM-KITTI labels that may be used for re-labeling. 
        self.build_semantic_ids = [45, 50, 0, 51, 52]
        self.road_semantic_ids = [46, 40, 10, 0, 44, 49, 60, 72]

        # These tags are passed into an OSM parser to grab the desired polyline data
        # These tags will be different depending on your use case.
        self.building_tags = {"building": True}
        self.road_tags = {"highway": ["motorway", "trunk", "primary", "secondary", 
                                      "tertiary", "unclassified", "residential", 
                                      "motorway_link", "trunk_link", "primary_link", 
                                      "secondary_link", "tertiary_link",
                                      "living_street", "service", "pedestrian", "road", 
                                      "cycleway", "foot", "footway", "path", "service",]}
    
    def setup_robot_data(self, environment, robot):
        self.env = environment
        self.robot = robot

        print(f"\nenvironment: {self.env}, robot: {self.robot}")

        # Set correct filepaths given seq
        self.set_paths()

        # self.get_frame_numbers(directory_path=self.label_path)
        self.init_frame, self.fin_frame = self.find_min_max_file_names()

        self.current_frame = self.init_frame
        self.labels_dict = {label.id: label.color for label in labels}

        # Set origin of sequence in lat lon
        self.set_gps_origin()

        # # Set lidar poses
        self.set_lidar_poses()

    def read_initial_utm_pose(self):
        """ Reads the initial utm pose from the ground truth utm poses csv. 
        """
        with open(self.gt_utm_poses_path, "r") as poses_file:
            gt_utm_poses = [line.strip().split() for line in poses_file]
        # Get the initial utm pose
        initial_utm_pose = gt_utm_poses[0]
        # Get the initial utm pose
        return gt_utm_poses[0]

    def load_csv(self, csv_file_path: str) -> List[Pose]:
        pose_list: List[Pose] = []
        with open(csv_file_path, "r", newline="") as csv_file:
            next(csv_file)  # Skip first header line
            next(csv_file)  # Skip second header line
            csv_reader = csv.DictReader(csv_file)
            if not {"# timestamp", " x", " y"}.issubset(csv_reader.fieldnames or set()):
                raise ValueError("CSV must have headers: '# timestamp',' x',' y',' z',' qx',' qy',' qz',' qw'")
            for csv_row in csv_reader:
                try:
                    pose_list.append(Pose(float(
                        csv_row["# timestamp"]), 
                        float(csv_row[" x"]), 
                        float(csv_row[" y"]),
                        float(csv_row[" z"]),
                        float(csv_row[" qx"]),
                        float(csv_row[" qy"]),
                        float(csv_row[" qz"]),
                        float(csv_row[" qw"])))
                except Exception:
                    continue
        pose_list.sort(key=lambda pose: pose.t)
        return pose_list

    def convert_utm_to_latlon(self, utm_pose):
        """ Converts a utm pose to a lat-lon pose. Zone is Boulder, CO.
        """
        return utm_pose.x, utm_pose.y

    def set_gps_origin(self):
        """ Sets the lat-lon origin used from the ground truth utm poses.
        """
        # Read lidar timestamps into a list
        with open(self.lidar_timestamps_path, "r") as ts_file:
            lidar_timestamps = [np.float128(line.strip()) for line in ts_file]

        # Read poses csv
        gt_utm_poses = self.load_csv(self.gt_utm_poses_path)
        initial_utm_pose = gt_utm_poses[0]
        print(f"initial_utm_pose: {initial_utm_pose}")

        # Set origin utm and latlon
        self.origin_utm = [initial_utm_pose.x, initial_utm_pose.y]
        self.origin_latlon = self.convert_utm_to_latlon(initial_utm_pose)
        print(f"origin_latlon: {self.origin_latlon}")

        lidar_poses_latlon = convert_utm_poses_to_(gt_utm_poses)

    # def set_origin(self):
    #     """ Sets the lat-lon origin used when projecting points into mercator map.
    #         Uses interpolation to get gps lat-lon at initial lidar timestamp.
    #     """
    #     # Read lidar timestamps into a list
    #     with open(self.lidar_timestamps_path, "r") as ts_file:
    #         lidar_timestamps = [np.float128(line.strip()) for line in ts_file]

    #     # Read lidar timestamps into a list
    #     with open(self.gps_1_timestamps_path, "r") as gps_ts_file:
    #         gnss1_timestamps = [np.float128(line.strip()) for line in gps_ts_file]

    #     gps_1_points = read_gps_data(self.gps_1_path)
    #     gps_1_points[:, 2] = 0
        
    #     # Interpolat lat-lon position
    #     idx = np.searchsorted(gnss1_timestamps, lidar_timestamps[0]) - 1
    #     t0, t1 = gnss1_timestamps[idx], gnss1_timestamps[idx + 1]
    #     p0, p1 = gps_1_points[idx], gps_1_points[idx + 1]
    #     ratio = (lidar_timestamps[0] - t0) / (t1 - t0)
    #     interp_gps_position = (1 - ratio) * p0 + ratio * p1

    #     self.origin_latlon = interp_gps_position[:2]

    def set_paths(self):
        """ Sets paths for necesary data."""
        env_dir = os.path.join(self.root_path, self.env)
        self.robot_dir_path = os.path.join(self.root_path, self.env, self.robot)

        self.raw_pc_path = os.path.join(self.robot_dir_path, "lidar_bin/data")
        self.lidar_timestamps_path = os.path.join(self.robot_dir_path, "lidar_bin/timestamps.txt")
        self.label_path = os.path.join(self.robot_dir_path, "lidar_labels")
        self.osm_label_path = os.path.join(self.robot_dir_path, "lidar_osm_labels")

        # Ground truth poses path
        self.gt_utm_poses_path = os.path.join(
            self.robot_dir_path, f"{self.robot}_{self.env}_gt_utm_poses.csv"
        )

        # GPS 1 path
        # self.gps_1_path = os.path.join(self.robot_dir_path, "gps_1/gnss_1_data.txt")
        # self.gps_1_timestamps_path = os.path.join(self.robot_dir_path, "gps_1/gnss_1_timestamps.txt")

        self.osm_file_path = os.path.join(env_dir, f"{self.env}.osm")

    def get_data_paths(self, frame_num):
        """ Returns full path to pc, pc labels, and pc osm labels"""
        raw_pc_path = os.path.join(self.raw_pc_path, f"lidar_pointcloud_{frame_num:01d}.bin")
        pc_label_path = os.path.join(self.label_path, f"lidar_pointcloud_{frame_num:01d}.bin")
        pc_osm_label_path = os.path.join(self.osm_label_path, f"lidar_pointcloud_{frame_num:01d}.bin")

        return pc_label_path, pc_osm_label_path, raw_pc_path
    
    def get_pointcloud(self, raw_pc_path):
        """ Returns pointcloud with shape (num_points, 4), where each point has a value for (x, y, z, intensity)"""
        return read_bin_file(raw_pc_path, dtype=np.float32, shape=(-1, 4))

    def get_labels(self, pc_label_path):
        """ Returns pointcloud with shape (num_points, 1), where each point has a distinct label int"""
        return read_bin_file(pc_label_path, dtype=np.int32, shape=(-1))
    
    def set_lidar_poses(self):
        # Get initial GPS datapoint
        if os.path.exists(self.lidar_poses_file):
            self.lidar_poses = self.read_poses(self.lidar_poses_file)
            self.lidar_poses_latlon = convert_ego_poses_to_latlon(self.lidar_poses, self.origin_latlon)

    def find_min_max_file_names(self):
        """
        Finds the minimum and maximum file names in a directory based on a number delimiter and file extension.

        Args:
            file_dir_path (str): The directory path containing the files.
            number_delimiter (str): The delimiter used to separate the number in the file names.
            file_extension (str): The file extension to look for.

        Returns:
            tuple: A tuple containing the minimum and maximum file numbers if files are found, otherwise (None, None).
        """
        all_files = os.listdir(self.label_path)

        # Filter out files ending with ".bin" and remove the filetype
        filenames = [
            int(re.search(r'\d+', os.path.splitext(file)[0]).group())
            for file in all_files if file.endswith(".bin") and re.search(r'\d+', os.path.splitext(file)[0])
        ]

        return min(filenames), max(filenames) if filenames else (None, None)

    # def get_frame_numbers(self, directory_path) -> list:
    #     """Count the total number of files in the directory"""
    #     frame_numbers = []
    #     all_files = os.listdir(directory_path)

    #     # Filter out files ending with ".bin" and remove the filetype
    #     filenames = [
    #         int(re.search(r'\d+', os.path.splitext(file)[0]).group())
    #         for file in all_files if file.endswith(".bin") and re.search(r'\d+', os.path.splitext(file)[0])
    #     ]

    #     for filename in filenames:
    #         frame_numbers.append(int(filename))

    #     return sorted(frame_numbers)

    def quaternion_pose_to_4x4(self, trans, quat):
        rotation_matrix = R.from_quat(quat).as_matrix()

        # Create the 4x4 transformation matrix
        transformation_matrix = np.eye(4)  # Start with an identity matrix
        transformation_matrix[:3, :3] = rotation_matrix  # Set the rotation part
        transformation_matrix[:3, 3] = trans  # Set the translation part

        return transformation_matrix

    def read_poses(self, poses_path):
        poses_xyz = {}
        with open(poses_path, "r") as file:
            for idx, line in enumerate(file):
                elements = line.strip().split()
                trans = np.array(elements[:3], dtype=float)
                quat = np.array(elements[3:], dtype=float)

                if len(trans) > 0:
                    transformation_matrix = self.quaternion_pose_to_4x4(trans, quat)
                    poses_xyz[idx] = transformation_matrix
                else:
                    poses_xyz[idx] = np.eye(4)
        return poses_xyz


Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        "id",  # An integer ID that is associated with this label.
        "color",  # The color of this label
    ],
)

# labels = [
#     #       name, id, color
#     Label("unlabeled", 0, (0, 0, 0)),               # OVERLAP
#     Label("outlier", 1, (0, 0, 0)),
#     Label("car", 10, (0, 0, 142)),                  # OVERLAP
#     Label("bicycle", 11, (119, 11, 32)),            # OVERLAP
#     Label("bus", 13, (250, 80, 100)),
#     Label("motorcycle", 15, (0, 0, 230)),           # OVERLAP
#     Label("on-rails", 16, (255, 0, 0)),
#     Label("truck", 18, (0, 0, 70)),                 # OVERLAP
#     Label("other-vehicle", 20, (51, 0, 51)),        # OVERLAP
#     Label("person", 30, (220, 20, 60)),             # OVERLAP
#     Label("bicyclist", 31, (200, 40, 255)),
#     Label("motorcyclist", 32, (90, 30, 150)),
#     Label("road", 40, (128, 64, 128)),              # OVERLAP
#     Label("parking", 44, (250, 170, 160)),          # OVERLAP
#     Label("OSM BUILDING", 45, (0, 0, 255)),         # ************ OSM
#     Label("OSM ROAD", 46, (255, 0, 0)),             # ************ OSM
#     Label("sidewalk", 48, (244, 35, 232)),          # OVERLAP
#     Label("other-ground", 49, (81, 0, 81)),         # OVERLAP
#     Label("building", 50, (0, 100, 0)),             # OVERLAP
#     Label("fence", 51, (190, 153, 153)),            # OVERLAP
#     Label("other-structure", 52, (0, 150, 255)),
#     Label("lane-marking", 60, (170, 255, 150)),
#     Label("vegetation", 70, (107, 142, 35)),        # OVERLAP
#     Label("trunk", 71, (0, 60, 135)),
#     Label("terrain", 72, (152, 251, 152)),          # OVERLAP
#     Label("pole", 80, (153, 153, 153)),             # OVERLAP
#     Label("traffic-sign", 81, (0, 0, 255)),
#     Label("other-object", 99, (255, 255, 50)),
#     # Label("moving-car", 252, (245, 150, 100)),
#     # Label("moving-bicyclist", 253, ()),
#     # Label("moving-person", 254, (30, 30, 25)),
#     # Label("moving-motorcyclist", 255, (90, 30, 150)),
#     # Label("moving-on-rails", 256, ()),
#     # Label("moving-bus", 257, ()),
#     # Label("moving-truck", 258, ()),
#     # Label("moving-other-vehicle", 259, ()),
# ]

# labels = [
#     #       name, id, color
#     Label("unlabeled", 0, (255, 0, 0)),               # OVERLAP
#     Label("outlier", 1, (0, 255, 0)),
#     Label("car", 10, (0, 0, 255)),                  # OVERLAP
#     Label("bicycle", 11, (0, 0, 0)),            # OVERLAP
#     Label("bus", 13, (0, 0, 255)),
#     Label("motorcycle", 15, (0, 0, 0)),           # OVERLAP
#     Label("on-rails", 16, (0, 0, 255)),
#     Label("truck", 18, (0, 0, 255)),                 # OVERLAP
#     Label("other-vehicle", 20, (0, 0, 0)),        # OVERLAP
#     Label("person", 30, (0, 0, 0)),             # OVERLAP
#     Label("bicyclist", 31, (0, 0, 0)),
#     Label("motorcyclist", 32, (0, 0, 0)),
#     Label("road", 40, (0, 0, 0)),              # OVERLAP
#     Label("parking", 44, (0, 0, 0)),          # OVERLAP
#     Label("OSM BUILDING", 45, (0, 0, 0)),         # ************ OSM
#     Label("OSM ROAD", 46, (0, 0, 0)),             # ************ OSM
#     Label("sidewalk", 48, (0, 0, 0)),          # OVERLAP
#     Label("other-ground", 49, (0, 0, 0)),         # OVERLAP
#     Label("building", 50, (0, 0, 0)),             # OVERLAP
#     Label("fence", 51, (0, 0, 0)),            # OVERLAP
#     Label("other-structure", 52, (0, 0, 0)),
#     Label("lane-marking", 60, (0, 0, 0)),
#     Label("vegetation", 70, (0, 255, 0)),        # OVERLAP
#     Label("trunk", 71, (0, 0, 0)),
#     Label("terrain", 72, (0, 0, 0)),          # OVERLAP
#     Label("pole", 80, (0, 0, 0)),             # OVERLAP
#     Label("traffic-sign", 81, (0, 0, 0)),
#     Label("other-object", 99, (0, 0, 0)),
# ]

labels = [
    # name, id, color
    Label("unlabeled", 0, (0, 0, 0)),               # OVERLAP
    Label("outlier", 1, (0, 0, 0)),
    Label("car", 10, (0, 0, 142)),                  # OVERLAP
    Label("bicycle", 11, (119, 11, 32)),            # OVERLAP
    Label("bus", 13, (250, 80, 100)),
    Label("motorcycle", 15, (0, 0, 230)),           # OVERLAP
    Label("on-rails", 16, (255, 0, 0)),
    Label("truck", 18, (0, 0, 70)),                 # OVERLAP
    Label("other-vehicle", 20, (51, 0, 51)),        # OVERLAP
    Label("person", 30, (220, 20, 60)),             # OVERLAP
    Label("bicyclist", 31, (200, 40, 255)),
    Label("motorcyclist", 32, (90, 30, 150)),
    Label("road", 40, (128, 64, 128)),              # OVERLAP
    Label("parking", 44, (250, 170, 160)),          # OVERLAP
    Label("OSM BUILDING", 45, (0, 0, 255)),         # ************ OSM
    Label("OSM ROAD", 46, (255, 0, 0)),             # ************ OSM
    Label("sidewalk", 48, (244, 35, 232)),          # OVERLAP
    Label("other-ground", 49, (81, 0, 81)),         # OVERLAP
    Label("building", 50, (0, 100, 0)),             # OVERLAP
    Label("fence", 51, (190, 153, 153)),            # OVERLAP
    Label("other-structure", 52, (0, 150, 255)),
    Label("lane-marking", 60, (170, 255, 150)),
    Label("vegetation", 70, (107, 142, 35)),        # OVERLAP
    Label("trunk", 71, (0, 60, 135)),
    Label("terrain", 72, (152, 251, 152)),          # OVERLAP
    Label("pole", 80, (153, 153, 153)),             # OVERLAP
    Label("traffic-sign", 81, (0, 0, 255)),
    Label("other-object", 99, (255, 255, 50)),
]