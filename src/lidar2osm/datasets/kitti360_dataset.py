#!/usr/bin/env python3

# External Imports
import os
import numpy as np
from collections import namedtuple
import glob

# Internal Imports
from lidar2osm.utils.file_io import save_poses
from lidar2osm.core.projection import convert_ego_poses_to_latlon
from lidar2osm.utils.file_io import read_bin_file

class KITTI360_DATASET:
    def __init__(self, root_path=None):
        self.root_path = root_path

        # Used when projecting points into mercator map
        self.origin_latlon = [48.9843445, 8.4295857]  # lake in Karlsruhe

        # Used in data extraction. Only points with these ids will be used for the
        # associated extraction
        self.build_semantic_ids = [45, 50, 0, 51, 52]
        self.road_semantic_ids = [46, 40, 0, 44, 49, 60, 72]
        # self.build_semantic_ids = [45, 11, 0]
        # self.road_semantic_ids = [46, 7, 0]

        # These tags are passed into an OSM parser to grab the desired polyline data
        # These tags will be different depending on your use case.
        self.building_tags = {"building": True}
        self.road_tags = {
                            "highway": [
                                "motorway",
                                "trunk",
                                "primary",
                                "secondary",
                                "tertiary",
                                "unclassified",
                                "residential",
                                "motorway_link",
                                "trunk_link",
                                "primary_link",
                                "secondary_link",
                                "tertiary_link",
                                "living_street",
                                "service",
                                "pedestrian",
                                "road",
                            ]
                        }

    def setup_sequence(self, seq):
        self.seq = seq

        # Set correct filepaths given seq
        self.set_paths()

        self.init_frame, self.fin_frame = self.find_min_max_file_names(
            file_dir_path=self.label_path, number_delimiter=".", file_extension=".bin"
        )
        self.current_frame = self.init_frame
        self.labels_dict = {label.id: label.color for label in labels}

        # Set lidar poses
        self.set_lidar_poses()

        # For visualization
        self.road_pcd_list = []
        self.building_pcd_list = []

    def get_data_paths(self, frame_num):
        """ Returns full path to pc, pc labels, and pc osm labels
        """
        pc_label_path = os.path.join(self.label_path, f"{frame_num:010d}.bin")
        raw_pc_path = os.path.join(self.raw_pc_path, f"{frame_num:010d}.bin")
        pc_osm_label_path = os.path.join(self.osm_label_path, f"{frame_num:010d}.bin")

        return pc_label_path, pc_osm_label_path, raw_pc_path
    
    def get_pointcloud(self, raw_pc_path):
        """ Returns pointcloud with shape (num_points, 4), where each point has a value for (x, y, z, intensity)
        """
        return read_bin_file(raw_pc_path, dtype=np.float32, shape=(-1, 4))

    def get_labels(self, pc_label_path):
        """ Returns pointcloud with shape (num_points, 1), where each point has a distinct label int
        
        KITTI-360 uses int16 datatype for its pointcloud labels.
        """
        return read_bin_file(pc_label_path, dtype=np.int32, shape=(-1))
    
    def set_paths(self):
        self.kitti360Path = self.root_path
        sequence_dir = f"2013_05_28_drive_{self.seq:04d}_sync"
        self.sequence_dir_path = os.path.join(self.kitti360Path, sequence_dir)
        self.raw_pc_path = os.path.join(
            self.kitti360Path, "data_3d_raw", sequence_dir, "velodyne_points", "data"
        )
        self.semantics_dir_path = os.path.join(
            self.kitti360Path, "data_3d_semantics", sequence_dir
        )
        self.label_path = os.path.join(self.semantics_dir_path, "labels")
        self.osm_label_path = os.path.join(self.semantics_dir_path, "osm_labels")
        self.inferred_osm_path = os.path.join(self.semantics_dir_path, "inferred_osm_labels")
        self.inferred_full_sem_path = os.path.join(self.semantics_dir_path, "inferred_full_semantics")
        self.accum_ply_path = os.path.join(self.semantics_dir_path, "accum_ply")
        self.imu_poses_file = os.path.join(
            self.kitti360Path, "data_poses", sequence_dir, "poses.txt"
        )
        self.lidar_poses_file = os.path.join(
            self.kitti360Path, "data_poses", sequence_dir, "velodyne_poses.txt"
        )
        self.lidar_poses_latlon_file = os.path.join(
            self.kitti360Path, "data_poses", sequence_dir, "poses_latlong.txt"
        )
        self.extracted_per_frame_dir = os.path.join(
            self.kitti360Path, "data_3d_extracted", sequence_dir, "per_frame"
        )
        self.extracted_bev_per_frame_dir = os.path.join(
            self.kitti360Path, "data_3d_extracted", sequence_dir, "bev"
        )
        self.extraced_map_segments_dir = os.path.join(
            self.kitti360Path, "data_3d_extracted", sequence_dir, "map_segments"
        )
        self.osm_file_path = os.path.join(
            self.kitti360Path, "data_osm", f"map_{self.seq:04d}.osm"
        )

    def set_lidar_poses(self):
        if os.path.exists(self.lidar_poses_file):
            self.lidar_poses = self.read_poses(self.lidar_poses_file)
        else:
            imu_poses_xyz = self.read_poses(self.imu_poses_file)
            self.lidar_poses = self.get_poses_lidar_from_imu(imu_poses_xyz)
            save_poses(self.lidar_poses_file, self.lidar_poses)
        
        self.lidar_poses_latlon = convert_ego_poses_to_latlon(self.lidar_poses, self.origin_latlon)
        save_poses(self.lidar_poses_latlon_file, self.lidar_poses_latlon)

    # TODO: Remove this function and make generalized in pose.py
    def get_poses_lidar_from_imu(self, imu_poses_xyz):
        """Get XYZ poses of Velodyne lidar using XYZ poses of imu (used in KITTI-360 dataset)

        Args:
            imu_poses_xyz (dict of 4x4 imu poses): Will use the poses of the imu to retrieve poses of velodyne.

        Returns:
            lidar_poses_xyz (dict of 4x4 Velodyne poses)
        """
        # Translation vector from IMU to LiDAR
        translation_vector = np.array([0.81, 0.32, -0.83])

        # Rotation matrix for a 180-degree rotation about the X-axis
        rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        # Create the transformation matrix from IMU to LiDAR
        imu_to_lidar_matrix = np.identity(4)
        imu_to_lidar_matrix[:3, :3] = rotation_matrix
        imu_to_lidar_matrix[:3, 3] = translation_vector

        # Apply the transformation to each pose in the dictionary
        lidar_poses_xyz = {}
        for idx, imu_matrix in imu_poses_xyz.items():
            # Perform the matrix multiplication
            lidar_pose = np.matmul(imu_matrix, imu_to_lidar_matrix)
            lidar_poses_xyz[idx] = lidar_pose

        return lidar_poses_xyz

    def find_min_max_file_names(self, file_dir_path, number_delimiter, file_extension):
        """
        Finds the minimum and maximum file names in a directory based on a number delimiter and file extension.

        Args:
            file_dir_path (str): The directory path containing the files.
            number_delimiter (str): The delimiter used to separate the number in the file names.
            file_extension (str): The file extension to look for.

        Returns:
            tuple: A tuple containing the minimum and maximum file numbers if files are found, otherwise (None, None).
        """

        pattern = os.path.join(file_dir_path, f"*{file_extension}")
        files = glob.glob(pattern)
        file_numbers = [
            int(os.path.basename(file).split(f"{number_delimiter}")[0]) for file in files
        ]
        return min(file_numbers), max(file_numbers) if file_numbers else (None, None)
    
    def read_poses(self, file_path):
        """
        Reads 4x4 or 3x4 transformation matrices from a file.

        Args:
            file_path (str): The path to the file containing poses.

        Returns:
            dict: A dictionary where keys are frame indices and values are 4x4 numpy arrays representing transformation matrices.
        """
        poses_xyz = {}
        with open(file_path, "r") as file:
            for line in file:
                elements = line.strip().split()
                frame_index = int(elements[0])
                if len(elements[1:]) == 16:
                    matrix_4x4 = np.array(elements[1:], dtype=float).reshape((4, 4))
                else:
                    matrix_3x4 = np.array(elements[1:], dtype=float).reshape((3, 4))
                    matrix_4x4 = np.vstack([matrix_3x4, np.array([0, 0, 0, 1])])
                poses_xyz[frame_index] = matrix_4x4
        return poses_xyz

Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        "id",  # An integer ID that is associated with this label.
        "color",  # The color of this label
    ],
)

labels = [
    #       name, id, color
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
    # Label("moving-car", 252, (245, 150, 100)),
    # Label("moving-bicyclist", 253, ()),
    # Label("moving-person", 254, (30, 30, 25)),
    # Label("moving-motorcyclist", 255, (90, 30, 150)),
    # Label("moving-on-rails", 256, ()),
    # Label("moving-bus", 257, ()),
    # Label("moving-truck", 258, ()),
    # Label("moving-other-vehicle", 259, ()),
]

# # a label and all meta information
# Label = namedtuple(
#     "Label",
#     [
#         "name",  # The identifier of this label, e.g. 'car', 'person', ... .
#         # We use them to uniquely name a class
#         "id",  # An integer ID that is associated with this label.
#         # The IDs are used to represent the label in ground truth images
#         # An ID of -1 means that this label does not have an ID and thus
#         # is ignored when creating ground truth images (e.g. license plate).
#         # Do not modify these IDs, since exactly these IDs are expected by the
#         # evaluation server.
#         "kittiId",  # An integer ID that is associated with this label for KITTI-360
#         # NOT FOR RELEASING
#         "trainId",  # Feel free to modify these IDs as suitable for your method. Then create
#         # ground truth images with train IDs, using the tools provided in the
#         # 'preparation' folder. However, make sure to validate or submit results
#         # to our evaluation server using the regular IDs above!
#         # For trainIds, multiple labels might have the same ID. Then, these labels
#         # are mapped to the same class in the ground truth images. For the inverse
#         # mapping, we use the label that is defined first in the list below.
#         # For example, mapping all void-type classes to the same ID in training,
#         # might make sense for some approaches.
#         # Max value is 255!
#         "category",  # The name of the category that this label belongs to
#         "categoryId",  # The ID of this category. Used to create ground truth images
#         # on category level.
#         "hasInstances",  # Whether this label distinguishes between single instances or not
#         "ignoreInEval",  # Whether pixels having this class as ground truth label are ignored
#         # during evaluations or not
#         "ignoreInInst",  # Whether pixels having this class as ground truth label are ignored
#         # during evaluations of instance segmentation or not
#         "color",  # The color of this label
#     ],
# )


# # --------------------------------------------------------------------------------
# # A list of all labels
# # --------------------------------------------------------------------------------

# # Please adapt the train IDs as appropriate for your approach.
# # Note that you might want to ignore labels with ID 255 during training.
# # Further note that the current train IDs are only a suggestion. You can use whatever you like.
# # Make sure to provide your results using the original IDs and not the training IDs.
# # Note that many IDs are ignored in evaluation and thus you never need to predict these!
# labels = [
#     #       name                     id    kittiId,    trainId   category            catId     hasInstances   ignoreInEval   ignoreInInst   color
#     Label("unlabeled", 0, -1, 255, "void", 0, False, True, True, (0, 0, 0)),
#     Label("ego vehicle", 1, -1, 255, "void", 0, False, True, True, (0, 0, 0)),
#     Label("rectification border", 2, -1, 255, "void", 0, False, True, True, (0, 0, 0)),
#     Label("out of roi", 3, -1, 255, "void", 0, False, True, True, (0, 0, 0)),
#     Label("static", 4, -1, 255, "void", 0, False, True, True, (0, 0, 0)),
#     Label("dynamic", 5, -1, 255, "void", 0, False, True, True, (111, 74, 0)),
#     Label("ground", 6, -1, 255, "void", 0, False, True, True, (81, 0, 81)),
#     Label("road", 7, 1, 0, "flat", 1, False, False, False, (128, 64, 128)),
#     Label("sidewalk", 8, 3, 1, "flat", 1, False, False, False, (244, 35, 232)),
#     Label("parking", 9, 2, 255, "flat", 1, False, True, True, (250, 170, 160)),
#     Label("rail track", 10, 10, 255, "flat", 1, False, True, True, (230, 150, 140)),
#     Label(
#         "building", 11, 11, 2, "construction", 2, True, False, False, (0, 100, 0)
#     ),  # ( 70, 70, 70) ),
#     Label("wall", 12, 7, 3, "construction", 2, False, False, False, (102, 102, 156)),
#     Label("fence", 13, 8, 4, "construction", 2, False, False, False, (190, 153, 153)),
#     Label(
#         "guard rail", 14, 30, 255, "construction", 2, False, True, True, (180, 165, 180)
#     ),
#     Label("bridge", 15, 31, 255, "construction", 2, False, True, True, (150, 100, 100)),
#     Label("tunnel", 16, 32, 255, "construction", 2, False, True, True, (150, 120, 90)),
#     Label("pole", 17, 21, 5, "object", 3, True, False, True, (153, 153, 153)),
#     Label("polegroup", 18, -1, 255, "object", 3, False, True, True, (153, 153, 153)),
#     Label("traffic light", 19, 23, 6, "object", 3, True, False, True, (250, 170, 30)),
#     Label("traffic sign", 20, 24, 7, "object", 3, True, False, True, (220, 220, 0)),
#     Label("vegetation", 21, 5, 8, "nature", 4, False, False, False, (107, 142, 35)),
#     Label("terrain", 22, 4, 9, "nature", 4, False, False, False, (152, 251, 152)),
#     Label("sky", 23, 9, 10, "sky", 5, False, False, False, (70, 130, 180)),
#     Label("person", 24, 19, 11, "human", 6, True, False, False, (220, 20, 60)),
#     Label("rider", 25, 20, 12, "human", 6, True, False, False, (255, 0, 0)),
#     Label("car", 26, 13, 13, "vehicle", 7, True, False, False, (0, 0, 142)),
#     Label("truck", 27, 14, 14, "vehicle", 7, True, False, False, (0, 0, 70)),
#     Label("bus", 28, 34, 15, "vehicle", 7, True, False, False, (0, 60, 100)),
#     Label("caravan", 29, 16, 255, "vehicle", 7, True, True, True, (0, 0, 90)),
#     Label("trailer", 30, 15, 255, "vehicle", 7, True, True, True, (0, 0, 110)),
#     Label("train", 31, 33, 16, "vehicle", 7, True, False, False, (0, 80, 100)),
#     Label("motorcycle", 32, 17, 17, "vehicle", 7, True, False, False, (0, 0, 230)),
#     Label("bicycle", 33, 18, 18, "vehicle", 7, True, False, False, (119, 11, 32)),
#     Label("garage", 34, 12, 2, "construction", 2, True, True, True, (64, 128, 128)),
#     Label("gate", 35, 6, 4, "construction", 2, False, True, True, (190, 153, 153)),
#     Label("stop", 36, 29, 255, "construction", 2, True, True, True, (150, 120, 90)),
#     Label("smallpole", 37, 22, 5, "object", 3, True, True, True, (153, 153, 153)),
#     Label("lamp", 38, 25, 255, "object", 3, True, True, True, (0, 64, 64)),
#     Label("trash bin", 39, 26, 255, "object", 3, True, True, True, (0, 128, 192)),
#     Label("vending machine", 40, 27, 255, "object", 3, True, True, True, (128, 64, 0)),
#     Label("box", 41, 28, 255, "object", 3, True, True, True, (64, 64, 128)),
#     Label(
#         "unknown construction", 42, 35, 255, "void", 0, False, True, True, (102, 0, 0)
#     ),
#     Label("unknown vehicle", 43, 36, 255, "void", 0, False, True, True, (51, 0, 51)),
#     Label("unknown object", 44, 37, 255, "void", 0, False, True, True, (32, 32, 32)),
#     Label("license plate", -1, -1, -1, "vehicle", 7, False, True, True, (0, 0, 142)),
#     Label("OSM BUILDING", 45, 3, 2, "flat", 7, False, True, True, (0, 0, 255)),
#     Label("OSM ROAD", 46, 21, 3, "flat", 7, False, False, False, (255, 0, 0)),
# ]
