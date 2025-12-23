import json
import os
import numpy as np
import open3d as o3d
import re
from scipy.spatial.transform import Rotation as R
from collections import defaultdict

classname_to_color = {                                          # RGB.
    "noise": (0, 0, 0),                                         # Black.
    "animal": (70, 130, 180),                                   # Steelblue
    "human.pedestrian.adult": (0, 0, 230),                      # Blue
    "human.pedestrian.child": (135, 206, 235),                  # Skyblue,
    "human.pedestrian.construction_worker": (100, 149, 237),    # Cornflowerblue
    "human.pedestrian.personal_mobility": (219, 112, 147),      # Palevioletred
    "human.pedestrian.police_officer": (0, 0, 128),             # Navy,
    "human.pedestrian.stroller": (240, 128, 128),               # Lightcoral
    "human.pedestrian.wheelchair": (138, 43, 226),              # Blueviolet
    "movable_object.barrier": (112, 128, 144),                  # Slategrey
    "movable_object.debris": (210, 105, 30),                    # Chocolate
    "movable_object.pushable_pullable": (105, 105, 105),        # Dimgrey
    "movable_object.trafficcone": (47, 79, 79),                 # Darkslategrey
    "static_object.bicycle_rack": (188, 143, 143),              # Rosybrown
    "vehicle.bicycle": (220, 20, 60),                           # Crimson
    "vehicle.bus.bendy": (255, 127, 80),                        # Coral
    "vehicle.bus.rigid": (255, 69, 0),                          # Orangered
    "vehicle.car": (255, 158, 0),                               # Orange
    "vehicle.construction": (233, 150, 70),                     # Darksalmon
    "vehicle.emergency.ambulance": (255, 83, 0),
    "vehicle.emergency.police": (255, 215, 0),                  # Gold
    "vehicle.motorcycle": (255, 61, 99),                        # Red
    "vehicle.trailer": (255, 140, 0),                           # Darkorange
    "vehicle.truck": (255, 99, 71),                             # Tomato
    "flat.driveable_surface": (0, 207, 191),                    # nuTonomy green
    "flat.other": (175, 0, 75),
    "flat.sidewalk": (75, 0, 75),
    "flat.terrain": (112, 180, 60),
    "static.manmade": (222, 184, 135),                          # Burlywood
    "static.other": (255, 228, 196),                            # Bisque
    "static.vegetation": (0, 175, 0),                           # Green
    "vehicle.ego": (255, 240, 245)
}

def quat_to_matrix(quat):
    '''
    Converts quaternion-based rotation to 3x3 rot matrix.
    '''
    def reorder_quaternion(nuscenes_quat):
        '''
        Util function for reordering nuscenes quat from [w,x,y,z] to [x,y,z,w].
        '''
        reordered_quat = [nuscenes_quat[1], nuscenes_quat[2], nuscenes_quat[3], nuscenes_quat[0]]
        return reordered_quat
    
    fixed_quat = reorder_quaternion(quat)
    return R.from_quat(fixed_quat).as_matrix()


class NuScenesVis:
    def __init__(self, nuscenes_data_path):
        self.nuscenes_data_path = nuscenes_data_path
        self.lidar_data_path = os.path.join(nuscenes_data_path, "samples/LIDAR_TOP")

    def visualize(self, train_val_sequence, max_scene_num):
        self.set_data_dicts(train_val_sequence)
        self.display_scene_maps(max_scene_num)
            
    def set_data_dicts(self, train_val_sequence):
        json_dir_path = os.path.join(nuscenes_data_path, train_val_sequence)

        def json_to_dict(json_file, key_to_order_by):
            json_file_path = os.path.join(json_dir_path, json_file)
            print(f"\n{json_file_path}\n")

            with open(json_file_path, 'r') as file:
                data_list = json.load(file)

            data_dict = {item[f'{key_to_order_by}']: item for item in data_list}
            return data_dict
    
        # NuScenes Dataset Scenes
        self.scene_dict = json_to_dict("scene.json", 'name')

        self.sensor_dict = json_to_dict("sensor.json", 'modality')

        self.sample_dict = json_to_dict("sample.json", 'token')

        self.sample_data_dict = json_to_dict("sample_data.json", 'token')

        self.lidar_seg_dict = json_to_dict("lidarseg.json", 'sample_data_token')

        self.category_dict = json_to_dict("category.json", 'index')

        self.ego_pose_dict = json_to_dict("ego_pose.json", 'token')

        self.cal_sensor_dict = json_to_dict("calibrated_sensor.json", 'token')


    def display_scene_lidar_map(self, first_sample_data_token):
        next_sample_data_token = first_sample_data_token

        while next_sample_data_token is not None:
            # print(f"next_sample_data_token: {next_sample_data_token}")
            sample_data = self.sample_data_dict[next_sample_data_token]

            if next_sample_data_token not in self.lidar_seg_dict:
                # print("NOT IN LIDAR SEG")
                if sample_data['next'] == "":
                    next_sample_data_token = None
                else:
                    next_sample_data_token = sample_data['next']
                continue
            lidar_seg_labels_filename = self.lidar_seg_dict[f'{next_sample_data_token}']['filename']
            lidar_seg_labels_path = os.path.join(self.nuscenes_data_path, lidar_seg_labels_filename)

            lidar_seg_labels = np.fromfile(lidar_seg_labels_path, dtype=np.uint8)
            lidar_seg_colors = []
            for lidar_label_index in lidar_seg_labels:
                label_name = self.category_dict[lidar_label_index]['name']
                label_color = classname_to_color[f'{label_name}']
                lidar_seg_colors.append(label_color)

            lidar_seg_colors = np.asarray(lidar_seg_colors)

            cal_sensor_token = sample_data['calibrated_sensor_token']
            cal_pose_trans = self.cal_sensor_dict[cal_sensor_token]['translation']
            cal_pose_rot = self.cal_sensor_dict[cal_sensor_token]['rotation']
            cal_rotation = quat_to_matrix(cal_pose_rot)

            ego_pose_data_token = sample_data['ego_pose_token']
            ego_pose_trans = self.ego_pose_dict[ego_pose_data_token]['translation']
            ego_pose_rot = self.ego_pose_dict[ego_pose_data_token]['rotation']
            ego_rotation = quat_to_matrix(ego_pose_rot)

            lidar_filepath = sample_data['filename']
            lidar_bin_path = os.path.join(self.nuscenes_data_path, lidar_filepath)
            lidar_data = np.fromfile(lidar_bin_path, dtype=np.float32)
            lidar_data = lidar_data.reshape(-1, 5)

            # Separate the point coordinates (x, y, z)
            points = lidar_data[:, :3]
            
            # TF with Cal sensor first
            rotated_points1 = np.dot(cal_rotation, points.T)
            transformed_points1 = rotated_points1.T + cal_pose_trans
            
            # then TF with Ego
            rotated_points2 = np.dot(ego_rotation, transformed_points1.T)
            transformed_points2 = rotated_points2.T + ego_pose_trans

            self.points_accum = np.concatenate((self.points_accum, transformed_points2), axis=0)

            # Optional: visualize intensity by setting point colors (normalize intensity values to [0, 1])
            intensities = lidar_data[:, 4]
            normalized_intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())
            intensity_colors = np.tile(normalized_intensities.reshape(-1, 1), (1, 3))  # Grayscale color based on intensity

            colors = intensity_colors * lidar_seg_colors / 255.0
            self.colors_accum = np.concatenate((self.colors_accum, colors), axis=0)
            
            if sample_data['next'] == "":
                next_sample_data_token = None
            else:
                next_sample_data_token = sample_data['next']


    def get_first_sample_data_token(self, first_sample_token):
        first_sample_data_token = None
        for sample_data_token in self.sample_data_dict.keys():
            sample_data = self.sample_data_dict[sample_data_token]

            cal_sensor_token = sample_data['calibrated_sensor_token']
            is_lidar = self.cal_sensor_dict[cal_sensor_token]['sensor_token'] == self.sensor_dict['lidar']['token']

            if sample_data['sample_token'] == first_sample_token and is_lidar:
                first_sample_data_token = sample_data_token
                break
        return first_sample_data_token
    
    def display_scene_maps(self, max_scene_num):
        self.points_accum = np.empty((0, 3))
        self.colors_accum = np.empty((0, 3))
        for scene_name in self.scene_dict.keys():
            first_sample_token = self.scene_dict[scene_name]['first_sample_token']
            first_sample_data_token = self.get_first_sample_data_token(first_sample_token)

            if first_sample_data_token:
                print(f"scene_name: {scene_name}, first_sample_data_token: {first_sample_data_token}")
                self.display_scene_lidar_map(first_sample_data_token)

            # Open3D point cloud object
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(self.points_accum)
            point_cloud.colors = o3d.utility.Vector3dVector(self.colors_accum)

            # Downsample for displaying
            downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size=1.0)
            self.points_accum = downsampled_point_cloud.points
            self.colors_accum = downsampled_point_cloud.colors
            
            # Break if max scene has been accumulated into points & labels
            if scene_name == f"scene-{max_scene_num:04d}":
                break

        # Visualize point cloud
        o3d.visualization.draw_geometries([downsampled_point_cloud])

nuscenes_data_path = "/media/donceykong/donceys_data_ssd/datasets/NUSCENES"
viz = NuScenesVis(nuscenes_data_path)

train_val_sequence = "v1.0-trainval"
max_scene_num = 20 #200
viz.visualize(train_val_sequence, max_scene_num)