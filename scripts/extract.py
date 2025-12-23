#!/usr/bin/env python3

# External
import os
import numpy as np
from multiprocessing import Pool
import yaml
from tqdm import tqdm
import sys

# Internal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lidar2osm.datasets.kitti360_dataset import KITTI360_DATASET
from lidar2osm.datasets.cu_multi_dataset import CU_MULTI_DATASET
from lidar2osm.core.pointcloud import transform_points_lat_lon, downsample_pointcloud
from lidar2osm.core.openstreetmap import (
    OSMItemList, 
    get_osm_labels,
    set_osm_buildings_lines,
    set_osm_roads_lines,
)
# from core.pcd import convert_pc_to_o3d, convert_polyline_points_to_o3d


class ExtractData:
    def __init__(self, dataset=None):
        self.dataset = dataset
        self.near_path_threshold_latlon = 0.002             # was 0.001
        self.within_scan_threshold_latlon_build = 0.00001   # 0.00002
        self.within_scan_threshold_latlon_road = 0.000005   # 0.000005

    # TODO: Move to correct file (osm.py)
    def relabel_points(self, osm_item_list, associated_osm_sem_labels, frame_num, semantic_3d_points, threshold, origin_latlon):
        """
        Relabels 3D points by projecting them into latitude-longitude coordinates and assigning labels
        based on the overlap with OpenStreetMap (OSM) items of a specified type (e.g., 'building' or 'road').
        """

        if len(semantic_3d_points) > 0:
            # Ref frame needs to be in 'latlon' here, since osm_item_list is in lat-lon
            semantic_3d_points_trans = transform_points_lat_lon(semantic_3d_points, 
                                                                frame_num, 
                                                                origin_latlon,
                                                                self.dataset.lidar_poses)

            osm_labels = get_osm_labels(osm_item_list, 
                                        associated_osm_sem_labels,
                                        semantic_3d_points_trans, 
                                        threshold)
        return osm_labels

    def save_semantic_labels(self, semantic_labels, file_path):
        semantic_labels = np.int32(semantic_labels)
        semantic_labels.tofile(file_path)
                
    def extract_osm_data_for_pointcloud(self, frame_num):
        pc_label_path, pc_osm_label_path, raw_pc_path = self.dataset.get_data_paths(frame_num)

        self.buildings_list = OSMItemList()
        self.roads_list = OSMItemList()

        # Make sure full pc label file exists and osm-label file not yet created
        if os.path.exists(pc_label_path) and not os.path.exists(pc_osm_label_path):
            points_np = self.dataset.get_pointcloud(raw_pc_path)
            labels_np = self.dataset.get_labels(pc_label_path)

            # Get points and downsample here
            frame_points_o3d_DS, downsampled_intensities, downsampled_labels = downsample_pointcloud(points_np, 
                                                                                                     labels_np, 
                                                                                                     self.dataset.labels_dict)

            if len(frame_points_o3d_DS.points) > 0:
                # o3d.visualization.draw_geometries([frame_points_o3d_DS])

                frame_points = np.copy(frame_points_o3d_DS.points)
                frame_points_semantic = np.concatenate(
                    (frame_points, downsampled_intensities, downsampled_labels), axis=1
                )

                self.show_colored_pc(frame_points_semantic, self.dataset.labels_dict)
                
                # Extract OSM Building points
                set_osm_buildings_lines(
                    self.buildings_list,
                    self.dataset.osm_file_path,
                    self.dataset.lidar_poses_latlon.get(frame_num),
                    self.near_path_threshold_latlon,
                    building_tags=self.dataset.building_tags,
                )
                frame_points_semantic[:, 4] = self.relabel_points(
                    self.buildings_list,
                    self.dataset.build_semantic_ids,
                    frame_num,
                    frame_points_semantic,
                    self.within_scan_threshold_latlon_build,
                    self.dataset.origin_latlon,
                )

                # Extract OSM Road points
                set_osm_roads_lines(
                    self.roads_list,
                    self.dataset.osm_file_path,
                    self.dataset.lidar_poses_latlon.get(frame_num),
                    self.near_path_threshold_latlon,
                    road_tags=self.dataset.road_tags,
                )
                frame_points_semantic[:, 4] = self.relabel_points(
                    self.roads_list,
                    self.dataset.road_semantic_ids,
                    frame_num,
                    frame_points_semantic,
                    self.within_scan_threshold_latlon_road,
                    self.dataset.origin_latlon,
                )
                
                # Finally, save labels after extracting new labels
                self.save_semantic_labels(semantic_labels=frame_points_semantic[:, 4],
                                          file_path=pc_osm_label_path)


def extract_osm_data(seq_extract, use_multiprocessing=True):
    frames = range(seq_extract.dataset.init_frame, seq_extract.dataset.fin_frame, 1)

    if use_multiprocessing:
        with Pool() as pool:
            list(
                tqdm(
                    pool.imap(
                        seq_extract.extract_osm_data_for_pointcloud, frames
                    ),
                    total=len(frames),
                    desc=f"      sequence {seq_extract.dataset.seq}",
                )
            )
    else:
        for frame in tqdm(frames, total=len(frames), desc=f"      sequence {seq_extract.dataset.seq}"):
            seq_extract.extract_osm_data_for_pointcloud(frame)


def load_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)

def load_KITTI360(dataset_config):

    dataset = KITTI360_DATASET(root_path = dataset_config["root_path"])

    print(f"\nExtracting OSM data from KITTI-360 Dataset.")

    for sequence in dataset_config["sequences"]:
        dataset.setup_sequence(sequence)
        seq_extract = ExtractData(dataset=dataset)
        extract_osm_data(
            seq_extract, 
            use_multiprocessing = dataset_config["use_multiprocessing"]
        )
            

def load_CU_MULTI(dataset_config):
    dataset = CU_MULTI_DATASET(root_path = dataset_config["root_path"])

    print(f"\nExtracting OSM data from CU-MULTI Dataset.")

    for environment in dataset_config["environments"]:
        for robot in dataset_config["robots"]:
            
            dataset.setup_robot_data(environment, robot)
            # seq_extract = ExtractData(dataset=dataset)
            # extract_osm_data(
            #     seq_extract, 
            #     use_multiprocessing = dataset_config["use_multiprocessing"]
            # )

def main():
    config_file = "lidar2osm/config/extraction_config.yaml"
    config = load_config(config_file)

    datasets = config["datasets"]
    for dataset_name in datasets:
        dataset_config = datasets[dataset_name]
        if dataset_config["enabled"]:
            if dataset_name == "KITTI_360":
                load_KITTI360(dataset_config)
            elif dataset_name == "CU_MULTI":
                load_CU_MULTI(dataset_config)

if __name__ == "__main__":
    main()
