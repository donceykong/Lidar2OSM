#!/usr/bin/env python3

# External
import os
import time
import open3d as o3d
import numpy as np
import copy
import re
from typing import Tuple
from tqdm import tqdm
import yaml
import osmnx as ox
from scipy.spatial.transform import Rotation as R, Slerp
from pathlib import Path

import multiprocessing as mp
from itertools import combinations, product
# import logging
import json
import time

import matplotlib.pyplot as plt
from collections import Counter

# Internal
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import clipperpy
# from lidar2osm.dataset.labels_kitti360 import labels as GT_labels
from lidar2osm.datasets.cu_multi_dataset import labels as sem_kitti_labels
from lidar2osm.utils.file_io import read_bin_file
from lidar2osm.core.pointcloud import labels2RGB
from lidar2osm.core.projection import *


class Robot:
    def __init__(self, robot_name, env_name, root_data_path):
        self.robot_name = robot_name
        self.env_name = env_name
        self.root_data_dir = Path(root_data_path) / self.env_name / self.robot_name
        self.set_paths()
        self.set_data()
        self.set_init_gps()
        
    def set_paths(self):
        lidar_dir = self.root_data_dir / "lidar"
        gps_dir = self.root_data_dir / "gps"

        self.paths = {
            "lidar_data": lidar_dir / "pointclouds",
            "lidar_ts": lidar_dir / "timestamps.txt",
            "label": lidar_dir / "labels/gt_osm_labels",
            "lidar_poses": lidar_dir / "poses_world.txt",
            "pose_ts": lidar_dir / "timestamps.txt",
            "gps_1": gps_dir / "gnss_1_data.txt",
            "gps_1_ts": gps_dir / "gnss_1_timestamps.txt",
        }

    def make_data_index_ts_dict(self, ts_list):
        file_path_dict = {}

        for instance_id, timestamp in enumerate(ts_list):
            file_path_dict[timestamp] = instance_id
        return file_path_dict

    def set_data(self):
        # Read timestamps efficiently using np.loadtxt
        self.lidar_timestamps = np.loadtxt(self.paths["lidar_ts"], dtype=np.float128)
        self.gnss1_timestamps = np.loadtxt(self.paths["gps_1_ts"], dtype=np.float128)

        # Load lidar poses
        self.lidar_poses = read_quat_poses(self.paths["lidar_poses"], self.paths["pose_ts"])
        self.sorted_timestamps = np.array(sorted(self.lidar_poses.keys()))  # Precompute sorted timestamps
        self.lidar_scan_idx_dict =  self.make_data_index_ts_dict(self.lidar_timestamps)
        self.labelled_frames = get_frame_numbers(self.paths["label"])

        self.first_lidar_frame, self.last_lidar_frame = 0, len(self.labelled_frames) - 1

    def get_nearest_timestamp(self, target_timestamp):
        """Use binary search to find the closest timestamp."""
        idx = np.searchsorted(self.sorted_timestamps, target_timestamp)
        
        # Find nearest by checking both sides
        if idx == 0:
            return self.sorted_timestamps[0]
        elif idx == len(self.sorted_timestamps):
            return self.sorted_timestamps[-1]
        else:
            before = self.sorted_timestamps[idx - 1]
            after = self.sorted_timestamps[idx]
            return before if abs(before - target_timestamp) < abs(after - target_timestamp) else after

    # def get_nearest_timestamp(self, target_timestamp):
    #     # Optimized nearest search using `min()`
    #     nearest_time = min(self.lidar_poses.keys(), key=lambda ts: abs(ts - target_timestamp))
    #     return nearest_time
    
    def set_init_gps(self):
        # Get initial GPS Point
        gps_1_points = read_gps_data(self.paths["gps_1"])
        gps_1_points[:, 2] = 0
        lidar_timestamp = self.lidar_timestamps[0]
        # Find the interval for interpolation
        idx = np.searchsorted(self.gnss1_timestamps, lidar_timestamp) - 1
        t0, t1 = self.gnss1_timestamps[idx], self.gnss1_timestamps[idx + 1]
        p0, p1 = gps_1_points[idx], gps_1_points[idx + 1]
        ratio = (lidar_timestamp - t0) / (t1 - t0)
        interp_gps_position = (1 - ratio) * p0 + ratio * p1
        self.init_gps_point = interp_gps_position

    def get_accum_pc(self, init_frame_index, end_frame_index, sem_type, frame_index_inc, use_latlon=False):
        semantic_points_np, tf_array = get_accum_points(
            first_frame_idx=init_frame_index,
            final_frame_idx=end_frame_index,
            frame_increment=frame_index_inc,
            osm_frames=self.labelled_frames,
            lidar_data_path=self.paths["lidar_data"],
            lidar_timestamps=self.lidar_timestamps,
            labels_path=self.paths["label"],
            lidar_poses=self.lidar_poses,
            init_gps_point = self.init_gps_point,
            gps_1_path=self.paths["gps_1"],
            gnss1_timestamps=self.gnss1_timestamps,
            semantics_type=sem_type,
            use_latlon=use_latlon,
        )

        return semantic_points_np, tf_array


def quaternion_pose_to_4x4(trans, quat):
    rotation_matrix = R.from_quat(quat).as_matrix()

    # Create the 4x4 transformation matrix
    transformation_matrix = np.eye(4)  # Start with an identity matrix
    transformation_matrix[:3, :3] = rotation_matrix  # Set the rotation part
    transformation_matrix[:3, 3] = trans  # Set the translation part

    return transformation_matrix


def pose4x4_to_quat(matrix):
    position = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    quaternion = R.from_matrix(rotation_matrix).as_quat()
    return position, quaternion


def read_quat_poses(poses_path, pose_ts_path):
    # Read timestamps into a list
    with open(pose_ts_path, "r") as ts_file:
        timestamps = [line.strip() for line in ts_file]

    poses_xyz = {}
    with open(poses_path, "r") as file:
        for idx, line in enumerate(file):
            elements = line.strip().split()
            trans = np.array(elements[:3], dtype=float)
            quat = np.array(elements[3:], dtype=float)

            if len(trans) > 0:
                transformation_matrix = quaternion_pose_to_4x4(trans, quat)
                timestamp = np.float128(timestamps[idx])
                poses_xyz[timestamp] = transformation_matrix
            else:
                timestamp = np.float128(timestamps[idx])
                poses_xyz[timestamp] = np.eye(4)
    return poses_xyz


def interpolate_pose(velodyne_poses, target_timestamp):
    print(f"target_timestamp: {target_timestamp}")
    timestamps = [timestamp for timestamp in velodyne_poses.keys()]
    positions = [pose4x4_to_quat(pose)[0] for pose in velodyne_poses.values()]
    quaternions = [pose4x4_to_quat(pose)[1] for pose in velodyne_poses.values()]

    # Find the interval for interpolation
    idx = np.searchsorted(timestamps, target_timestamp) - 1
    print(f"target_timestamp: {target_timestamp}, idx: {idx}")
    t0, t1 = timestamps[idx], timestamps[idx + 1]
    p0, p1 = positions[idx], positions[idx + 1]
    q0, q1 = quaternions[idx], quaternions[idx + 1]

    target_timestamp = np.float128(target_timestamp)
    t0 = np.double(t0)
    t1 = np.double(t1)
    p0 = np.double(p0)
    p1 = np.double(p1)
    q0 = np.double(q0)
    q1 = np.double(q1)

    # Perform linear interpolation for position
    ratio = (target_timestamp - t0) / (t1 - t0)
    interp_position = (1 - ratio) * p0 + ratio * p1

    # Perform SLERP for orientation
    rotations = R.from_quat([q0, q1])
    # print(f"t0: {t0}, t1: {t1}")
    slerp = Slerp([t0, t1], rotations)
    interp_orientation = slerp(target_timestamp).as_quat()

    return interp_position, interp_orientation


def random_downsample_pc(semantic_pc, max_points):
    """Downsamples a sematic pointcloud and shifts the points to origin.
    Args:
        semantic_pc (numpy.ndarray): Nx4 array of point positions and semantic IDs.
        max_points (float): number of indices from semantic pc to randomly sample.

    Returns:
        semantic_pc_ds (ndarray): 
            This array is the downsampled points and labels of the semantically-labelled pc.
    """

    downsampled_indices = np.random.choice(len(semantic_pc), max_points, replace=False)
    semantic_pc_ds = np.asarray(semantic_pc)[downsampled_indices, :]

    return semantic_pc_ds


def get_frame_numbers(directory_path) -> list:
    """
    Count the total number of files in the directory
    """
    frame_numbers = []
    all_files = os.listdir(directory_path)

    # Filter out files ending with ".bin" and remove the filetype
    filenames = [
        int(re.search(r'\d+', os.path.splitext(file)[0]).group())
        for file in all_files if file.endswith(".bin") and re.search(r'\d+', os.path.splitext(file)[0])
    ]

    for filename in filenames:
        frame_numbers.append(int(filename))

    return sorted(frame_numbers)


def get_err(tf_ground_truth, tf_estimate) -> Tuple[float, float]:
    """
    Returns the error of the estimated transformation matrix.
    """

    Terr = np.linalg.inv(tf_ground_truth) @ tf_estimate
    rerr = abs(np.arccos(min(max(((Terr[0:3, 0:3]).trace() - 1) / 2, -1.0), 1.0)))
    terr = np.linalg.norm(Terr[0:3, 3])

    return (rerr, terr)


def downsample_association_matrix(A, max_size_A=10000, corr_labels=None) -> None:    
    # randomly downsample A indices
    rand_ds_A_idxs = np.random.choice(A.shape[0], size=max_size_A, replace=False)

    # Downsample correlation labels too if passed in
    if corr_labels is not None:
        corr_labels = corr_labels[rand_ds_A_idxs]
    
    return A[rand_ds_A_idxs], corr_labels


def get_a2a_assoc_matrix(N1, N2):
   assoc_matrix = np.zeros((N1*N2,2),np.int32)
 
   i = 0
   for n1 in range(N1):
       for n2 in range(N2):
           assoc_matrix[i,0] = n1
           assoc_matrix[i,1] = n2
           i += 1
 
   return assoc_matrix


def filter_by_ego_distance(pc1, pc2, A, max_ego_dist):
    """Filter pairs in A based on ego distance using vectorized operations."""

    # Extract indices from A
    indices1 = A[:, 0]
    indices2 = A[:, 1]

    # Compute ego distances for all points in A using the indices
    ego_dist1 = np.linalg.norm(pc1[indices1], axis=1)
    ego_dist2 = np.linalg.norm(pc2[indices2], axis=1)

    # Compute the relative ego distances
    relative_ego_dist = np.abs(ego_dist1 - ego_dist2)

    # Filter entries where relative ego distance is within the threshold
    mask = relative_ego_dist <= max_ego_dist
    Anew = A[mask]

    return Anew


def random_downsample_pc(semantic_pc, max_points):
    """Downsamples a sematic pointcloud.
    Args:
        semantic_pc (numpy.ndarray): Nx4 array of point positions and semantic IDs.
        max_points (float): number of indices from semantic pc to randomly sample.

    Returns:
        semantic_pc_ds (ndarray): 
            This array is the downsampled points and labels of the semantically-labelled pc.
    """

    downsampled_indices = np.random.choice(len(semantic_pc), max_points, replace=False)
    semantic_pc_ds = np.asarray(semantic_pc)[downsampled_indices, :]

    return semantic_pc_ds


def voxel_downsample_pc(semantic_pc, voxel_leaf_size):
    sem_labels_3x3 = np.stack([semantic_pc[:, 3]] * 3, axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(semantic_pc[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(sem_labels_3x3)
    
    pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_leaf_size)

    points_ds = np.asarray(pcd_downsampled.points)
    labels_ds = np.asarray(pcd_downsampled.colors)[:, 0].reshape(-1, 1)
    
    semantic_pc_ds = np.concatenate((points_ds, labels_ds), axis=1)

    return semantic_pc_ds


def transform_semantic_pts(sem_pc, pose, inverse=False):
    ''' TF points by 'pose' variable. If inverse is true, tf is done by the inverse of 'pose'
    '''
    if inverse:
        pose = np.linalg.inv(pose)
    homogeneous_points = np.hstack((sem_pc[:, :3], np.ones((sem_pc[:, :3].shape[0], 1))))
    transformed_points = homogeneous_points @ pose.T
    sem_pc[:, :3] = transformed_points[:, :3]
    return sem_pc


def umeyama_alignment(target_pts, source_pts):
    """
    Computes the optimal rigid transformation using Umeyama's method.

    :param source: (N, 3) array of source points
    :param target: (N, 3) array of corresponding target points
    :return: 4x4 transformation matrix
    """
    assert target_pts.shape == source_pts.shape

    # Compute centroids
    target_mean = np.mean(target_pts, axis=0)
    source_mean = np.mean(source_pts, axis=0)

    # Center the points
    target_centered = target_pts - target_mean
    source_centered = source_pts - source_mean

    # Compute covariance matrix
    H = target_centered.T @ source_centered
    # H = np.dot(target_centered.T, source_centered)

    # Perform SVD
    U, _, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = Vt.T @ U.T
    # R = np.dot(Vt.T, U.T)

    # Ensure proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    # if np.linalg.det(R) < 0:
    #     Vt[-1, :] *= -1
    #     R = np.dot(Vt.T, U.T)

    # Compute translation
    t = source_mean - R @ target_mean
    # t = source_mean - np.dot(R, target_mean)

    # Construct transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

import copy
def compute_transformation_point_to_point(target, source, corres):
    """
    Computes the best-fit rigid transformation (rotation + translation)
    that aligns the source point cloud to the target given correspondences.
    """
    # Convert to NumPy arrays
    target_points = np.asarray(target.points)
    source_points = np.asarray(source.points)
    
    # Extract corresponding points
    target_corr = target_points[np.asarray(corres)[:, 0]]
    source_corr = source_points[np.asarray(corres)[:, 1]]

    # Calculate alignment using
    T = umeyama_alignment(target_corr, source_corr)

    # # Run ICP
    # source_t = copy.deepcopy(source)
    # source_t.transform(np.linalg.inv(T))   # Apply transformation to source
    # o3d.visualization.draw_geometries([target, source_t], window_name="Random Point Cloud", width=800, height=600)

    # threshold = 0.5        # Distance threshold for correspondence
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     source_t, target, threshold,
    #     np.eye(4),
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint()
    # )
    # T = reg_p2p.transformation

    # result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
    #     source, target,
    #     o3d.pipelines.registration.Feature(),
    #     o3d.pipelines.registration.Feature(),
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint()
    # )
    # T = result.transformation

    return T


def calculate_tf_estimate(A_sel, target_sem_pc, source_sem_pc, labels_dict):
    # Unaligned Source Cloud PCD
    labels_rgb_ds = labels2RGB(target_sem_pc[:, 3], labels_dict)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_sem_pc[:, :3])
    target_pcd.colors = o3d.utility.Vector3dVector(labels_rgb_ds)
    source_labels_rgb = labels2RGB(source_sem_pc[:, 3], labels_dict)

    # Create source pcd
    source_labels_rgb[:, :] = 0 #+=  0.5     # Adjust source cloud colors and normalize bw 0:1 for open3D viewing
    source_labels_rgb = np.clip(source_labels_rgb, 0.0, 1.0)
    source_pcd_tf = o3d.geometry.PointCloud()
    source_pcd_tf.points = o3d.utility.Vector3dVector(source_sem_pc[:, :3])
    source_pcd_tf.colors = o3d.utility.Vector3dVector(source_labels_rgb)

    tf_est = compute_transformation_point_to_point(target_pcd, source_pcd_tf, A_sel)
    # print(f"tf_est: {tf_est}")

    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    # p2p.with_scaling = True
    tf_estimate = p2p.compute_transformation(target_pcd, source_pcd_tf, o3d.utility.Vector2iVector(A_sel))

    return tf_estimate, tf_est

# Append log entry
def log_result(log_file, test_params, test_results, contact_ts_pair):
    # Unpack logging information
    target_robot_eff_ts, source_robot_eff_ts = contact_ts_pair
    sem_type, prefilter_type, env, (target_robot_name, source_robot_name) = test_params
    number_A_in, number_A_sel, time_to_score, time_to_solve, trans_error, rot_error = test_results

    log_entry = {
        # "test_id": test_id,
        "sem_type": sem_type,
        "prefilter_type": prefilter_type,
        "env": env,
        "target_robot_name": target_robot_name,
        "target_robot_eff_ts": str(target_robot_eff_ts),
        "source_robot_name": source_robot_name,
        "source_robot_eff_ts": str(source_robot_eff_ts),
        "time_to_score": time_to_score,
        "time_to_solve": time_to_solve,
        "number_A_in": number_A_in,
        "number_A_sel": number_A_sel,
        "trans_error": trans_error,
        "rot_error": rot_error
    }
    
    with open(log_file, mode="a") as file:
        file.write(json.dumps(log_entry) + "\n")

def align_maps(target_sem_pc, target_tf_array, source_sem_pc, source_tf_array, test, contact_ts_pair, log_file):
    epsilon = 1.0
    max_A_size = 10000
    # max_points = 10000
    voxel_size = 0.1
    random_downsample = True
    voxel_downsample = True
    eff_comm_dist = 10 # Effective Communication Distance of BT recievers used

    sem_type, prefilter_type, env, (target_robot_name, source_robot_name) = test

    if voxel_downsample:
        print("Resetting epsilon to twice the voxel leaf size.")
        epsilon = voxel_size * 5

    # TODO: Move to dataset class
    labels_dict = {label.id: label.color for label in sem_kitti_labels}

    # Get list of labels as int32 elements
    target_labels_list = [np.int32(x) for x in target_sem_pc[:, 3]]
    source_labels_list = [np.int32(x) for x in source_sem_pc[:, 3]]

    # Get the transform needed to tf target to source
    target_final_pose = target_tf_array[-1]
    source_final_pose = source_tf_array[-1]
    relative_transform_t2s = np.linalg.inv(source_final_pose) @ target_final_pose

    # # Calc true relative distance of robots and use that as eff
    # true_ego_dist = np.linalg.norm(relative_transform_t2s[:3, 3])
    # eff_comm_dist = true_ego_dist * 2

    # Center PCS and TFs about current robot positions
    target_sem_pc = transform_semantic_pts(target_sem_pc, target_final_pose, inverse=True)
    target_tf_array = np.einsum('ij,njk->nik', np.linalg.inv(target_final_pose), target_tf_array)

    source_sem_pc = transform_semantic_pts(source_sem_pc, source_final_pose, inverse=True)
    source_tf_array = np.einsum('ij,njk->nik', np.linalg.inv(source_final_pose), source_tf_array)

    # ---- Generate dataset (ALL-TO-ALL) ---- #
    A_all_to_all = get_a2a_assoc_matrix(len(target_sem_pc), len(source_sem_pc))

    # Set up vanilla invariant function for vanilla CLIPPER
    iparams = clipperpy.invariants.EuclideanDistanceParams()
    # iparams.mindist = 0.01
    iparams.epsilon = epsilon
    iparams.sigma = 0.5 * iparams.epsilon
    invariant = clipperpy.invariants.EuclideanDistance(iparams)

    # Set up CLIPPER rounding parameters
    params = clipperpy.Params()
    params.rounding = clipperpy.Rounding.DSD_HEU
    clipper = clipperpy.CLIPPER(invariant, params)

    # Prefilter all-to-all associations using sensor-based and inference-based prior knowledge
    if prefilter_type=="all_prefiltering":
        A_filtered = filter_by_ego_distance(target_sem_pc[:, :3], source_sem_pc[:, :3], A_all_to_all, max_ego_dist=eff_comm_dist)
        A_filtered, corr_filtered_labels = clipper.filter_semantic_associations(target_labels_list, source_labels_list, A_filtered)
        corr_filtered_labels = np.array(corr_filtered_labels, dtype=np.int32)
        A_filtered, corr_filtered_labels = downsample_association_matrix(A_filtered, max_size_A=max_A_size, corr_labels=corr_filtered_labels)
    elif prefilter_type=="comms_aware_prefiltering":
        A_filtered = filter_by_ego_distance(target_sem_pc[:, :3], source_sem_pc[:, :3], A_all_to_all, max_ego_dist=eff_comm_dist)
        A_filtered, _ = downsample_association_matrix(A_filtered, max_size_A=max_A_size)
    elif prefilter_type=="semantic_prefiltering":
        A_filtered, corr_filtered_labels = clipper.filter_semantic_associations(target_labels_list, source_labels_list, A_all_to_all)
        corr_filtered_labels = np.array(corr_filtered_labels, dtype=np.int32)
        A_filtered, corr_filtered_labels = downsample_association_matrix(A_filtered, max_size_A=max_A_size, corr_labels=corr_filtered_labels)
    elif prefilter_type=="no_prefiltering":
        A_filtered, _ = downsample_association_matrix(A_all_to_all, max_size_A=max_A_size)

    # Score associations using pairwise distance
    D1 = np.asarray(target_sem_pc[:, :3]).T
    D2 = np.asarray(source_sem_pc[:, :3]).T
    t0 = time.perf_counter()
    clipper.score_pairwise_consistency(D1, D2, A_filtered)
    t1 = time.perf_counter()
    time_to_score = t1 - t0

    # Solve for max clique
    t0 = time.perf_counter()
    clipper.solve() # Solve as dense
    # clipper.solve_as_maximum_clique()
    t1 = time.perf_counter()
    time_to_solve = t1 - t0

    Ain = clipper.get_initial_associations()
    A_sel = clipper.get_selected_associations()

    # Estimate rel tf using filtered associations
    tf_estimate_filtered, tf_est2_filtered = calculate_tf_estimate(Ain, target_sem_pc, source_sem_pc, labels_dict)
    rerr2_filtered, terr2_filtered = get_err(relative_transform_t2s, tf_est2_filtered)
    rerr_filtered, terr_filtered = get_err(relative_transform_t2s, tf_estimate_filtered)
    print(f"\n\nrerror_filtered = {rerr_filtered}, terror_filtered = {terr_filtered}")
    print(f"err2_filtered = {rerr2_filtered}, terr2_filtered = {terr2_filtered}")

    # Estimate relative transform and calc err from true
    tf_estimate, tf_est2 = calculate_tf_estimate(A_sel, target_sem_pc, source_sem_pc, labels_dict)
    rerr2, terr2 = get_err(relative_transform_t2s, tf_est2)
    rerr, terr = get_err(relative_transform_t2s, tf_estimate)
    print(f"\nError: rerror = {rerr}, terror = {terr}")
    print(f"rerr2 = {rerr2}, terr2 = {terr2}")

    # Log results
    test_results = [len(Ain), len(A_sel), time_to_score, time_to_solve, terr, rerr]
    log_result(log_file, test, test_results, contact_ts_pair)


def draw_semantic_pointcloud(semantic_pointcloud, sigma_values):
    # Set a uniform color for robots
    # robot1_pcd.colors = o3d.utility.Vector3dVector(np.tile([0.0, 1.0, 0.0], (len(robot1_pcd.points), 1)))
    # robot2_pcd.colors = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 1.0], (len(robot1_pcd.points), 1)))

    labels_dict = {label.id: label.color for label in sem_kitti_labels}
    labels_rgb = labels2RGB(semantic_pointcloud[:, 3], labels_dict)

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(semantic_pointcloud[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(labels_rgb)

    o3d.visualization.draw_geometries([o3d_pcd])


def get_osm_grassland_points(osm_file_path):
    # Filter features for buildings and sidewalks
    buildings = ox.features_from_xml(osm_file_path, tags={"natural": True, "landuse": True})
    osm_building_list = []
    # Process Buildings as LineSets
    for _, building in buildings.iterrows():
        if building.geometry.geom_type == "Polygon":
            exterior_coords = building.geometry.exterior.coords

            for i in range(len(exterior_coords) - 1):
                start_point = [exterior_coords[i][0], exterior_coords[i][1], 0]
                end_point = [exterior_coords[i + 1][0], exterior_coords[i + 1][1], 0]

                osm_building_list.append(start_point)
                osm_building_list.append(end_point)
    osm_building_points = np.array(osm_building_list)
    return osm_building_points


def get_osm_buildings_points(osm_file_path):
    # Filter features for buildings and sidewalks
    buildings = ox.features_from_xml(osm_file_path, tags={"building": True})
    osm_building_list = []

    # Process Buildings as LineSets
    for _, building in buildings.iterrows():
        if building.geometry.geom_type == "Polygon":
            exterior_coords = np.array(building.geometry.exterior.coords)
            exterior_coords_fixed = np.copy(exterior_coords)
            exterior_coords_fixed[:, 0] = exterior_coords[:, 1]
            exterior_coords_fixed[:, 1] = exterior_coords[:, 0]

            for i in range(len(exterior_coords_fixed) - 1):
                start_point = [exterior_coords_fixed[i][0], exterior_coords_fixed[i][1], 0]
                end_point = [exterior_coords_fixed[i + 1][0],exterior_coords_fixed[i + 1][1], 0]

                osm_building_list.append(start_point)
                osm_building_list.append(end_point)
    osm_building_points = np.array(osm_building_list)
    return osm_building_points


def get_osm_road_points(osm_file_path):
    # For below tags, see: https://wiki.openstreetmap.org/wiki/Key:highway#Roads
    tags = {"highway": True}  # This will fetch tertiary and residential roads

    # Fetch roads using defined tags
    roads = ox.features_from_xml(osm_file_path, tags=tags)

    # Process Roads as LineSets with width
    osm_road_list = []
    for _, road in roads.iterrows():
        if road.geometry.geom_type == "LineString":
            # print(f"coords[0]: {coords = np.array(road.geometry.xy)[0]}")
            coords = np.array(road.geometry.xy).T
            coords_fixed = np.copy(coords)
            coords_fixed[:, 0] = coords[:, 1]
            coords_fixed[:, 1] = coords[:, 0]

            road_center = [
                np.mean(np.array(coords_fixed)[:, 0]),
                np.mean(np.array(coords_fixed)[:, 1]),
            ]
            for i in range(len(coords_fixed) - 1):
                start_point = [coords_fixed[i][0], coords_fixed[i][1],0]  # Assuming roads are at ground level (z=0)
                end_point = [coords_fixed[i + 1][0], coords_fixed[i + 1][1], 0]

                osm_road_list.append(start_point)
                osm_road_list.append(end_point)
    osm_road_points = np.array(osm_road_list)
    return osm_road_points


def convert_polyline_points_to_o3d(polyline_points, rgb_color):
    polyline_pcd = o3d.geometry.LineSet()
    if len(polyline_points) > 0:
        polyline_lines_idx = [[i, i + 1] for i in range(0, len(polyline_points) - 1, 2)]
        polyline_pcd.points = o3d.utility.Vector3dVector(polyline_points)
        polyline_pcd.lines = o3d.utility.Vector2iVector(polyline_lines_idx)
        polyline_pcd.paint_uniform_color(rgb_color)
    return polyline_pcd


def transform_points_lat_lon(xyz_homogeneous, transformation_matrix, init_gps_point=None):
    # Convert 30 degrees to radians
    # theta = np.deg2rad(25)    # KL
    theta = np.deg2rad(-4.5)    # MC

    # Create the 4x4 transformation matrix
    T = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0,              0,             1, 0],
        [0,              0,             0, 1]
    ])

    # transformation_matrix = T @ transformation_matrix
    pc = np.dot(xyz_homogeneous, transformation_matrix.T)[:, :3]
    pc_reshaped = np.array([np.eye(4) for _ in range(pc.shape[0])])
    pc_reshaped[:, 0:3, 3] = pc[:, :3]
    pc_lla = np.asarray(post_process_points(pc_reshaped))
    pc_lla = np.asarray(convert_pointcloud_to_latlon(pc_lla, origin_latlon=init_gps_point))[:, :3]
    pc_lla[:, 2] *= 0.0#0001
    return pc_lla


# Function to read GPS data from a file
def read_gps_data(file_path):
    points = []
    with open(file_path, "r") as file:
        for line in file:
            lat, lon, alt = map(float, line.strip().split())
            points.append([lat, lon, alt])
    return np.array(points)


def get_accum_points(
    first_frame_idx, 
    final_frame_idx, 
    frame_increment, 
    osm_frames, 
    lidar_data_path, 
    lidar_timestamps, 
    labels_path, 
    lidar_poses,
    init_gps_point, 
    gps_1_path, 
    gnss1_timestamps,
    semantics_type,
    use_latlon
) -> Tuple[np.array, np.array]:
    """
    Iterate through osm_frames and accumulate points (x,y,z,semantic) in their respective semantic class, road or building
    """

    semantic_points_list = []
    tf_list = []
    for frame_idx in tqdm(range(first_frame_idx, final_frame_idx, frame_increment)):
        frame_number = osm_frames[frame_idx]

        raw_pc_frame_path = os.path.join(lidar_data_path, f"lidar_pointcloud_{frame_number}.bin")
        points_np = read_bin_file(raw_pc_frame_path, dtype=np.float32, shape=(-1, 4))

        pc_frame_label_path = os.path.join(labels_path, f"lidar_pointcloud_{frame_number}.bin")
        labels_np = read_bin_file(pc_frame_label_path, dtype=np.int32, shape=(-1))
        
        pc_xyz = points_np[:, :3]
        
        # for intensity in pc_xyz:
        #     print(intensity)
        # points_transformed = get_transformed_point_cloud(pc_xyz, velodyne_poses, frame_number)

        # # if using groundtruth_odom
        # interp_trans, interp_quat = interpolate_pose(lidar_poses, lidar_timestamps[frame_number])
        # transformation_matrix = quaternion_pose_to_4x4(interp_trans, interp_quat)

        # Compute distances once and apply distance filter
        distances = np.linalg.norm(pc_xyz, axis=1)
        mask = (distances > 3) #& (distances < 80)
        pc_xyz = pc_xyz[mask]
        labels_np = labels_np[mask]

        # If using lidar/poses_world.txt
        transformation_matrix = lidar_poses[lidar_timestamps[frame_number]]
        tf_list.append(transformation_matrix)

        # Shift points from lidar to gnss receiver using lidar's relative xy plane
        # Doesnt need to change as only the first gps datapoint is used for all succeeding lidar scans
        # transformation_matrix[:3, 3] += np.array([-0.757, 0.16, 0])

        xyz_homogeneous = np.hstack([pc_xyz, np.ones((pc_xyz.shape[0], 1))])

        if use_latlon:
            # Transform the point cloud using the pose matrix
            transformed_xyz = transform_points_lat_lon(xyz_homogeneous, transformation_matrix, init_gps_point)
        else:
            transformed_xyz = np.dot(xyz_homogeneous, transformation_matrix.T)[:, :3]

        semantic_points = np.concatenate(
            (transformed_xyz, labels_np.reshape(-1, 1).astype(np.int32)), axis=1
        )

        # Extract only points with building or road labels 45 46
        if semantics_type == "osm_semantics":
            desired_semantic_indices = np.where((labels_np == 45) | (labels_np == 46))[0]
            semantic_points = semantic_points[desired_semantic_indices, :]
        # semantic_points[:, 2] = 0

        semantic_points_list.extend(semantic_points)
    
    return  np.asarray(semantic_points_list), np.asarray(tf_list)


def get_all_robot_timestamps(environments, robots, root_data_path):
    # Get all unique robot timestamps per environment
    all_robot_times = [
        np.unique(np.concatenate([
            np.float128(Robot(robot_name, env, root_data_path).lidar_timestamps)
            for robot_name in robots
        ]))
        for env in environments
    ]
    return all_robot_times


from dataclasses import dataclass

@dataclass
class ContactParams:
    eff_comms_dist: float
    min_dist: float
    min_contact_separation: float

def get_all_contact_timestamps(env, target_robot_name, source_robot_name, root_data_path, all_robot_times, params: ContactParams):
    """Finds contact timestamps where two robots are within the effective communication distance."""
    target_robot = Robot(robot_name=target_robot_name, env_name=env, root_data_path=root_data_path)
    source_robot = Robot(robot_name=source_robot_name, env_name=env, root_data_path=root_data_path)

    eff_comms_dist = params.eff_comms_dist
    min_dist = params.min_dist
    min_contact_separation = params.min_contact_separation

    distances = []
    eff_ts_pair_list = []
    effective_timestamps = []
    effective_distances = []

    # Ensure each contact location is at least min_contact_separation meters apart
    previous_position = [0, 0, 0]
    min_contact_separation = 10

    # Get initial earliest and latest timestamps of robot data
    initial_search_time = min(min(target_robot.lidar_poses.keys()), min(source_robot.lidar_poses.keys()))
    final_search_time = max(max(target_robot.lidar_poses.keys()), max(source_robot.lidar_poses.keys()))
    # print(f"initial_search_time: {initial_search_time}")

    # Get all timestamps where robots are within effective communication range
    if env == "kittredge_loop":
        timestamps = all_robot_times[0]
    elif env == "main_campus":
        timestamps = all_robot_times[1]

    for timestamp in timestamps:
        if timestamp >= initial_search_time and timestamp <= final_search_time:
            target_robot_nearest_ts = target_robot.get_nearest_timestamp(timestamp)
            target_robot_pose = target_robot.lidar_poses[target_robot_nearest_ts]

            source_robot_nearest_ts = source_robot.get_nearest_timestamp(timestamp)
            source_robot_pose = source_robot.lidar_poses[source_robot_nearest_ts]

            pose_offset = np.linalg.inv(target_robot_pose) @ source_robot_pose
            distance = np.linalg.norm(pose_offset[:3, 3])

            # For plotting
            # distances.append(distance)

            separation_from_prev = target_robot_pose[:3, 3] - previous_position
            separation_from_prev_norm = np.linalg.norm(separation_from_prev)

            if distance < eff_comms_dist and distance > min_dist and separation_from_prev_norm > min_contact_separation:
                previous_position = target_robot_pose[:3, 3]
                # print(f"separation_distance: {separation_from_prev_norm}")
                # effective_timestamps.append(target_robot_nearest_ts)
                # effective_distances.append(distance)
                eff_ts_pair_list.append([target_robot_nearest_ts, source_robot_nearest_ts])

    # if save_figure:
    #     # Plot results
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(timestamps, distances, label="All Distances", color="blue", alpha=0.6)
    #     plt.scatter(effective_timestamps, effective_distances, color="green", label="Effective Contacts (< {}m)".format(eff_comms_dist), marker='o')

    #     plt.xlabel("Timestamp")
    #     plt.ylabel("Distance (m)")
    #     plt.title(f"Robot Contact Distances Over Time \n({target_robot.env_name}: {target_robot.robot_name} and {source_robot.robot_name})")
    #     plt.legend()
    #     plt.grid()

    #     plt.savefig(f"{target_robot.env_name}_{target_robot.robot_name}_{source_robot.robot_name}.png", dpi=1000, bbox_inches="tight")
    #     # plt.show()

    return eff_ts_pair_list


    '''
    TODO:   - Make sure to get a count of the number of times CLIPPER will be tested in total. This can help indicate how long all tests will
              take.

            We could then do some np array difference to filter out all instances where the abs(dist) < eff_dist and get those times. 
            Right now I am brute-forcing through every timestamp to get the distances, which seems quite inefficient.

            - Interpolate pose at contact times? This would really slow things down.
            
            3) Figure out how to reduce memory consumption when using MP. 
                - The full dense maps can likely be voxel-downsampled before sending off. This is probably the same 
                  everytime anyhow. Only caveat is that any random sampling needs to be done in each individual proc.
    '''

def retrieve_existing_results(log_file):
    # Load existing results
    existing_results = set()
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            for line in f:
                try:
                    result = json.loads(line.strip())  # Read each line as a JSON object
                    key = (result["sem_type"], result["prefilter_type"], result["env"],
                        result["target_robot_name"], result["target_robot_eff_ts"],
                        result["source_robot_name"], result["source_robot_eff_ts"])
                    existing_results.add(key)  # Store unique identifier
                except json.JSONDecodeError:
                    print("Skipping corrupted JSON entry.")
    return existing_results

if __name__ == "__main__":
    root_data_path = f"/media/donceykong/donceys_data_ssd/datasets/CU_MULTI/data"
    environments = ["kittredge_loop", "main_campus"]
    robots = ["robot1", "robot2", "robot3", "robot4"]
    semantics_types = ["all_semantics", "osm_semantics"]
    prefiltering_types = ["all_prefiltering", "no_prefiltering", "semantic_prefiltering", "comms_aware_prefiltering"]

    # Set logging file
    log_file = "logs/results.json"

    # BLE peer-to-peer contact settings
    contact_params = ContactParams(eff_comms_dist=10.0, min_dist=2.0, min_contact_separation=10.0)

    # Num times to run map-matching using rand-sampled points
    per_sample_iter = 10

    # Map config
    voxel_size = 0.1
    random_downsample = True
    max_points = 10_000

    # Generate all test configs
    robot_pairs = list(combinations(robots, 2))
    test_configs = list(product(semantics_types, prefiltering_types, environments, robot_pairs))
    
    # Precompute timestamps
    all_robot_times = get_all_robot_timestamps(environments, robots, root_data_path=root_data_path)

    # Number of cores to utilize for multiprocessing
    use_multiprocessing = True
    num_processes = 1

    # **Step 1: Precompute Total Number of Tests (Before Running)**
    total_tests_count = 0
    eff_ts_pair_lists = []            # Set up multiprocessing
    if use_multiprocessing:
        num_workers = min(mp.cpu_count(), 10) 
        pool = mp.Pool(processes=num_workers)
    extraction_tasks = []

    # progress_bar = tqdm(total=len(test_configs), desc="extracting all contact timestamps", unit="test")
    for test in test_configs:
        env, (target_robot_name, source_robot_name) = test[2], test[3]

        if use_multiprocessing:
            extraction_tasks.append((env,target_robot_name,source_robot_name,root_data_path,all_robot_times,contact_params))
        else:
            eff_ts_pair_list = get_all_contact_timestamps(env=env, 
                                                          target_robot_name=target_robot_name, 
                                                          source_robot_name=source_robot_name,
                                                          root_data_path=root_data_path,
                                                          all_robot_times=all_robot_times,
                                                          params=contact_params)
            eff_ts_pair_lists.append(eff_ts_pair_list)
            total_tests_count += len(eff_ts_pair_list) * per_sample_iter  # Multiply by per-sample iterations

    if use_multiprocessing:
        for eff_ts_pair_list in pool.starmap(get_all_contact_timestamps, extraction_tasks):
            eff_ts_pair_lists.append(eff_ts_pair_list)
            total_tests_count += len(eff_ts_pair_list) * per_sample_iter
            # progress_bar.update(1)  # Update progress bar for each completed test

    pool.close()
    pool.join()
    # progress_bar.close()
    print(f"\n📊 Total tests to be run: {total_tests_count}\n")
    
    # Check for all existing results
    existing_results = retrieve_existing_results(log_file)

    use_multiprocessing = False
    # **Step 2: Run Tests and Track Progress**
    completed_tests = 0
    progress_bar = tqdm(total=total_tests_count, desc="Running Tests", unit="test")
    
    for test, eff_ts_pair_list in zip(test_configs, eff_ts_pair_lists):
        sem_type, prefilter_type, env, (target_robot_name, source_robot_name) = test

        print(f"\n {env}: {target_robot_name} and {source_robot_name}")

        target_robot = Robot(robot_name=target_robot_name, env_name=env, root_data_path=root_data_path)
        source_robot = Robot(robot_name=source_robot_name, env_name=env, root_data_path=root_data_path)

        for contact_ts_pair in eff_ts_pair_list:
            target_robot_eff_ts, source_robot_eff_ts = contact_ts_pair
            test_key = (sem_type, prefilter_type, env, target_robot_name, str(target_robot_eff_ts),
                    source_robot_name, str(source_robot_eff_ts))
            if test_key in existing_results:
                progress_bar.update(per_sample_iter)
                continue

            # Accumulate points to create maps
            target_robot_eff_ts, source_robot_eff_ts = contact_ts_pair
            target_pc, target_tf_array = target_robot.get_accum_pc(target_robot.first_lidar_frame,
                                                                   target_robot.lidar_scan_idx_dict[target_robot_eff_ts],
                                                                   sem_type, 
                                                                   frame_index_inc=100)
            source_pc, source_tf_array = source_robot.get_accum_pc(source_robot.first_lidar_frame, 
                                                                   source_robot.lidar_scan_idx_dict[source_robot_eff_ts],
                                                                   sem_type, 
                                                                   frame_index_inc=100)

            target_pc = voxel_downsample_pc(target_pc, voxel_leaf_size=voxel_size)
            source_pc = voxel_downsample_pc(source_pc, voxel_leaf_size=voxel_size)

            # Set up multiprocessing
            if use_multiprocessing:
                num_workers = min(mp.cpu_count(), num_processes) 
                pool = mp.Pool(processes=num_workers)

            tasks = []
            for sample_idx in range(per_sample_iter):
                if random_downsample:
                    target_pc_rand_ds = random_downsample_pc(target_pc, max_points)
                    source_pc_rand_ds = random_downsample_pc(source_pc, max_points)

                print(f"test: {test}")
                if use_multiprocessing:
                    tasks.append((target_pc_rand_ds, target_tf_array, source_pc_rand_ds, source_tf_array, test, contact_ts_pair, log_file))
                else:
                    align_maps(target_pc_rand_ds, target_tf_array, source_pc_rand_ds, source_tf_array, test, contact_ts_pair, log_file)
                    progress_bar.update(1)  # 2739/3280
            
            if use_multiprocessing:
                for _ in pool.starmap(align_maps, tasks):
                    completed_tests += 1
                    progress_bar.update(1)

                pool.close()
                pool.join()

    progress_bar.close()
    print(f"\n✅ All {completed_tests} tests completed!")