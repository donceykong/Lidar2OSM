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
    mask = relative_ego_dist < max_ego_dist
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

def align_maps(target_data, source_data, iter):
    epsilon = 1.0
    max_A_size = 10000
    max_points = 1000
    voxel_size = 0.1
    random_downsample = True
    voxel_downsample = True
    eff_comm_dist = 10 # Effective Communication Distance of BT recievers used
    use_ego_filtering = True
    use_sem_filtering = True

    if voxel_downsample:
        print("Resetting epsilon to twice the voxel leaf size.")
        epsilon = voxel_size * 5

    # TODO: Move to dataset class
    labels_dict = {label.id: label.color for label in sem_kitti_labels}

    target_sem_pc = np.copy(target_data[0])
    target_tf_array = np.copy(target_data[1])
    source_sem_pc = np.copy(source_data[0])
    source_tf_array = np.copy(source_data[1])

    # Downsample Pointclouds
    if voxel_downsample:
        init_target_size = target_sem_pc.shape[0]
        init_source_size = source_sem_pc.shape[0]
        target_sem_pc = voxel_downsample_pc(target_sem_pc, voxel_leaf_size=voxel_size)
        source_sem_pc = voxel_downsample_pc(source_sem_pc, voxel_leaf_size=voxel_size)
        print(f"target pc size:\n   -BEFORE downsampling: {init_target_size}\n   -AFTER: {target_sem_pc.shape[0]}")
        print(f"source pc size:\n   -BEFORE downsampling: {init_source_size}\n   -AFTER: {source_sem_pc.shape[0]}")
    if random_downsample:
        target_sem_pc = random_downsample_pc(target_sem_pc, max_points)
        source_sem_pc = random_downsample_pc(source_sem_pc, max_points)

    target_map_mbytes = 1000 * target_sem_pc.nbytes / (1024 ** 2)
    source_map_mbytes = 1000 * source_sem_pc.nbytes / (1024 ** 2)
    print(f"\nsize of robot3 DSmap: {target_map_mbytes} KB")
    print(f"size of robot4 DSmap: {source_map_mbytes} KB")

    # Create label lists
    target_labels_list = [np.int32(x) for x in target_sem_pc[:, 3]]
    source_labels_list = [np.int32(x) for x in source_sem_pc[:, 3]]

    # Get the transform needed to tf target to source
    target_final_pose = target_tf_array[-1]
    source_final_pose = source_tf_array[-1]
    relative_transform_t2s = np.linalg.inv(source_final_pose) @ target_final_pose

    # Calc true relative distance of robots and use that as eff
    true_ego_dist = np.linalg.norm(relative_transform_t2s[:3, 3])
    eff_comm_dist = true_ego_dist * 2

    # Center PCS and TFs about robot positions
    homogeneous_points = np.hstack((target_sem_pc[:, :3], np.ones((target_sem_pc[:, :3].shape[0], 1))))
    transformed_points = homogeneous_points @ np.linalg.inv(target_final_pose).T
    target_sem_pc[:, :3] = transformed_points[:, :3]
    target_tf_array = np.einsum('ij,njk->nik', np.linalg.inv(target_final_pose), target_tf_array)

    homogeneous_points = np.hstack((source_sem_pc[:, :3], np.ones((source_sem_pc[:, :3].shape[0], 1))))
    transformed_points = homogeneous_points @ np.linalg.inv(source_final_pose).T
    source_sem_pc[:, :3] = transformed_points[:, :3]
    source_tf_array = np.einsum('ij,njk->nik', np.linalg.inv(source_final_pose), source_tf_array)

    # Create target PCD
    labels_rgb_ds = labels2RGB(target_sem_pc[:, 3], labels_dict)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_sem_pc[:, :3])
    target_pcd.colors = o3d.utility.Vector3dVector(labels_rgb_ds)
    o3d.visualization.draw_geometries([target_pcd])

    # Create target PCD
    target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    target_frame.transform(target_tf_array[-1])

    # Create source PCD
    source_labels_rgb = labels2RGB(source_sem_pc[:, 3], labels_dict)
    # Increase brightness of source cloud colors and normalize bw 0:1
    source_labels_rgb[:, :] = 0 #+=  0.5
    source_labels_rgb = np.clip(source_labels_rgb, 0.0, 1.0)
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_sem_pc[:, :3])
    source_pcd.colors = o3d.utility.Vector3dVector(source_labels_rgb)

    source_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    source_frame.transform(source_tf_array[-1])

    # Draw frames and pointclouds
    o3d.visualization.draw_geometries([target_pcd, source_pcd, target_frame, source_frame])

    # # Generate ground truth transform(R,t) for the source cloud
    # tf_ground_truth = np.eye(4)
    # tf_ground_truth[0:3, 0:3] = R.random().as_matrix()
    # tf_ground_truth[0:3, 3] = np.random.uniform(low=-1000, high=1000, size=(3,))

    # ---- Generate dataset (ALL-TO-ALL) ---- #
    D1 = np.asarray(target_pcd.points).T
    D2 = np.asarray(source_pcd.points).T
    # D2 = tf_ground_truth[0:3, 0:3] @ D2 + tf_ground_truth[0:3, 3].reshape(-1, 1)
    A_all_to_all = get_a2a_assoc_matrix(len(target_pcd.points), len(source_pcd.points))

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

    print(f"\nInitial size of A: {len(A_all_to_all)}")
    print("FILTERING ASSOCIATIONS NOW:")
    # Further downsample A and correlation labels to max A size
    if use_ego_filtering==True and use_sem_filtering==True:
        A_ego_filtered = filter_by_ego_distance(target_sem_pc[:, :3], source_sem_pc[:, :3], A_all_to_all, max_ego_dist=eff_comm_dist)
        A_sem_filtered, corr_filtered_labels = clipper.filter_semantic_associations(target_labels_list, source_labels_list, A_ego_filtered)
        corr_filtered_labels = np.array(corr_filtered_labels, dtype=np.int32)
        A_filtered, corr_filtered_labels = downsample_association_matrix(A_sem_filtered, max_size_A=max_A_size, corr_labels=corr_filtered_labels)
    elif use_ego_filtering==False and use_sem_filtering==True:
        A_sem_filtered, corr_filtered_labels = clipper.filter_semantic_associations(target_labels_list, source_labels_list, A_all_to_all)
        corr_filtered_labels = np.array(corr_filtered_labels, dtype=np.int32)
        A_filtered, corr_filtered_labels = downsample_association_matrix(A_sem_filtered, max_size_A=max_A_size, corr_labels=corr_filtered_labels)
    elif use_ego_filtering==True and use_sem_filtering==False:
        A_ego_filtered = filter_by_ego_distance(target_sem_pc[:, :3], source_sem_pc[:, :3], A_all_to_all, max_ego_dist=eff_comm_dist)
        A_filtered, _ = downsample_association_matrix(A_ego_filtered, max_size_A=max_A_size)
    elif use_ego_filtering==False and use_sem_filtering==False:
        A_filtered, _ = downsample_association_matrix(A_all_to_all, max_size_A=max_A_size)
    print(f"Size of A_filtered: {len(A_filtered)}")

    t0 = time.perf_counter()
    # L1 = target_labels_list
    # L2 = source_labels_list
    # clipper.score_semantic_pairwise_consistency(D1, D2, L1, L2, A_filtered)
    clipper.score_pairwise_consistency(D1, D2, A_filtered)
    t1 = time.perf_counter()
    time_to_score = t1 - t0
    
    # # Create image of affinity matrix
    affinity_mat = clipper.get_affinity_matrix()
    # plt.imshow(affinity_mat, cmap='viridis', interpolation='nearest')  # 'viridis' colormap
    # plt.colorbar()  # Add a colorbar to show the value scale
    # plt.title("Affinity Matrix as Image")
    # plt.show()

    # Show array of doubles ranging from 0 to 1
    affinity_mat_flat = sorted(np.array(affinity_mat).flatten())
    max_affin = np.max(affinity_mat_flat)
    min_affin = np.min(affinity_mat)
    print(f"max affin: {max_affin}, min affin: {min_affin}")
    # plt.scatter(range(len(affinity_mat_flat)), affinity_mat_flat, c='blue', edgecolor='black')
    # plt.title('Scatter Plot of Affinity Matrix Values')
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.show()

    t0 = time.perf_counter()
    clipper.solve() # Solve as dense
    # clipper.solve_as_maximum_clique()
    t1 = time.perf_counter()
    time_to_solve = t1 - t0

    Ain = clipper.get_initial_associations()
    Ain_len = Ain.shape[0]
    A_sel = clipper.get_selected_associations()
    A_sel_len = A_sel.shape[0]

    # print(f"Max elt in A selected: {np.max(Ain)}")
    print(f"Selected {len(A_sel)} associations.")

    if use_sem_filtering:
        # get labels for semantic associations (for vis/saving pcds)
        _, asel_corr_filtered_labels = clipper.filter_semantic_associations(target_labels_list, source_labels_list, A_sel)

    # # Create bar chart of selected semantics
    # label_counts = Counter(asel_corr_filtered_labels)
    # sorted_labels = sorted(label_counts.keys())
    # sorted_frequencies = [label_counts[label] for label in sorted_labels]
    # plt.bar(sorted_labels, sorted_frequencies, edgecolor='black', align='center') #, width=.8)
    # plt.title('Frequency of Labels')
    # plt.xlabel('Labels')
    # plt.ylabel('Frequency')
    # plt.show()

    # Unaligned Source Cloud PCD
    source_pcd_tf = o3d.geometry.PointCloud()
    source_pcd_tf.points = o3d.utility.Vector3dVector(D2.T)
    source_pcd_tf.colors = o3d.utility.Vector3dVector(source_labels_rgb)

    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    # p2p.with_scaling = True
    tf_estimate = p2p.compute_transformation(target_pcd, source_pcd_tf, o3d.utility.Vector2iVector(A_sel))

    # rerr, terr = get_err(tf_ground_truth, tf_estimate)
    rerr, terr = get_err(relative_transform_t2s, tf_estimate)

    print(f"Ground truth tf: {relative_transform_t2s}\n")
    print(f"Estimated tf: {tf_estimate}")
    print(f"\n\nError: rerror = {rerr}, terror = {terr}")

    # trans_gt = tf_ground_truth[0:3, 3]
    trans_gt = relative_transform_t2s[0:3, 3]
    trans_est = tf_estimate[0:3, 3]
    
    if iter > 0:
        # Create a sphere mesh with a radius of 1.0
        sphere_GT = o3d.geometry.TriangleMesh.create_sphere(radius=10.0)
        sphere_GT.paint_uniform_color([0, 1.0, 0])
        sphere_GT.compute_vertex_normals()
        sphere_GT.translate(trans_gt)
        
        sphere_EST = o3d.geometry.TriangleMesh.create_sphere(radius=10.0)
        sphere_EST.paint_uniform_color([1.0, 0, 0])
        sphere_EST.compute_vertex_normals()
        sphere_EST.translate(trans_est)

        # O3d geometry for initial point cloud correspondances
        corr_initial = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            target_pcd, source_pcd_tf, Ain
        )
        
        corr_filtered = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            target_pcd, source_pcd_tf, A_filtered
        )

        # O3d geometry for selected point cloud correspondances
        corr_selected = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            target_pcd, source_pcd_tf, A_sel
        )

        if use_sem_filtering:
            corr_filtered_rgb = labels2RGB(corr_filtered_labels, labels_dict)
            corr_filtered.colors = o3d.utility.Vector3dVector(corr_filtered_rgb)

            # Add semantic colors to selected associations
            corr_selected_rgb = labels2RGB(asel_corr_filtered_labels, labels_dict)
            corr_selected.colors = o3d.utility.Vector3dVector(corr_selected_rgb)

        source_fixed = copy.deepcopy(source_pcd_tf).transform(np.linalg.inv(tf_estimate))
        
        # if show_visualizations or terr < 2.0:
        # Ground truth
        o3d.visualization.draw_geometries([target_pcd, source_pcd])
        # Unaligned source
        o3d.visualization.draw_geometries([target_pcd, source_pcd_tf, sphere_GT, sphere_EST])
        o3d.visualization.draw_geometries([target_pcd, source_pcd_tf])
        # Initial associations
        o3d.visualization.draw_geometries([target_pcd, source_pcd_tf, corr_initial])
        # Filtered associations
        o3d.visualization.draw_geometries([target_pcd, source_pcd_tf, corr_filtered])
        # Selected associations
        o3d.visualization.draw_geometries([target_pcd, source_pcd_tf, corr_selected])
        # Draw registration result: Map matching with correspondances and estimated transformation
        o3d.visualization.draw_geometries([source_fixed, target_pcd])

    return rerr, terr

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


def compare_with_osm(semantic_pointcloud, osm_file_path, gps_1_points):
    labels_dict = {label.id: label.color for label in sem_kitti_labels}
    labels_rgb = labels2RGB(semantic_pointcloud[:, 3], labels_dict)

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(semantic_pointcloud[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(labels_rgb)

    # # Downsample the point cloud using voxel downsampling
    voxel_size = 0.000002  # Adjust this value to control downsampling resolution
    o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)

    gps_1_point_cloud = o3d.geometry.PointCloud()
    gps_1_point_cloud.points = o3d.utility.Vector3dVector(gps_1_points[:10000])
    gps_1_point_cloud.paint_uniform_color([1, 0, 0])

    osm_building_points = get_osm_buildings_points(osm_file_path)
    osm_road_points = get_osm_road_points(osm_file_path)
    osm_road_pcd = convert_polyline_points_to_o3d(osm_road_points, [1, 0, 0])
    osm_building_pcd = convert_polyline_points_to_o3d(osm_building_points, [0, 0, 1])

    # Add coordinate frames to the scene
    geometry_list = [osm_road_pcd, osm_building_pcd, gps_1_point_cloud, o3d_pcd]

    # Visualize using Open3D
    o3d.visualization.draw_geometries(
        geometry_list,
        window_name="Downsampled Accumulated Point Cloud with Pose Frames",
        width=800,
        height=600,
    )


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
        desired_semantic_indices = np.where((labels_np == 45) | (labels_np == 46))[0]
        semantic_points = semantic_points[desired_semantic_indices, :]
        # semantic_points[:, 2] = 0

        # frame_kbytes = 1000 * semantic_points.nbytes / (1024 ** 2)
        # print(f"\nsize of map frame: {frame_kbytes} KB")

        semantic_points_list.extend(semantic_points)
    
    return  np.asarray(semantic_points_list), np.asarray(tf_list)


def get_accum_pc(root_dir, frame_inc, use_latlon=False):
    lidar_data_path = os.path.join(root_dir, "lidar/pointclouds")
    lidar_ts_path = os.path.join(root_dir, "lidar/timestamps.txt")
    label_path = os.path.join(root_dir, "lidar/labels/gt_osm_labels")
    # label_path = os.path.join(root_dir, "lidar/labels/gt_labels")

    # If using world poses (angle in latlon accounted for)
    lidar_poses_path = os.path.join(root_dir, "lidar/poses_world.txt")
    pose_ts_path = os.path.join(root_dir, "lidar/timestamps.txt")

    # If using ego poses (angle in latlon NOT accounted for)
    # gt_poses_path = os.path.join(root_dir, "poses/groundtruth_odom.txt")
    # gt_poses_timestamps_path = os.path.join(root_dir, "poses/odom_timestamps.txt")

    gps_1_path = os.path.join(root_dir, "gps/gnss_1_data.txt")
    gps_1_ts_path = os.path.join(root_dir, "gps/gnss_1_timestamps.txt")

    # Read lidar timestamps into a list
    with open(lidar_ts_path, "r") as ts_file:
        lidar_timestamps = [np.float128(line.strip()) for line in ts_file]

    # Read lidar timestamps into a list
    with open(gps_1_ts_path, "r") as gps_ts_file:
        gnss1_timestamps = [np.float128(line.strip()) for line in gps_ts_file]

    # dict with ts as key and pose as value
    lidar_poses = read_quat_poses(lidar_poses_path, pose_ts_path)
    labelled_frames = get_frame_numbers(label_path)

    # Get initial GPS Point
    gps_1_points = read_gps_data(gps_1_path)
    gps_1_points[:, 2] = 0
    lidar_timestamp = lidar_timestamps[0]
    # Find the interval for interpolation
    idx = np.searchsorted(gnss1_timestamps, lidar_timestamp) - 1
    t0, t1 = gnss1_timestamps[idx], gnss1_timestamps[idx + 1]
    p0, p1 = gps_1_points[idx], gps_1_points[idx + 1]
    ratio = (lidar_timestamp - t0) / (t1 - t0)
    interp_gps_position = (1 - ratio) * p0 + ratio * p1
    init_gps_point = interp_gps_position

    # Interp lidar position from GPS readings
    gnss1_points_ego = convert_pointcloud_to_ego(gps_1_points, origin_latlon=init_gps_point)
    # num_ts = len(gnss1_timestamps)
    # for timestamp in lidar_poses.keys():
    #     idx = np.searchsorted(gnss1_timestamps, timestamp) - 1
    #     print(f"idx: {idx}")
    #     if idx < num_ts-1:
    #         t0, t1 = gnss1_timestamps[idx], gnss1_timestamps[idx + 1]
    #         p0, p1 = gnss1_points_ego[idx], gnss1_points_ego[idx + 1]
    #         ratio = (timestamp - t0) / (t1 - t0)
    #         interp_gps_position = (1 - ratio) * p0 + ratio * p1   
    #         new_pose = lidar_poses[timestamp]
    #         new_pose[:3, 3] = interp_gps_position
    #         lidar_poses[timestamp] = new_pose 

    print(f"labelled_frames: {len(labelled_frames)}")
    # Accumulate the target points
    semantic_points_np, tf_array = get_accum_points(
        first_frame_idx=0,#len(labelled_frames) - 1000,#0,
        final_frame_idx=len(labelled_frames) - 1,
        frame_increment=frame_inc,
        osm_frames=labelled_frames,
        lidar_data_path=lidar_data_path,
        lidar_timestamps=lidar_timestamps,
        labels_path=label_path,
        lidar_poses=lidar_poses,
        init_gps_point = init_gps_point,
        gps_1_path=gps_1_path,
        gnss1_timestamps=gnss1_timestamps,
        use_latlon=use_latlon,
    )

    # # # Get OSM data as points
    # osm_file_path = os.path.join(root_dir, "kittredge_loop.osm")
    # osm_building_points = get_osm_buildings_points(osm_file_path)
    # osm_road_points = get_osm_road_points(osm_file_path)

    # osm_building_points_ego = convert_pointcloud_to_ego(osm_building_points, origin_latlon=init_gps_point)
    # osm_road_points_ego = convert_pointcloud_to_ego(osm_road_points, origin_latlon=init_gps_point)


    # semantic_osm_points = np.concatenate(
    #     (osm_building_points_ego, labels_np.reshape(-1, 1).astype(np.int32)), axis=1
    # )

    # osm_grassland_pcd = convert_polyline_points_to_o3d(osm_grassland_points, [0.0, 1.0, 0.0])
    # osm_road_pcd = convert_polyline_points_to_o3d(osm_road_points_ego, [1, 0.2, 0.2])
    # osm_building_pcd = convert_polyline_points_to_o3d(osm_building_points_ego, [0.2, 0.2, 1])

    # labels_dict = {label.id: label.color for label in sem_kitti_labels}
    # labels_rgb = labels2RGB(semantic_points_np[:, 3], labels_dict)
    # semantic_points_o3d = o3d.geometry.PointCloud()
    # semantic_points_o3d.points = o3d.utility.Vector3dVector(semantic_points_np[:, :3])
    # semantic_points_o3d.colors = o3d.utility.Vector3dVector(labels_rgb)

    # # gnss1_points_ego = convert_pointcloud_to_ego(gps_1_points, origin_latlon=init_gps_point)
    # gnss1_points_o3d = o3d.geometry.PointCloud()
    # gnss1_points_o3d.points = o3d.utility.Vector3dVector(gnss1_points_ego[:, :3])
    # gnss1_points_o3d.paint_uniform_color([0.0, 0.0, 0.0])

    # # voxel_size = 0.000001  # Set the size of each voxel (adjust as needed)
    # # semantic_points_o3d = semantic_points_o3d.voxel_down_sample(voxel_size=voxel_size)

    # # Add coordinate frames to the scene
    # geometry_list = [osm_road_pcd, osm_building_pcd, semantic_points_o3d, gnss1_points_o3d]

    # # Visualize using Open3D
    # o3d.visualization.draw_geometries(
    #     geometry_list,
    #     window_name="Downsampled Accumulated Point Cloud with Pose Frames",
    #     width=800,
    #     height=600,
    # )
    return semantic_points_np, tf_array

if __name__ == "__main__":
    # Path to YAML-based ablation config file
    ablation_config = 'lidar2osm/config/ablation_config.yaml'
    with open(ablation_config, 'r') as file:
        ablation_data = yaml.safe_load(file)
    
    env = "kittredge_loop"
    target_robot = "robot3"
    source_robot = "robot4"
    env_path = f"/media/donceykong/donceys_data_ssd/datasets/CU_MULTI/data/{env}"
    # env_path = f"/home/donceykong/Desktop/ARPG/projects/spring2025/lidar2osm_full/cu-multi-dataset/data/binarized/{env}"

    target_root_dir = os.path.join(env_path, target_robot)
    target_data = get_accum_pc(target_root_dir, frame_inc=10, use_latlon=False)

    source_root_dir = os.path.join(env_path, source_robot)
    source_data = get_accum_pc(source_root_dir, frame_inc=100, use_latlon=False)

    # Get the size in bytes
    target_map_mbytes = 1000 * target_data[0].nbytes / (1024 ** 2)
    source_map_mbytes = 1000 * source_data[0].nbytes / (1024 ** 2)

    print(f"\nsize of robot3 map: {target_map_mbytes} KB")
    print(f"size of robot4 map: {source_map_mbytes} KB")

    rerr_list = []; terr_list = []
    for i in range(10):
        iter = i + 1
        print(f"\niter: {iter}")
        rerr, terr = align_maps(target_data=target_data, source_data=source_data, iter=iter)
        rerr_list.append(rerr)
        terr_list.append(terr)

    rerr_ave = np.average(rerr_list)
    terr_ave = np.average(terr_list)

    print(f"rerr_ave: {rerr_ave}, terr_ave: {terr_ave}")