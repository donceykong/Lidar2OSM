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


# def random_downsample_pc(semantic_pc, max_points):
#     """Downsamples a sematic pointcloud and shifts the points to origin.
#     Args:
#         semantic_pc (numpy.ndarray): Nx4 array of point positions and semantic IDs.
#         max_points (float): number of indices from semantic pc to randomly sample.

#     Returns:
#         semantic_pc_ds (ndarray): 
#             This array is the downsampled points and labels of the semantically-labelled pc.
#     """

#     downsampled_indices = np.random.choice(len(semantic_pc), max_points, replace=False)
#     semantic_pc_ds = np.asarray(semantic_pc)[downsampled_indices, :]

#     return semantic_pc_ds


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


# def downsample_association_matrix(A, max_size_A=10000, corr_labels=None) -> None:    
#     # randomly downsample A indices
#     rand_ds_A_idxs = np.random.choice(A.shape[0], size=max_size_A, replace=False)   #rand_ds_A_idxs

#     # Downsample correlation labels too if passed in
#     if corr_labels is not None:
#         corr_labels = corr_labels[rand_ds_A_idxs]
    
#     return A[rand_ds_A_idxs], corr_labels

import numpy as np

def downsample_association_matrix(
    A, 
    stats_obj,
    max_size_A=10000, 
    corr_labels=None
):
    """
    Downsamples the association matrix `A` while ensuring specific indices are retained.
    
    Args:
        A (numpy.ndarray): The association matrix to downsample (NxM).
        accum_target_sel_indices (list or ndarray): Indices of rows that must be included from the target.
        accum_source_sel_indices (list or ndarray): Indices of rows that must be included from the source.
        max_size_A (int): Maximum number of rows to retain in the downsampled matrix.
        corr_labels (numpy.ndarray, optional): Labels associated with rows of `A`.

    Returns:
        A_ds (numpy.ndarray): Downsampled association matrix.
        corr_labels_ds (numpy.ndarray): Downsampled correlation labels (if provided).
        downsampled_indices (numpy.ndarray): Original indices of downsampled rows.
    """
    # accum_target_sel_indices_list = [key for key in accum_target_sel_indices.keys()]
    # accum_source_sel_indices_list = [key for key in accum_source_sel_indices.keys()]
    
    # # Combine required target and source indices to keep
    # required_indices = np.concatenate([accum_target_sel_indices_list, accum_source_sel_indices_list])

    # Exclude required indices from the random pool
    required_indices = []
    for A_index in range(len(A)-1):
        entry = A[A_index]
        target_index = entry[0]
        source_index = entry[1]
        
        if target_index in stats_obj.accum_sel_target_indices or source_index in stats_obj.accum_sel_source_indices:
            required_indices.append(A_index)

    required_indices = np.array(required_indices)
    
    if len(required_indices) > max_size_A:
        raise ValueError("Number of required indices exceeds max_size_A.")
    
    remaining_indices = np.setdiff1d(np.arange(A.shape[0]), required_indices, assume_unique=True)
    num_random_indices = max_size_A - len(required_indices)

    # Randomly select additional indices
    random_indices = np.random.choice(remaining_indices, size=num_random_indices, replace=False)

    # Combine required indices with random indices
    if len(required_indices) > 0:
        downsampled_indices = np.concatenate([required_indices, random_indices])
    else:
        downsampled_indices = random_indices

    # Downsample the matrix and correlation labels
    A_ds = A[downsampled_indices]
    corr_labels_ds = corr_labels[downsampled_indices] if corr_labels is not None else None

    return A_ds, corr_labels_ds


def get_a2a_assoc_matrix(N1, N2):
   assoc_matrix = np.zeros((N1*N2,2),np.int32)
 
   i = 0
   for n1 in range(N1):
       for n2 in range(N2):
           assoc_matrix[i,0] = n1
           assoc_matrix[i,1] = n2
           i += 1
 
   return assoc_matrix


def random_downsample_pc(semantic_pc, max_points, accum_sel_indices):
    """
    Downsamples a semantic point cloud and shifts the points to the origin.
    
    Args:
        semantic_pc (numpy.ndarray): Nx4 array of point positions and semantic IDs.
        max_points (int): Number of indices from the semantic point cloud to randomly sample.
        accum_sel_indices (list or ndarray): List of indices that must be included in the downsampling.
    
    Returns:
        semantic_pc_ds (numpy.ndarray): Downsampled points and labels of the semantically-labelled point cloud.
        index_map (dict): Mapping from new indices in the downsampled array to original indices.
    """

#    downsampled_indices = np.random.choice(len(semantic_pc), max_points, replace=False)
    
    # Make sure accum_sel_indices are in downsampled_indices
    # Randomly select remaining indices, excluding accum_sel_indices
    accum_sel_indices_list = [key for key in accum_sel_indices.keys()]

    remaining_indices = np.setdiff1d(np.arange(len(semantic_pc)), accum_sel_indices_list, assume_unique=True)
    num_random_indices = max_points - len(accum_sel_indices_list)
    random_indices = np.random.choice(remaining_indices, num_random_indices, replace=False)

    # Combine accum_sel_indices and random_indices
    if len(accum_sel_indices_list) > 0:
        downsampled_indices = np.concatenate([accum_sel_indices_list, random_indices])
    else:
        downsampled_indices = random_indices
        
    semantic_pc_ds = np.asarray(semantic_pc)[downsampled_indices, :]
    index_map = {new_idx: original_idx for new_idx, original_idx in enumerate(downsampled_indices)}

    return semantic_pc_ds, index_map

def get_selected_indices(A_sel):
    # Initialize two arrays to store the selected indices
    sel_target_indices = []
    sel_source_indices = []

    # Iterate through each entry in array A
    for entry in A_sel:
        # Get the indices from dataset1 and dataset2
        sel_target_indices.append(entry[0])
        sel_source_indices.append(entry[1])
    
    return sel_target_indices, sel_source_indices

def visualize_geodesic_path(mesh, path):
    """
    Visualize the geodesic path on the mesh.

    Parameters:
        mesh (o3d.geometry.TriangleMesh): The input mesh.
        path (list): List of vertex indices forming the geodesic path.
    """
    V = np.asarray(mesh.vertices)

    # Create lines for the path
    path_edges = [[path[i], path[i + 1]] for i in range(len(path) - 1)]
    path_edges = np.array(path_edges)

    # Create a LineSet for the path
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(V),
        lines=o3d.utility.Vector2iVector(path_edges),
    )
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in path_edges])  # Red for path

    # Visualize the mesh and the geodesic path
    o3d.visualization.draw_geometries([line_set])

from geodesic2 import *
def solve_shortest_path(mesh, source_idx, target_idx):
    # mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=100)
    mesh.compute_vertex_normals()

    print(len(mesh.vertices))

    # # Compute geodesic distances
    # geodesic_distances = compute_heat_geodesic(mesh, source_idx)

    # Get shortest path
    path = astar_geodesic_path(mesh, source_idx, target_idx)
    # path = find_geodesic_path(mesh, geodesic_distances, source_idx, target_idx)

    # Visualize
    visualize_geodesic_path(mesh, path)

def make_mesh(pcd):
    alpha = 0.5  # Adjust alpha based on point cloud density
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

    # # Estimate normals for the point cloud
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))
    # # Create a mesh using Ball-Pivoting
    # radii = [0.5, 0.5, 0.5]  # Radii for ball-pivoting
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    
    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh])
    
    return mesh

import matplotlib.pyplot as plt
def align_maps(target_sem_pc, source_sem_pc, stats_obj, tf_ground_truth, iter):
    epsilon = 0.5
    max_A_size = 300
    max_points = 400000
    
    # Choose 'max_points'-number of points at random from both tar and source pcs
    # and add back in indices of points that were previously chosen
    # target_sem_pc, target_index_map = random_downsample_pc(target_sem_pc, max_points, stats_obj.accum_sel_target_indices)
    # source_sem_pc, source_index_map = random_downsample_pc(source_sem_pc, max_points, stats_obj.accum_sel_source_indices)

    labels_dict = {label.id: label.color for label in sem_kitti_labels}

    target_labels = target_sem_pc[:, 3]
    target_labels_rgb = labels2RGB(target_sem_pc[:, 3], labels_dict)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_sem_pc[:, :3])
    target_pcd.colors = o3d.utility.Vector3dVector(target_labels_rgb)

    # source_labels = source_sem_pc[:, 3]
    # source_labels_rgb = labels2RGB(source_sem_pc[:, 3], labels_dict)
    # # Increase brightness of source cloud colors and normalize bw 0:1
    # source_labels_rgb[:, :] +=  0.5
    # source_labels_rgb = np.clip(source_labels_rgb, 0.0, 1.0)
    # source_pcd = o3d.geometry.PointCloud()
    # source_pcd.points = o3d.utility.Vector3dVector(source_sem_pc[:, :3])
    # source_pcd.colors = o3d.utility.Vector3dVector(source_labels_rgb)

    voxel_size = 0.5
    target_pcd = o3d.geometry.VoxelGrid.create_from_point_cloud(target_pcd, voxel_size=voxel_size)

    # Make and solve mesh
    o3d.visualization.draw_geometries([target_pcd])
    mesh = make_mesh(target_pcd)
    solve_shortest_path(mesh, source_idx=10, target_idx=100)


    # ---- Generate dataset (ALL-TO-ALL) ---- #
    D1 = np.asarray(target_pcd.points).T
    D2 = np.asarray(source_pcd.points).T
    D2 = tf_ground_truth[0:3, 0:3] @ D2 + tf_ground_truth[0:3, 3].reshape(-1, 1)
    A_all_to_all = get_a2a_assoc_matrix(len(target_pcd.points), len(source_pcd.points))

    # print(f"\nInitial size of A: {len(A_all_to_all)}")

    # # Set up custom invariant function for semantic CLIPPER
    # iparams = clipperpy.invariants.SemanticsConstrainedEuclideanDistanceParams()
    # iparams.epsilon = epsilon
    # iparams.sigma =  0.5 * iparams.epsilon
    # invariant = clipperpy.invariants.SemanticsConstrainedEuclideanDistance(iparams)
    # # Add the semantic information to the point cloud data
    # target_pc_sc_reshaped = target_labels.reshape(1, -1)
    # source_pc_sc_reshaped = source_labels.reshape(1, -1)
    # D1 = np.concatenate((D1, target_pc_sc_reshaped), axis=0)
    # D2 = np.concatenate((D2, source_pc_sc_reshaped), axis=0)

    # Set up vanilla invariant function for vanilla CLIPPER
    iparams = clipperpy.invariants.EuclideanDistanceParams()
    # iparams.mindist = 10
    iparams.epsilon = epsilon
    iparams.sigma = 0.5 * iparams.epsilon
    invariant = clipperpy.invariants.EuclideanDistance(iparams)

    # Set up CLIPPER rounding parameters
    params = clipperpy.Params()
    params.rounding = clipperpy.Rounding.DSD_HEU
    clipper = clipperpy.CLIPPER(invariant, params)

    # Do semantic filtering of A
    target_labels_list = [np.int32(x) for x in target_labels]
    source_labels_list = [np.int32(x) for x in source_labels]
    A_sem_filtered, corr_filtered_labels = clipper.filter_semantic_associations(target_labels_list, source_labels_list, A_all_to_all)
    corr_filtered_labels = np.array(corr_filtered_labels, dtype=np.int32)

    # # Further downsample A and correlation labels to max A size
    # # TODO: Make sure accum_sel_indicies are kept
    A_filtered, corr_filtered_labels = downsample_association_matrix(A_sem_filtered, 
                                                                     stats_obj,
                                                                     max_size_A=max_A_size, 
                                                                     corr_labels=corr_filtered_labels)
    A_filtered = A_sem_filtered
    corr_filtered_labels = corr_filtered_labels
    
    # print(f"Size of A_filtered: {len(A_filtered)}")
    t0 = time.perf_counter()
    clipper.score_pairwise_consistency(D1, D2, A_filtered)
    t1 = time.perf_counter()
    time_to_score = t1 - t0

    t0 = time.perf_counter()
    clipper.solve()
    t1 = time.perf_counter()
    time_to_solve = t1 - t0

    # # Remove the semantic information from the point cloud data
    # D1 = D1[:-1, :]
    # D2 = D2[:-1, :]

    Ain = clipper.get_initial_associations()
    Ain_len = Ain.shape[0]
    A_sel = clipper.get_selected_associations()
    A_sel_len = A_sel.shape[0]
    
    # Save
    # np.save('A_in.npy', Ain)
    # np.save('A_sel.npy', A_sel)

    # print(f"Max elt in A selected: {np.max(Ain)}")
    print(f"    - Selected {A_sel_len} associations.")

    # get labels for semantic associations (for vis/saving pcds)
    _, asel_corr_filtered_labels = clipper.filter_semantic_associations(target_labels_list, source_labels_list, A_sel)

    # print(f"asel_corr_filtered_labels: {asel_corr_filtered_labels}")


    # # Create the histogram
    # plt.hist(asel_corr_filtered_labels, bins=len(list(set(asel_corr_filtered_labels))), edgecolor='black')

    # # Customize the plot
    # plt.title('Histogram Example')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')

    # # Show the plot
    # plt.show()


    # Unaligned Source Cloud PCD
    source_pcd_tf = o3d.geometry.PointCloud()
    source_pcd_tf.points = o3d.utility.Vector3dVector(D2.T)
    source_pcd_tf.colors = o3d.utility.Vector3dVector(source_labels_rgb)

    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    # p2p.with_scaling = True
    tf_estimate = p2p.compute_transformation(target_pcd, source_pcd_tf, o3d.utility.Vector2iVector(A_sel))

    rerr, terr = get_err(tf_ground_truth, tf_estimate)

    # print(f"Ground truth tf: {tf_ground_truth}\n")
    # print(f"Estimated tf: {tf_estimate}")
    print(f"    - Error: rerror = {rerr}, terror = {terr}")

    trans_gt = tf_ground_truth[0:3, 3]
    trans_est = tf_estimate[0:3, 3]
    rot_est = tf_estimate[0:3, 0:3]
    
    # Extend the accumulated selected index lists with the original pc indices
    # TODO: These will accumulate indices sometimes more than once, so we can actually just
    # use a map instead, where the indices are the keys and the number of times that index was added
    # is the value.
    # This will keep the list shorter as well as provide a gradient as to how these points may be colored ...
    # The 'coloring' of these points can be used as an additional class or even distance ...
    sel_target_indices, sel_source_indices = get_selected_indices(A_sel)
    sel_orig_target_indices = [int(target_index_map[sel_target_index]) for sel_target_index in sel_target_indices]
    sel_orig_source_indices = [int(source_index_map[sel_source_index]) for sel_source_index in sel_source_indices]
    
    if (A_sel_len) > 1:
        gt_rotation = R.from_matrix(rot_est)
        gt_quaternion_est = gt_rotation.as_quat()
        
        stats_obj.trans_pred_list.append(trans_est)
        stats_obj.rot_pred_list.append(gt_quaternion_est)
        
        for index in sel_orig_target_indices:
            if index in stats_obj.accum_sel_target_indices:
                stats_obj.accum_sel_target_indices[index] += 1
            else:
                stats_obj.accum_sel_target_indices[index] = 1
        
        for index in sel_orig_source_indices:
            if index in stats_obj.accum_sel_source_indices:
                stats_obj.accum_sel_source_indices[index] += 1
            else:
                stats_obj.accum_sel_source_indices[index] = 1
            
    print(f"    - len(accum_sel_target_indices): {len(stats_obj.accum_sel_target_indices.keys())}")
    print(f"    - accum_sel_source_indices: {len(stats_obj.accum_sel_source_indices.keys())}")
        
    # if iter > 0:
    #     # Create a sphere mesh with a radius of 1.0
    #     sphere_GT = o3d.geometry.TriangleMesh.create_sphere(radius=10.0)
    #     sphere_GT.paint_uniform_color([0, 1.0, 0])
    #     sphere_GT.compute_vertex_normals()
    #     sphere_GT.translate(trans_gt)
        
    #     sphere_EST = o3d.geometry.TriangleMesh.create_sphere(radius=10.0)
    #     sphere_EST.paint_uniform_color([1.0, 0, 0])
    #     sphere_EST.compute_vertex_normals()
    #     sphere_EST.translate(trans_est)

    #     # O3d geometry for initial point cloud correspondances
    #     corr_initial = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
    #         target_pcd, source_pcd_tf, Ain
    #     )

    #     # O3d geometry for selected point cloud correspondances
    #     corr_selected = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
    #         target_pcd, source_pcd_tf, A_sel
    #     )

    #     corr_filtered = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
    #         target_pcd, source_pcd_tf, A_filtered
    #     )
    #     corr_filtered_rgb = labels2RGB(corr_filtered_labels, labels_dict)
    #     corr_filtered.colors = o3d.utility.Vector3dVector(corr_filtered_rgb)

    #     # Add semantic colors to selected associations
    #     corr_selected_rgb = labels2RGB(asel_corr_filtered_labels, labels_dict)
    #     corr_selected.colors = o3d.utility.Vector3dVector(corr_selected_rgb)

    #     source_fixed = copy.deepcopy(source_pcd_tf).transform(np.linalg.inv(tf_estimate))
        
    #     # if show_visualizations or terr < 2.0:
    #     # Ground truth
    #     o3d.visualization.draw_geometries([target_pcd, source_pcd])
    #     # Unaligned source
    #     o3d.visualization.draw_geometries([target_pcd, source_pcd_tf, sphere_GT, sphere_EST])
    #     o3d.visualization.draw_geometries([target_pcd, source_pcd_tf])
    #     # Initial associations
    #     o3d.visualization.draw_geometries([target_pcd, source_pcd_tf, corr_initial])
    #     # Filtered associations
    #     o3d.visualization.draw_geometries([target_pcd, source_pcd_tf, corr_filtered])
    #     # Selected associations
    #     o3d.visualization.draw_geometries([target_pcd, source_pcd_tf, corr_selected])
    #     # Draw registration result: Map matching with correspondances and estimated transformation
    #     o3d.visualization.draw_geometries([source_fixed, target_pcd])


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

        # If using lidar/poses_world.txt
        transformation_matrix = lidar_poses[lidar_timestamps[frame_number]]

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
            (transformed_xyz, labels_np.reshape(-1, 1)), axis=1
        )

        # Extract only points with building or road labels 45 46
        desired_semantic_indices = np.where((labels_np == 45) | (labels_np == 46))[0]
        # desired_semantic_indices = np.where(labels_np == 72)[0]
        # semantic_points = semantic_points[desired_semantic_indices, :]
        # semantic_points[:, 2] = 0

        semantic_points_list.extend(semantic_points)

    return  np.asarray(semantic_points_list)


def get_accum_pc(root_dir, frame_inc, osm_file_path, use_latlon=False):
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
    semantic_points_np = get_accum_points(
        first_frame_idx=3000,
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

    # Get OSM data as points
    osm_building_points = get_osm_buildings_points(osm_file_path)
    osm_road_points = get_osm_road_points(osm_file_path)
    osm_grassland_points = get_osm_grassland_points(osm_file_path)

    osm_building_points_ego = convert_pointcloud_to_ego(osm_building_points, origin_latlon=init_gps_point)
    osm_road_points_ego = convert_pointcloud_to_ego(osm_road_points, origin_latlon=init_gps_point)

    osm_grassland_pcd = convert_polyline_points_to_o3d(osm_grassland_points, [0.0, 1.0, 0.0])
    osm_road_pcd = convert_polyline_points_to_o3d(osm_road_points_ego, [1, 0.2, 0.2])
    osm_building_pcd = convert_polyline_points_to_o3d(osm_building_points_ego, [0.2, 0.2, 1])

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
    return semantic_points_np, osm_road_points_ego

# def selectedPair():
#     def __init__(self):
        
class StatisticsManager:
    def __init__(self):
        self.accum_sel_target_indices = {}
        self.accum_sel_source_indices = {}
        self.rot_pred_list = []
        self.trans_pred_list = []
        self.rot_err_list = []
        self.trans_err_list = []
        self.rot_mean_err_list = []
        
        # Plot to see how mean error acts over iterations
        self.trans_mean_err_list = []
        
    # def make_and_save_plots():
    
if __name__ == "__main__":
    # Path to YAML-based ablation config file
    ablation_config = 'lidar2osm/config/ablation_config.yaml'
    with open(ablation_config, 'r') as file:
        ablation_data = yaml.safe_load(file)
    
    env = "kittredge_loop"
    env_path = f"/media/donceykong/donceys_data_ssd/datasets/CU_MULTI/data/{env}"

    root_dir = os.path.join(env_path, "robot1")
    robot1_sem_pc, r1_osm_road_points = get_accum_pc(root_dir, frame_inc=50, osm_file_path=f"{env_path}/{env}.osm", use_latlon=False)
    # r1_osm_sem_road_points = np.empty((len(r1_osm_road_points), 4))
    # r1_osm_sem_road_points[:, :3] = r1_osm_road_points
    # r1_osm_sem_road_points[:, 3] = 46

    root_dir = os.path.join(env_path, "robot3")
    robot3_sem_pc, r3_osm_road_points = get_accum_pc(root_dir, frame_inc=10000, osm_file_path=f"{env_path}/{env}.osm", use_latlon=False)

    # Get the size in bytes
    robot1_map_mbytes = robot1_sem_pc.nbytes / (1024 ** 2)
    robot3_map_mbytes = robot3_sem_pc.nbytes / (1024 ** 2)

    print(f"\nsize of robot1 map: {robot1_map_mbytes} MB")
    print(f"size of robot3 map: {robot3_map_mbytes} MB")

    # Generate ground truth transform(R,t) for the source cloud
    tf_ground_truth = np.eye(4)
    tf_ground_truth[0:3, 0:3] = R.random().as_matrix()
    tf_ground_truth[0:3, 3] = [1200, 6000, -100]
    # tf_ground_truth[0:3, 3] = np.random.uniform(low=-1000, high=1000, size=(3,))
    gt_rotation = R.from_matrix(tf_ground_truth[0:3, 0:3])
    gt_quaternion = gt_rotation.as_quat()
        
    stats_obj = StatisticsManager()
    for i in range(1000):
        iter = i + 1
        print(f"\niter: {iter}")
    
        align_maps(target_sem_pc=robot1_sem_pc,
                   source_sem_pc=robot3_sem_pc, 
                   stats_obj = stats_obj,
                   tf_ground_truth=tf_ground_truth,
                   iter=iter)
    
        if iter > 0:
            # Extract keys (indices) and values (good clique counts)
            sel_target_indices = list(stats_obj.accum_sel_target_indices.keys())
            sel_target_counts = list(stats_obj.accum_sel_target_indices.values())
            sel_source_indices = list(stats_obj.accum_sel_source_indices.keys())
            sel_source_counts = list(stats_obj.accum_sel_source_indices.values())

            # Plot the bar chart
            plt.subplot(2, 1, 1)
            plt.bar(sel_target_indices, sel_target_counts, color='skyblue', edgecolor='black')
            plt.xlabel('Selected TARGET Point Cloud Indices')
            plt.ylabel('Number of Times in selected as a max clique')
            plt.title('Good Clique Counts by Point Cloud Index')
            plt.xticks(sel_target_indices)

            plt.subplot(2, 1, 2)
            plt.bar(sel_source_indices, sel_source_counts, color='skyblue', edgecolor='black')
            plt.xlabel('Selected Source Point Cloud Indices')
            plt.ylabel('Number of Times in selected as a max clique')
            plt.title('Good Clique Counts by Point Cloud Index')
            plt.xticks(sel_source_indices)
            plt.tight_layout()
            # plt.show()
            plt.savefig(f'/media/donceykong/doncey_ssd_02/lidar2osm_plots/sel_indices/index_distribution_{iter}.png', format='png', dpi=300, bbox_inches='tight')
            plt.close()


            # Separate x, y, and z coordinates
            x_vals = [p[0] for p in stats_obj.trans_pred_list]
            x_mean = np.mean(x_vals)
            x_gt = tf_ground_truth[0, 3]
            
            y_vals = [p[1] for p in stats_obj.trans_pred_list]
            y_mean = np.mean(y_vals)
            y_gt = tf_ground_truth[1, 3]
            
            z_vals = [p[2] for p in stats_obj.trans_pred_list]
            z_mean = np.mean(z_vals)
            z_gt = tf_ground_truth[2, 3]
            
            # Create histograms for x, y, and z distributions
            plt.figure(figsize=(12, 6))

            # Plot x distribution
            plt.subplot(1, 3, 1)
            plt.hist(x_vals, bins=100, alpha=0.7, color='blue', edgecolor='black')
            plt.axvline(x_mean, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {x_mean:.2f}')
            plt.axvline(x_gt, color='green', linestyle='dashed', linewidth=2)
            plt.title('X Distribution')
            plt.xlabel('X')
            plt.ylabel('Frequency')

            # Plot y distribution
            plt.subplot(1, 3, 2)
            plt.hist(y_vals, bins=100, alpha=0.7, color='green', edgecolor='black')
            plt.axvline(y_mean, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {y_mean:.2f}')
            plt.axvline(y_gt, color='green', linestyle='dashed', linewidth=2)
            plt.title('Y Distribution')
            plt.xlabel('Y')
            plt.ylabel('Frequency')

            # Plot z distribution
            plt.subplot(1, 3, 3)
            plt.hist(z_vals, bins=100, alpha=0.7, color='red', edgecolor='black')
            plt.axvline(z_mean, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {z_mean:.2f}')
            plt.axvline(z_gt, color='green', linestyle='dashed', linewidth=2)
            plt.title('Z Distribution')
            plt.xlabel('Z')
            plt.ylabel('Frequency')

            # Show the plots
            plt.tight_layout()
            # plt.show()
            plt.savefig(f'/media/donceykong/doncey_ssd_02/lidar2osm_plots/trans/trans_distribution_{iter}.png', format='png', dpi=300, bbox_inches='tight')
            plt.close()



            trans_mean_err = [x_mean-x_gt, y_mean-y_gt, z_mean-z_gt]
            stats_obj.trans_mean_err_list.append(trans_mean_err)

            trans_mean_err_x = [p[0] for p in stats_obj.trans_mean_err_list]
            trans_mean_err_y = [p[1] for p in stats_obj.trans_mean_err_list]
            trans_mean_err_z = [p[2] for p in stats_obj.trans_mean_err_list]

            indices = range(len(stats_obj.trans_mean_err_list))
            plt.figure(figsize=(8, 6))
            plt.plot(indices, trans_mean_err_x, label='X Error', marker='o', linestyle='-', color='blue')
            plt.plot(indices, trans_mean_err_y, label='Y Error', marker='o', linestyle='-', color='green')
            plt.plot(indices, trans_mean_err_z, label='Z Error', marker='o', linestyle='-', color='red')
            plt.title('Translation Mean Error Over Iterations')
            plt.xlabel('Iteration')
            plt.ylabel('Mean Error')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            # plt.show()
            plt.savefig(f'/media/donceykong/doncey_ssd_02/lidar2osm_plots/trans_mean_err/trans_mean_err_{iter}.png', format='png', dpi=300, bbox_inches='tight')
            plt.close()
            
            
            
            
            # Separate x-roll, y-pitch, z-yaw, and w
            roll_vals = [p[0] for p in stats_obj.rot_pred_list]
            roll_mean = np.mean(roll_vals)
            roll_gt = gt_quaternion[0]
            
            pitch_vals = [p[1] for p in stats_obj.rot_pred_list]
            pitch_mean = np.mean(pitch_vals)
            pitch_gt = gt_quaternion[1]
            
            yaw_vals = [p[2] for p in stats_obj.rot_pred_list]
            yaw_mean = np.mean(yaw_vals)
            yaw_gt = gt_quaternion[2]

            w_vals = [p[3] for p in stats_obj.rot_pred_list]
            w_mean = np.mean(w_vals)
            w_gt = gt_quaternion[3]
            
            # Create histograms for x, y, and z distributions
            plt.figure(figsize=(12, 6))

            # Plot roll distribution
            plt.subplot(1, 4, 1)
            plt.hist(roll_vals, bins=100, alpha=0.7, color='blue', edgecolor='black')
            plt.axvline(roll_mean, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {roll_mean:.4f}')
            plt.axvline(roll_gt, color='green', linestyle='dashed', linewidth=2)
            plt.title('ROLL Distribution')
            plt.xlabel('ROLL')
            plt.ylabel('Frequency')

            # Plot pitch distribution
            plt.subplot(1, 4, 2)
            plt.hist(pitch_vals, bins=100, alpha=0.7, color='green', edgecolor='black')
            plt.axvline(pitch_mean, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {pitch_mean:.4f}')
            plt.axvline(pitch_gt, color='green', linestyle='dashed', linewidth=2)
            plt.title('PITCH Distribution')
            plt.xlabel('PITCH')
            plt.ylabel('Frequency')

            # Plot yaw distribution
            plt.subplot(1, 4, 3)
            plt.hist(yaw_vals, bins=100, alpha=0.7, color='red', edgecolor='black')
            plt.axvline(yaw_mean, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {yaw_mean:.4}')
            plt.axvline(yaw_gt, color='green', linestyle='dashed', linewidth=2)
            plt.title('YAW Distribution')
            plt.xlabel('YAW')
            plt.ylabel('Frequency')

            # Plot yaw distribution
            plt.subplot(1, 4, 4)
            plt.hist(w_vals, bins=100, alpha=0.7, color='orange', edgecolor='black')
            plt.axvline(w_mean, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {w_mean:.4}')
            plt.axvline(w_gt, color='green', linestyle='dashed', linewidth=2)
            plt.title('W Distribution')
            plt.xlabel('W')
            plt.ylabel('Frequency')
            
            # Show the plots
            plt.tight_layout()
            # plt.show()
            plt.savefig(f'/media/donceykong/doncey_ssd_02/lidar2osm_plots/roll/roll_distribution_{iter}.png', format='png', dpi=300, bbox_inches='tight')
            plt.close()
            
            
            
            
            rot_mean_err = [roll_mean-roll_gt, pitch_mean-pitch_gt, yaw_mean-yaw_gt, w_mean-w_gt]
            stats_obj.rot_mean_err_list.append(rot_mean_err)

            rot_mean_err_roll = [p[0] for p in stats_obj.rot_mean_err_list]
            rot_mean_err_pitch = [p[1] for p in stats_obj.rot_mean_err_list]
            rot_mean_err_yaw = [p[2] for p in stats_obj.rot_mean_err_list]
            rot_mean_err_w = [p[3] for p in stats_obj.rot_mean_err_list]

            indices = range(len(stats_obj.rot_mean_err_list))
            plt.figure(figsize=(8, 6))
            plt.plot(indices, rot_mean_err_roll, label='ROLL Error', marker='o', linestyle='-', color='blue')
            plt.plot(indices, rot_mean_err_pitch, label='PITCH Error', marker='o', linestyle='-', color='green')
            plt.plot(indices, rot_mean_err_yaw, label='YAW Error', marker='o', linestyle='-', color='red')
            plt.plot(indices, rot_mean_err_w, label='W Error', marker='o', linestyle='-', color='orange')
            plt.title('Rotation Mean Error Over Iterations')
            plt.xlabel('Iteration')
            plt.ylabel('Mean Error')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            # plt.show()
            plt.savefig(f'/media/donceykong/doncey_ssd_02/lidar2osm_plots/rot_mean_err/rot_mean_err_{iter}.png', format='png', dpi=300, bbox_inches='tight')
            plt.close()