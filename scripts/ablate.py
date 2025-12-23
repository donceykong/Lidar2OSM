#!/usr/bin/env python3

import copy
import os
import re
import time
from typing import Tuple
import numpy as np
np.set_printoptions(suppress=True)
import open3d as o3d
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import yaml
import sys

# Testing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Internal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import clipperpy as clipperpy_import
from lidar2osm.datasets import kitti360_dataset
from lidar2osm.datasets import cu_multi_dataset
from lidar2osm.datasets.kitti360_dataset import KITTI360_DATASET
from lidar2osm.datasets.cu_multi_dataset import CU_MULTI_DATASET
from lidar2osm.utils.file_io import read_bin_file
from lidar2osm.core.pointcloud import labels2RGB, get_transformed_point_cloud
from lidar2osm.core.ablation.processor import L2O_AblationStudyProcessor
from lidar2osm.core.ablation.visualizer import L2O_Visualization

def evaluate(log_dir):
    # Main log directory
    # log_dir = "/home/donceykong/Desktop/ARPG/projects/fall_2024/Lidar2OSM_FULL/Lidar2OSM/results/ablation_logs1"
    # log_dir = "/home/donceykong/Desktop/ARPG/projects/fall_2024/Lidar2OSM_FULL/Lidar2OSM/results/ablation_large"

    # Initialize processor
    processor = L2O_AblationStudyProcessor(log_dir=log_dir, verbose_output=True)

    # Try setting min and max values for different resulting stats
    min_key_values = {} #{'terr': -20.0, 'rerr': -0.5}
    max_key_values = {} # {'terr': 20.00, 'rerr': 0.5}

    ablation_args1 = {"clipper_style": ["baseline_clipper"],      # Filter by "semantic_filtering" or "baseline_clipper"
                      # "epsilon": [9.5],                         # Filter by epsilon range(0.5, 10, 0.5)
                      "pointcloud_style": ["regular"],            # "regular" or "osm"
                      # Note: Leave other keys as empty lists to include all possible values for those keys
                      }
    # Set ablation arguments with a dictionary specifying which values to filter by
    processor.set_ablation_args(ablation_args1)
    results_1 = processor.find_and_process_files(min_val_key_values=min_key_values, max_val_key_values=max_key_values)

    # Set ablation arguments with a dictionary specifying which values to filter by
    ablation_args2 = {"clipper_style": ["baseline_clipper"],      # Filter by "semantic_filtering" or "baseline_clipper"
                      # "epsilon": [9.5],                         # Filter by epsilon range(0.5, 10, 0.5)
                      "pointcloud_style": ["osm"],            # "regular" or "osm"
                      # Note: Leave other keys as empty lists to include all possible values for those keys
                      }
    
    processor.set_ablation_args(ablation_args2)
    results_2 = processor.find_and_process_files(min_val_key_values=min_key_values, max_val_key_values=max_key_values)

    ablation_args3 = {"clipper_style": ["semantic_filtering"],      # Filter by "semantic_filtering" or "baseline_clipper"
                      # "epsilon": [9.5],                         # Filter by epsilon range(0.5, 10, 0.5)
                      "pointcloud_style": ["regular"],            # "regular" or "osm"
                      # Note: Leave other keys as empty lists to include all possible values for those keys
                      }
    processor.set_ablation_args(ablation_args3)
    results_3 = processor.find_and_process_files(min_val_key_values=min_key_values, max_val_key_values=max_key_values)

    # Set ablation arguments with a dictionary specifying which values to filter by
    ablation_args4 = {"clipper_style": ["semantic_filtering"],      # Filter by "semantic_filtering" or "baseline_clipper"
                      # "epsilon": [9.5],                         # Filter by epsilon range(0.5, 10, 0.5)
                      "pointcloud_style": ["osm"],            # "regular" or "osm"
                      # Note: Leave other keys as empty lists to include all possible values for those keys
                      }
    processor.set_ablation_args(ablation_args4)
    results_4 = processor.find_and_process_files(min_val_key_values=min_key_values, max_val_key_values=max_key_values)

    data_dicts = [results_1, results_2, results_3, results_4]

     # Initialize visualizer
    visualizer = L2O_Visualization()

    # Visualize Scatterplots
    titles = ["basline/regular", "basline/osm", "Semantic-Clipper/regular", "Semantic-Clipper/OSM"]

    # 2D Scatter plots
    variables = ['num_sel_assoc', 'terr']
    visualizer.scatterplot_2D(data_dicts, variables, titles=titles, xlabel=variables[0], ylabel=variables[1], color_low="green", color_high="black")
    
    variables = ['num_sel_assoc', 'rerr']
    visualizer.scatterplot_2D(data_dicts, variables, titles=titles, xlabel=variables[0], ylabel=variables[1], color_low="green", color_high="black")

    variables = ['epsilon', 'num_sel_assoc']
    visualizer.scatterplot_2D(data_dicts, variables, titles=titles, xlabel=variables[0], ylabel=variables[1], color_low="black", color_high="green")

    variables = ['epsilon', 'terr']
    visualizer.scatterplot_2D(data_dicts, variables, titles=titles, xlabel=variables[0], ylabel=variables[1], color_low="green", color_high="black")

    variables = ['epsilon', 'terr']
    visualizer.scatterplot_2D(data_dicts, variables, titles=titles, xlabel=variables[0], ylabel=variables[1], color_low="green", color_high="black")

    # 2D Histogram
    variable = 'terr'
    visualizer.histogram_2D(data_dicts, variable, bins=50, titles=titles, color_low="green", color_high="black")

    variable = 'rerr'
    visualizer.histogram_2D(data_dicts, variable, bins=50, titles=titles, color_low="green", color_high="black")

    variable = 'time_to_solve'
    visualizer.histogram_2D(data_dicts, variable, bins=50, titles=titles, color_low="green", color_high="black")

    variable = 'time_to_score'
    visualizer.histogram_2D(data_dicts, variable, bins=50, titles=titles, color_low="green", color_high="black")

    # 3D Scatterplot
    variables = ['terr', 'rerr', 'num_sel_assoc']
    visualizer.scatterplot_3d(data_dicts, variables, titles=titles, color_low="green", color_high="black")

    # 3D Histogram
    variables = ['epsilon', 'num_sel_assoc']
    visualizer.histogram_3D(data_dicts, variables, bins=25, titles=titles, color_low="black", color_high="green")

    variables = ['rerr', 'terr']
    visualizer.histogram_3D(data_dicts, variables, bins=25, titles=titles, color_low="green", color_high="red")
    
# # TODO: Delete or keep this class?
# class Custom(clipperpy.invariants.PairwiseInvariant):
#     def __init__(self, σ=0.06, ϵ=0.01):
#         clipperpy.invariants.PairwiseInvariant.__init__(self)
#         self.σ = σ
#         self.ϵ = ϵ

#     def __call__(self, ai, aj, bi, bj):
#         l1 = np.linalg.norm(ai - aj)
#         l2 = np.linalg.norm(bi - bj)

#         c = np.abs(l1 - l2)

#         return np.exp(-0.5 * c**2 / self.σ**2) if c < self.ϵ else 0

def generate_Abad(n1, n2) -> np.array:
    """
    Create incorrect associations from the indices of view1 and view2
    """
    # Create a grid of all index combinations
    I, J = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")

    # Flatten the arrays to get all combinations as column vectors
    I_flat = I.flatten()
    J_flat = J.flatten()

    # Filter out the pairs where indices are the same
    mask = I_flat != J_flat
    Abad = np.vstack((I_flat[mask], J_flat[mask])).T

    return Abad

def createSemanticallyFilteredArray(labels1, labels2, A):
    """
    Filter out the associations to only those that have the same semantic label.

    Args:
        - labels1: The semantic labels of the points in view 1
        - labels2: The semantic labels of the points in view 2

    Returns:
        - Anew: The filtered array of associations
    """
    # Initialize the new array to store the filtered results and the corresponding labels
    Anew = []
    corr_filtered_labels = []

    # Iterate through each entry in array A
    for entry in A:
        # Get the indices from dataset1 and dataset2
        index1 = entry[0]
        index2 = entry[1]

        # Get the corresponding labels from dataset1 and dataset2
        label1 = labels1[index1]
        label2 = labels2[index2]

        # Check if the labels are the same.
        if label1 == label2 and label1 != -1:
            # print(f"label1: {label1}")
            # If they are, add the entry to the new array.
            Anew.append(entry)
            corr_filtered_labels.append(label1)

    Anew = np.array(Anew)
    corr_filtered_labels = np.array(corr_filtered_labels)

    return Anew, corr_filtered_labels

def generate_associations(
    target_pcd, source_pcd, n1, n2, nia, noa, tf_ground_truth
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate Dataset: view1, view2, associations ground truth, associations with inliers and outliers"""

    if nia > n1:
        raise ValueError(
            "Cannot have more inlier associations "
            "than there are model points. Increase"
            "the number of points to sample from the"
            "original point cloud model."
        )
    
    # The target point cloud
    D1 = np.asarray(target_pcd.points).T

    # Transform the source point cloud
    D2 = np.asarray(source_pcd.points).T
    D2 = tf_ground_truth[0:3, 0:3] @ D2 + tf_ground_truth[0:3, 3].reshape(-1, 1)

    # Correct associations to draw from
    Agood = np.tile(np.arange(n1).reshape(-1, 1), (1, 2))
    Abad = generate_Abad(n1, n2)

    # Sample good and bad associations to satisfy total
    # num of associations with the requested outlier ratio
    IAgood = np.random.choice(Agood.shape[0], nia, replace=False)
    IAbad = np.random.choice(Abad.shape[0], noa, replace=False)

    # Create association matrix
    A = np.concatenate((Agood[IAgood, :], Abad[IAbad, :])).astype(np.int32)

    # Ground truth associations
    Agt = Agood[IAgood, :]

    return (D1, D2, Agt, A)

def get_a2a_assoc_matrix(N1, N2):
   assoc_matrix = np.zeros((N1*N2,2),np.int32)
 
   i = 0
   for n1 in range(N1):
       for n2 in range(N2):
           assoc_matrix[i,0] = n1
           assoc_matrix[i,1] = n2
           i += 1
 
   return assoc_matrix

def downsample_association_matrix(A, max_size_A=10000, corr_labels=None) -> None:    
    ''' Downsamples an association matrix in-place

    Args:
        A (ndarray):
            The nxn association matrix for the target and source pointclouds to be matched.
        max_size_A (int): 
            The size of what the assocition matrix A should be downsampled to.
        corr_labels (ndarray):
            The 

    Returns:
        A,corr_labels (Tuple [ndarray, ndarray]):
        Downsampled association matrix.
    '''

    # randomly downsample A indices
    rand_ds_A_idxs = np.random.choice(A.shape[0], size=max_size_A, replace=False)   #rand_ds_A_idxs

    # Downsample correlation labels too if passed in
    if corr_labels is not None:
        corr_labels = corr_labels[rand_ds_A_idxs]
    
    return A[rand_ds_A_idxs], corr_labels

# TODO: --------------------- end class SemanticCLIPPER ------------------------


# --------------------- class Transform ---------------------
def get_err(tf_ground_truth, tf_estimate) -> Tuple[float, float]:
    """
    Returns the error of the estimated transformation matrix.
    """

    Terr = np.linalg.inv(tf_ground_truth) @ tf_estimate
    rerr = abs(np.arccos(min(max(((Terr[0:3, 0:3]).trace() - 1) / 2, -1.0), 1.0)))
    terr = np.linalg.norm(Terr[0:3, 3])

    return (rerr, terr)
# TODO: --------------------- end class Transform -------------------------

def read_poses(file_path):
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
    
# TODO: --------------------- class Dataset ---------------------
def get_data_file_paths(root_path, seq):
    print(f"\ndataset: {dataset}\n")
    seq = int(seq)
    sequence_dir = f"2013_05_28_drive_{seq:04d}_sync"
    data_file_paths = [
        os.path.join(root_path, "data_poses", sequence_dir, "velodyne_poses.txt"),
        os.path.join(root_path, "data_3d_raw", sequence_dir, "velodyne_points", "data"),
        os.path.join(root_path, "data_3d_semantics", sequence_dir, "labels"),
        os.path.join(root_path, "data_3d_semantics", sequence_dir, "osm_labels"),
        os.path.join(root_path, "data_3d_semantics", sequence_dir, "inferred"),
    ]
    return data_file_paths


def get_frame_numbers(directory_path) -> list:
    """
    Count the total number of files in the directory
    """
    frame_numbers = []
    all_files = os.listdir(directory_path)

    # Filter out files ending with ".bin" and remove the filetype
    filenames = [
        os.path.splitext(file)[0] for file in all_files if file.endswith(".bin")
    ]

    for filename in filenames:
        frame_numbers.append(int(filename))

    return sorted(frame_numbers)

# TODO: --------------------- end class Dataset -------------------------

def get_accum_points(
    dataset, first_frame_idx, final_frame_idx, frame_increment, pointcloud_style
) -> Tuple[np.array, np.array]:
    """
    Iterate through osm_frames and accumulate points (x,y,z,semantic) in their respective semantic class, road or building
    """
    osm_frames = get_frame_numbers(dataset.osm_label_path)
    # print(f"Len OSM frames: {len(osm_frames)}")
    semantic_points_list = []
    velodyne_poses = dataset.read_poses(dataset.lidar_poses_file)
    for frame_idx in range(first_frame_idx, final_frame_idx, frame_increment):
        frame_number = osm_frames[frame_idx]

        raw_pc_frame_path = os.path.join(dataset.raw_pc_path, f"{frame_number:010d}.bin")
        points_np = read_bin_file(raw_pc_frame_path, dtype=np.float32, shape=(-1, 4))

        if pointcloud_style=="full":
            pc_frame_label_path = os.path.join(dataset.label_path, f"{frame_number:010d}.bin")
            labels_np = read_bin_file(pc_frame_label_path, dtype=np.int16, shape=(-1))
        elif pointcloud_style=="inferred_full":
            pc_frame_label_path = os.path.join(dataset.inferred_full_sem_path, f"{frame_number:010d}.bin")
            labels_np = read_bin_file(pc_frame_label_path, dtype=np.int32, shape=(-1))
        elif pointcloud_style=="gt_osm":
            pc_frame_label_path = os.path.join(dataset.osm_label_path, f"{frame_number:010d}.bin")
            labels_np = read_bin_file(pc_frame_label_path, dtype=np.int32, shape=(-1))
        elif pointcloud_style=="inferred_osm":
            pc_frame_label_path = os.path.join(dataset.inferred_osm_path, f"{frame_number:010d}.bin")
            labels_np = read_bin_file(pc_frame_label_path, dtype=np.int32, shape=(-1))

        pc_xyz = points_np[:, :3]
        intensities_np = points_np[:, 3]
        # semantic_points = np.concatenate(
        #     (pc, intensities_np.reshape(-1, 1), labels_np.reshape(-1, 1)), axis=1
        # )
        points_transformed = get_transformed_point_cloud(pc_xyz, velodyne_poses, frame_number)
        semantic_points = enate(
            (points_transformed, labels_np.reshape(-1, 1)), axis=1
        )

        # Filter for desired semantics
        if pointcloud_style=="gt_osm" or pointcloud_style=="inferred_osm":
            desired_semantic_indices = np.where((labels_np == 45) | (labels_np == 46))[0]
            semantic_points = semantic_points[desired_semantic_indices, :]

        semantic_points_list.extend(semantic_points)
    semantic_points = np.asarray(semantic_points_list)
    return semantic_points

def get_semantic_pointclouds(dataset, target_source_frames, pointcloud_style) :
    """Generates downsampled semantic point clouds from association matrices.

    This function processes association matrices and downsamples them in-place
    to extract 4D semantic point clouds (num_points, x, y, z, semantic_id).

    Args:
        target_source_frames (list[int]): 
            A list containing start and end frame indices and increments 
            for both target and source point cloud generation.
        data_file_paths (list[str]): 
            File paths needed to extract the required data, including the 
            location of pointcloud scans and associated semantic labels.
        pointcloud_style (str): 
            The style or format of the point cloud data (e.g., "regular", "osm").

    Returns:
        tuple: 
            - target_semantic_pointcloud (ndarray): A 4D array of shape 
              (num_points, x, y, z, semantic_id) with `np.float32` data type 
              for the target point cloud.
            - source_semantic_pointcloud (ndarray): A 4D array of shape 
              (num_points, x, y, z, semantic_id) with `np.float32` data type 
              for the source point cloud.

    Notes:
        - The function retrieves the appropriate pointcloud scans from the specified 
          `data_file_paths` (constrained by pointcloud_style) and processes the point clouds 
          based on frame indices and increments specified in `target_source_frames`.
    """

    # Accumulate the target points
    target_semantic_pointcloud = get_accum_points(
        dataset = dataset,
        first_frame_idx=target_source_frames[0],
        final_frame_idx=target_source_frames[1],
        frame_increment=target_source_frames[2],
        pointcloud_style=pointcloud_style,
    )

    # Accumulate source points
    source_semantic_pointcloud = get_accum_points(
        dataset = dataset,
        first_frame_idx=target_source_frames[3],
        final_frame_idx=target_source_frames[4],
        frame_increment=target_source_frames[5],
        pointcloud_style=pointcloud_style,
    )

    return target_semantic_pointcloud, source_semantic_pointcloud

def voxel_downsample_pc(semantic_pc, max_points, epsilon_init, voxel_leaf_size_init):
    """
    Increase epsilon so that it is at least as large as voxel leaf size, or else neighboring
    points may not even be evaluated for pairwise consistency.

    """
    points = semantic_pc[:, :3]
    sem_labels = semantic_pc[:, 3]

    # Covert labels to RGB stack for downsampling
    sem_labels_3x3 = np.stack([sem_labels] * 3, axis=-1)

    # Create o3d object and voxel-downsample
    semantic_pcd = o3d.geometry.PointCloud()
    semantic_pcd.points = o3d.utility.Vector3dVector(points)
    semantic_pcd.colors = o3d.utility.Vector3dVector(sem_labels_3x3)

    current_voxel_leaf_size = voxel_leaf_size_init
    current_epsilon = epsilon_init + current_voxel_leaf_size
    
    # Do initial downsampling
    semantic_ds_pcd = semantic_pcd.voxel_down_sample(current_voxel_leaf_size)

    print_init_stats = True
    while (len(semantic_ds_pcd.points) > max_points):
        if print_init_stats:
            print(f"    - Point clouds are beyond max number of points.")
            print(f"        - size_semantic_cloud: {len(semantic_ds_pcd.points)}")
            print_init_stats = False
        current_voxel_leaf_size += 0.5
        current_epsilon = current_voxel_leaf_size + epsilon_init
        semantic_ds_pcd = semantic_pcd.voxel_down_sample(current_voxel_leaf_size)
    if not print_init_stats:
        print(f"        - Final voxel leaf size needed to reduce points below max: {current_voxel_leaf_size}")

    sem_labels_ds = np.asarray(semantic_ds_pcd.colors, dtype=np.int16)[:, 0]

    semantic_pc_ds = np.zeros((len(semantic_ds_pcd.points), 4))
    semantic_pc_ds[:, :3] = semantic_ds_pcd.points
    semantic_pc_ds[:, 3] = sem_labels_ds

    return semantic_pc_ds, current_epsilon, current_voxel_leaf_size

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


# TODO: --------------------- class Ablation -------------------------
def ablate(test_metadata):
    clipperpy = clipperpy_import

    # Get vars from metadata dict
    ablation_dir = os.path.join(
        test_metadata["log_dir"],
        f"test_{test_metadata['test_id']:05d}"
    )
    
    seq = test_metadata["sequence"]
    if test_metadata["dataset"]  == "KITTI_360":
        dataset = KITTI360_DATASET(root_path = test_metadata["root_path"])
        seq = int(seq)
    elif test_metadata["dataset"] == "CU_MULTI":
        dataset = CU_MULTI_DATASET(root_path = test_metadata["root_path"])
        seq = str(seq)
    dataset.setup_sequence(seq)

    semantic_pointclouds = get_semantic_pointclouds(dataset,
                                                    test_metadata["target_source_frames"], 
                                                    test_metadata["pointcloud_style"])
    
    clipper_style=test_metadata["clipper_style"]
    show_visualizations=test_metadata["show_vis"] 
    num_test_iteratons=test_metadata["trial_count"]
    pointcloud_style=test_metadata["pointcloud_style"]
    max_points=test_metadata["max_points"]
    max_A_size = test_metadata["max_A_size"]
    downsampling_style = test_metadata["downsampling_style"]
    epsilon_param=test_metadata["epsilon"]
    verbose=test_metadata["verbose_output"]
    save_pcds=test_metadata["save_pcds"]
    voxel_leaf_size = test_metadata["voxel_leaf_size"]

    # Print 
    if verbose:
        print(f"\n**********************************************************************")

        if clipper_style=="semantic_filtering":
            print(f"Vanilla CLIPPER with semantic filtering: {num_test_iteratons} tests.")
        elif clipper_style=="semantic_invariant":
            print(f"Semantic CLIPPER with custom invariant function: {num_test_iteratons} tests.")
        elif clipper_style=="baseline_clipper":
            print(f"Baseline CLIPPER: {num_test_iteratons} tests.")
        else:
            print("ERROR")
    
    # Inferred data uses KITTI labeling style by default...
    if pointcloud_style == "inferred":
        labels_dict = {label.id: label.color for label in cu_multi_dataset.labels}
    else:
        labels_dict = {label.id: label.color for label in kitti360_dataset.labels} # Semantic label colors, given a semantic label id number

    # Open log file to store ablation results
    log_path = os.path.join(ablation_dir, "ablation_results.json")
    log_data = []

    target_semantic_pointcloud, source_semantic_pointcloud = semantic_pointclouds

    # Shift semantic pcs closer to origin using average location in target pc
    center_point_x = (np.average(target_semantic_pointcloud[:, 0]) + np.average(source_semantic_pointcloud[:, 0])) / 2
    center_point_y = (np.average(target_semantic_pointcloud[:, 1]) + np.average(source_semantic_pointcloud[:, 1])) / 2
    center_point_z = (np.average(target_semantic_pointcloud[:, 2]) + np.average(source_semantic_pointcloud[:, 2])) / 2

    center_point = np.array([center_point_x, center_point_y, center_point_z])

    target_semantic_pointcloud[:, :3] = target_semantic_pointcloud[:, :3] - center_point
    source_semantic_pointcloud[:, :3] = source_semantic_pointcloud[:, :3] - center_point
    
    # Run tests
    for test_iter in range(num_test_iteratons):
        if verbose:
            print(f"--------------------------------------------")
            print(f"Trial {test_iter + 1}")
        
        epsilon = None  # Init eps to none
        target_semantic_pointcloud_ds = None
        source_semantic_pointcloud_ds = None
        # Downsample semantic pcs
        if downsampling_style == "voxel_downsample":
            target_voxel_ds_list = voxel_downsample_pc(target_semantic_pointcloud,
                                                       max_points=max_points,
                                                       epsilon_init=epsilon_param,
                                                       voxel_leaf_size_init=voxel_leaf_size)
            target_semantic_pointcloud_ds, current_epsilon, current_voxel_leaf_size = target_voxel_ds_list
            
            source_voxel_ds_list = voxel_downsample_pc(source_semantic_pointcloud,
                                                       max_points=max_points,
                                                       epsilon_init=epsilon_param,
                                                       voxel_leaf_size_init=voxel_leaf_size)
            source_semantic_pointcloud_ds, current_epsilon, current_voxel_leaf_size = source_voxel_ds_list
            
            # Set eps as current_epsilon after voxel-downsampling source cloud
            # TODO: log these if voxel ds used
            init_voxel_leaf_size = voxel_leaf_size
            final_voxel_leaf_size = current_voxel_leaf_size
            epsilon = current_epsilon

            # Make sure target and source have sane number of points due to issue in
            # filtering associations requiring A to be square
            target_sem_pc_size = len(target_semantic_pointcloud_ds)
            source_sem_pc_size = len(source_semantic_pointcloud_ds)
            if target_sem_pc_size > source_sem_pc_size:
                I_target = np.random.choice(target_sem_pc_size, source_sem_pc_size, replace=False)
                target_semantic_pointcloud_ds = target_semantic_pointcloud_ds[I_target, :]
            elif target_sem_pc_size < source_sem_pc_size:
                I_source = np.random.choice(source_sem_pc_size, target_sem_pc_size, replace=False)
                source_semantic_pointcloud_ds = source_semantic_pointcloud_ds[I_source, :]

        elif downsampling_style == "random_downsample":
            target_semantic_pointcloud_ds = random_downsample_pc(target_semantic_pointcloud,
                                                                 max_points=max_points)
            source_semantic_pointcloud_ds = random_downsample_pc(source_semantic_pointcloud, 
                                                                 max_points=max_points)
            epsilon = epsilon_param
        
        # separate downsampled points and labels
        target_labels = target_semantic_pointcloud_ds[:, 3]
        target_pointcloud = target_semantic_pointcloud_ds[:, :3]

        source_labels = source_semantic_pointcloud_ds[:, 3]
        source_pointcloud = source_semantic_pointcloud_ds[:, :3]

        # Convert semantic ids to rgb arrays
        target_labels_rgb = labels2RGB(target_labels, labels_dict)
        source_labels_rgb = labels2RGB(source_labels, labels_dict)

        # Increase brightness of source cloud colors and normalize bw 0:1
        source_labels_rgb[:, :] +=  0.5
        source_labels_rgb = np.clip(source_labels_rgb, 0.0, 1.0)

        # # Apply a transformation matrix to shift and rotate the colors in RGB space
        # color_trans_matrix = np.array([
        #     [0.5, 0.3, 0.5],
        #     [0.3, 0.5, 0.3],
        #     [0.5, 0.3, 0.5]
        # ])
        # source_labels_rgb = np.clip(np.dot(source_labels_rgb, color_trans_matrix.T), 0, 1)

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_pointcloud)
        target_pcd.colors = o3d.utility.Vector3dVector(target_labels_rgb)

        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_pointcloud)
        source_pcd.colors = o3d.utility.Vector3dVector(source_labels_rgb)
    
        # Generate ground truth transform(R,t) for the source cloud
        tf_ground_truth = np.eye(4)
        tf_ground_truth[0:3, 0:3] = Rotation.random().as_matrix()
        tf_ground_truth[0:3, 3] = np.random.uniform(low=-1000, high=1000, size=(3,))

        # # ---- Generate dataset ---- #
        # n1 = int(len(source_pcd.points))            # number of inliers
        # n2o = int(len(target_pcd.points)) - n1      # number of outliers in data (points from both target and source that do not overlap)
        # n2 = n1 + n2o                               # Total # points (number of points in view 2)
        # m = n1                                      # total number of associations in problem
        # outrat = 0.01                               # outlier ratio of initial association set
        # noa = n2 #round(m * outrat)                     # number of outlier associations
        # nia = n1 #m - noa                               # number of inlier associations
        # D1, D2, Agt, A = generate_associations(target_pcd, source_pcd, n1, n2, nia, noa, tf_ground_truth)

        # ---- Generate dataset (ALL-TO-ALL) ---- #
        D1 = np.asarray(target_pcd.points).T
        D2 = np.asarray(source_pcd.points).T
        D2 = tf_ground_truth[0:3, 0:3] @ D2 + tf_ground_truth[0:3, 3].reshape(-1, 1)
        A_all_to_all = get_a2a_assoc_matrix(len(target_pcd.points), len(source_pcd.points))

        if verbose:
            print(f"\nInitial size of A: {len(A_all_to_all)}")

        # min_dist_param = 1.0
        if clipper_style=="sem_invariant":
            # Set up custom invariant function for semantic CLIPPER
            iparams = clipperpy.invariants.SemanticsConstrainedEuclideanDistanceParams()
            iparams.epsilon = epsilon
            iparams.sigma =  0.5 * iparams.epsilon
            # iparams.mindist = min_dist_param
            invariant = clipperpy.invariants.SemanticsConstrainedEuclideanDistance(iparams)

            # Add the semantic information to the point cloud data
            target_pc_sc_reshaped = target_labels.reshape(1, -1)
            source_pc_sc_reshaped = source_labels.reshape(1, -1)
            D1 = np.concatenate((D1, target_pc_sc_reshaped), axis=0)
            D2 = np.concatenate((D2, source_pc_sc_reshaped), axis=0)
        else:
            # Set up vanilla invariant function for vanilla CLIPPER
            iparams = clipperpy.invariants.EuclideanDistanceParams()
            iparams.epsilon = epsilon
            iparams.sigma = 0.5 * iparams.epsilon
            # iparams.mindist = min_dist_param
            invariant = clipperpy.invariants.EuclideanDistance(iparams)

        # # TODO: Just testing
        # # Add the semantic information to the point cloud data
        # target_pc_sc_reshaped = target_labels.reshape(1, -1)
        # source_pc_sc_reshaped = source_labels.reshape(1, -1)
        # D1 = np.concatenate((D1, target_pc_sc_reshaped), axis=0)
        # D2 = np.concatenate((D2, source_pc_sc_reshaped), axis=0)

        # Set up CLIPPER rounding parameters
        params = clipperpy.Params()
        params.rounding = clipperpy.Rounding.DSD_HEU
        clipper = clipperpy.CLIPPER(invariant, params)

        # Do semantic filtering of A
        if clipper_style=="semantic_filtering":
            #     A, corr_filtered_labels = createSemanticallyFilteredArray(
            #         target_labels, source_labels, A
            #     )
            target_labels_list = [np.int32(x) for x in target_labels]
            source_labels_list = [np.int32(x) for x in source_labels]
            A_sem_filtered, corr_filtered_labels = clipper.filter_semantic_associations(target_labels_list, source_labels_list, A_all_to_all)
            corr_filtered_labels = np.array(corr_filtered_labels, dtype=np.int32)
            # if verbose:
            #     print(f"    - Size of A (after semantic filtering): {len(A_sem_filtered)}")

            # Further downsample A and correlation labels to max A size
            A_filtered, corr_filtered_labels = downsample_association_matrix(A_sem_filtered, 
                                                                             max_size_A=max_A_size, 
                                                                             corr_labels=corr_filtered_labels)
        else:
            # Downsample A to max A size
            A_filtered, _ = downsample_association_matrix(A_all_to_all, max_size_A=max_A_size)

        # if verbose:
        #     print(f"    - Size of A (after reducing): {len(A_filtered)}")

        t0 = time.perf_counter()
        clipper.score_pairwise_consistency(D1, D2, A_filtered)
        t1 = time.perf_counter()
        time_to_score = t1 - t0
            
        # if verbose:
        #     print(f"Affinity matrix creation took {t1-t0:.3f} seconds")

        t0 = time.perf_counter()
        clipper.solve()
        t1 = time.perf_counter()
        time_to_solve = t1 - t0

        Ain = clipper.get_initial_associations()
        Ain_len = Ain.shape[0]
        A_sel = clipper.get_selected_associations()
        A_sel_len = A_sel.shape[0]

        # get labels for semantic associations (for vis/saving pcds)
        asel_corr_filtered_labels = None
        if clipper_style=="semantic_filtering":
            _, asel_corr_filtered_labels = createSemanticallyFilteredArray(
                target_labels, source_labels, A_sel
            )

        # # p = np.isin(Ain, Agt)[:, 0].sum() / Ain.shape[0]
        # # r = np.isin(Ain, Agt)[:, 0].sum() / Agt.shape[0]
        # if verbose:
        #     print(
        #         f"CLIPPER selected {A_sel.shape[0]} inliers from {Ain.shape[0]} "
        #         # f"putative associations (precision {p:.2f}, recall {r:.2f}) in {t1-t0:.3f} s"
        #     )

        # if clipper_style=="sem_invariant":
        #     # Remove the semantic information from the point cloud data
        #     D1 = D1[:-1, :]
        #     D2 = D2[:-1, :]

        # Unaligned Source Cloud PCD
        source_pcd_tf = o3d.geometry.PointCloud()
        source_pcd_tf.points = o3d.utility.Vector3dVector(D2.T)
        source_pcd_tf.colors = o3d.utility.Vector3dVector(source_labels_rgb)

        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        # p2p.with_scaling = True
        tf_estimate = p2p.compute_transformation(target_pcd, source_pcd_tf, o3d.utility.Vector2iVector(A_sel))

        rerr, terr = get_err(tf_ground_truth, tf_estimate)

        test_result = {
            "trial_id": f"{test_iter + 1:05d}",
            "epsilon_param": epsilon_param,
            "epsilon": epsilon,
            "rerr": rerr,
            "terr": terr,
            "num_init_assoc": int(Ain_len),
            "num_sel_assoc": int(A_sel_len),
            "time_to_score": round(time_to_score, 3),
            "time_to_solve": round(time_to_solve, 3),
            "ground_truth_tf": tf_ground_truth.flatten().tolist(),
            "estimated_tf": tf_estimate.flatten().tolist(),
        }
        log_data.append(test_result)

        if verbose:
            print(f"Ground truth tf: {tf_ground_truth}\n")
            print(f"Estimated tf: {tf_estimate}")
            print(f"Error: rerror = {rerr}, terror = {terr}")

        trans_gt = tf_ground_truth[0:3, 3]
        trans_est = tf_estimate[0:3, 3]
        
        # Only create o3d objects if saving pcds or visualizing
        if save_pcds or show_visualizations:
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

            # O3d geometry for selected point cloud correspondances
            corr_selected = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
                target_pcd, source_pcd_tf, A_sel
            )

            # Filtered point cloud correspondances
            if clipper_style=="semantic_filtering":
                corr_filtered = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
                    target_pcd, source_pcd_tf, A_filtered
                )
                corr_filtered_rgb = labels2RGB(corr_filtered_labels, labels_dict)
                corr_filtered.colors = o3d.utility.Vector3dVector(corr_filtered_rgb)

                # Add semantic colors to selected associations
                corr_selected_rgb = labels2RGB(asel_corr_filtered_labels, labels_dict)
                corr_selected.colors = o3d.utility.Vector3dVector(corr_selected_rgb)

        if show_visualizations or verbose:
            # Fixed source point cloud
            source_fixed = copy.deepcopy(source_pcd_tf).transform(np.linalg.inv(tf_estimate))
        
        if show_visualizations:
        # if show_visualizations or terr < 2.0:
            # Ground truth
            o3d.visualization.draw_geometries([target_pcd, source_pcd])
            # Unaligned source
            o3d.visualization.draw_geometries([target_pcd, source_pcd_tf, sphere_GT, sphere_EST])
            # Initial associations
            o3d.visualization.draw_geometries([target_pcd, source_pcd_tf, corr_initial])
            # Filtered associations
            if clipper_style=="semantic_filtering":
                o3d.visualization.draw_geometries([target_pcd, source_pcd_tf, corr_filtered])
            # Selected associations
            o3d.visualization.draw_geometries([target_pcd, source_pcd_tf, corr_selected])
            # Draw registration result: Map matching with correspondances and estimated transformation
            o3d.visualization.draw_geometries([source_fixed, target_pcd])

        # # Save the downsampled point clouds before tf, after tf, the correspondances, and the fixed tf
        # if save_pcds:
        #     # Create a directory to store the ply files
        #     ply_dir = f"{ablation_dir}/ply"
        #     os.makedirs(ply_dir, exist_ok=True)

        #     # Save the point clouds
        #     if pointcloud_style=="regular":
        #         o3d.io.write_point_cloud(f"{ablation_dir}/ply/trial_{test_iter + 1:05d}_target.ply", target_pcd)
        #         o3d.io.write_point_cloud(f"{ablation_dir}/ply/trial_{test_iter + 1:05d}_source.ply", source_pcd)
            
        #     # Save the correspondances
        #     o3d.io.write_line_set(f"{ablation_dir}/ply/trial_{test_iter + 1:05d}_corr_initial.ply", corr_initial)
        #     if clipper_style=="semantic_filtering":
        #         o3d.io.write_line_set(f"{ablation_dir}/ply/trial_{test_iter + 1:05d}_corr_filtered.ply", corr_filtered)
        #     o3d.io.write_line_set(f"{ablation_dir}/ply/trial_{test_iter + 1:05d}_corr_selected.ply", corr_selected)

        # if verbose:
        #     # Get tf differences betwean points after inverse transforming
        #     source_points_np = np.asarray(source_pcd.points)
        #     source_fixed_points_np = np.asarray(source_fixed.points)
        #     # source_trans_np = source_points_np + tf_ground_truth[0:3, 3]
        #     # source_fixed_points_np = source_trans_np - tf_estimate[0:3, 3]
        #     diff = source_points_np - source_fixed_points_np
        #     ave_diff_x = np.average(diff[:, 0])
        #     ave_diff_y = np.average(diff[:, 1])
        #     ave_diff_z = np.average(diff[:, 2])
        #     ave_diff = np.asarray([ave_diff_x, ave_diff_y, ave_diff_z])
        #     terr_points = np.linalg.norm(ave_diff)
        #     print(f"\n\n ave_diff: {ave_diff}, terr_points: {terr_points}")
        # Save all results to JSON log

    # Write all results to the JSON log file
    with open(log_path, "w") as json_file:
        json.dump(log_data, json_file, indent=4)


def run_ablation_study(log_dir):
    # Load the overall ablation metadata
    ablation_metadata_file = os.path.join(log_dir, "ablation_metadata.json")
    with open(ablation_metadata_file, 'r') as f:
        ablation_metadata = json.load(f)

    total_iterations = len(ablation_metadata)
    print("\n ***Running ablation study*** \n")
    
    with tqdm(total=total_iterations, desc="Ablation Study") as pbar:
        for test_metadata in ablation_metadata:
            # Get directory path for this specific test
            test_dir = os.path.join(log_dir, f"test_{test_metadata['test_id']:05d}")
            
            # Continue if the test directory has ablation results log
            if os.path.exists(os.path.join(test_dir, "ablation_results.json")):
                pbar.update(1)
                pbar.refresh()  # Force an immediate update of the progress bar
                continue

            ablate(test_metadata)
            
            # Update progress bar
            pbar.update(1)


# from multiprocessing import Pool
# def run_ablation_study(log_dir):
#     # Load the overall ablation metadata
#     ablation_metadata_file = os.path.join(log_dir, "ablation_metadata.json")
#     with open(ablation_metadata_file, 'r') as f:
#         ablation_metadata = json.load(f)

#     all_tests = [test for test in ablation_metadata]
#     print("\n*** Running ablation study ***\n")

#     # Filter out already completed tests
#     pending_tests = []
#     for test_metadata in all_tests:
#         test_dir = os.path.join(log_dir, f"test_{test_metadata['test_id']:05d}")
#         if not os.path.exists(os.path.join(test_dir, "ablation_results.json")):
#             pending_tests.append(test_metadata)

#     # Prepare arguments for the ablate function
#     args_list = [test_metadata for test_metadata in pending_tests]

#     # Run the ablation study with multiprocessing
#     with Pool(processes=2) as pool:
#         with tqdm(total=len(args_list), desc="Ablation Study") as pbar:
#             for _ in pool.imap(ablate, args_list):  # No separate `log_dir` argument here
#                 pbar.update(1)
#                 pbar.refresh()


def create_test_directories(
    log_dir_path,
    datasets,
    pointcloud_styles,
    epsilon_list,
    max_assoc_matrix_list,
    max_points_list,
    downsampling_styles,
    clipper_styles,
    trial_count,
    show_vis,
    verbose_output,
    voxel_leaf_size,
    save_pcds,
):
    """
    Creates test directories for ablation studies and saves metadata for each test.

    Parameters:
        log_dir_path (str): The base directory where test directories will be created.
        datasets (list): List of datasets to be used in the study.
        sequences (list): List of sequences from each dataset.
        epsilon_list (list): List of epsilon parameter values.
        max_points_list (list): List of max points to sample from point clouds.
        pointcloud_styles (list): List of point cloud styles to use.
        clipper_styles (list): List of clipper styles to apply.
        target_source_frames_list (list): List of frame indices for target and source clouds.
        trial_count (int): Number of trials to run for each parameter set.

    Returns:
        list: A list of paths to the created test directories.
    """
    test_dirs = []
    test_dir_idx = -1
    ablation_metadata = []

    for dataset in datasets:
        for seq in datasets[dataset]["sequences"]:
            for target_source_frames in datasets[dataset]["target_source_frames"]:
                for pointcloud_style in pointcloud_styles:
                    for epsilon in epsilon_list:
                        for max_A_size in max_assoc_matrix_list:
                            for max_points in max_points_list:
                                for clipper_style in clipper_styles:
                                    for downsampling_style in downsampling_styles:
                                        test_dir_idx += 1
                                        test_dir = os.path.join(log_dir_path, f"test_{test_dir_idx+1:05d}")
                                        os.makedirs(test_dir, exist_ok=True)

                                        # Convert numpy types to native Python types
                                        metadata = {
                                            "log_dir": str(log_dir_path),
                                            "test_id": int(test_dir_idx + 1),
                                            "dataset": str(dataset),
                                            "root_path": datasets[dataset]["root_path"],
                                            "sequence": str(seq),
                                            "epsilon": float(epsilon),
                                            "max_A_size": int(max_A_size),
                                            "max_points": int(max_points),
                                            "downsampling_style": str(downsampling_style),
                                            "pointcloud_style": str(pointcloud_style),
                                            "clipper_style": str(clipper_style),
                                            "target_source_frames": [int(x) for x in target_source_frames],
                                            "trial_count": int(trial_count),
                                            "show_vis": bool(show_vis),
                                            "verbose_output": bool(verbose_output),
                                            "voxel_leaf_size": float(voxel_leaf_size),
                                            "save_pcds": bool(save_pcds),
                                        }

                                        # Save individual metadata for each test directory
                                        metadata_file = os.path.join(test_dir, "metadata.json")
                                        with open(metadata_file, 'w') as f:
                                            json.dump(metadata, f, indent=4)

                                        # Append to overall ablation study metadata
                                        ablation_metadata.append(metadata)
                                        test_dirs.append(test_dir)

    # Save overall ablation metadata
    ablation_metadata_file = os.path.join(log_dir_path, "ablation_metadata.json")
    with open(ablation_metadata_file, 'w') as f:
        json.dump(ablation_metadata, f, indent=4)

    return test_dirs


if __name__ == "__main__":
    # Path to YAML-based ablation config file
    ablation_config = 'config/ablation_config.yaml'
    with open(ablation_config, 'r') as file:
        ablation_data = yaml.safe_load(file)
    
    # Main log directory
    log_dir = ablation_data['log_dir_path']
    os.makedirs(log_dir, exist_ok=True)
    
    # Global ablation study parameters (not limited to particular dataset)
    trial_count = ablation_data['trial_count']                      # Number of tests to run for each ablation parameter set
    max_points_list = ablation_data['max_points_list']              # max number of points to use in source-target PCs
    pointcloud_styles = ablation_data["pointcloud_styles"]          # "regular" and/or "osm" point cloud styles                
    clipper_styles = ["semantic_filtering", "baseline_clipper"]     # "baseline_clipper", "semantic_filtering", and/or "sem_invariant" clipper styles
    downsampling_styles = ablation_data["downsampling_styles"]      # "random_downsample" or "voxel_downsample"

    # Epsilon parameter values: these are the distance beyond the voxel leaf size for a map
    epsilon_params = ablation_data['epsilon_params']
    epsilon_list = np.arange(epsilon_params["min"], 
                             epsilon_params["max"],
                             epsilon_params["step"])

    # Max association mat size parameter values
    max_A_params = ablation_data['max_A_params']
    max_assoc_matrix_list = np.arange(max_A_params["min"], 
                                      max_A_params["max"],
                                      max_A_params["step"])

    datasets = {}
    for dataset in ablation_data['datasets']:
        dataset_params = ablation_data['datasets'][dataset]
        if (dataset_params['enabled']):
            target_source_frames_list = []
            target_source_frames = dataset_params['target_source_frames']

            for target_source_frame in target_source_frames.values():
                frames = target_source_frame["target_frames"]
                frames.extend(target_source_frame["source_frames"])
                target_source_frames_list.append(frames)
            datasets[dataset] = {"target_source_frames": target_source_frames_list, 
                                 "sequences": dataset_params["sequences"],
                                 "root_path": dataset_params["root_path"]}
    
    test_dirs = create_test_directories(
        log_dir,
        datasets,
        pointcloud_styles,
        epsilon_list,
        max_assoc_matrix_list,
        max_points_list,
        downsampling_styles,
        clipper_styles,
        trial_count,
        show_vis=ablation_data["show_vis"],
        verbose_output=ablation_data["verbose_output"],
        voxel_leaf_size = ablation_data['voxel_leaf_size'], # If "voxel_downsample" is in downsampling_styles, this will be used
        save_pcds=ablation_data["save_pcds"],
    )

    print(f"\nPreparing to run ablation study...")
    print(f"    - Number of test directories created: {len(test_dirs)}.")
    print(f"    - Number of trials per test: {trial_count}.")
    print(f"    - Total iterations (num_tests * num_trials_per_test): {len(test_dirs) * trial_count}")

    run_ablation_study(log_dir=log_dir)