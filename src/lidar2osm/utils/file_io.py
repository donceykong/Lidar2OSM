#!/usr/bin/env python3

# External
import os
import numpy as np
import numpy.typing as npt


def save_poses(file_path: str, xyz_poses: dict[int, npt.NDArray[np.float64]]):
    """
    Saves 4x4 transformation matrices to a file.

    Args:
        file_path (str): The path to the file where poses will be saved.
        xyz_poses (dict): A dictionary where keys are indices and values are 4x4 numpy arrays representing transformation matrices.

    Returns:
        None
    """
    with open(file_path, "w") as file:
        for idx, matrix_4x4 in xyz_poses.items():
            flattened_matrix = matrix_4x4.flatten()
            line = f"{idx} " + " ".join(map(str, flattened_matrix)) + "\n"
            file.write(line)


# Function to read GPS data from a file
def read_gps_data(file_path):
    points = []
    with open(file_path, "r") as file:
        for line in file:
            lat, lon, alt = map(float, line.strip().split())
            points.append([lat, lon, alt])
    return np.array(points)


def read_bin_file(file_path, dtype, shape=None):
    """
    Reads a .bin file and reshapes the data according to the provided shape.

    Args:
        file_path (str): The path to the .bin file.
        dtype (data-type): The data type of the file content (e.g., np.float32, np.int16).
        shape (tuple, optional): The desired shape of the output array. If None, the data is returned as a 1D array.

    Returns:
        np.ndarray: The data read from the .bin file, reshaped according to the provided shape.
    """
    data = np.fromfile(file_path, dtype=dtype)
    if shape:
        return data.reshape(shape)
    return data


def save_extracted_data(
    extracted_per_frame_dir,
    frame_num,
    full_pcd_points,
    osm_points,
):
    """
    Saves extracted point cloud data to files.

    Args:
        extracted_per_frame_dir (str): The directory where extracted data will be saved.
        frame_num (int): The frame number.
        full_pcd_points (np.ndarray): The full point cloud points.
        osm_points (np.ndarray): The OSM building and road points.
    """
    lidar_points_file = os.path.join(
        extracted_per_frame_dir, f"{frame_num:010d}_lidar_points"
    )
    osm_points_file = os.path.join(
        extracted_per_frame_dir, f"{frame_num:010d}_osm_points"
    )
    np.save(lidar_points_file, full_pcd_points)
    np.save(osm_points_file, osm_points)


def read_extracted_data(extracted_per_frame_dir, frame_num):
    """
    Retrieves saved extracted data for a specific frame.

    Args:
        extracted_per_frame_dir (str): The directory containing the saved data.
        frame_num (int): The frame number.

    Returns:
        tuple: A tuple containing OSM building points, OSM road points, and lidar points if the files exist, otherwise None.
    """
    osm_points_file = os.path.join(
        extracted_per_frame_dir, f"{frame_num:010d}_osm_points.npy"
    )
    lidar_points_file = os.path.join(
        extracted_per_frame_dir, f"{frame_num:010d}_lidar_points.npy"
    )

    if os.path.exists(osm_points_file):
        osm_points = np.load(osm_points_file, allow_pickle=True)
        lidar_points = np.load(lidar_points_file, allow_pickle=True)

        return osm_points, lidar_points
    else:
        return None, None
