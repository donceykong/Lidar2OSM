#!/usr/bin/env python3

import numpy as np
from .utils import *


def ego_pose_to_latlon(pose, origin_oxts):
    single_value = not isinstance(pose, list)
    if single_value:
        pose = [pose]

    scale = lat_to_scale(origin_oxts[0])
    ox, oy = latlon_to_mercator(origin_oxts[0], origin_oxts[1], scale)
    origin = np.array([ox, oy, 0])

    tf_matrices = []

    for i in range(len(pose)):
        if not len(pose[i]):
            tf_matrices.append([])
            continue

        R = pose[i][0:3, 0:3]
        t = pose[i][0:3, 3] + origin

        # Convert to latitude, longitude, altitude
        lat, lon = mercator_to_latlon(t[0], t[1], scale)
        alt = t[2]

        # Combine rotation matrix R and translation t into a 4x4 matrix
        T = np.eye(4)  # Start with an identity matrix
        T[0:3, 0:3] = R
        T[0:3, 3] = [lat, lon, alt]  # Using lon, lat, alt as the translation part

        tf_matrices.append(T)

    if single_value:
        return tf_matrices[0]
    else:
        return tf_matrices


def convert_ego_poses_to_latlon(poses, origin_oxts):
    poses_latlon = {}
    for idx, pose in poses.items():
        poses_latlon[idx] = np.asarray(ego_pose_to_latlon(pose, origin_oxts))
    return poses_latlon


def convert_pose_to_latlon(points, origin_latlon):
    """Convert a numpy array of points from XYZ to lat-long.

    Args:
        points (): _description_

    Notes:
        All points must be relative to some (0, 0) coordinate, which is 
        aligned with 'origin_latlon'.
    """
    # compute scale from lat value of the origin
    scale = lat_to_scale(origin_latlon[0])

    # origin in Mercator coordinate
    ox, oy = latlon_to_mercator(origin_latlon[0], origin_latlon[1], scale)
    origin = np.array([ox, oy, 0])

    poses_latlon = []
    for i in range(len(points)):
        # if there is no data => no pose
        if not len(points[i]):
            poses_latlon.append([])
            continue

        # rotation and translation
        R = points[i, 0:3, 0:3]
        t = points[i, 0:3, 3]

        # unnormalize translation
        t = t + origin

        # translation vector
        lat, lon = mercator_to_latlon(t[0], t[1], scale)

        alt = t[2]

        # rotation matrix (OXTS RT3000 user manual, page 71/92)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        roll = np.arctan2(R[2, 1], R[2, 2])

        # add oxts
        poses_latlon.append([lat, lon, alt, roll, pitch, yaw])

    return poses_latlon

def convert_pointcloud_to_latlon(points, origin_latlon):
    """ Converts entire pc in ego frame to latlon frame.
    """
    pc_reshaped = np.array([np.eye(4) for _ in range(points.shape[0])])
    pc_reshaped[:, 0:3, 3] = points[:, :3]
    pc_lla = np.asarray(post_process_points(pc_reshaped))
    pc_lla = np.asarray(convert_pose_to_latlon(pc_lla, origin_latlon=origin_latlon))[:, :3]

    return pc_lla