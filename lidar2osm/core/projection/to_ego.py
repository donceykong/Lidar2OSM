#!/usr/bin/env python3

import numpy as np
from .utils import *


def convert_pose_to_ego(points, origin_latlon):
    """Convert a numpy array of poses from lat-long to XYZ. 
    """
    # compute scale from lat value of the origin
    scale = lat_to_scale(origin_latlon[0])

    # origin in Mercator coordinate
    ox, oy = latlon_to_mercator(origin_latlon[0], origin_latlon[1], scale)
    origin = np.array([ox, oy, 0])
    
    ego_poses = []
    for i in range(len(points)):
        # if there is no data => not a pointcloud
        if not len(points[i]):
            ego_poses.append([])
            continue

        # # rotation and translation
        R = points[i, 0:3, 0:3]
        trans = points[i, 0:3, 3]

        # translation vector
        tx, ty = latlon_to_mercator(trans[0], trans[1], scale)
        t = np.array([tx, ty, trans[2]])
  
        # rotation matrix (OXTS RT3000 user manual, page 71/92)
        # Extract elements of the rotation matrix
        r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
        r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
        r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]

        # Compute roll, pitch, and yaw
        pitch = np.arcsin(-r31)
        yaw = np.arctan2(r21, r11)
        roll = np.arctan2(r32, r33)

        rx = roll   #points[i][3] # roll
        ry = pitch  #points[i][4] # pitch
        rz = yaw    #points[i][5] # heading
        Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]]) # base => nav  (level oxts => rotated oxts)
        Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]]) # base => nav  (level oxts => rotated oxts)
        Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]]) # base => nav  (level oxts => rotated oxts)
        R  = np.matmul(np.matmul(Rz, Ry), Rx)
    
        # normalize translation
        t = t - origin
        
        # add points
        ego_pose = np.vstack((np.hstack((R,t.reshape(3,1))),np.array([0,0,0,1])))
        ego_poses.append(ego_pose)

    return ego_poses


def convert_pointcloud_to_ego(points, origin_latlon):
    """ Converts entire pc to ego frame.
    """
    points_as_poses_latlon = np.array([np.eye(4) for _ in range(points.shape[0])])
    points_as_poses_latlon[:, 0:3, 3] = points[:, :3]
    points_as_poses_ego = np.asarray(convert_pose_to_ego(points_as_poses_latlon, origin_latlon))[:, :3]
    points_as_poses_ego = np.asarray(post_process_points(points_as_poses_ego))
    points_ego = points_as_poses_ego[:, 0:3, 3]

    return points_ego