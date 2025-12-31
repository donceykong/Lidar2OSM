#!/usr/bin/env python3

import numpy as np
import open3d as o3d

def get_edge_pcd(edge_points_file, rgb_color):
    """
    Reads edge points from a file and returns an Open3D LineSet.

    Args:
        edge_points_file (str): The path to the file containing edge points.
        rgb_color (list): A list of three elements specifying the RGB color for the edges.

    Returns:
        o3d.geometry.LineSet: An Open3D LineSet object representing the edges.
    """
    build_edges_points = np.load(edge_points_file)
    build_edges_pcd = o3d.geometry.LineSet()
    if len(build_edges_points) > 0:
        build_edges_lines_idx = [
            [i, i + 1] for i in range(0, len(build_edges_points) - 1, 2)
        ]
        build_edges_pcd.points = o3d.utility.Vector3dVector(build_edges_points)
        build_edges_pcd.lines = o3d.utility.Vector2iVector(build_edges_lines_idx)
        build_edges_pcd.paint_uniform_color(rgb_color)
    return build_edges_pcd

def get_gnss_data_pcd(file_path, color_array):
    gnss_data = []

    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            # Split the line by comma and extract latitude, longitude, and altitude
            parts = line.strip().split(",")
            latitude = float(parts[0].split(":")[1].strip())
            longitude = float(parts[1].split(":")[1].strip())
            altitude = float(parts[2].split(":")[1].strip())

            gnss_data.append([longitude, latitude, 0])

    gnss_data_points_pcd = o3d.geometry.PointCloud()
    gnss_data_points_pcd.points = o3d.utility.Vector3dVector(gnss_data)
    gnss_data_points_pcd.paint_uniform_color(
        color_array
    )  # Black color for gnss frame points

    return gnss_data_points_pcd


def convert_OSM_list_to_o3d_rect(osm_list, rgb_color):
    # Initialize the LineSet object
    osm_line_set = o3d.geometry.LineSet()

    # Initialize an empty list to store points and lines
    points = []
    lines = []
    point_index = 0  # This will keep track of the index for points to form lines

    # Function to compute perpendicular vector to a line segment
    def perpendicular_vector(v):
        perp = np.array([-v[1], v[0], 0])
        norm = np.linalg.norm(perp)
        if norm == 0:
            return perp  # To avoid division by zero
        return perp / norm

    # Process each road line
    for road_line in osm_list:
        start_point = np.array(road_line["start_point"])
        end_point = np.array(road_line["end_point"])
        width = np.float64(road_line["width"]) / (
            63781.37 / 2.0
        )  # TODO: Need to make sure width is not a string and Need to convert the points to latlon

        # Compute the unit vector perpendicular to the line segment
        line_vec = end_point - start_point
        perp_vec = perpendicular_vector(line_vec) * (width / 2)

        # Calculate vertices for the polygon (rectangle)
        v1 = start_point - perp_vec
        v2 = start_point + perp_vec
        v3 = end_point + perp_vec
        v4 = end_point - perp_vec

        # Add points to the list
        points.extend([v1, v2, v3, v4])

        # Add lines to connect these points into a rectangle
        # Connect v1-v2, v2-v3, v3-v4, and v4-v1 to close the rectangle
        lines.extend(
            [
                [point_index, point_index + 1],
                [point_index + 1, point_index + 2],
                [point_index + 2, point_index + 3],
                [point_index + 3, point_index],
            ]
        )
        point_index += 4  # Move to the next set of vertices

    if len(points) != 0:
        # Convert list of points and lines into Open3D Vector format
        osm_line_set.points = o3d.utility.Vector3dVector(np.array(points))
        osm_line_set.lines = o3d.utility.Vector2iVector(np.array(lines))

        # Paint all lines with the specified color
        osm_line_set.paint_uniform_color(rgb_color)

    return osm_line_set


def convert_pc_to_o3d(point_cloud, rgb_color):
    colored_pcd = o3d.geometry.PointCloud()
    if len(point_cloud) > 0:
        colored_pcd.points = o3d.utility.Vector3dVector(point_cloud)
        colored_pcd.paint_uniform_color(rgb_color)
    return colored_pcd


def convert_polyline_points_to_o3d(polyline_points, rgb_color):
    polyline_pcd = o3d.geometry.LineSet()
    if len(polyline_points) > 0:
        polyline_lines_idx = [[i, i + 1] for i in range(0, len(polyline_points) - 1, 2)]
        polyline_pcd.points = o3d.utility.Vector3dVector(polyline_points)
        polyline_pcd.lines = o3d.utility.Vector2iVector(polyline_lines_idx)
        polyline_pcd.paint_uniform_color(rgb_color)
    return polyline_pcd


def get_circle_pcd(center, radius, num_points=30):
    """
    Create a circle at a given center point.

    :param center: Center of the circle (x, y, z).
    :param radius: Radius of the circle.
    :param num_points: Number of points to approximate the circle.
    :return: Open3D point cloud representing the circle.
    """
    points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = 0  # center[2]
        points.append([x, y, z])

    circle_pcd = o3d.geometry.PointCloud()
    circle_pcd.points = o3d.utility.Vector3dVector(points)
    return circle_pcd
