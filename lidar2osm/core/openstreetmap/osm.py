#!/usr/bin/env python3

import numpy as np
import osmnx as ox
from scipy.spatial import cKDTree

class OSM:
    def __init__(self):
        self.type = "None"  # Either "building" or "road" for now
        self.edges = []     # both buildings and roads are composed of edges

    def add_edge(self, start_point, end_point):
        self.edges.append([start_point, end_point])


class OSMRoadPolyline(OSM):
    def __init__(self):
        super().__init__()
        self.type = "road"


class OSMBuildingPolygon(OSM):
    def __init__(self):
        super().__init__()
        self.type = "building"


class OSMItemList:
    def __init__(self):
        super().__init__()
        self.item_list = []  # List of OSM Feature objects
        self.points = None

    def add_item(self, item):
        if isinstance(item, OSMBuildingPolygon) or isinstance(item, OSMRoadPolyline):
            self.item_list.append(item)
        else:
            raise ValueError("Only OSMBuildingPolygon or OSMRoadPolyline objects can be added.")

    def get_items(self):
        return self.item_list
    
    def get_egdes_of_items_as_list_of_points(self):
        points = []
        for item in self.item_list:
            for edge in item.edges:
                edge = np.asarray(edge)
                if len(edge.shape) > 2: # Used if inlier edges not removed (method below), len of shape will be 3
                    print(len(edge.shape))
                    # print(f"\nedge.shape: {edge.shape}\n")
                    edge = edge.reshape(edge.shape[0]*edge.shape[1], edge.shape[2])
                for vertex in edge:
                    points.append(vertex)
        self.points = np.asarray(points)
        return points
    
    def remove_inlier_edges(self):
        for item in self.item_list:
            reduced_edges = []
            for edge in item.edges:
                edge = np.asarray(edge)
                edge = edge.reshape(edge.shape[0]*edge.shape[1], edge.shape[2])
                edge = sort_points_along_edge(edge)
                edge = [edge[0], edge[-1]]
                reduced_edges.append(edge)
            item.edges = reduced_edges


def sort_points_along_edge(points):
    if len(points) < 2:
        raise ValueError("Need at least two points to define an edge")

    points = np.asarray(points)
    # Calculate the direction vector from the first two points
    direction = points[1] - points[0]
    unit_vector = direction / np.linalg.norm(direction)

    # Calculate the projection of each point onto the direction vector
    projections = np.dot(points, unit_vector)
    
    # Get the indices that would sort the projections
    sorted_indices = np.argsort(projections)
    
    # Sort the points array using the sorted indices
    sorted_points = points[sorted_indices]
    
    return sorted_points


def convert_OSM_list_to_points(osm_list):
    points = []
    for edge in osm_list:
        # Add the start and end points to the points list
        points.append(edge["start_point"])
        points.append(edge["end_point"])

    return np.array(points)


def road_near_pose(road_vertex, pos, threshold):
    pos = pos[0:3, 3]
    road_vertex = np.array(road_vertex)
    vert_dist = np.sqrt(
        (pos[0] - road_vertex[0]) * (pos[0] - road_vertex[0])
        + (pos[1] - road_vertex[1]) * (pos[1] - road_vertex[1])
    )
    return vert_dist <= threshold


def building_near_pose(building_vertex, pos, threshold) -> bool:
    """
    Returns true or false if the OSM building center is less than the given threshold distance
    from pos.
    """
    pos = pos[:3, 3] # Extract translation vector
    building_vertex = np.asarray(building_vertex)
    vert_dist_sq = np.sum((pos[:2] - building_vertex[:2])**2)  # squared distance
    return vert_dist_sq <= threshold**2


def set_osm_roads_lines(osm_roads_list, osm_file_path, pos_lat_lon, threshold_dist, road_tags):
    # For below tags, see: https://wiki.openstreetmap.org/wiki/Key:highway#Roads
    # tags_OG = {'highway': ['residential', 'tertiary']}  # This will fetch tertiary and residential roads

    # Fetch roads using defined tags
    roads = ox.features_from_xml(osm_file_path, tags=road_tags)

    # Process Roads as LineSets with width
    road_lines = []
    for _, road in roads.iterrows():
        if road.geometry.geom_type == "LineString":
            coords = np.array(road.geometry.xy).T
            coords_fixed = np.copy(coords)
            coords_fixed[:, 0] = coords[:, 1]
            coords_fixed[:, 1] = coords[:, 0]

            road_center = [
                np.mean(np.array(coords_fixed)[:, 0]),
                np.mean(np.array(coords_fixed)[:, 1]),
            ]
            if road_near_pose(road_center, np.asarray(pos_lat_lon), threshold_dist):
                osm_road = OSMRoadPolyline()
                for i in range(len(coords_fixed) - 1):
                    start_point = [coords_fixed[i][0], coords_fixed[i][1], 0]  # Assuming roads are at ground level (z=0)
                    end_point = [coords_fixed[i + 1][0], coords_fixed[i + 1][1], 0]
                    osm_road.add_edge(start_point, end_point)
                osm_roads_list.add_item(osm_road)


def set_osm_buildings_lines(
    osm_building_list, osm_file_path, pos_lat_lon, threshold_dist, building_tags
):
    # Filter features for buildings and sidewalks
    buildings = ox.features_from_xml(osm_file_path, tags=building_tags)

    # Process Buildings as LineSets
    for _, building in buildings.iterrows():
        if building.geometry.geom_type == "Polygon":
            exterior_coords = np.array(building.geometry.exterior.coords)
            exterior_coords_fixed = np.copy(exterior_coords)
            exterior_coords_fixed[:, 0] = exterior_coords[:, 1]
            exterior_coords_fixed[:, 1] = exterior_coords[:, 0]

            build_center = [
                np.mean(np.array(exterior_coords_fixed)[:, 0]),
                np.mean(np.array(exterior_coords_fixed)[:, 1]),
            ]
            if building_near_pose(
                build_center, np.asarray(pos_lat_lon), threshold_dist
            ):
                osm_building = OSMBuildingPolygon()
                for i in range(len(exterior_coords_fixed) - 1):
                    start_point = [exterior_coords_fixed[i][0], exterior_coords_fixed[i][1], 0]
                    end_point = [exterior_coords_fixed[i + 1][0], exterior_coords_fixed[i + 1][1], 0]
                    osm_building.add_edge(start_point, end_point)
                osm_building_list.add_item(osm_building)


# def get_osm_tree_points(osm_file_path):
#     # Define tags for querying trees
#     tags = {"natural": "tree"}

#     # Fetch tree points using defined tags
#     trees = ox.geometries_from_xml(osm_file_path, tags=tags)

#     # Process Trees
#     tree_points = []
#     for _, tree in trees.iterrows():
#         if tree.geometry.type == "Point":
#             x, y = tree.geometry.xy
#             tree_points.append(
#                 [x[0], y[0], 0]
#             )  # Assuming trees are at ground level (z=0)

#     return tree_points


# def get_osm_sidewalk_list(osm_file_path):
#     # Define tags for querying roads
#     tags = {
#         "highway": ["footway", "path", "cycleway"]
#     }  # This will fetch tertiary and residential roads

#     # Fetch roads using defined tags
#     sidewalks = ox.features_from_xml(osm_file_path, tags=tags)

#     # Process Roads as LineSets with width
#     sidewalk_lines = []
#     for _, sidewalk in sidewalks.iterrows():
#         if sidewalk.geometry.geom_type == "LineString":
#             coords = np.array(sidewalk.geometry.xy).T
#             for i in range(len(coords) - 1):
#                 start_point = [coords[i][0], coords[i][1], 0]  # Assuming sidewalks are at ground level (z=0)
#                 end_point = [coords[i + 1][0], coords[i + 1][1], 0]
#                 sidewalk_lines.append(
#                     {
#                         "start_point": start_point,
#                         "end_point": end_point,
#                     }
#                 )
#     return sidewalk_lines


def get_edges_near_semantic_points(
    osm_item_list, semantic_2d_points, near_path_threshold
):
    """
    Filter edges based on proximity to semantic points.

    Args:
    edge_list (list): List of edges, where each edge is represented as [start_point, end_point].
    semantic_2d_points (list): List of 2D points representing semantic features.
    near_path_threshold (float): Distance threshold to consider an edge close to a semantic point.
    """
    new_osm_item_list = OSMItemList()
    point_cloud_2D_kdtree = cKDTree(semantic_2d_points)

    for osm_item in osm_item_list.item_list:
        if osm_item.type == "building":
            new_osm_item = OSMBuildingPolygon()
        elif osm_item.type == "road":
            new_osm_item = OSMRoadPolyline()
        for edge in osm_item.edges:
            new_edge = []
            for sub_edge in edge:
                sub_edge_center = (sub_edge[1] - sub_edge[0]) / 2 + sub_edge[0]
                # Query the KDTree for points within the threshold distance from the center
                indices = point_cloud_2D_kdtree.query_ball_point(
                    sub_edge_center, near_path_threshold
                )
                if len(indices):
                    new_edge.extend(sub_edge)
            if len(new_edge) > 0:
                new_osm_item.edges.append(edge)
        if (
            len(new_osm_item.edges) > 0
        ):  # Each item (building/road) must have at least one edge
            if osm_item.type == "building":
                new_osm_item_list.add_item(new_osm_item)
            elif osm_item.type == "road":
                new_osm_item_list.add_item(new_osm_item)
    osm_item_list.item_list = new_osm_item_list.get_items()


def get_semantic_points_near_osm_edges(
    osm_item_list, semantic_3d_points, near_path_threshold
):
    """
    Filter semantic points based on proximity to OSM edges.

    Args:
    edge_list (list): List of edges, where each edge is represented as [start_point, end_point].
    semantic_2d_points (list): List of 2D points representing semantic features.
    near_path_threshold (float): Distance threshold to consider an edge close to a semantic point.
    """
    semantic_3d_xyz = semantic_3d_points[:, :3]
    semantic_2d_xyz = np.copy(semantic_3d_xyz) * np.array([1, 1, 0])
    semantic_2d_xyz_kdtree = cKDTree(semantic_2d_xyz)

    near_points = set()
    near_points_intensities = set()
    near_points_labels = set()
    for osm_item in osm_item_list.item_list:
        for edge in osm_item.edges:
            for sub_edge in edge:
                sub_edge_center = (sub_edge[1] - sub_edge[0]) / 2 + sub_edge[0]
                # Query the KDTree for points within the threshold distance from the center
                indices = semantic_2d_xyz_kdtree.query_ball_point(
                    sub_edge_center, near_path_threshold
                )
                for i in indices:
                    near_points.add(tuple(semantic_3d_points[i]))
    return list(set(near_points))


def subdivide_line_segments(osm_item_list, subdivision_factor):
    """Subdivide each line segment in the road_lines list into smaller segments.

    Args:
    osm_list (list): List of OSM artifacts, from either building class or road class.
    subdivision_factor (int): Number of subdivisions per line segment.

    Returns:
    list: New list of subdivided line segments.
    """
    for osm_item in osm_item_list.item_list:
        for edge in osm_item.edges:
            # print(f"edge: {edge}")
            start_point = np.asarray(edge[0])
            end_point = np.asarray(edge[1])
            edge.pop(1)
            edge.pop(0)
            # Generate points along the segment
            for i in range(subdivision_factor):
                fraction = i / subdivision_factor
                intermediate_start = start_point * (1 - fraction) + end_point * fraction
                intermediate_end = start_point * (
                    1 - (fraction + 1 / subdivision_factor)
                ) + end_point * (fraction + 1 / subdivision_factor)
                edge.append([intermediate_start, intermediate_end])


def get_rotation_matrix(edge):
    direction = edge[1] - edge[0]

    # Check if the direction vector is a zero vector
    if np.linalg.norm(direction) == 0:
        raise ValueError(
            "The two points in the edge are the same; cannot compute rotation."
        )

    direction = direction / np.linalg.norm(direction)
    x_axis = np.array([1, 0, 0])

    # Check for parallel vectors
    if np.allclose(direction, x_axis):
        return np.eye(3)
    elif np.allclose(direction, -x_axis):
        return -np.eye(3)

    # Compute the rotation axis and angle
    v = np.cross(direction, x_axis)
    c = np.dot(direction, x_axis)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    rotation_matrix = np.eye(3) + k + k @ k * (1 / (1 + c))

    return rotation_matrix


def get_osm_labels(osm_item_list, assoc_sem_labels, semantic_3d_points, near_path_threshold):
    """
    Filter semantic points based on proximity to OSM edges and relabel them with OSM-style labels.
    """

    # Extract the XYZ coordinates and copy labels
    semantic_3d_xyz = semantic_3d_points[:, :3]
    osm_labels = np.copy(semantic_3d_points[:, 4])

    mask = np.isin(semantic_3d_points[:, 4], assoc_sem_labels)
    filtered_points_xyz = semantic_3d_xyz[mask] * np.array([1, 1, 0])  # Flatten points
    original_indices = np.where(mask)[0]

    # Flatten the edges_np array and reshape it
    edges_list = []
    for osm_item in osm_item_list.item_list:
        for edge in osm_item.edges:
            edges_list.extend(edge)
    edges_np = np.array(edges_list).reshape(-1, 2, 3)

    for edge in edges_np:
        rotation_matrix = get_rotation_matrix(edge)

        # Rotate points by unit vector
        rotated_points = (np.array(filtered_points_xyz) - edge[0]) @ rotation_matrix.T
        edge_len = np.linalg.norm(edge[1] - edge[0])
        # rotated_points_filtered = rotated_points[(rotated_points[:, 0] >= 0) &
        #                                             (rotated_points[:, 0] <= edge_len) &
        #                                             (rotated_points[:, 1] >= -near_path_threshold) &
        #                                             (rotated_points[:, 1] <= near_path_threshold)]

        # Get the indices of these filtered points
        filtered_indices = np.where((rotated_points[:, 0] >= 0) & 
                                    (rotated_points[:, 0] <= edge_len) &
                                    (rotated_points[:, 1] >= -near_path_threshold) &
                                    (rotated_points[:, 1] <= near_path_threshold))[0]
        
        # Map these indices back to the original indices
        original_indices_to_update = original_indices[filtered_indices]

        # Update the labels (first sem id in assoc_sem_labels is the osm-replacement label)
        osm_labels[original_indices_to_update] = assoc_sem_labels[0]

    return osm_labels



# OG
# def get_osm_labels(osm_item_list, semantic_3d_points, near_path_threshold, osm_label):
#     """
#     Filter semantic points based on proximity to OSM edges and relabel them with OSM-style labels.
#     """
#     # Extract the XYZ coordinates and copy labels
#     semantic_3d_xyz = semantic_3d_points[:, :3]
#     osm_labels = np.copy(semantic_3d_points[:, 4])

#     # Only use points with specific labels
#     if osm_label == 45:
#         associated_semantic_labels = [11, 0]
#     elif osm_label == 46:
#         associated_semantic_labels = [7]

#     mask = np.isin(semantic_3d_points[:, 4], associated_semantic_labels)
#     filtered_2d_xyz = semantic_3d_xyz[mask, :2]  # Only take the first two dimensions for 2D projection
#     original_indices = np.where(mask)[0]

#     # Build the KDTree with filtered points
#     filtered_2d_xyz_kdtree = cKDTree(filtered_2d_xyz)

#     # Flatten the edges_np array and reshape it
#     edges_list = []
#     for osm_item in osm_item_list.item_list:
#         for edge in osm_item.edges:
#             edges_list.extend(edge)
#     edges_np = np.array(edges_list)
#     vertices = edges_np.reshape(-1, 3)

#     # Convert vertices to 2D
#     vertices_2d = vertices[:, :2]

#     # Query the KDTree for points within the threshold distance from the vertices
#     indices = filtered_2d_xyz_kdtree.query_ball_point(vertices_2d, near_path_threshold)

#     # Flatten the list of indices and map them back to the original indices
#     if any(indices):  # Check if there are any indices found
#         flat_indices = np.unique(np.hstack(indices).astype(int))
#         original_indices_to_update = original_indices[flat_indices]
#         # Update the labels using advanced indexing
#         osm_labels[original_indices_to_update] = osm_label

#     return osm_labels

# def get_rotation_matrix(edge):
#     direction = edge[1] - edge[0]
#     direction = direction / np.linalg.norm(direction)
#     x_axis = np.array([1, 0, 0])

#     # Compute rotation matrix using Rodrigues' rotation formula
#     v = np.cross(direction, x_axis)
#     c = np.dot(direction, x_axis)
#     k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
#     rotation_matrix = np.eye(3) + k + k @ k * (1 / (1 + c))

#     return rotation_matrix


# def subdivide_line_segments(osm_item_list, subdivision_factor):
#     """Subdivide each line segment in the road_lines list into smaller segments.

#     Args:
#     osm_list (list): List of OSM artifacts, from either building class or road class.
#     subdivision_factor (int): Number of subdivisions per line segment.

#     Returns:
#     list: New list of subdivided line segments.
#     """
#     for osm_item in osm_item_list.buildings_list:
#         for edge in osm_item.edges:
#             start_point = np.asarray(edge[0])
#             end_point = np.asarray(edge[1])
#             # Clear the original edge or create a new list if needed
#             new_edge_points = [start_point]  # Start with the initial point
#             # Generate points along the segment
#             for i in range(1, subdivision_factor):
#                 fraction = i / subdivision_factor
#                 intermediate_point = start_point * (1 - fraction) + end_point * fraction
#                 new_edge_points.append(intermediate_point)  # Only append the start of each segment

#             new_edge_points.append(end_point)  # Append the final endpoint outside the loop
#             # Replace or modify the edge as needed in your data structure
#             edge[:] = new_edge_points  # If you want to replace the points in the original edge