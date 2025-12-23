import open3d as o3d
import numpy as np
import osmnx as ox

def get_osm_buildings_points(osm_file_path):
    buildings = ox.features_from_xml(osm_file_path, tags={"building": True})
    osm_building_list = []
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

def get_osm_road_points(osm_file_path):
    tags = {
        "highway": [
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "unclassified",
            "residential",
            "motorway_link",
            "trunk_link",
            "primary_link",
            "secondary_link",
            "tertiary_link",
            "living_street",
            "service",
            "pedestrian",
            "road",
            "cycleway",
            "foot",
            "footway",
            "path",
            "service",
        ]
    }
    roads = ox.features_from_xml(osm_file_path, tags=tags)
    osm_road_list = []
    for _, road in roads.iterrows():
        if road.geometry.geom_type == "LineString":
            coords = np.array(road.geometry.xy).T
            for i in range(len(coords) - 1):
                start_point = [coords[i][0], coords[i][1], 0]
                end_point = [coords[i + 1][0], coords[i + 1][1], 0]
                osm_road_list.append(start_point)
                osm_road_list.append(end_point)
    osm_road_points = np.array(osm_road_list)
    return osm_road_points

def get_osm_stair_points(osm_file_path):
    tags = {
        "highway": [
            "steps"
        ]
    }
    roads = ox.features_from_xml(osm_file_path, tags=tags)
    osm_road_list = []
    for _, road in roads.iterrows():
        if road.geometry.geom_type == "LineString":
            coords = np.array(road.geometry.xy).T
            for i in range(len(coords) - 1):
                start_point = [coords[i][0], coords[i][1], 0]
                end_point = [coords[i + 1][0], coords[i + 1][1], 0]
                osm_road_list.append(start_point)
                osm_road_list.append(end_point)
    osm_road_points = np.array(osm_road_list)
    return osm_road_points

def get_osm_grass_points(osm_file_path):
    grass = ox.features_from_xml(osm_file_path, tags={"landuse": ["grass", "recreation_ground"]})
    osm_grass_list = []
    for _, g in grass.iterrows():
        if g.geometry.geom_type == "Polygon":
            exterior_coords = g.geometry.exterior.coords
            for i in range(len(exterior_coords) - 1):
                start_point = [exterior_coords[i][0], exterior_coords[i][1], 0]
                end_point = [exterior_coords[i + 1][0], exterior_coords[i + 1][1], 0]
                osm_grass_list.append(start_point)
                osm_grass_list.append(end_point)
    osm_grass_points = np.array(osm_grass_list)
    return osm_grass_points

def convert_polyline_points_to_o3d(polyline_points, rgb_color):
    polyline_pcd = o3d.geometry.LineSet()
    if len(polyline_points) > 0:
        polyline_lines_idx = [[i, i + 1] for i in range(0, len(polyline_points) - 1, 2)]
        polyline_pcd.points = o3d.utility.Vector3dVector(polyline_points)
        polyline_pcd.lines = o3d.utility.Vector2iVector(polyline_lines_idx)
        polyline_pcd.paint_uniform_color(rgb_color)
    return polyline_pcd

def read_gps_data(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            lat, lon, alt = map(float, line.strip().split())
            points.append([lon, lat, alt])
    return np.array(points)

def latlon_to_ecef(lat, lon, alt):
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    lat = np.radians(lat)
    lon = np.radians(lon)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    X = (N + alt) * np.cos(lat) * np.cos(lon)
    Y = (N + alt) * np.cos(lat) * np.sin(lon)
    Z = (N * (1 - e2) + alt) * np.sin(lat)
    return X, Y, Z

def convert_to_local_coordinates(points):
    ecef_points = []
    for point in points:
        ecef_points.append(latlon_to_ecef(point[0], point[1], point[2]))
    return np.array(ecef_points)

def display(osm_road_points, osm_stair_points, osm_building_points, osm_grass_points, gps_1_points=None, gps_2_points=None):
    geometries = []

    osm_road_pcd = convert_polyline_points_to_o3d(osm_road_points, [0.5, 0.5, 0.5])
    geometries.append(osm_road_pcd)
    osm_stair_pcd = convert_polyline_points_to_o3d(osm_stair_points, [1, 0, 0])
    geometries.append(osm_stair_pcd)
    osm_building_pcd = convert_polyline_points_to_o3d(osm_building_points, [0, 0, 1])
    geometries.append(osm_building_pcd)
    osm_grass_pcd = convert_polyline_points_to_o3d(osm_grass_points, [0, 1, 0])
    geometries.append(osm_grass_pcd)

    if gps_1_points is not None:
        gps_1_point_cloud = o3d.geometry.PointCloud()
        gps_1_point_cloud.points = o3d.utility.Vector3dVector(gps_1_points)
        gps_1_point_cloud.paint_uniform_color([1, 0, 0])
        geometries.append(gps_1_point_cloud)
    if gps_2_points is not None:
        gps_2_point_cloud = o3d.geometry.PointCloud()
        gps_2_point_cloud.points = o3d.utility.Vector3dVector(gps_2_points)
        gps_2_point_cloud.paint_uniform_color([0, 1, 0])
        geometries.append(gps_2_point_cloud)

    o3d.visualization.draw_geometries(geometries)

# Main execution
osm_file_path = './UCD.osm'
osm_building_points = get_osm_buildings_points(osm_file_path)
osm_road_points = get_osm_road_points(osm_file_path)
osm_stair_points = get_osm_stair_points(osm_file_path)
osm_grass_points = get_osm_grass_points(osm_file_path)

display(osm_road_points, osm_stair_points, osm_building_points, osm_grass_points, gps_1_points=None, gps_2_points=None)
