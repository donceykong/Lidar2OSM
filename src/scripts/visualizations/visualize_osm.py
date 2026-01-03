#!/usr/bin/env python3

import argparse

import numpy as np
import open3d as o3d
import osmnx as ox


def _polyline_points_from_linestring(ls) -> list[list[float]]:
    coords = np.array(ls.xy).T
    pts: list[list[float]] = []
    for i in range(len(coords) - 1):
        pts.append([coords[i][0], coords[i][1], 0.0])
        pts.append([coords[i + 1][0], coords[i + 1][1], 0.0])
    return pts


def _polyline_points_from_polygon(poly) -> list[list[float]]:
    pts: list[list[float]] = []
    exterior_coords = poly.exterior.coords
    for i in range(len(exterior_coords) - 1):
        pts.append([exterior_coords[i][0], exterior_coords[i][1], 0.0])
        pts.append([exterior_coords[i + 1][0], exterior_coords[i + 1][1], 0.0])
    return pts


def convert_polyline_points_to_o3d(polyline_points: np.ndarray, rgb_color: list[float]) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    if len(polyline_points) > 0:
        lines = [[i, i + 1] for i in range(0, len(polyline_points) - 1, 2)]
        ls.points = o3d.utility.Vector3dVector(polyline_points)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.paint_uniform_color(rgb_color)
    return ls


def get_osm_buildings_points(osm_file_path: str) -> np.ndarray:
    buildings = ox.features_from_xml(osm_file_path, tags={"building": True})
    pts: list[list[float]] = []
    for _, b in buildings.iterrows():
        if b.geometry.geom_type == "Polygon":
            pts.extend(_polyline_points_from_polygon(b.geometry))
    return np.array(pts)


def get_osm_road_points(osm_file_path: str) -> np.ndarray:
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
        ]
    }
    roads = ox.features_from_xml(osm_file_path, tags=tags)
    pts: list[list[float]] = []
    for _, r in roads.iterrows():
        if r.geometry.geom_type == "LineString":
            pts.extend(_polyline_points_from_linestring(r.geometry))
    return np.array(pts)


def get_osm_stair_points(osm_file_path: str) -> np.ndarray:
    stairs = ox.features_from_xml(osm_file_path, tags={"highway": ["steps"]})
    pts: list[list[float]] = []
    for _, r in stairs.iterrows():
        if r.geometry.geom_type == "LineString":
            pts.extend(_polyline_points_from_linestring(r.geometry))
    return np.array(pts)


def get_osm_grass_points(osm_file_path: str, *, include_parks: bool) -> np.ndarray:
    tags = {"landuse": ["grass", "recreation_ground"]}
    if include_parks:
        tags["leisure"] = "park"
    grass = ox.features_from_xml(osm_file_path, tags=tags)
    pts: list[list[float]] = []
    for _, g in grass.iterrows():
        if g.geometry.geom_type == "Polygon":
            pts.extend(_polyline_points_from_polygon(g.geometry))
    return np.array(pts)


def display(*, roads: np.ndarray, stairs: np.ndarray, buildings: np.ndarray, grass: np.ndarray) -> None:
    geometries = [
        convert_polyline_points_to_o3d(roads, [0.5, 0.5, 0.5]),
        convert_polyline_points_to_o3d(stairs, [1, 0, 0]),
        convert_polyline_points_to_o3d(buildings, [0, 0, 1]),
        convert_polyline_points_to_o3d(grass, [0, 1, 0]),
    ]
    o3d.visualization.draw_geometries(geometries)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize OSM geometries as Open3D line sets.")
    ap.add_argument("--osm_file", required=True, help="Path to an .osm file.")
    ap.add_argument("--include_parks", action="store_true", help="Include leisure=park in grass selection.")
    args = ap.parse_args()

    buildings = get_osm_buildings_points(args.osm_file)
    roads = get_osm_road_points(args.osm_file)
    stairs = get_osm_stair_points(args.osm_file)
    grass = get_osm_grass_points(args.osm_file, include_parks=args.include_parks)
    display(roads=roads, stairs=stairs, buildings=buildings, grass=grass)


if __name__ == "__main__":
    main()


