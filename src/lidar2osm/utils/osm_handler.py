#!/usr/bin/env python3
"""
OSM Data Handler Class

A clean class to handle all OpenStreetMap data operations including:
- Loading OSM data from files
- Filtering OSM elements by distance to poses
- Plotting OSM elements (roads, buildings, trees, etc.)
- Managing OSM data visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import osmnx as ox

class OSMDataHandler:
    """
    A class to handle OpenStreetMap data operations.
    
    This class provides a clean interface for:
    - Loading OSM data from XML files
    - Filtering OSM elements by proximity to robot poses
    - Plotting various OSM elements (roads, buildings, trees, etc.)
    - Managing visualization parameters
    """
    
    def __init__(self, osm_file: Optional[Union[str, Path]] = None, 
                 filter_distance: float = 100.0, enable_semantics: Optional[Dict[str, bool]] = None):
        """
        Initialize OSM Data Handler.
        
        Args:
            osm_file: Path to OSM XML file
            filter_distance: Distance in meters to filter OSM elements around poses
            enable_semantics: Dictionary controlling which OSM semantics to load/plot
        """
        self.osm_file = Path(osm_file) if osm_file else None
        self.filter_distance = filter_distance
        self.osm_graph = None
        self.osm_geometries = {}
        self._haversine_cache = {}
        
        # Default semantics to enable
        self.enable_semantics = enable_semantics or {
            'roads': True,
            'buildings': True,
            'trees': True,
            'grassland': True,
            'gardens': True,
            'water': True,
            'parking': True,
            'amenities': True
        }
    
    def load_osm_data(self, osm_file: Optional[Union[str, Path]] = None) -> bool:
        """
        Load OSM data from XML file.
        
        Args:
            osm_file: Path to OSM XML file. If None, uses the file from initialization.
            
        Returns:
            bool: True if successful, False otherwise
        """
            
        if osm_file:
            self.osm_file = Path(osm_file)
        
        if not self.osm_file or not self.osm_file.exists():
            print(f"OSM file not found: {self.osm_file}")
            return False
        
        try:
            print(f"\nLoading OSM data from {self.osm_file}")
            
            # Load road network
            self.osm_graph = ox.graph_from_xml(self.osm_file)
            print(f"Loaded OSM graph with {len(self.osm_graph.nodes)} nodes and {len(self.osm_graph.edges)} edges")
            
            # Load geometries for different features
            self._load_osm_geometries()
            print("OSM data loaded successfully.\n")

            return True
            
        except Exception as e:
            print(f"Error loading OSM data: {e}")
            return False
    
    def _load_osm_geometries(self):
        """Load OSM geometries for different feature types."""
        try:
            # Load roads
            if self.enable_semantics.get('roads', True):
                print("\nLoading roads...\n")
                highways = ox.features_from_xml(self.osm_file, tags={'highway': True})
                self.osm_geometries['highways'] = highways
                print(f"Found {len(highways)} highways")
            

            # Load buildings
            if self.enable_semantics.get('buildings', True):
                buildings = ox.features_from_xml(self.osm_file, tags={'building': True})
                self.osm_geometries['buildings'] = buildings
                print(f"Found {len(buildings)} buildings")
            
            # Load trees
            if self.enable_semantics.get('trees', True):
                trees = ox.features_from_xml(self.osm_file, tags={'natural': 'tree'})
                self.osm_geometries['trees'] = trees
                print(f"Found {len(trees)} trees")
            
            # Load grassland/landuse
            if self.enable_semantics.get('grassland', True):
                grassland = ox.features_from_xml(self.osm_file, tags={'natural': True, 'landuse': True})
                self.osm_geometries['grassland'] = grassland
                print(f"Found {len(grassland)} grassland areas")
            
            # Load gardens (leisure=garden)
            if self.enable_semantics.get('gardens', True):
                gardens = ox.features_from_xml(self.osm_file, tags={'leisure': 'garden'})
                self.osm_geometries['gardens'] = gardens
                print(f"Found {len(gardens)} garden areas")
            
            # Load water features
            if self.enable_semantics.get('water', True):
                water = ox.features_from_xml(self.osm_file, tags={'natural': 'water'})
                self.osm_geometries['water'] = water
                print(f"Found {len(water)} water features")
            
            # Load parking areas
            if self.enable_semantics.get('parking', True):
                parking = ox.features_from_xml(self.osm_file, tags={'amenity': 'parking'})
                self.osm_geometries['parking'] = parking
                print(f"Found {len(parking)} parking areas")
            
            # Load other amenities
            if self.enable_semantics.get('amenities', True):
                amenities = ox.features_from_xml(self.osm_file, tags={'amenity': True})
                self.osm_geometries['amenities'] = amenities
                print(f"Found {len(amenities)} amenities")
            
        except Exception as e:
            print(f"Error loading OSM geometries: {e}")
            self.osm_geometries = {}
    
    def haversine_distance(self, lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """
        Calculate the great circle distance between two points on Earth.
        
        Args:
            lon1, lat1: Longitude and latitude of first point
            lon2, lat2: Longitude and latitude of second point
            
        Returns:
            Distance in meters
        """
        from math import radians, cos, sin, asin, sqrt
        
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371000  # Radius of earth in meters
        return c * r
    
    def filter_nodes_by_distance(self, poses_latlon: np.ndarray) -> List[int]:
        """
        Filter OSM nodes within filter_distance of any pose.
        
        Args:
            poses_latlon: Array of poses in lat/lon format (N, 2) or (N, 3)
            
        Returns:
            List of node IDs within the filter distance
        """
        if self.osm_graph is None:
            print("OSM graph not loaded. Call load_osm_data() first.")
            return []
        
        filtered_nodes = []
        poses_2d = poses_latlon[:, :2]  # Take only lat, lon columns
        
        print(f"Filtering OSM nodes within {self.filter_distance}m of {len(poses_2d)} poses...")
        
        for node_id, data in self.osm_graph.nodes(data=True):
            node_lat = data['y']
            node_lon = data['x']
            
            # Check distance to all poses
            min_distance = float('inf')
            for pose in poses_2d:
                pose_lat = pose[0]
                pose_lon = pose[1]
                distance = self.haversine_distance(pose_lon, pose_lat, node_lon, node_lat)
                min_distance = min(min_distance, distance)
            
            if min_distance <= self.filter_distance:
                filtered_nodes.append(node_id)
        
        print(f"Found {len(filtered_nodes)} nodes within {self.filter_distance}m of any pose")
        return filtered_nodes
    

    def filter_geometries_by_pose_center(self, geometry_type: str, poses_center: np.ndarray, poses_max_point: np.ndarray) -> List:
        """
        Filter OSM geometries by distance to pose center.
        
        Args:
            geometry_type: Type of geometry ('buildings', 'trees', 'grassland', etc.)
            poses_center: Array of pose centers in lat/lon format
            poses_radius: Radius in meters
        """
        if geometry_type not in self.osm_geometries:
            print(f"No geometries loaded for type: {geometry_type}")
            return []
        
        geometries = self.osm_geometries[geometry_type]
        filtered_geometries = []

        geometries_length = len(geometries)

        # calculate radius using haversine distance between poses_center and poses_max_point
        poses_radius = self.haversine_distance(poses_center[1], poses_center[0], poses_max_point[1], poses_max_point[0])
        print(f"Pose radius in meters: {poses_radius}")
        print(f"Filtering {geometry_type} within {poses_radius}m of pose CENTER...")
        
        from tqdm import tqdm
        for idx, geometry in tqdm(geometries.iterrows(), total=geometries_length, desc=f"Filtering {geometry_type}"):
            geometry_added = False
            if geometry.geometry.geom_type == "Polygon":
                coords = list(geometry.geometry.exterior.coords)
                for coord in coords:
                    if geometry_added:
                        break
                    pose_lat = poses_center[0]
                    pose_lon = poses_center[1]
                    distance = self.haversine_distance(pose_lon, pose_lat, coord[0], coord[1])
                    # print(f"Polygon (exterior) distance: {distance}")
                    if distance <= poses_radius:
                        filtered_geometries.append(geometry)
                        geometry_added = True
                        break
            elif geometry.geometry.geom_type == "LineString":
                coords = list(geometry.geometry.coords)
                for coord in coords:
                    if geometry_added:
                        break
                    pose_lat = poses_center[0]
                    pose_lon = poses_center[1]
                    distance = self.haversine_distance(pose_lon, pose_lat, coord[0], coord[1])
                    # print(f"LineString distance: {distance}")
                    if distance <= poses_radius:
                        filtered_geometries.append(geometry)
                        geometry_added = True
                        break
        print(f"    - Found {len(filtered_geometries)} {geometry_type} within {self.filter_distance}m of any pose")
        return filtered_geometries


    def filter_geometries_by_distance(self, 
                                      geometry_type: str, 
                                      poses_latlon: np.ndarray,
                                      use_centroid: bool = False) -> List:
        """
        Filter OSM geometries by distance to poses.
        
        Args:
            geometry_type: Type of geometry ('buildings', 'trees', 'grassland', etc.)
            poses_latlon: Array of poses in lat/lon format
            use_centroid: If True, use geometry centroid for distance calculation instead of all points
            
        Returns:
            List of filtered geometries
        """
        print(f"Checking if geometries are loaded for type: {geometry_type}")
        if geometry_type not in self.osm_geometries:
            print(f"No geometries loaded for type: {geometry_type}")
            return []
        
        geometries = self.osm_geometries[geometry_type]
        poses_2d = poses_latlon[:, :2]  # Take only lat, lon columns
        filtered_geometries = []
        
        print(f"Filtering {geometry_type} within {self.filter_distance}m of poses...")
        
        geometries_length = len(geometries)

        # Use tqdm to show progress
        from tqdm import tqdm
        for idx, geometry in tqdm(geometries.iterrows(), total=geometries_length, desc=f"Filtering {geometry_type}"):
        # for idx, geometry in geometries.iterrows():
            geometry_added = False
            # print(f"Percentage complete: {geometry_idx / geometries_length * 100:.2f}%")
            if geometry.geometry.geom_type == "Polygon":
                if use_centroid:
                    # Use centroid for distance calculation
                    centroid = geometry.geometry.centroid
                    min_distance = float('inf')
                    for pose in poses_2d:
                        pose_lat = pose[0]
                        pose_lon = pose[1]
                        distance = self.haversine_distance(pose_lon, pose_lat, centroid.x, centroid.y)
                        min_distance = min(min_distance, distance)
                    # print(f"Polygon (centroid) min distance: {min_distance}")
                    if min_distance <= self.filter_distance:
                        filtered_geometries.append(geometry)
                        geometry_added = True
                        break
                        # print(f"Polygon (centroid) distance: {distance}")
                        # if distance <= self.filter_distance:
                        #     filtered_geometries.append(geometry)
                        #     geometry_added = True
                        #     break
                else:
                    # Check all exterior coordinates
                    coords = list(geometry.geometry.exterior.coords)
                    for coord in coords:
                        if geometry_added:
                            break
                        min_distance = float('inf')
                        for pose in poses_2d:
                            pose_lat = pose[0]
                            pose_lon = pose[1]
                            distance = self.haversine_distance(pose_lon, pose_lat, coord[0], coord[1])
                            min_distance = min(min_distance, distance)
                        
                        # print(f"Polygon (exterior) mindistance: {min_distance}")
                        if min_distance <= self.filter_distance:
                            filtered_geometries.append(geometry)
                            geometry_added = True
                            break
                            # print(f"Polygon (exterior) distance: {distance}")
                            # if distance <= self.filter_distance:
                            #     filtered_geometries.append(geometry)
                            #     geometry_added = True
                            #     break

            elif geometry.geometry.geom_type == "LineString":
                coords = list(geometry.geometry.coords)
                for coord in coords:
                    if geometry_added:
                        break
                    min_distance = float('inf')
                    for pose in poses_2d:
                        pose_lat = pose[0]
                        pose_lon = pose[1]
                        distance = self.haversine_distance(pose_lon, pose_lat, coord[0], coord[1])
                        min_distance = min(min_distance, distance)
                    #print(f"LineString min distance: {min_distance}")
                    if min_distance <= self.filter_distance:
                        filtered_geometries.append(geometry)
                        geometry_added = True
                        break
                        # print(f"LineString distance: {distance}")
                        # if distance <= self.filter_distance:
                        #     filtered_geometries.append(geometry)
                        #     geometry_added = True
                        #     break

            # geometry_idx += 1

            # elif geometry.geometry.geom_type == "MultiPolygon":
            #     for polygon in geometry.geometry.geoms:
            #         if geometry_added:
            #             break
            #         if use_centroid:
            #             # Use centroid for distance calculation
            #             centroid = polygon.centroid
            #             for pose in poses_2d:
            #                 pose_lat = pose[0]
            #                 pose_lon = pose[1]
            #                 distance = self.haversine_distance(pose_lon, pose_lat, centroid.x, centroid.y)
            #                 if distance <= self.filter_distance:
            #                     filtered_geometries.append(geometry)
            #                     geometry_added = True
            #                     break
            #         else:
            #             # Check all exterior coordinates
            #             coords = list(polygon.exterior.coords)
            #             for coord in coords:
            #                 if geometry_added:
            #                     break
            #                 for pose in poses_2d:
            #                     pose_lat = pose[0]
            #                     pose_lon = pose[1]
            #                     distance = self.haversine_distance(pose_lon, pose_lat, coord[0], coord[1])
            #                     if distance <= self.filter_distance:
            #                         filtered_geometries.append(geometry)
            #                         geometry_added = True
            #                         break
                                    
            # elif geometry.geometry.geom_type == "Point":
            #     # Handle point geometries
            #     point = geometry.geometry
            #     for pose in poses_2d:
            #         pose_lat = pose[0]
            #         pose_lon = pose[1]
            #         distance = self.haversine_distance(pose_lon, pose_lat, point.x, point.y)
            #         if distance <= self.filter_distance:
            #             filtered_geometries.append(geometry)
            #             geometry_added = True
            #             break
        
        print(f"    - Found {len(filtered_geometries)} {geometry_type} within {self.filter_distance}m of any pose")
        return filtered_geometries
    
    def plot_highways(self, poses_latlon: np.ndarray, ax: Optional[plt.Axes] = None) -> bool:
    
        """
        Plot filtered OSM highways.
        
        Args:
            poses_latlon: Array of poses in lat/lon format
            ax: Matplotlib axes to plot on. If None, uses current axes.
            
        Returns:
            bool: True if successful, False otherwise
        """
    
        filtered_highways = self.filter_geometries_by_distance('highways', poses_latlon)
        
        if not filtered_highways:
            return False
        
        for highway in filtered_highways:
            if highway.geometry.geom_type == "LineString":
                lons, lats = np.array(highway.geometry.xy)
                # print(f"shape of road_coords: {road_coords.shape}")
                plt.plot(lons, lats, color='black', linewidth=1, zorder=2)
            else:
                print(f"\nhighway.geometry.geom_type: {highway.geometry.geom_type}")

        return True
    
    def plot_buildings(self, poses_latlon: np.ndarray, ax: Optional[plt.Axes] = None, 
                      color: str = 'blue', alpha: float = 0.3, use_centroid: bool = False) -> bool:
        """
        Plot filtered OSM buildings.
        
        Args:
            poses_latlon: Array of poses in lat/lon format
            ax: Matplotlib axes to plot on. If None, uses current axes.
            color: Color for building fill
            alpha: Transparency for building fill
            use_centroid: If True, use centroid for distance calculation
            
        Returns:
            bool: True if successful, False otherwise
        """
        filtered_buildings = self.filter_geometries_by_distance('buildings', poses_latlon, use_centroid)
        
        if not filtered_buildings:
            return False
        
        for building in filtered_buildings:
            if building.geometry.geom_type == "Polygon":
                coords = list(building.geometry.exterior.coords)
                lons, lats = zip(*coords)
                plt.fill(lons, lats, color=color, alpha=alpha, zorder=1)
                plt.plot(lons, lats, color=color, linewidth=1, zorder=2)
            elif building.geometry.geom_type == "MultiPolygon":
                for polygon in building.geometry.geoms:
                    coords = list(polygon.exterior.coords)
                    lons, lats = zip(*coords)
                    plt.fill(lons, lats, color=color, alpha=alpha, zorder=1)
                    plt.plot(lons, lats, color=color, linewidth=1, zorder=2)
        
        return True
    
    def plot_trees(self, poses_latlon: np.ndarray, ax: Optional[plt.Axes] = None, 
                  color: str = 'green', alpha: float = 0.3) -> bool:
        """
        Plot filtered OSM trees.
        
        Args:
            poses_latlon: Array of poses in lat/lon format
            ax: Matplotlib axes to plot on. If None, uses current axes.
            color: Color for tree fill
            alpha: Transparency for tree fill
            
        Returns:
            bool: True if successful, False otherwise
        """
        filtered_trees = self.filter_geometries_by_distance('trees', poses_latlon)
        
        if not filtered_trees:
            return False
        
        for tree in filtered_trees:
            coords = list(tree.geometry.exterior.coords)
            lons, lats = zip(*coords)
            plt.fill(lons, lats, color=color, alpha=alpha, zorder=1)
            plt.plot(lons, lats, color=color, linewidth=1, zorder=2)
        
        return True
    
    def plot_grassland(self, poses_latlon: np.ndarray, ax: Optional[plt.Axes] = None, 
                      color: str = 'lightgreen', alpha: float = 0.2) -> bool:
        """
        Plot filtered OSM grassland areas.
        
        Args:
            poses_latlon: Array of poses in lat/lon format
            ax: Matplotlib axes to plot on. If None, uses current axes.
            color: Color for grassland fill
            alpha: Transparency for grassland fill
            
        Returns:
            bool: True if successful, False otherwise
        """
        filtered_grassland = self.filter_geometries_by_distance('grassland', poses_latlon)
        
        if not filtered_grassland:
            return False
        
        for grass in filtered_grassland:
            if grass.geometry.geom_type == "Polygon":
                coords = list(grass.geometry.exterior.coords)
                lons, lats = zip(*coords)
                plt.fill(lons, lats, color=color, alpha=alpha, zorder=1)
                plt.plot(lons, lats, color=color, linewidth=1, zorder=2)
        
        return True
    
    def plot_water(self, poses_latlon: np.ndarray, ax: Optional[plt.Axes] = None, 
                  color: str = 'cyan', alpha: float = 0.4, use_centroid: bool = False) -> bool:
        """Plot filtered OSM water features."""
        filtered_water = self.filter_geometries_by_distance('water', poses_latlon, use_centroid)
        
        if not filtered_water:
            return False
        
        for water in filtered_water:
            if water.geometry.geom_type == "Polygon":
                coords = list(water.geometry.exterior.coords)
                lons, lats = zip(*coords)
                plt.fill(lons, lats, color=color, alpha=alpha, zorder=1)
                plt.plot(lons, lats, color=color, linewidth=1, zorder=2)
            # elif water.geometry.geom_type == "MultiPolygon":
            #     for polygon in water.geometry.geoms:
            #         coords = list(polygon.exterior.coords)
            #         lons, lats = zip(*coords)
            #         plt.fill(lons, lats, color=color, alpha=alpha, zorder=1)
            #         plt.plot(lons, lats, color=color, linewidth=1, zorder=2)
        
        return True
    
    def plot_parking(self, poses_latlon: np.ndarray, ax: Optional[plt.Axes] = None, 
                    color: str = 'orange', alpha: float = 0.3, use_centroid: bool = False) -> bool:
        """Plot filtered OSM parking areas."""
        filtered_parking = self.filter_geometries_by_distance('parking', poses_latlon, use_centroid)
        
        if not filtered_parking:
            return False
        
        for parking in filtered_parking:
            if parking.geometry.geom_type == "Polygon":
                coords = list(parking.geometry.exterior.coords)
                lons, lats = zip(*coords)
                plt.fill(lons, lats, color=color, alpha=alpha, zorder=1)
                plt.plot(lons, lats, color=color, linewidth=1, zorder=2)
            elif parking.geometry.geom_type == "MultiPolygon":
                for polygon in parking.geometry.geoms:
                    coords = list(polygon.exterior.coords)
                    lons, lats = zip(*coords)
                    plt.fill(lons, lats, color=color, alpha=alpha, zorder=1)
                    plt.plot(lons, lats, color=color, linewidth=1, zorder=2)
            # elif parking.geometry.geom_type == "Point":
            #     plt.plot(parking.geometry.x, parking.geometry.y, 's', 
            #             color=color, markersize=4, zorder=2)
        
        return True
    
    def plot_amenities(self, poses_latlon: np.ndarray, ax: Optional[plt.Axes] = None, 
                      color: str = 'purple', alpha: float = 0.3, use_centroid: bool = False) -> bool:
        """Plot filtered OSM amenities."""
        filtered_amenities = self.filter_geometries_by_distance('amenities', poses_latlon, use_centroid)
        
        if not filtered_amenities:
            return False
        
        for amenity in filtered_amenities:
            if amenity.geometry.geom_type == "Polygon":
                coords = list(amenity.geometry.exterior.coords)
                lons, lats = zip(*coords)
                plt.fill(lons, lats, color=color, alpha=alpha, zorder=1)
                plt.plot(lons, lats, color=color, linewidth=1, zorder=2)
            elif amenity.geometry.geom_type == "MultiPolygon":
                for polygon in amenity.geometry.geoms:
                    coords = list(polygon.exterior.coords)
                    lons, lats = zip(*coords)
                    plt.fill(lons, lats, color=color, alpha=alpha, zorder=1)
                    plt.plot(lons, lats, color=color, linewidth=1, zorder=2)
            # elif amenity.geometry.geom_type == "Point":
            #     plt.plot(amenity.geometry.x, amenity.geometry.y, '^', 
            #             color=color, markersize=4, zorder=2)
        
        return True
    
    def plot_all_osm_data(self, poses_latlon: np.ndarray, ax: Optional[plt.Axes] = None,
                         plot_highways: bool = True,
                         plot_buildings: bool = True,
                         plot_trees: bool = True, plot_grassland: bool = True,
                         plot_water: bool = True, plot_parking: bool = True,
                         plot_amenities: bool = True, use_centroid: bool = False) -> bool:
        """
        Plot all available OSM data elements.
        
        Args:
            poses_latlon: Array of poses in lat/lon format
            ax: Matplotlib axes to plot on. If None, uses current axes.
            plot_buildings: Whether to plot buildings
            plot_trees: Whether to plot trees
            plot_grassland: Whether to plot grassland
            plot_water: Whether to plot water features
            plot_parking: Whether to plot parking areas
            plot_amenities: Whether to plot amenities
            use_centroid: If True, use centroid for distance calculation
            
        Returns:
            bool: True if any data was plotted, False otherwise
        """
        success = False
        
        # Plot in order of z-order (background to foreground)
        if plot_grassland:
            if self.plot_grassland(poses_latlon, ax):
                success = True
        
        if plot_water:
            if self.plot_water(poses_latlon, ax):
                success = True
        
        if plot_parking:
            if self.plot_parking(poses_latlon, ax):
                success = True
        
        if plot_buildings:
            if self.plot_buildings(poses_latlon, ax, use_centroid=use_centroid):
                success = True
        
        if plot_trees:
            if self.plot_trees(poses_latlon, ax):
                success = True
        
        if plot_amenities:
            if self.plot_amenities(poses_latlon, ax):
                success = True
        
        if plot_highways:
            if self.plot_highways(poses_latlon, ax):
                success = True
        
        return success
    
    def get_osm_summary(self) -> Dict[str, int]:
        """
        Get summary of loaded OSM data.
        
        Returns:
            Dictionary with counts of different OSM elements
        """
        summary = {}
        
        if self.osm_graph is not None:
            summary['nodes'] = len(self.osm_graph.nodes)
            summary['edges'] = len(self.osm_graph.edges)
        
        for geom_type, geometries in self.osm_geometries.items():
            summary[geom_type] = len(geometries)
        
        return summary
    
    def set_semantics(self, semantics: Dict[str, bool]):
        """
        Set which OSM semantics to enable.
        
        Args:
            semantics: Dictionary with semantic types as keys and boolean values
        """
        self.enable_semantics.update(semantics)
        print(f" -> Updated semantics: {self.enable_semantics}")
    
    def set_filter_distance(self, distance: float):
        """
        Set the filter distance for OSM elements.
        
        Args:
            distance: Distance in meters
        """
        self.filter_distance = distance
        print(f" ->Filter distance set to {distance}m")
