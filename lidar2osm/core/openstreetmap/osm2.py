import numpy as np

# Class
#   - file path
#   - 

class OSM():
    def __init__(self, osm_file_path):
        self.file_path = osm_file_path
        self.edges = []     # If feature is composed of one or more edges
        self.points = []    # If feature is composed of one or more points
        
        print(f"osm_file_path: {self.file_path}")
        
    def extract_features(self, 
                         osm_list, 
                         pos_lat_lon, 
                         threshold_dist, 
                         desired_tags, 
                         geometry_type="Polygon"
                         ):
        
        # Filter features for buildings and sidewalks
        filtered_features = ox.features_from_xml(self.file_path, tags=desired_tags)

        # Process Buildings as LineSets
        for _, feature in filtered_features.iterrows():
            if feature.geometry.geom_type == geometry_type:
                exterior_coords = feature.geometry.exterior.coords
                build_center = [
                    np.mean(np.array(exterior_coords)[:, 0]),
                    np.mean(np.array(exterior_coords)[:, 1]),
                ]
                if geometry_near_pose(
                    build_center, np.asarray(pos_lat_lon), threshold_dist
                ):
                    osm_building = OSMBuildingPolygon()
                    for i in range(len(exterior_coords) - 1):
                        start_point = [exterior_coords[i][0], exterior_coords[i][1], 0]
                        end_point = [
                            exterior_coords[i + 1][0],
                            exterior_coords[i + 1][1],
                            0,
                        ]
                        osm_building.add_edge(start_point, end_point)
                    osm_building_list.add_item(osm_building)


def main():
    osm = OSM(osm_file_path="/users/donceykong/Desktop/test.txt")

if __name__ == "__main__":
    main()
