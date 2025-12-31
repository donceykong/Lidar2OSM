"""
visualize.py

Author: Doncey Albin

Description:
"""

# External imports
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

# Internal Imports
from lidar2osm.utils.projection import convertOxtsToPoints # TODO: Remove after testing
from lidar2osm.utils.file_io import (
    find_min_max_file_names,
    read_bin_file,
    read_extracted_data,
    save_extracted_data,
)
# from data_extraction.utils.lidar_to_bev import (
#     birdseye_to_point_cloud,
#     point_cloud_2_birdseye,
#     point_cloud_2_birdseye_auto_params,
# )
from Lidar2OSM.lidar2osm.core.pointcloud.pointcloud import labels2RGB, get_transformed_point_cloud
from lidar2osm.utils.pcd import convert_pc_to_o3d, convert_polyline_points_to_o3d


def change_frame(vis, key_code, seq_extract):
    if key_code == ord("N") and seq_extract.current_frame < seq_extract.fin_frame:
        seq_extract.current_frame += seq_extract.inc_frame
    elif key_code == ord("P") and seq_extract.current_frame > seq_extract.init_frame:
        seq_extract.current_frame -= seq_extract.inc_frame

    pc_frame_label_path = os.path.join(
        seq_extract.label_path, f"{seq_extract.current_frame:010d}.bin"
    )
    frame_build_points_file = os.path.join(
        seq_extract.extracted_per_frame_dir,
        f"{seq_extract.current_frame:010d}_build_points.npy",
    )

    if os.path.exists(frame_build_points_file):
        print(f"    Frame: {seq_extract.current_frame}")
        osm_buildings_points, osm_roads_points, lidar_points = read_extracted_data(
            seq_extract.extracted_per_frame_dir, seq_extract.current_frame
        )
        velodyne_poses = (
            seq_extract.velodyne_poses_latlon
            if seq_extract.coordinate_frame == "latlon"
            else seq_extract.velodyne_poses
        )

        lidar_points = np.array(lidar_points[:, :3])
        lidar_points_trans = get_transformed_point_cloud(
            lidar_points, velodyne_poses, seq_extract.current_frame
        )

        if osm_buildings_points.shape[0] > 0:
            # osm_buildings_points[:, 2] = 0
            osm_buildings_points_trans = get_transformed_point_cloud(
                osm_buildings_points, velodyne_poses, seq_extract.current_frame
            )
            seq_extract.building_pcd_list.extend(osm_buildings_points_trans)
        else:
            osm_buildings_points_trans = osm_buildings_points

        if osm_roads_points.shape[0] > 0:
            # osm_roads_points[:, 2] = 0
            osm_roads_points_trans = get_transformed_point_cloud(
                osm_roads_points, velodyne_poses, seq_extract.current_frame
            )
            seq_extract.road_pcd_list.extend(osm_roads_points_trans)
        else:
            osm_roads_points_trans = osm_roads_points

        # osm_buildings_points_trans[:, 2] = 1
        # osm_roads_points_trans[:, 2] = 1

        # lidar_points[:,2] = 0
        # lidar_points[:,2] -= np.min(lidar_points[:,2])

        osm_roads_points_cp = osm_roads_points[1:-1].copy()
        extended_osm_roads_points = np.concatenate(
            (osm_roads_points, osm_roads_points_cp)
        )

        osm_buildings_o3d = convert_pc_to_o3d(osm_buildings_points, [0, 0, 1])
        osm_roads_o3d = convert_pc_to_o3d(osm_roads_points, [1, 0, 0])
        lidar_points_o3d = convert_pc_to_o3d(lidar_points, [0, 0, 0])

        # semantic_ids = [7, 11, 0]
        # raw_pc_frame_path = os.path.join(
        #     seq_extract.raw_pc_path, f"{seq_extract.current_frame:010d}.bin"
        # )
        # semantic_points_o3d = seq_extract.process_pointcloud(
        #     raw_pc_frame_path,
        #     pc_frame_label_path,
        #     seq_extract.current_frame,
        #     semantic_ids,
        #     transform=False,
        # )
        # voxel_size = 1.0  # Specify the voxel size

        # # lidar_points_o3d_DS = semantic_points_o3d.voxel_down_sample(voxel_size=voxel_size)
        # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        #     osm_roads_o3d, voxel_size=voxel_size
        # )

        # TODO: Check if all points in osm_roads_o3d are in lidar_points_o3d
        osm_roads_points = np.asarray(osm_roads_o3d.points)
        lidar_points = np.asarray(lidar_points_o3d.points)

        # Create a KDTree for the lidar points
        lidar_kdtree = o3d.geometry.KDTreeFlann(lidar_points_o3d)

        # Define a distance threshold
        distance_threshold = 0.001  # Adjust this threshold as needed

        # Check if all points in osm_roads_o3d are within lidar_points_o3d
        all_points_found = True
        for point in osm_roads_points:
            [_, idx, _] = lidar_kdtree.search_knn_vector_3d(point, 1)
            nearest_point = lidar_points[idx[0]]
            distance = np.linalg.norm(point - nearest_point)
            if distance > distance_threshold:
                all_points_found = False
                break

        if all_points_found:
            print(
                "All points in osm_roads_o3d are in lidar_points_o3d within the specified distance threshold."
            )
        else:
            print(
                "Not all points in osm_roads_o3d are in lidar_points_o3d within the specified distance threshold."
            )

        # Visualize
        vis.clear_geometries()
        # vis.add_geometry(semantic_points_o3d)
        vis.add_geometry(lidar_points_o3d)
        vis.add_geometry(osm_roads_o3d)
        vis.add_geometry(osm_buildings_o3d)
    return True


def create_wireframe_cube_mesh(
    origin=(0, 0, 0), size=1.0, color=(1, 0, 0), thickness=0.02
):
    """
    Create a wireframe cube using LineSet in Open3D.

    Parameters:
    - origin: Tuple of (x, y, z) coordinates for the center of the cube.
    - size: The edge length of the cube.
    - color: Tuple of (r, g, b) values for the color of the cube.
    - thickness: Thickness of the lines.

    Returns:
    - o3d.geometry.LineSet object representing the cube.
    """
    # Cube vertices
    vertices = (
        np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ]
        )
        * size
        + np.array(origin)
        - np.array([size / 2, size / 2, size / 2])
    )

    # Cube edges
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]

    # Create LineSet object
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(edges)

    # Set colors
    colors = np.array([color for _ in range(len(edges))])
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # To create a thick line effect, we use cylinders instead of direct lines
    # This is a workaround since Open3D doesn't directly support thick lines
    mesh = o3d.geometry.TriangleMesh()
    for start, end in edges:
        cyl_mesh = o3d.geometry.TriangleMesh.create_cylinder(
            radius=thickness, height=np.linalg.norm(vertices[start] - vertices[end])
        )
        cyl_mesh.translate(
            (vertices[start] + vertices[end]) / 2
            - np.array([0, 0, cyl_mesh.get_center()[2]])
        )
        cyl_mesh.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz(
                [
                    0,
                    np.arccos(
                        np.dot(
                            (vertices[end] - vertices[start])
                            / np.linalg.norm(vertices[end] - vertices[start]),
                            [0, 0, 1],
                        )
                    ),
                    np.arctan2(
                        vertices[end][1] - vertices[start][1],
                        vertices[end][0] - vertices[start][0],
                    ),
                ]
            )
        )
        cyl_mesh.paint_uniform_color(color)
        mesh += cyl_mesh

    return mesh


def create_wireframe_cube(tf_matrix, width=1.0, height=1.0, depth=1.0, color=(1, 0, 0)):
    """
    Create a wireframe cube using LineSet in Open3D.

    Parameters:
    - tf_matrix: 4x4 transformation matrix for the cube.
    - width: The width of the cube.
    - height: The height of the cube.
    - depth: The depth of the cube.
    - color: Tuple of (r, g, b) values for the color of the cube.

    Returns:
    - o3d.geometry.LineSet object representing the cube.
    """
    # Cube vertices (unit cube centered at the origin)
    vertices = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ]
    )

    # Scale vertices by width, height, and depth
    vertices[:, 0] *= width
    vertices[:, 1] *= depth
    vertices[:, 2] *= height

    # Transform vertices
    ones = np.ones((vertices.shape[0], 1))
    vertices_homogeneous = np.hstack((vertices, ones))
    transformed_vertices = (tf_matrix @ vertices_homogeneous.T).T[:, :3]

    # Cube edges
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]

    # Create LineSet object
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(transformed_vertices)
    line_set.lines = o3d.utility.Vector2iVector(edges)

    # Set colors
    colors = np.array([color for _ in range(len(edges))])
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def vis_lidar_BEV(seq_extract):
    first_frame, last_extracted_frame = find_min_max_file_names(
        seq_extract.extracted_per_frame_dir2,
        number_delimiter="_",
        file_extension=".npy",
    )

    first_frame = 9
    current_frame = first_frame
    last_extracted_frame = 500
    inc_frame = seq_extract.inc_frame

    lidar_points_bev_accum = []
    road_points_bev_accum = []
    build_points_bev_accum = []

    lidar_points_3d_accum = []
    build_points_3d_accum = []
    road_points_3d_accum = []

    axis_frame_accum = []
    wireframe_cube_accum = []
    while current_frame < last_extracted_frame:
        frame_build_points_file = os.path.join(
            seq_extract.extracted_per_frame_dir2,
            f"{current_frame:010d}_osm_points.npy",
        )
        print(f"    Frame: {current_frame}")
        if os.path.exists(frame_build_points_file):
            osm_points_BEV, lidar_points_all_BEV = read_extracted_data(
                seq_extract.extracted_per_frame_dir2, current_frame
            )
            osm_points_3d, lidar_points_all_3d = read_extracted_data(
                seq_extract.extracted_per_frame_dir, current_frame
            )

            # Separate building and road points
            build_points_3d = osm_points_3d[osm_points_3d[:, 4] == 11]
            road_points_3d = osm_points_3d[osm_points_3d[:, 4] == 7]

            # Init np array with zeros, but same shape as osm_BEV
            mask_buildings = osm_points_BEV[:, :, 2] == 11
            mask_roads = osm_points_BEV[:, :, 2] == 7
            build_points_BEV = np.zeros_like(osm_points_BEV)
            road_points_BEV = np.zeros_like(osm_points_BEV)
            build_points_BEV[mask_buildings] = osm_points_BEV[mask_buildings]
            road_points_BEV[mask_roads] = osm_points_BEV[mask_roads]

            # Create an RGB image from the BEV data
            # lidar_points_BEV = lidar_points_all_BEV[:, :, 0]
            lidar_classes_BEV = lidar_points_all_BEV[:, :, 2].flatten()
            rgb_np = labels2RGB(lidar_classes_BEV, seq_extract.labels_dict)
            rgb_np_reshaped = rgb_np.reshape((801, 801, 3))

            # build_points_BEV = build_points_BEV[:, :, 0]
            # road_points_BEV = road_points_BEV[:, :, 0]

            # lidar_rgb = np.stack([lidar_points_BEV] * 3, axis=-1)
            # lidar_rgb[:] = rgb_np_reshaped
            # print(f"shape lidar_rgb: {lidar_rgb[:, :, -1].shape}")
            # lidar_rgb[:, :, -1] *= lidar_points_BEV[:, :] / 255

            # # Assign blue color to building points (assumes build_points_BEV is a binary mask)
            # # lidar_rgb[build_points_BEV > 0] = [0, 0, 255]  # Blue

            # # Assign red color to road points (assumes road_points_BEV is a binary mask)
            # # lidar_rgb[road_points_BEV > 0] = [255, 0, 0]  # Red

            # # Display the image
            # plt.imshow(osm_points_BEV)
            # plt.colorbar()  # Optionally add a color bar
            # plt.show()
            # time.sleep(0.5)
            # plt.close()

            ### 3D
            lidar_points_3d_tf = get_transformed_point_cloud(
                lidar_points_all_3d[:, :3],
                seq_extract.velodyne_poses,
                current_frame,
            )
            lidar_points_3d = np.concatenate(
                (
                    lidar_points_3d_tf,
                    lidar_points_all_3d[:, 3].reshape(lidar_points_3d_tf.shape[0], 1),
                    lidar_points_all_3d[:, 4].reshape(lidar_points_3d_tf.shape[0], 1),
                ),
                axis=1,
            )
            lidar_points_3d_accum.extend(lidar_points_3d)

            build_points_3d_tf = get_transformed_point_cloud(
                build_points_3d,
                seq_extract.velodyne_poses,
                current_frame,
            )
            build_points_3d_accum.extend(build_points_3d_tf)

            road_points_3d_tf = get_transformed_point_cloud(
                road_points_3d,
                seq_extract.velodyne_poses,
                current_frame,
            )
            road_points_3d_accum.extend(road_points_3d_tf)

            ### BEV
            (
                lidar_points_bev_xyz,
                lidar_points_bev_intensity,
                lidar_points_bev_semantic,
            ) = birdseye_to_point_cloud(lidar_points_all_BEV)
            lidar_points_from_bev_tf = get_transformed_point_cloud(
                lidar_points_bev_xyz,
                seq_extract.velodyne_poses,
                current_frame,
            )
            lidar_points_from_BEV = np.concatenate(
                (
                    lidar_points_from_bev_tf,
                    lidar_points_bev_intensity,
                    lidar_points_bev_semantic,
                ),
                axis=1,
            )
            lidar_points_bev_accum.extend(lidar_points_from_BEV)

            (
                build_points_bev_xyz,
                build_points_bev_intensity,
                build_points_bev_semantic,
            ) = birdseye_to_point_cloud(build_points_BEV)
            build_points_from_bev_tf = get_transformed_point_cloud(
                build_points_bev_xyz,
                seq_extract.velodyne_poses,
                current_frame,
            )
            build_points_from_BEV = np.concatenate(
                (
                    build_points_from_bev_tf,
                    build_points_bev_intensity,
                    build_points_bev_semantic,
                ),
                axis=1,
            )
            build_points_bev_accum.extend(build_points_from_BEV)

            (
                road_points_bev_xyz,
                road_points_bev_intensity,
                road_points_bev_semantic,
            ) = birdseye_to_point_cloud(road_points_BEV)
            road_points_from_bev_tf = get_transformed_point_cloud(
                road_points_bev_xyz,
                seq_extract.velodyne_poses,
                current_frame,
            )
            road_points_from_BEV = np.concatenate(
                (
                    road_points_from_bev_tf,
                    road_points_bev_intensity,
                    road_points_bev_semantic,
                ),
                axis=1,
            )
            road_points_bev_accum.extend(road_points_from_BEV)

            frame_build_points_file = os.path.join(
                seq_extract.extraced_map_segments_dir,
                f"frame_{current_frame:010d}_build_points",
            )
            frame_road_points_file = os.path.join(
                seq_extract.extraced_map_segments_dir,
                f"frame_{current_frame:010d}_road_points",
            )
            np.save(frame_build_points_file, build_points_from_BEV)
            np.save(frame_road_points_file, road_points_from_BEV)

            ### POSE FRAME
            tf_matrix = seq_extract.velodyne_poses.get(current_frame)
            axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=3.0, origin=[0, 0, 0]
            )
            axis_frame.transform(tf_matrix)
            axis_frame_accum.append(axis_frame)

            ## CUBE
            red_darkness = current_frame / last_extracted_frame
            wireframe_cube = create_wireframe_cube(
                tf_matrix,
                width=80.0,
                height=16.0,
                depth=80.0,
                color=(red_darkness, 0, 0),
            )
            wireframe_cube_accum.append(wireframe_cube)
        current_frame += inc_frame

    lidar_points_3d_accum = np.asarray(lidar_points_3d_accum)
    build_points_3d_accum = np.asarray(build_points_3d_accum)
    road_points_3d_accum = np.asarray(road_points_3d_accum)

    lidar_points_3d_o3d = convert_pc_to_o3d(lidar_points_3d_accum[:, :3], [0, 0, 0])
    rgb_np = labels2RGB(lidar_points_3d_accum[:, 4], seq_extract.labels_dict)
    lidar_points_3d_o3d.colors = o3d.utility.Vector3dVector(rgb_np)
    build_points_3d_o3d = convert_pc_to_o3d(build_points_3d_accum[:, :3], [0, 0, 0])
    road_points_3d_o3d = convert_pc_to_o3d(road_points_3d_accum[:, :3], [0, 0, 0])

    lidar_points_bev_accum = np.asarray(lidar_points_bev_accum)
    build_points_bev_accum = np.asarray(build_points_bev_accum)
    road_points_bev_accum = np.asarray(road_points_bev_accum)

    # frame_build_points_file = os.path.join(
    #     seq_extract.extraced_map_segments_dir,
    #     f"firstframe_{first_frame:010d}_lastframe_{last_extracted_frame:010d}_build_points_map",
    # )
    # frame_road_points_file = os.path.join(
    #     seq_extract.extraced_map_segments_dir,
    #     f"firstframe_{first_frame:010d}_lastframe_{last_extracted_frame:010d}_road_points_map",
    # )
    # np.save(frame_build_points_file, build_points_bev_accum)
    # np.save(frame_road_points_file, road_points_bev_accum)

    lidar_points_bev_o3d = convert_pc_to_o3d(lidar_points_bev_accum[:, :3], [1, 0, 0])
    build_points_bev_o3d = convert_pc_to_o3d(build_points_bev_accum[:, :3], [0, 0, 1])
    road_points_bev_o3d = convert_pc_to_o3d(road_points_bev_accum[:, :3], [1, 0, 0])

    # Combine all geometries into a single list
    all_geometries = (
        [
            # lidar_points_3d_o3d,
            # build_points_3d_o3d,
            # road_points_3d_o3d,
            # lidar_points_bev_o3d,
            build_points_bev_o3d,
            road_points_bev_o3d,
        ]
        + axis_frame_accum
        + wireframe_cube_accum
    )

    # Visualize all accumulated frames and point clouds together
    o3d.visualization.draw_geometries(all_geometries)


with open("data_extraction/config/o3d_color_palette.yaml", "r") as file:
    palette = yaml.safe_load(file)
    o3d_colors = palette["colors"]


def vis_all_extracted_data(seq_extract):
    # road_pcd_list = []
    # building_pcd_list = []
    # lidar_points_list = []
    # labels_list = []

    robot1 = robot()
    robot2 = robot()
    overlapping_data = robot()

    first_frame, last_extracted_frame = find_min_max_file_names(
        seq_extract.extracted_per_frame_dir, number_delimiter="_", file_extension=".npy"
    )
    mid_frame = int((last_extracted_frame - first_frame) / 2)

    overlap = 500

    robot1.min_frame = first_frame
    robot1.max_frame = mid_frame - overlap / 2

    overlapping_data.min_frame = mid_frame - overlap / 2
    overlapping_data.max_frame = mid_frame + overlap / 2

    robot2.min_frame = mid_frame + overlap / 2
    robot2.max_frame = last_extracted_frame

    print(
        f"\n \nSequence: {seq_extract.seq}, First frame extracted: {first_frame}, final frame extracted: {last_extracted_frame}"
    )

    velodyne_poses = (
        seq_extract.velodyne_poses_latlon
        if seq_extract.coordinate_frame == "latlon"
        else seq_extract.velodyne_poses
    )
    zero_road = 0
    zero_building = 0
    total_scans = 0
    while seq_extract.current_frame < last_extracted_frame:
        frame_num = seq_extract.current_frame
        extracted_per_frame_dir = seq_extract.extracted_per_frame_dir

        pc_frame_label_path = os.path.join(
            seq_extract.label_path, f"{seq_extract.current_frame:010d}.bin"
        )
        frame_build_points_file = os.path.join(
            seq_extract.extracted_per_frame_dir,
            f"{seq_extract.current_frame:010d}_build_points.npy",
        )
        frame_road_points_file = os.path.join(
            extracted_per_frame_dir, f"{frame_num:010d}_road_points.npy"
        )
        frame_points_file = os.path.join(
            seq_extract.extracted_per_frame_dir,
            f"{seq_extract.current_frame:010d}_lidar_scan_points.npy",
        )
        frame_points_file_size_list = []
        OSM_file_size_list = []
        times_bigger_list = []
        if os.path.exists(frame_build_points_file):
            frame_points_file_size = os.path.getsize(frame_points_file)
            frame_points_file_size_list.append(frame_points_file_size)
            OSM_file_size = os.path.getsize(frame_build_points_file) + os.path.getsize(
                frame_road_points_file
            )
            OSM_file_size_list.append(OSM_file_size)

            # print(f" frame_points_file_size: {frame_points_file_size} | OSM_file_size: {OSM_file_size}")
            times_bigger = frame_points_file_size / OSM_file_size
            times_bigger_list.append(times_bigger)
            times_bigger_ave = np.average(np.asarray(times_bigger_list))
            print(
                f"frame_points_file is {times_bigger_ave} times bigger than OSM file on average"
            )
            extracted_data = read_extracted_data(
                seq_extract.extracted_per_frame_dir, seq_extract.current_frame
            )
            if extracted_data is not None:
                osm_buildings_points, osm_roads_points, lidar_points = extracted_data

                if osm_buildings_points.shape[0] == 0:
                    zero_building += 1
                    osm_buildings_points_trans = osm_buildings_points
                elif robot1.min_frame < frame_num < robot1.max_frame:
                    osm_buildings_points_trans = get_transformed_point_cloud(
                        osm_buildings_points, velodyne_poses, seq_extract.current_frame
                    )
                    robot1.building_pcd_list.extend(osm_buildings_points_trans)
                elif (
                    overlapping_data.min_frame < frame_num < overlapping_data.max_frame
                ):
                    osm_buildings_points_trans = get_transformed_point_cloud(
                        osm_buildings_points, velodyne_poses, seq_extract.current_frame
                    )
                    overlapping_data.building_pcd_list.extend(
                        osm_buildings_points_trans
                    )
                elif robot2.min_frame < frame_num < robot2.max_frame:
                    osm_buildings_points_trans = get_transformed_point_cloud(
                        osm_buildings_points, velodyne_poses, seq_extract.current_frame
                    )
                    robot2.building_pcd_list.extend(osm_buildings_points_trans)

                if osm_roads_points.shape[0] == 0:
                    zero_road += 1
                    osm_roads_points_trans = osm_roads_points
                    # os.remove(frame_road_points_file)
                    # os.remove(frame_build_points_file)
                    # os.remove(frame_points_file)
                elif robot1.min_frame < frame_num < robot1.max_frame:
                    osm_roads_points_trans = get_transformed_point_cloud(
                        osm_roads_points, velodyne_poses, seq_extract.current_frame
                    )
                    robot1.road_pcd_list.extend(osm_roads_points_trans)
                elif (
                    overlapping_data.min_frame < frame_num < overlapping_data.max_frame
                ):
                    osm_roads_points_trans = get_transformed_point_cloud(
                        osm_roads_points, velodyne_poses, seq_extract.current_frame
                    )
                    overlapping_data.road_pcd_list.extend(osm_roads_points_trans)
                elif robot2.min_frame < frame_num < robot2.max_frame:
                    osm_roads_points_trans = get_transformed_point_cloud(
                        osm_roads_points, velodyne_poses, seq_extract.current_frame
                    )
                    robot2.road_pcd_list.extend(osm_roads_points_trans)

                total_scans += 1

                lidar_points_trans = get_transformed_point_cloud(
                    lidar_points, velodyne_poses, seq_extract.current_frame
                )
                labels_np = read_bin_file(pc_frame_label_path, dtype=np.uint16) # Note: Use uint32 for inferred labels
                rgb_np = labels2RGB(labels_np, seq_extract.labels_dict)
                if robot1.min_frame < frame_num < robot1.max_frame:
                    # robot1.lidar_points_list.extend(lidar_points_trans)
                    robot1.labels_list.extend(rgb_np)
                elif (
                    overlapping_data.min_frame < frame_num < overlapping_data.max_frame
                ):
                    # overlapping_data.lidar_points_list.extend(lidar_points_trans)
                    overlapping_data.labels_list.extend(rgb_np)
                elif robot2.min_frame < frame_num < robot2.max_frame:
                    # robot2.lidar_points_list.extend(lidar_points_trans)
                    robot2.labels_list.extend(rgb_np)

        print(f"seq_extract.current_frame: {seq_extract.current_frame}")
        seq_extract.current_frame += seq_extract.inc_frame

    print(
        f"    zero_building: {zero_building}/{total_scans}, zero_road: {zero_road}/{total_scans}"
    )
    print("\nConverting lists of points to o3D PCDs...")
    robot1.building_pcd_list = convert_polyline_points_to_o3d(
        robot1.building_pcd_list, [0, 0, 1]
    )
    robot1.road_pcd_list = convert_polyline_points_to_o3d(
        robot1.road_pcd_list, [1, 0, 0]
    )
    print("\n   - Robot 1 Done.")

    overlapping_data.building_pcd_list = convert_polyline_points_to_o3d(
        overlapping_data.building_pcd_list, [0, 0, 0]
    )
    overlapping_data.road_pcd_list = convert_polyline_points_to_o3d(
        overlapping_data.road_pcd_list, [0, 0, 0]
    )
    print("\n   - Robot 2 Done.")

    robot2.building_pcd_list = convert_polyline_points_to_o3d(
        robot2.building_pcd_list, o3d_colors["dark_green"]
    )
    robot2.road_pcd_list = convert_polyline_points_to_o3d(
        robot2.road_pcd_list, o3d_colors["magenta"]
    )
    print("\n   - Robot 3 Done.")

    # lidar_points_o3d = convert_pc_to_o3d(robot1.lidar_points_list, [0, 0, 0])
    # lidar_points_o3d.colors = o3d.utility.Vector3dVector(robot1.labels_list)
    print("DONE.")

    voxel_size = 0.5  # Specify the voxel size
    # lidar_points_o3d_DS = lidar_points_o3d.voxel_down_sample(voxel_size=voxel_size)

    o3d.visualization.draw_geometries(
        [
            robot1.building_pcd_list,
            robot1.road_pcd_list,
            # overlapping_data.building_pcd_list, overlapping_data.road_pcd_list,
            robot2.building_pcd_list,
            robot2.road_pcd_list,
        ]
    )
    # o3d.visualization.draw_geometries([lidar_points_o3d_DS])


def animate_extracted_data(seq_extract):
    key_to_callback = {
        ord("N"): lambda vis: change_frame(vis, ord("N"), seq_extract),
        ord("P"): lambda vis: change_frame(vis, ord("P"), seq_extract),
    }
    o3d.visualization.draw_geometries_with_key_callbacks([], key_to_callback)


def visualize_extracted_osm_data(seq_extract, animate="3d"):
    if animate == "3d":
        animate_extracted_data(seq_extract)
    elif animate == "bev":
        vis_lidar_BEV(seq_extract)
    elif animate == "all":
        vis_all_extracted_data(seq_extract)
    else:
        print(
            "Error: No visualization style chosen. Please select available animation style in config."
        )
