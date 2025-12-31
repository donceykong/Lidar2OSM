#!/usr/bin/env python3
"""
Simple script to convert robot poses to lat/lon and overlay on OSM map.
Hardcoded for main_campus robot1.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import folium
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Internal imports
from lidar2osm.utils.file_io import read_bin_file
from lidar2osm.utils.osm_handler import OSMDataHandler
from lidar2osm.datasets.cu_multi_dataset import labels as sem_kitti_labels
from lidar2osm.core.pointcloud import labels2RGB
from lidar2osm.utils.label_filter import filter_building_labels_with_osm, filter_building_labels_with_osm_fast


def load_poses(poses_file):
    """Load UTM poses from CSV file."""
    import pandas as pd
    
    print(f"\nReading UTM posses CSV file: {poses_file}")
    
    # Try different CSV reading approaches
    try:
        # Skip comment lines starting with #
        df = pd.read_csv(poses_file, comment='#')
        print(f"Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"CSV reading with comment skipping failed: {e}")
        try:
            # Try reading with different separators
            df = pd.read_csv(poses_file, sep=r'\s+', comment='#')  # whitespace separator
            print(f"Successfully read CSV with whitespace separator: {len(df)} rows")
        except Exception as e2:
            print(f"Whitespace separator failed: {e2}")
            try:
                # Try reading with comma separator and skip problematic lines
                df = pd.read_csv(poses_file, sep=',', on_bad_lines='skip', comment='#')
                print(f"Successfully read CSV with comma separator (skipping bad lines): {len(df)} rows")
            except Exception as e3:
                print(f"All CSV reading attempts failed: {e3}")
                return {}
    
    poses = {}
    for _, row in df.iterrows():
        try:
            # Try to get values by column name first, then by position
            if 'timestamp' in df.columns:
                timestamp = row['timestamp']
                x = row['x']
                y = row['y']
                z = row['z']
                qx = row['qx']
                qy = row['qy']
                qz = row['qz']
                qw = row['qw']
            else:
                # Use positional indexing - check if we have enough columns
                if len(row) >= 8:
                    timestamp = row.iloc[0]
                    x = row.iloc[1]
                    y = row.iloc[2]
                    z = row.iloc[3]
                    qx = row.iloc[4]
                    qy = row.iloc[5]
                    qz = row.iloc[6]
                    qw = row.iloc[7]
                else:
                    print(f"Row has only {len(row)} columns, skipping")
                    continue
            
            pose = [float(x), float(y), float(z), float(qx), float(qy), float(qz), float(qw)]
            poses[float(timestamp)] = pose
        except Exception as e:
            print(f"Error processing row: {e}")
            print(f"Row data: {row}")
            continue
    
    print(f"Successfully loaded {len(poses)} poses\n")
    return poses


def transform_imu_to_lidar(poses):
    """Transform poses from IMU frame to LiDAR frame."""
    from scipy.spatial.transform import Rotation as R
    
    # IMU to LiDAR transformation
    IMU_TO_LIDAR_T = np.array([-0.058038, 0.015573, 0.049603])
    IMU_TO_LIDAR_Q = [0.0, 0.0, 1.0, 0.0]  # [qx, qy, qz, qw]
    
    # Convert quaternion to rotation matrix
    imu_to_lidar_rot = R.from_quat(IMU_TO_LIDAR_Q)
    imu_to_lidar_rot_matrix = imu_to_lidar_rot.as_matrix()
    
    transformed_poses = {}
    
    for timestamp, pose in poses.items():
        # pose format: [x, y, z, qx, qy, qz, qw]
        imu_position = np.array(pose[:3])
        imu_quat = pose[3:7]  # [qx, qy, qz, qw]
        
        # Convert IMU quaternion to rotation matrix
        imu_rot = R.from_quat(imu_quat)
        imu_rot_matrix = imu_rot.as_matrix()
        
        # Transform position: lidar_pos = imu_pos + imu_rot * imu_to_lidar_translation
        lidar_position = imu_position + imu_rot_matrix @ IMU_TO_LIDAR_T
        
        # Transform orientation: lidar_rot = imu_rot * imu_to_lidar_rot
        lidar_rot_matrix = imu_rot_matrix @ imu_to_lidar_rot_matrix
        lidar_quat = R.from_matrix(lidar_rot_matrix).as_quat()  # [qx, qy, qz, qw]
        
        # Combine position and orientation
        transformed_pose = np.concatenate([lidar_position, lidar_quat])
        transformed_poses[timestamp] = transformed_pose
    
    return transformed_poses


def voxel_downsample(points, intensities, voxel_size=1.0):
    """
    Downsample point cloud using voxel centers.
    
    Args:
        points: numpy array of shape (N, 3) containing xyz coordinates
        intensities: numpy array of shape (N,) containing intensity values
        voxel_size: size of voxel cube in meters (default: 1.0)
    
    Returns:
        downsampled_points: numpy array of shape (M, 3) with voxel centers
        downsampled_intensities: numpy array of shape (M,) with mean intensities
    """
    if len(points) == 0:
        return np.array([]), np.array([])
    
    # Calculate voxel grid dimensions
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    voxel_dims = np.ceil((max_coords - min_coords) / voxel_size).astype(int)
    
    # Assign points to voxels
    voxel_indices = np.floor((points - min_coords) / voxel_size).astype(int)
    voxel_indices = np.clip(voxel_indices, 0, voxel_dims - 1)
    
    # Create unique voxel keys using a more robust approach
    # Use string concatenation to avoid integer overflow
    voxel_keys = np.array([f"{x}_{y}_{z}" for x, y, z in voxel_indices])
    
    # Find unique voxels and calculate centers
    unique_voxels, inverse_indices = np.unique(voxel_keys, return_inverse=True)
    
    downsampled_points = []
    downsampled_intensities = []
    
    for i, voxel_id in enumerate(unique_voxels):
        # Find all points in this voxel
        voxel_mask = inverse_indices == i  # Use index, not voxel_id
        voxel_points = points[voxel_mask]
        voxel_intensities = intensities[voxel_mask]
        
        # Use voxel center (mean of all points in voxel)
        if len(voxel_points) > 0:
            voxel_center = np.mean(voxel_points, axis=0)
            mean_intensity = np.mean(voxel_intensities)
            
            downsampled_points.append(voxel_center)
            downsampled_intensities.append(mean_intensity)
    
    return np.array(downsampled_points), np.array(downsampled_intensities)


def utm_to_latlon(poses):
    """Convert UTM poses to lat/lon coordinates."""
    # from pyproj import Proj, transform # old way
    from pyproj import Transformer
    
    # UTM zone for Colorado (zone 13N)
    # utm_proj = Proj(proj='utm', zone=13, ellps='WGS84')
    # wgs84_proj = Proj(proj='latlong', ellps='WGS84')
    transformer = Transformer.from_crs("EPSG:32613", "EPSG:4326", always_xy=True)
    
    # Extract positions (x, y, z) from poses
    positions = []
    timestamps = []
    
    for timestamp, pose in poses.items():
        # pose format: [x, y, z, qx, qy, qz, qw]
        position = pose[:3]  # x, y, z
        positions.append(position)
        timestamps.append(timestamp)
    
    positions = np.array(positions)
    timestamps = np.array(timestamps)
    
    # Convert UTM to lat/lon
    # lons, lats = transform(utm_proj, wgs84_proj, positions[:, 0], positions[:, 1])
    lons, lats = transformer.transform(positions[:, 0], positions[:, 1])
    
    # Combine with z coordinates
    latlon_positions = np.column_stack([lats, lons, positions[:, 2]])
    
    return latlon_positions, timestamps


def create_osm_map_with_poses(osm_file, poses_latlon, output_file="robot_poses_on_osm.html"):
    """Create an interactive map with OSM data and robot poses."""
    
    # Extract only lat, lon coordinates for Folium (remove z coordinate)
    poses_2d = poses_latlon[:, :2]  # Take only lat, lon columns
    
    # Calculate map center from poses
    lats = poses_2d[:, 0]
    lons = poses_2d[:, 1]
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=18,
        tiles='OpenStreetMap'
    )
    
    # Add robot path
    folium.PolyLine(
        locations=poses_2d,
        color='red',
        weight=3,
        opacity=0.8,
        popup='Robot Path'
    ).add_to(m)
    
    # Add start and end markers
    folium.Marker(
        location=poses_2d[0],
        popup='Start',
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    
    folium.Marker(
        location=poses_2d[-1],
        popup='End',
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)
    
    # Add some intermediate markers for reference
    num_markers = min(10, len(poses_2d))
    step = len(poses_2d) // num_markers
    for i in range(0, len(poses_2d), step):
        folium.CircleMarker(
            location=poses_2d[i],
            radius=3,
            color='blue',
            fill=True,
            popup=f'Frame {i}'
        ).add_to(m)
    
    # Save map
    m.save(output_file)
    print(f"Map saved to {output_file}")
    
    return m


def create_simple_plot(poses_latlon, poses_dict=None, osm_file=None, show_lidar=False, 
                      dataset_path=None, robot="robot1", environment="main_campus", 
                      output_file="robot_poses_plot.png", show_semantic=False):
    """Create a simple matplotlib plot of the robot path with optional OSM data."""
    
    plt.figure(figsize=(12, 8))
    
    # Load and plot OSM data if available
    if osm_file and Path(osm_file).exists():
        # Initialize OSM data handler with larger filter distance and semantic control
        osm_handler = OSMDataHandler(osm_file, filter_distance=100.0)
        
        # Configure which semantics to load/plot
        osm_handler.set_semantics({
            'roads': False,
            'highways': True,
            'buildings': True,
            'trees': False,
            'grassland': True,
            'water': False,
            'parking': False,
            'amenities': False  # Disable amenities to reduce clutter
        })
        
        # Load OSM data
        if osm_handler.load_osm_data():
            # Plot all OSM elements with centroid-based filtering for better performance
            osm_handler.plot_all_osm_data(poses_latlon, ax=plt.gca(), use_centroid=False)
            
            # Print summary
            summary = osm_handler.get_osm_summary()
            print(f"OSM Summary: {summary}")
        else:
            print("Failed to load OSM data")
    
    # Sample 100 poses from start to finish
    num_poses = len(poses_latlon)
    if num_poses > 100:
        step = num_poses // 100
        sampled_indices = list(range(0, num_poses, step))[:100]
    else:
        sampled_indices = list(range(num_poses))
    
    sampled_poses = poses_latlon[sampled_indices]
    print(f"Sampled {len(sampled_poses)} poses from {num_poses} total poses")
    
    # Plot the robot path
    plt.plot(poses_latlon[:, 1], poses_latlon[:, 0], 'r-', linewidth=3, label='Robot Path', zorder=10)
    
    # Mark start and end
    plt.plot(poses_latlon[0, 1], poses_latlon[0, 0], 'go', markersize=12, label='Start', zorder=11)
    plt.plot(poses_latlon[-1, 1], poses_latlon[-1, 0], 'ro', markersize=12, label='End', zorder=11)
    
    # Add arrows showing x-direction (heading) for sampled poses
    if poses_dict is not None:
        # Get timestamps for sampled poses
        timestamps = list(poses_dict.keys())
        timestamps.sort()
        
        for i, pose_idx in enumerate(sampled_indices):
            if pose_idx < len(timestamps):
                timestamp = timestamps[pose_idx]
                pose_data = poses_dict[timestamp]
                
                # Extract quaternion (qx, qy, qz, qw)
                qx, qy, qz, qw = pose_data[3], pose_data[4], pose_data[5], pose_data[6]
                
                # Convert quaternion to heading angle (yaw)
                # For a quaternion (qx, qy, qz, qw), yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
                yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                
                # Convert yaw to direction vector
                # In robot frame, x is forward, so heading direction is (cos(yaw), sin(yaw))
                dx = np.cos(yaw)
                dy = np.sin(yaw)
                
                # Scale for arrow length (make them smaller and more visible)
                length = 0.00005  # Smaller arrow length in degrees
                dx_scaled = dx * length
                dy_scaled = dy * length
                
                # Get position
                lat, lon = poses_latlon[pose_idx][0], poses_latlon[pose_idx][1]
                
                # Draw arrow with better visibility
                plt.arrow(lon, lat, dx_scaled, dy_scaled, 
                        head_width=0.00002, head_length=0.000015, 
                        fc='darkorange', ec='darkorange', alpha=0.8, zorder=12, width=0.000005)
    else:
        # Fallback: calculate heading from adjacent poses
        for i, pose_idx in enumerate(sampled_indices):
            pose = poses_latlon[pose_idx]
            lat, lon = pose[0], pose[1]
            
            if i < len(sampled_indices) - 1:
                next_pose = poses_latlon[sampled_indices[i + 1]]
                # Calculate direction vector
                dx = next_pose[1] - lon  # longitude difference
                dy = next_pose[0] - lat  # latitude difference
                
                # Normalize and scale for arrow
                length = 0.00005  # Smaller arrow length in degrees
                if dx != 0 or dy != 0:
                    norm = np.sqrt(dx*dx + dy*dy)
                    dx_norm = dx / norm * length
                    dy_norm = dy / norm * length
                    
                    # Draw arrow with better visibility
                    plt.arrow(lon, lat, dx_norm, dy_norm, 
                            head_width=0.00002, head_length=0.000015, 
                            fc='darkorange', ec='darkorange', alpha=0.8, zorder=12, width=0.000005)

    from lidar2osm.utils.label_filter import filter_building_labels_alg2
    filter_building_labels_alg2(osm_handler)

    # Add LiDAR scan overlay if requested
    if show_lidar and dataset_path is not None:
        print("Loading and overlaying LiDAR scans...")
        try:
            # Use 500 poses for LiDAR overlay
            velodyne_path = Path(dataset_path) / environment / robot / "lidar_bin/data"
            velodyne_files = sorted([f for f in velodyne_path.glob("*.bin")])
            
            # Load semantic labels if requested
            labels_path = Path(dataset_path) / environment / robot / "lidar_labels"
            label_files = sorted([f for f in labels_path.glob("*.bin")]) if show_semantic and labels_path.exists() else None
            
            # Convert labels to dictionary for coloring
            labels_dict = {label.id: label.color for label in sem_kitti_labels} if show_semantic else None
            
            # Sample 500 poses evenly distributed
            total_poses = len(poses_latlon)
            lidar_sample_count = min(500, total_poses)
            lidar_sample_indices = np.linspace(0, total_poses - 1, lidar_sample_count, dtype=int)
            
            print(f"Found {len(velodyne_files)} LiDAR scans, using {lidar_sample_count} poses for overlay")
            if show_semantic and label_files:
                print(f"Found {len(label_files)} semantic label files")
            
            # Accumulate point clouds from sampled poses
            all_world_points = []
            all_intensities = []
            per_scan_voxel_size = 4.0  # meters - appropriate for UTM coordinate scale
            global_voxel_size = 1.0  # meters - appropriate for UTM coordinate scale

            print("Processing LiDAR scans...")
            for pose_idx in tqdm(lidar_sample_indices, desc="Loading LiDAR scans", unit="scan"):
                if pose_idx < len(velodyne_files):
                    try:
                        print(f"\\n\\nProcessing pose {pose_idx}")

                        # Load LiDAR scan for this pose
                        points = read_bin_file(velodyne_files[pose_idx], dtype=np.float32, shape=(-1, 4))
                        print(f"Points shape: {points.shape}")
                        points_xyz = points[:, :3]  # Extract xyz coordinates
                        intensities = points[:, 3]  # Extract intensity
                        
                        # Load semantic labels if available
                        labels = None
                        if show_semantic and label_files and pose_idx < len(label_files):
                            try:
                                labels = read_bin_file(label_files[pose_idx], dtype=np.int32)
                                print(f"Labels shape: {labels.shape}")
                                # Ensure labels have the same length as points
                                if len(labels) != len(points_xyz):
                                    print(f"Warning: Labels length ({len(labels)}) doesn't match points length "
                                          f"({len(points_xyz)}) for pose {pose_idx}, truncating to match")
                                    min_length = min(len(labels), len(points_xyz))
                                    labels = labels[:min_length]
                                    points_xyz = points_xyz[:min_length]
                                    intensities = intensities[:min_length]  
                                    print(f"After truncation - Points: {len(points_xyz)}, Labels: {len(labels)}")
                                else:
                                    print(f"Labels length ({len(labels)}) matches points length ({len(points_xyz)}) for pose {pose_idx}")
                            except Exception as e:
                                print(f"Warning: Could not load semantic labels for pose {pose_idx}: {e}")
                                labels = None
                        
                        # Get pose for this frame
                        if poses_dict is not None:
                            timestamps = list(poses_dict.keys())
                            timestamps.sort()
                            if pose_idx < len(timestamps):
                                timestamp = timestamps[pose_idx]
                                pose_data = poses_dict[timestamp]
                                
                                # Create transformation matrix from pose
                                from scipy.spatial.transform import Rotation as R
                                
                                # Position and orientation
                                position = pose_data[:3]
                                quat = pose_data[3:7]  # [qx, qy, qz, qw]
                                
                                # Create 4x4 transformation matrix
                                rotation_matrix = R.from_quat(quat).as_matrix()
                                transform_matrix = np.eye(4)
                                transform_matrix[:3, :3] = rotation_matrix
                                transform_matrix[:3, 3] = position
                                
                                # Transform points to world coordinates
                                points_homogeneous = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1))])
                                world_points = (transform_matrix @ points_homogeneous.T).T
                                world_points_xyz_full = world_points[:, :3]
                                
                                # Downsample points
                                print(f"Voxel downsampling points with voxel size {per_scan_voxel_size}")
                                world_points_xyz, intensities = voxel_downsample(world_points_xyz_full, intensities, voxel_size=per_scan_voxel_size)
                                
                                # Downsample labels if available
                                if labels is not None:
                                    print(f"\\n\\nBefore label downsampling - World points: {len(world_points_xyz_full)}, Labels: {len(labels)}")
                                    # Ensure labels have the same length as world_points_xyz before downsampling
                                    # if len(labels) != len(world_points_xyz):
                                    #     print(f"Warning: Labels length ({len(labels)}) doesn't match world points length ({len(world_points_xyz)}) for pose {pose_idx}, skipping label downsampling")
                                    #     labels = None
                                    # else:
                                    # Apply same voxel downsampling to labels
                                    print(f"Downsampling labels with voxel size {per_scan_voxel_size}")
                                    _, labels = voxel_downsample(world_points_xyz_full, labels.astype(np.float32), voxel_size=per_scan_voxel_size)
                                    labels = labels.astype(np.int32)
                                    print(f"After label downsampling - Labels: {len(labels)}")

                                    # Filter out labels that are only buildings (ie 50)
                                    mask = np.isin(labels, [50])
                                    world_points_xyz = world_points_xyz[mask]
                                    intensities = intensities[mask]
                                    labels = labels[mask]
                                    print(f"After label filtering - Labels: {len(labels)}")

#                                    # TODO: Fix below relabeling to cycle through per building geom
#                                    # and then filter points that are within a circular radius about the building

                                    # Filter labels that are not within building polygons
                                    # Relabel building points not within OSM building polygons as vegetation
                                    if osm_handler is not None and hasattr(osm_handler, 'osm_geometries'):
                                        labels = filter_building_labels_with_osm_fast(
                                            world_points_xyz, 
                                            labels, 
                                            osm_handler,
                                            poses_latlon=poses_latlon,
                                            building_label_id=50,
                                            vegetation_label_id=70
                                        )
                                    
                                # Accumulate points, intensities, and labels
                                all_world_points.append(world_points_xyz)
                                all_intensities.append(intensities)
                                if labels is not None:
                                    if 'all_labels' not in locals():
                                        all_labels = []
                                    all_labels.append(labels)
                                
                    except Exception as e:
                        print(f"Error loading LiDAR scan for pose {pose_idx}: {e}")
                        continue
            
            if all_world_points:
                # Combine all point clouds
                combined_points = np.vstack(all_world_points)
                combined_intensities = np.hstack(all_intensities)
                
                # Combine labels if available
                combined_labels = None
                if show_semantic and 'all_labels' in locals() and all_labels:
                    combined_labels = np.hstack(all_labels)
                    print(f"Combined labels shape: {combined_labels.shape}")
                    print(f"Unique labels: {np.unique(combined_labels)}")
                    print(f"Labels dict keys: {list(labels_dict.keys()) if labels_dict else 'None'}")
                else:
                    print("No combined labels available for semantic coloring")
                
                print(f"Combined {len(combined_points)} points from {len(all_world_points)} scans")
                
                # Trim outlier intensities for the entire map (remove top and bottom 5%)
                print("Trimming outlier intensities for entire map...")
                sorted_intensities = np.sort(combined_intensities)
                trim_count = max(1, len(sorted_intensities) // 20)  # 5% on each side
                intensity_min = sorted_intensities[trim_count]
                intensity_max = sorted_intensities[-trim_count]
                print(f"Trim count: {trim_count}")
                print(f"Intensity min: {intensity_min}")
                print(f"Intensity max: {intensity_max}")
                
                # Clip intensities to trimmed range
                combined_intensities = np.clip(combined_intensities, intensity_min, intensity_max)
                print(f"Intensity range trimmed to [{intensity_min:.3f}, {intensity_max:.3f}]")
                
                # Voxel downsampling
                print(f"Applying final voxel downsampling with {global_voxel_size}m voxel size...")
                
                # Downsample all points
                downsampled_points, downsampled_intensities = voxel_downsample(
                    combined_points, combined_intensities, voxel_size=global_voxel_size
                )
                
                print(f"Downsampled to {len(downsampled_points)} points")
                
                # Downsample labels if available (this is the key fix!)
                if show_semantic and combined_labels is not None:
                    print(f"Downsampling labels from {len(combined_labels)} to match {len(downsampled_points)} points")
                    _, downsampled_labels = voxel_downsample(
                        combined_points, combined_labels.astype(np.float32), voxel_size=global_voxel_size
                    )
                    combined_labels = downsampled_labels.astype(np.int32)
                    print(f"Downsampled labels to {len(combined_labels)} labels")
                
                # Convert to lat/lon
                # from pyproj import Proj, transform # old way
                # utm_proj = Proj(proj='utm', zone=13, ellps='WGS84')
                # wgs84_proj = Proj(proj='latlong', ellps='WGS84')
                # lons, lats = transform(utm_proj, wgs84_proj, 
                #                      downsampled_points[:, 0], downsampled_points[:, 1])
                # new way
                from pyproj import Transformer

                transformer = Transformer.from_crs("EPSG:32613", "EPSG:4326", always_xy=True)
                lons, lats = transformer.transform(downsampled_points[:, 0], downsampled_points[:, 1])

                # Determine coloring scheme
                if show_semantic and combined_labels is not None:
                    # Use semantic labels for coloring
                    print("Using semantic labels for point cloud coloring")
                    print(f"Combined labels shape: {combined_labels.shape}")
                    print(f"Downsampled points shape: {downsampled_points.shape}")
                    print(f"Labels dict: {labels_dict}")
                    
                    # Get semantic colors
                    semantic_colors = labels2RGB(combined_labels, labels_dict)
                    print(f"Semantic colors shape: {semantic_colors.shape}")
                    print(f"Semantic colors range: [{semantic_colors.min():.3f}, {semantic_colors.max():.3f}]")
                    
                    # Calculate alpha based on Z height for semantic points
                    z_values = downsampled_points[:, 2]
                    normalized_z = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values))
                    alpha_values = 0.3 + 0.6 * normalized_z  # Alpha between 0.3 and 0.9 based on height
                    alpha_values = np.clip(alpha_values, 0.3, 0.9)
                    
                    # Plot semantic points with RGB colors
                    scatter = plt.scatter(lons, lats, c=semantic_colors, s=2.0, alpha=alpha_values, 
                                        zorder=5)
                    
                    # Create a custom legend for semantic classes
                    unique_labels = np.unique(combined_labels)
                    legend_elements = []
                    for label_id in unique_labels[:10]:  # Show top 10 most common classes
                        if label_id in labels_dict:
                            color = np.array(labels_dict[label_id]) / 255.0
                            # Find the label name
                            label_name = "Unknown"
                            for label in sem_kitti_labels:
                                if label.id == label_id:
                                    label_name = label.name
                                    break
                            legend_elements.append(
                                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                            markersize=8, label=label_name)
                            )
                    
                    if legend_elements:
                        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
                else:
                    # Use intensity coloring (original behavior)
                    print("Using intensity values for point cloud coloring")
                    
                    # Normalize intensities for coloring (0-1 range)
                    normalized_intensities = (downsampled_intensities - np.min(downsampled_intensities)) / \
                                           (np.max(downsampled_intensities) - np.min(downsampled_intensities))
                    
                    # Calculate alpha based on intensity and Z (height)
                    z_values = downsampled_points[:, 2]  # Z coordinates
                    normalized_z = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values))
                    
                    # Combine intensity and Z for alpha calculation
                    alpha_values = 0.9 * normalized_intensities + 0.1 * normalized_z
                    alpha_values = np.clip(alpha_values, 0.1, 0.9)
                    
                    # Plot LiDAR points with intensity coloring
                    scatter = plt.scatter(lons, lats, c=normalized_intensities, s=1.0, alpha=alpha_values, 
                                        zorder=5, cmap='viridis', vmin=0, vmax=1)
                    
                    # Add colorbar for intensity
                    cbar = plt.colorbar(scatter, ax=plt.gca(), shrink=0.8)
                    cbar.set_label('LiDAR Intensity', rotation=270, labelpad=15)
                        
        except Exception as e:
            print(f"Error overlaying LiDAR scans: {e}")
    
    print(f"\nPlotting now!\n")

    # Add some intermediate points
    for i in range(0, len(poses_latlon), max(1, len(poses_latlon) // 20)):
        plt.plot(poses_latlon[i, 1], poses_latlon[i, 0], 'bo', markersize=4, zorder=9)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Robot Poses on Main Campus with OSM Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to {output_file}")


def main():
    """Main function - hardcoded for main_campus robot1."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Visualize robot poses on OSM map")
    parser.add_argument("--show_lidar", action="store_true", 
                       help="Overlay LiDAR scans on the plot")
    parser.add_argument("--show_semantic", action="store_true",
                       help="Use semantic labels for LiDAR point coloring (requires --show_lidar)")
    args = parser.parse_args()
    
    # Hardcoded paths
    dataset_path = "/media/donceykong/donceys_data_ssd/datasets/CU_MULTI/data"
    environment = "main_campus"
    robot = "robot1"
    
    # Construct file paths
    poses_file = Path(dataset_path) / environment / robot / f"{robot}_{environment}_gt_utm_poses.csv"
    osm_file = Path(dataset_path) / environment / f"{environment}.osm"
    
    # Check if files exist
    if not poses_file.exists():
        print(f"Error: Poses file not found: {poses_file}")
        return
    
    if not osm_file.exists():
        print(f"Warning: OSM file not found: {osm_file}")
        print("Will create map without OSM overlay")
    
    # Load poses
    poses = load_poses(poses_file)
    print(f"Loaded {len(poses)} poses")
    
    if len(poses) == 0:
        print("No poses found!")
        return
    
    # Transform poses from IMU to LiDAR frame
    print("Transforming poses from IMU to LiDAR frame...")
    poses = transform_imu_to_lidar(poses)
    
    # Convert UTM poses to lat/lon
    poses_latlon, timestamps = utm_to_latlon(poses)
    print(f"Converted {len(poses_latlon)} poses to lat/lon")
    
    # Print some statistics
    print(f"Latitude range: {poses_latlon[:, 0].min():.6f} to {poses_latlon[:, 0].max():.6f}")
    print(f"Longitude range: {poses_latlon[:, 1].min():.6f} to {poses_latlon[:, 1].max():.6f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Create interactive map
    if osm_file.exists():
        create_osm_map_with_poses(osm_file, poses_latlon)
    else:
        # Create map without OSM file
        create_osm_map_with_poses(None, poses_latlon)
    
    print(f"Dataset path: {dataset_path}, Show LiDAR: {args.show_lidar}, Show Semantic: {args.show_semantic}")
    # Create simple plot with OSM data
    create_simple_plot(poses_latlon, poses, osm_file, show_lidar=args.show_lidar, 
                      dataset_path=dataset_path, robot=robot, environment=environment,
                      show_semantic=args.show_semantic)
    
    print("\nDone! Check the generated files:")
    print("- robot_poses_on_osm.html (interactive map)")
    print("- robot_poses_plot.png (static plot)")


if __name__ == "__main__":
    main()
