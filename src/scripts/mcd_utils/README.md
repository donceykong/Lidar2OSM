# MCD Utility Scripts

#### Custom MCD Annotation Pipeline (under `mcd_annotation/`).
The following scripts were used to create the interpolated ground trut (GT) data. In essence, we use an annotated global map as the source of truth and map underlying point clouds to this map for relabeling. This enables creating a training dataset beyond the key-frames alone.

- **Step 1: binarize_ros2_lidar.py** was used to binarize the lidar scans from a ROS2 bag. I used rosbags-convert to convert the ROS1 bags to ROS2.
- **Step 2: create_global_map.py** is the unified entrypoint for creating a global semantic map. 
  - Use the `octree` subcommand (default) to accumulate `.bin` scans + inferred labels.
  - Use `voxel` to merge GT labeled `.pcd` scans.
  - Both strategies support saving to `.ply` (colored point cloud) and `.npy` (numpy array of points+colors/labels) via `--output-ply` and `--output-npy` arguments.
- **Step 3: extract_perscan_semantics.py** extracts per-scan semantic labels from a global GT map (.npy). Requires `--scan_dir`, `--pose_file`, `--merged_npy`, and `--output_dir`. Supports parallel processing via `--jobs`.

To view the annotated point clouds, the following script can be used to see how well the above methods work for annotating the underlying MCD data.

- **mcd_annotation/plot_semantic_map.py** - Unified script to visualize semantic maps from BIN or PCD files.
  - `bin` subcommand: Create maps using GT semantics (.bin) from **extract_perscan_semantics.py**.
  - `pcd` subcommand: Create maps using GT PCD files that ship with the dataset.

#### Aligning OSM data to MCD (under `osm_annotation/`)
This process for aligning the OSM data was done as follows:

1. **view_initial_pose.py** lets a user simply see view the initial pose as a coordinate frame for one of the sequences. This will become our anchor pose. Take a screenshot for reference in the next step.
2. **visualize_osm.py** is the unified script for visualizing OSM data, robot paths, and point clouds. It replaces the previous separate scripts.
    - **Usage**:
      ```bash
      python3 osm_annotation/visualize_osm.py --osm <path_to_osm> [--pose <path_to_pose_csv>] [--npy <path_to_pointcloud_npy>]
      ```
    - **Interactive Features**:
        - **Click**: Print lat-lon coordinates of the mouse cursor (useful for finding initial anchor).
        - **Arrow Keys**: Shift the overlay (robot path and point cloud) to align with the map.
        - **'S' Key**: Save the shifted poses to a new CSV file (`_shifted_utm.csv`).

    - **Workflow**:
        1. Run with just `--osm` to find the start coordinate.
        2. Run with `--osm` and `--pose` (and optionally `--npy`) to visualize alignment.
        3. Use arrow keys to refine alignment.
        4. Press 'S' to save the aligned poses.