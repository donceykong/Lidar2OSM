# MCD Utility Scripts

#### Custom MCD Annotation Pipeline (under `mcd_annotation/`).
The following scripts were used to create the interpolated ground truth data. In essence, we use an annotated global map as the source of truth and map underlying point clouds to this map for relabeling. This enables creating a training dataset beyond the key-frames alone.

- **Step 1: binarize_ros2_lidar.py** was used to binarize the lidar scans from a ROS2 bag. I used rosbags-convert to convert the ROS1 bags to ROS2.
- **Step 2: create_global_map.py** is the unified entrypoint for creating a global semantic map. 
  - Use the `octree` subcommand (default) to accumulate `.bin` scans + inferred labels.
  - Use `voxel` to merge GT labeled `.pcd` scans.
  - Both strategies support saving to `.ply` (colored point cloud) and `.npy` (numpy array of points+colors/labels) via `--output-ply` and `--output-npy` arguments.
- **Step 3: extract_perscan_semantics.py** extracts per-scan semantic labels from a global GT map (.npy). Requires `--scan_dir`, `--pose_file`, `--merged_npy`, and `--output_dir`. Supports parallel processing via `--jobs`.

To view the annotated point clouds, the following script can be used to see how well the above methods work for annotating the underlying MCD data.

- **visualization/plot_semantic_map.py** - Unified script to visualize semantic maps from BIN or PCD files.
  - `bin` subcommand: Create maps using GT semantics (.bin) from **extract_perscan_semantics.py**.
  - `pcd` subcommand: Create maps using GT PCD files that ship with the dataset.

#### Aligning OSM data to MCD (under `mcd_annotation/`)
This process for aligning the OSM data was done as follows:

1. **visualization/view_initial_pose.py** lets a user simply see view the initial pose as a coordinate frame for one of the sequences. This will become our anchor pose. Take a screenshot for reference in the next step.
2. **view_osm_interactive.py** will plot OSM data in matplotlib and let a user select a rough coordinate with their mouse where the initial pose in the last step lies. Once the mouse is pressed, the lat-lon coordinates will be printed in the console.
3. **plot_path_onto_osm.py** lets a user then plot the path from the above sequence onto the OSM map to ensure scaling is decent. Additions to this script should let a user translate and rotate the path in the chance it is far too off.
4. **view_points_on_osm_interactive.py** plots a map of points onto the OSM data using the anchor pose and the global semantic map (.npy) file. Additionally, it allows users to translate/rotate the points on the map for better alignment. At every keypress, a new lat-lon will be printed. Once satisfied, use this latlong as the anchor pose in lat-lon and visualize again.
5. **plot_points_onto_osm.py** can be used to visualize the points projected on OSM again and will utilize the anchor pose for other sequences in the same environment.

- Note: **plot_points_onto_osm.py** and **plot_path_onto_osm.py** can both be reconciled in **view_points_on_osm_interactive.py** quite simply. Just make sure the path is translated with the points.