# MCD Utility Scripts

#### Functional
- **binarize_ros2_lidar.py** was used to binarize the lidar scans from a ROS2 bag. I used rosbags-convert to convert the ROS1 bags to ROS2.
- **create_global_map.py** is the unified entrypoint for creating a global semantic map. 
  - Use the `octree` subcommand (default) to accumulate `.bin` scans + inferred labels.
  - Use `voxel` to merge GT labeled `.pcd` scans.
  - Both strategies support saving to `.ply` (colored point cloud) and `.npy` (numpy array of points+colors/labels) via `--output-ply` and `--output-npy` arguments.
- **create_global_gt_map_octree.py** and **create_global_gt_map_voxel.py** are legacy wrappers that forward to `create_global_map.py`.
- **extract_per_scan_semantics.py** can be used once a global gt semantic map is made, I then find semantics for all scans that have a GT pose associated with it in the sequence's GT pose csv (pose_inW.csv).

#### Visualization
- **plot_bin_map_semantic.py** will create a global semantic map using the GT semantics (.bin) we extracted using **extract_per_scan_semantics.py** above. It will also save a .ply for viewing later of if desired.
- **plot_pcd_map.py** will create a global semantic map using the GT pcd files that the dataset ships with. It will also save a ply file.

#### OSM alignment
This process for aligning the OSM data was done as follows:

1. **visualization/view_initial_pose.py** lets a user simply see view the initial pose as a coordinate frame for one of the sequences. This will become our anchor pose. Take a screenshot for reference in the next step.
2. **view_osm_interactive.py** will plot OSM data in matplotlib and let a user select a rough coordinate with their mouse where the initial pose in the last step lies. Once the mouse is pressed, the lat-lon coordinates will be printed in the console.
3. **plot_path_onto_osm.py** lets a user then plot the path from the above sequence onto the OSM map to ensure scaling is decent. Additions to this script should let a user translate and rotate the path in the chance it is far too off.
4. **view_points_on_osm_interactive.py** plots a map of points onto the OSM data using the anchor pose and the global semantic map (.npy) file. Additionally, it allows users to translate/rotate the points on the map for better alignment. At every keypress, a new lat-lon will be printed. Once satisfied, use this latlong as the anchor pose in lat-lon and visualize again.
5. **plot_points_onto_osm.py** can be used to visualize the points projected on OSM again and will utilize the anchor pose for other sequences in the same environment.

- Note: **plot_points_onto_osm.py** and **plot_path_onto_osm.py** can both be reconciled in **view_points_on_osm_interactive.py** quite simply. Just make sure the path is translated with the points.

