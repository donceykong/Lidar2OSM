#!/usr/bin/env bash

# Resolve absolute paths
DATASETS_DIR="$(realpath /media/donceykong/doncey_ssd_02/datasets/CU_MULTI/ros2_bags/with_gt)"
ROS2_WS_DIR="$(realpath /home/donceykong/Desktop/ARPG/projects/fall_2024/Lidar2OSM_FULL/lidar2osm_ws)"

# Give privledge for screen sharing
xhost +local:root

# Run Docker container
docker run -it -d --rm --privileged \
  --name lidar2osm_ros2 \
  --net=host \
  --env="DISPLAY=$DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="$DATASETS_DIR:/root/Datasets:rw" \
  --volume="$ROS2_WS_DIR:/root/lidar2osm_ws:rw" \
  lidar2osm_ros2

# Optional mounts and GPU support (uncomment as needed):
#   --gpus all \
#   --env="NVIDIA_VISIBLE_DEVICES=all" \
#   --env="NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility" \

docker exec -it lidar2osm_ros2 /bin/bash

