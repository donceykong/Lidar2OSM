"""
Script to report the size of extracted data per frame in a sequence.

This script processes a sequence of extracted LiDAR and OSM data,
reporting the file sizes of the extracted data for each frame and
visualizing the results with a bar chart.

Author: Doncey Albin
Date: 08/22/2024

Usage:
    This script is intended to be used as part of a data extraction pipeline.
    It requires the sequence extraction object (`seq_extract`) to be initialized
    with the proper directories and parameters.

"""

# External Imports
import os
import matplotlib.pyplot as plt

# Internal Imports
from lidar2osm.utils.file_io import find_min_max_file_names


def report_size_extracted_data(seq_extract):
    first_frame, last_extracted_frame = find_min_max_file_names(seq_extract.extracted_per_frame_dir, number_delimiter='_', file_extension='.npy')
    last_extracted_frame = 200
    print(f"\n \nSequence: {seq_extract.seq}, First frame extracted: {first_frame}, final frame extracted: {last_extracted_frame}")
    
    frame_points_file_size_list = []
    OSM_file_size_list = []
    frames = []
    
    while seq_extract.current_frame < last_extracted_frame:
        frame_num = seq_extract.current_frame
        extracted_per_frame_dir = seq_extract.extracted_per_frame_dir

        frame_build_points_file = os.path.join(seq_extract.extracted_per_frame_dir, f'{seq_extract.current_frame:010d}_build_points.npy')
        frame_road_points_file = os.path.join(extracted_per_frame_dir, f'{frame_num:010d}_road_points.npy')
        frame_points_file = os.path.join(seq_extract.extracted_per_frame_dir, f'{seq_extract.current_frame:010d}_lidar_scan_points.npy')

        if os.path.exists(frame_build_points_file):
            frame_points_file_size = os.path.getsize(frame_points_file) / 1e6
            frame_points_file_size_list.append(frame_points_file_size)
            OSM_file_size = os.path.getsize(frame_build_points_file) + os.path.getsize(frame_road_points_file)
            OSM_file_size_list.append(OSM_file_size)
            frames.append(frame_num)

        print(f"seq_extract.current_frame: {seq_extract.current_frame}")
        seq_extract.current_frame += seq_extract.inc_frame
    
    # Plotting the barchart with dual y-axis
    fig, ax1 = plt.subplots()

    index = range(len(frame_points_file_size_list))
    bar_width = 0.35

    rects1 = ax1.bar(index, frame_points_file_size_list, bar_width, label='LiDAR Scan Points', color='b')
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('LiDAR File Size (Megabytes)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    # ax1.set_xticks([i + bar_width / 2 for i in index])
    # ax1.set_xticklabels([str(index) for f, index in enumerate(frames)], rotation=45)

    # Create a second y-axis for the smaller data
    ax2 = ax1.twinx()
    rects2 = ax2.bar([i + bar_width for i in index], OSM_file_size_list, bar_width, label='OSM File Sizes', color='r')
    ax2.set_ylabel('OSM File Size (bytes)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Adding legends
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    
    plt.title(f'File Sizes per Frame in seq {seq_extract.seq}')
    plt.show()