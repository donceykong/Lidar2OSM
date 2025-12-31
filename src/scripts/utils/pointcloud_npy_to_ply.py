#!/usr/bin/env python3
"""
Convert semantic map .npy file to PLY format with colored points based on semantic labels.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Internal imports
from lidar2osm.datasets.cu_multi_dataset import labels as sem_kitti_labels
from lidar2osm.core.pointcloud import labels2RGB


def load_semantic_map(npy_file):
    """
    Load semantic map from numpy file.
    
    Args:
        npy_file: Path to .npy file with columns [x, y, z, intensity, semantic_id]
    
    Returns:
        points: (N, 3) array of UTM coordinates [x, y, z]
        intensities: (N,) array of intensities
        labels: (N,) array of semantic labels
    """
    print(f"\nLoading semantic map from {npy_file}")
    data = np.load(npy_file)
    print(f"Loaded data shape: {data.shape}")
    print(f"Data columns: [x, y, z, intensity, semantic_id]")
    
    points = data[:, :3]  # x, y, z in UTM
    intensities = data[:, 3]
    labels = data[:, 4].astype(np.int32)
    
    print(f"  Points: {len(points)}")
    print(f"  UTM X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  UTM Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  UTM Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    print(f"  Unique labels: {np.unique(labels)}")
    
    return points, intensities, labels


def get_semantic_colors(labels):
    """
    Get RGB colors for semantic labels.
    
    Args:
        labels: (N,) array of semantic label IDs
    
    Returns:
        colors: (N, 3) array of RGB colors in [0, 1] range
    """
    from lidar2osm.core.pointcloud.pointcloud import labels2RGB_tqdm
    # Create label dictionary mapping label ID to RGB color
    labels_dict = {label.id: label.color for label in sem_kitti_labels}
    
    # Convert labels to RGB colors
    colors = labels2RGB_tqdm(labels, labels_dict)
    
    return colors


# def save_topdown_view(points, colors, labels, output_file, downsample_factor=1):
#     """
#     Save a top-down view (X-Y projection) of the point cloud as a PNG image.
    
#     Args:
#         points: (N, 3) array of xyz coordinates
#         colors: (N, 3) array of RGB colors in [0, 1] range
#         labels: (N,) array of semantic labels
#         output_file: Output PNG filename
#         downsample_factor: Downsample factor (1 = no downsampling, 2 = every 2nd point, etc.)
#     """
#     print(f"\nSaving top-down view to: {output_file}")

#     points_plot = points
#     colors_plot = colors
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=(16, 16))
    
#     # Plot points (X-Y view, top-down)
#     # Convert colors from [0, 1] to [0, 255] for matplotlib
#     colors_uint8 = (colors_plot * 255).astype(np.uint8)
#     colors_rgb = colors_uint8 / 255.0
    
#     # Scatter plot with colors
#     ax.scatter(points_plot[:, 0], points_plot[:, 1], 
#               c=colors_rgb, s=0.1, alpha=0.6, rasterized=True)
    
#     # Set labels and title
#     ax.set_xlabel('X (UTM Easting)', fontsize=12)
#     ax.set_ylabel('Y (UTM Northing)', fontsize=12)
#     ax.set_title(f'Top-Down View of Point Cloud ({len(points_plot):,} points)', fontsize=14, fontweight='bold')
    
#     # Set equal aspect ratio
#     ax.set_aspect('equal', adjustable='box')
    
#     # Add grid
#     ax.grid(True, alpha=0.3)
    
#     # Print coordinate ranges
#     print(f"  X range: [{points_plot[:, 0].min():.2f}, {points_plot[:, 0].max():.2f}]")
#     print(f"  Y range: [{points_plot[:, 1].min():.2f}, {points_plot[:, 1].max():.2f}]")
#     print(f"  Z range: [{points_plot[:, 2].min():.2f}, {points_plot[:, 2].max():.2f}]")
    
#     # Save figure
#     plt.tight_layout()
#     plt.savefig(output_file, dpi=150, bbox_inches='tight')
#     print(f"  Successfully saved to {output_file}")
#     plt.close()


def save_pointcloud_to_ply(points, colors, output_file):
    """
    Save point cloud to PLY file with colors.
    
    Args:
        points: (N, 3) array of xyz coordinates
        colors: (N, 3) array of RGB colors in [0, 1] range
        output_file: Output PLY filename
    """
    print(f"\nSaving point cloud to PLY file: {output_file}")
    print(f"  Points: {len(points)}")
    
    # Convert colors from [0, 1] to [0, 255] for PLY format
    colors_uint8 = (colors * 255).astype(np.uint8)
    
    # Calculate expected file size
    # Header: ~200 bytes (approximate)
    # Each point: 3 floats (4 bytes each) + 3 uchars (1 byte each) = 15 bytes
    expected_size = 200 + len(points) * 15
    print(f"  Expected file size: {expected_size / (1024 * 1024):.2f} MB")
    
    try:
        # Write PLY file
        with open(output_file, 'wb') as f:
            # Write PLY header (no extra indentation, proper newlines)
            header = f"""ply
format binary_little_endian 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
            f.write(header.encode('ascii'))
            
            # Write point data in batches for better performance
            batch_size = 10000
            for i in range(0, len(points), batch_size):
                end_idx = min(i + batch_size, len(points))
                batch_points = points[i:end_idx]
                batch_colors = colors_uint8[i:end_idx]
                
                # Write all points in batch at once
                for j in range(len(batch_points)):
                    # Write x, y, z as float32
                    f.write(np.array([batch_points[j, 0], batch_points[j, 1], batch_points[j, 2]], 
                                    dtype=np.float32).tobytes())
                    # Write r, g, b as uint8
                    f.write(np.array([batch_colors[j, 0], batch_colors[j, 1], batch_colors[j, 2]], 
                                    dtype=np.uint8).tobytes())
        
        # Verify file was written correctly
        if not Path(output_file).exists():
            raise FileNotFoundError(f"PLY file was not created: {output_file}")
        
        actual_size = Path(output_file).stat().st_size
        print(f"  Successfully saved to {output_file}")
        print(f"  Actual file size: {actual_size / (1024 * 1024):.2f} MB")
        
        # Check if size is reasonable (within 10% of expected)
        size_diff = abs(actual_size - expected_size) / expected_size
        if size_diff > 0.1:
            print(f"  WARNING: File size differs significantly from expected ({size_diff*100:.1f}% difference)")
        else:
            print(f"  File size matches expected (difference: {size_diff*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: Failed to save PLY file: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_ply_file(ply_file, expected_num_points):
    """
    Validate that a PLY file was written correctly.
    
    Args:
        ply_file: Path to PLY file
        expected_num_points: Expected number of points
    
    Returns:
        True if valid, False otherwise
    """
    print(f"\nValidating PLY file: {ply_file}")
    
    if not Path(ply_file).exists():
        print(f"  ERROR: File does not exist")
        return False
    
    file_size = Path(ply_file).stat().st_size
    print(f"  File size: {file_size / (1024 * 1024):.2f} MB")
    
    try:
        # Read and check header
        with open(ply_file, 'rb') as f:
            # Read header (first 200 bytes should be enough)
            header_bytes = f.read(200)
            header_text = header_bytes.decode('ascii', errors='ignore')
            
            # Check for PLY magic number
            if not header_text.startswith('ply'):
                print(f"  ERROR: File does not start with 'ply' magic number")
                return False
            
            # Check for vertex count
            if f'element vertex {expected_num_points}' not in header_text:
                print(f"  WARNING: Vertex count in header may not match expected")
                # Try to extract actual count
                import re
                match = re.search(r'element vertex (\d+)', header_text)
                if match:
                    actual_count = int(match.group(1))
                    print(f"    Header says: {actual_count} vertices")
                    print(f"    Expected: {expected_num_points} vertices")
                    if actual_count != expected_num_points:
                        print(f"    ERROR: Vertex count mismatch!")
                        return False
            
            # Check file size is reasonable
            # Header: ~200 bytes
            # Each point: 15 bytes (3 floats + 3 uchars)
            expected_size = 200 + expected_num_points * 15
            size_diff = abs(file_size - expected_size) / expected_size
            
            if size_diff > 0.1:
                print(f"  WARNING: File size differs from expected by {size_diff*100:.1f}%")
            else:
                print(f"  File size is correct (difference: {size_diff*100:.1f}%)")
            
            # Try to read a few points to verify structure
            f.seek(0)
            # Find end of header
            header_end = header_text.find('end_header')
            if header_end == -1:
                print(f"  ERROR: Could not find 'end_header' in header")
                return False
            
            header_end += len('end_header') + 1  # +1 for newline
            f.seek(header_end)
            
            # Read first point
            point_data = f.read(15)  # 3 floats (12 bytes) + 3 uchars (3 bytes)
            if len(point_data) != 15:
                print(f"  ERROR: Could not read complete first point (got {len(point_data)} bytes)")
                return False
            
            # Unpack first point to verify
            x, y, z = np.frombuffer(point_data[:12], dtype=np.float32)
            r, g, b = np.frombuffer(point_data[12:], dtype=np.uint8)
            print(f"  First point: x={x:.2f}, y={y:.2f}, z={z:.2f}, rgb=({r}, {g}, {b})")
            
            # Check if values are reasonable
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                print(f"  ERROR: First point contains non-finite values")
                return False
            
            if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                print(f"  ERROR: First point has invalid color values")
                return False
        
        print(f"  ✓ PLY file appears to be valid")
        return True
        
    except Exception as e:
        print(f"  ERROR: Failed to validate PLY file: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_point_clouds(original_file, relabeled_file):
    """
    Compare two point cloud files to verify they have the same number of points.
    
    Args:
        original_file: Path to original .npy file
        relabeled_file: Path to relabeled .npy file
    """
    print(f"\n{'='*80}")
    print("Comparing Point Clouds")
    print(f"{'='*80}")
    
    # Load original file
    print(f"\nLoading original file: {original_file}")
    if not Path(original_file).exists():
        print(f"  ERROR: File not found: {original_file}")
        return False
    
    original_data = np.load(original_file)
    original_points = original_data[:, :3]
    original_intensities = original_data[:, 3]
    original_labels = original_data[:, 4].astype(np.int32)
    
    print(f"  Original points: {len(original_points)}")
    print(f"  Original shape: {original_data.shape}")
    
    # Load relabeled file
    print(f"\nLoading relabeled file: {relabeled_file}")
    if not Path(relabeled_file).exists():
        print(f"  ERROR: File not found: {relabeled_file}")
        return False
    
    relabeled_data = np.load(relabeled_file)
    relabeled_points = relabeled_data[:, :3]
    relabeled_intensities = relabeled_data[:, 3]
    relabeled_labels = relabeled_data[:, 4].astype(np.int32)
    
    print(f"  Relabeled points: {len(relabeled_points)}")
    print(f"  Relabeled shape: {relabeled_data.shape}")
    
    # Compare
    print(f"\n{'='*80}")
    print("Comparison Results")
    print(f"{'='*80}")
    
    num_points_match = len(original_points) == len(relabeled_points)
    print(f"  Number of points match: {num_points_match}")
    if not num_points_match:
        print(f"    Original: {len(original_points)} points")
        print(f"    Relabeled: {len(relabeled_points)} points")
        print(f"    Difference: {abs(len(original_points) - len(relabeled_points))} points")
    
    # Check if coordinates match (they should be identical)
    coords_match = np.allclose(original_points, relabeled_points, rtol=1e-5, atol=1e-8)
    print(f"  Coordinates match: {coords_match}")
    
    # Check if intensities match (they should be identical)
    intensities_match = np.allclose(original_intensities, relabeled_intensities, rtol=1e-5, atol=1e-8)
    print(f"  Intensities match: {intensities_match}")
    
    # Check label differences
    labels_match = np.array_equal(original_labels, relabeled_labels)
    print(f"  Labels match: {labels_match}")
    if not labels_match:
        num_different = np.sum(original_labels != relabeled_labels)
        print(f"    Number of points with different labels: {num_different} ({100*num_different/len(original_labels):.2f}%)")
        
        # Show label distribution comparison
        print(f"\n  Original label distribution:")
        orig_unique, orig_counts = np.unique(original_labels, return_counts=True)
        for label_id, count in zip(orig_unique, orig_counts):
            label_name = "Unknown"
            for label in sem_kitti_labels:
                if label.id == label_id:
                    label_name = label.name
                    break
            print(f"    {label_name} (ID {label_id}): {count} points ({100*count/len(original_labels):.1f}%)")
        
        print(f"\n  Relabeled label distribution:")
        rel_unique, rel_counts = np.unique(relabeled_labels, return_counts=True)
        for label_id, count in zip(rel_unique, rel_counts):
            label_name = "Unknown"
            for label in sem_kitti_labels:
                if label.id == label_id:
                    label_name = label.name
                    break
            print(f"    {label_name} (ID {label_id}): {count} points ({100*count/len(relabeled_labels):.1f}%)")
    
    # Overall result
    all_match = num_points_match and coords_match and intensities_match
    print(f"\n  Overall match: {all_match}")
    
    return all_match


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert semantic map .npy file to PLY format with colors")

    # Dataset path
    parser.add_argument("--dataset_path", type=str, default="/media/donceykong/donceys_data_ssd/datasets/CU_MULTI/data",
                       help="Path to dataset root")

    # Environment name
    parser.add_argument("--environment", type=str, default="main_campus",
                       help="Environment name (default: main_campus)")

    # Input and output file postfixes
    parser.add_argument("--input_postfix", type=str, default="sem_map_confident_relabeled",
                       help="Input file postfix (default: sem_map_orig_utm_knn_smoothed)")

    # Original .npy file postfix to compare against (default: same as input)
    parser.add_argument("--original_postfix", type=str, default="sem_map_orig_utm",
                       help="Original postfix for .npy file to compare against (default: sem_map_orig_utm)")
    
    # Compare with original file
    parser.add_argument("--compare", action="store_true", default=False,
                       help="Compare with original file")
    
    args = parser.parse_args()
    
    dataset_path = os.path.join(args.dataset_path)
    environment = args.environment
    file_dir = os.path.join(dataset_path, environment, "additional")
    input_file = os.path.join(file_dir, f"{environment}_{args.input_postfix}.npy")
    output_file = os.path.join(file_dir, f"{environment}_{args.input_postfix}.ply")
    original_file = os.path.join(file_dir, f"{environment}_{args.original_postfix}.npy")
    
    # Compare files if requested and files exist
    if args.compare and original_file and input_file:
        compare_point_clouds(original_file, input_file)
        print("\n")
    
    # Load semantic map
    print(f"\n{'='*80}")
    print("Loading Semantic Map")
    print(f"{'='*80}")
    points, intensities, labels = load_semantic_map(input_file)
    
    # Get semantic colors
    print(f"\n{'='*80}")
    print("Generating Semantic Colors")
    print(f"{'='*80}")
    colors = get_semantic_colors(labels)
    
    # # Save top-down view
    # print(f"\n{'='*80}")
    # print("Creating Top-Down View")
    # print(f"{'='*80}")
    # topdown_output = Path(input_file).with_suffix('.png').name
    # save_topdown_view(points, colors, labels, topdown_output, downsample_factor=args.downsample)
    
    # Print label statistics
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\nLabel distribution:")
    for label_id, count in zip(unique_labels, counts):
        label_name = "Unknown"
        for label in sem_kitti_labels:
            if label.id == label_id:
                label_name = label.name
                break
        print(f"  {label_name} (ID {label_id}): {count} points ({100*count/len(labels):.1f}%)")
    
    # Save to PLY
    print(f"\n{'='*80}")
    print("Saving to PLY Format")
    print(f"{'='*80}")
    success = save_pointcloud_to_ply(points, colors, output_file)
    
    if success:
        # Validate PLY file
        print(f"\n{'='*80}")
        print("Validating PLY File")
        print(f"{'='*80}")
        is_valid = validate_ply_file(output_file, len(points))
        
        if is_valid:
            print(f"\n{'='*80}")
            print("SUCCESS!")
            print(f"{'='*80}")
            print(f"Output saved to: {output_file}")
            print(f"PLY file validated successfully")
        else:
            print(f"\n{'='*80}")
            print("WARNING!")
            print(f"{'='*80}")
            print(f"PLY file was created but validation failed!")
            print(f"File location: {output_file}")
    else:
        print(f"\n{'='*80}")
        print("ERROR!")
        print(f"{'='*80}")
        print(f"Failed to save PLY file!")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

