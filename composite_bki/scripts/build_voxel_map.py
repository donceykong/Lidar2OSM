#!/usr/bin/env python3
"""
Build a voxel map from multiple LiDAR scans with labels.

This script processes a directory of point cloud scans and aggregates them into 
a unified voxel grid with semantic information. For MCD dataset, applies proper
calibration transforms (LiDAR->Body->World) using poses.
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
from tqdm import tqdm
import pickle


# Body to LiDAR transformation matrix for MCD dataset
# This calibration accounts for the sensor mounting offset and rotation
BODY_TO_LIDAR_TF = np.array(
    [
        [0.9999135040741837, -0.011166365511073898, -0.006949579221822984, -0.04894521120494695],
        [-0.011356389542502144, -0.9995453006865824, -0.02793249526856565, -0.03126929060348084],
        [-0.006634514801117132, 0.02800900135032654, -0.999585653686922, -0.01755515794222565],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


class VoxelMap:
    """
    A voxel-based 3D map that aggregates multiple point clouds with semantic labels.
    
    Each voxel stores:
    - Point count
    - Mean position
    - Label histogram
    - Mean intensity
    """
    
    def __init__(self, voxel_size=0.2, origin=None):
        """
        Initialize voxel map.
        
        Args:
            voxel_size: Size of each voxel in meters
            origin: Optional origin point [x, y, z]. If None, computed from first scan
        """
        self.voxel_size = voxel_size
        self.origin = origin if origin is not None else np.array([0.0, 0.0, 0.0])
        
        # Dictionary mapping voxel indices (i,j,k) to voxel data
        self.voxels = defaultdict(lambda: {
            'points': [],
            'intensities': [],
            'labels': []
        })
        
        self.bounds_min = None
        self.bounds_max = None
        
    def world_to_voxel(self, points):
        """Convert world coordinates to voxel indices."""
        shifted = points - self.origin
        indices = np.floor(shifted / self.voxel_size).astype(np.int32)
        return indices
    
    def voxel_to_world(self, indices):
        """Convert voxel indices to world coordinates (voxel centers)."""
        return (indices + 0.5) * self.voxel_size + self.origin
    
    def add_points(self, points, intensities=None, labels=None):
        """
        Add points to the voxel map.
        
        Args:
            points: (N, 3) array of xyz coordinates
            intensities: (N,) array of intensity values (optional)
            labels: (N,) array of semantic labels (optional)
        """
        if len(points) == 0:
            return
        
        # Update bounds
        if self.bounds_min is None:
            self.bounds_min = points.min(axis=0)
            self.bounds_max = points.max(axis=0)
        else:
            self.bounds_min = np.minimum(self.bounds_min, points.min(axis=0))
            self.bounds_max = np.maximum(self.bounds_max, points.max(axis=0))
        
        # Convert to voxel indices
        voxel_indices = self.world_to_voxel(points)
        
        # Add points to corresponding voxels
        for i in range(len(points)):
            voxel_key = tuple(voxel_indices[i])
            
            self.voxels[voxel_key]['points'].append(points[i])
            
            if intensities is not None:
                self.voxels[voxel_key]['intensities'].append(intensities[i])
            
            if labels is not None:
                self.voxels[voxel_key]['labels'].append(labels[i])
    
    def finalize(self):
        """
        Finalize the voxel map by computing statistics for each voxel.
        Converts lists to arrays and computes means.
        """
        print("Finalizing voxel map...")
        
        finalized_voxels = {}
        
        for voxel_key, data in tqdm(self.voxels.items(), desc="Computing voxel statistics"):
            points = np.array(data['points'])
            
            voxel_data = {
                'count': len(points),
                'mean_position': points.mean(axis=0),
            }
            
            if data['intensities']:
                voxel_data['mean_intensity'] = np.mean(data['intensities'])
                voxel_data['std_intensity'] = np.std(data['intensities'])
            
            if data['labels']:
                labels = np.array(data['labels'], dtype=np.int32)
                # Compute label histogram
                unique_labels, counts = np.unique(labels, return_counts=True)
                voxel_data['label_histogram'] = dict(zip(unique_labels.tolist(), counts.tolist()))
                # Majority vote label
                voxel_data['majority_label'] = unique_labels[np.argmax(counts)]
                voxel_data['label_confidence'] = counts.max() / len(labels)
            
            finalized_voxels[voxel_key] = voxel_data
        
        self.voxels = finalized_voxels
        print(f"Finalized {len(self.voxels)} voxels")
    
    def get_occupied_voxel_centers(self, min_points=1):
        """
        Get centers of all occupied voxels.
        
        Args:
            min_points: Minimum number of points required for a voxel to be considered occupied
        
        Returns:
            (N, 3) array of voxel centers
        """
        centers = []
        for voxel_key, data in self.voxels.items():
            if data['count'] >= min_points:
                centers.append(data['mean_position'])
        return np.array(centers)
    
    def get_voxel_labels(self, min_points=1):
        """
        Get majority labels for all occupied voxels.
        
        Returns:
            Array of labels corresponding to get_occupied_voxel_centers()
        """
        labels = []
        for voxel_key, data in self.voxels.items():
            if data['count'] >= min_points:
                labels.append(data.get('majority_label', 0))
        return np.array(labels)
    
    def get_voxel_intensities(self, min_points=1):
        """
        Get mean intensities for all occupied voxels.
        
        Returns:
            Array of intensities corresponding to get_occupied_voxel_centers()
        """
        intensities = []
        for voxel_key, data in self.voxels.items():
            if data['count'] >= min_points:
                intensities.append(data.get('mean_intensity', 0.0))
        return np.array(intensities)
    
    def export_to_bin(self, output_bin, output_labels, min_points=1):
        """
        Export voxel map to point cloud .bin and .label files.
        
        Args:
            output_bin: Path to output .bin file (x,y,z,intensity format)
            output_labels: Path to output .label file (uint32 format)
            min_points: Minimum points per voxel to export
        """
        print(f"Exporting voxel map to point cloud format...")
        
        # Collect data for all valid voxels
        points = []
        labels = []
        
        for voxel_key, data in self.voxels.items():
            if data['count'] >= min_points:
                # Get voxel center position
                position = data['mean_position']
                
                # Get mean intensity (default to 0 if not available)
                intensity = data.get('mean_intensity', 0.0)
                
                # Get majority label (default to 0 if not available)
                label = data.get('majority_label', 0)
                
                # Create point with [x, y, z, intensity]
                point = np.array([position[0], position[1], position[2], intensity])
                points.append(point)
                labels.append(label)
        
        # Convert to arrays
        points = np.array(points, dtype=np.float32)
        labels = np.array(labels, dtype=np.uint32)
        
        print(f"Exporting {len(points)} voxel centers")
        
        # Save point cloud (.bin format: x,y,z,intensity)
        points.tofile(output_bin)
        print(f"Saved point cloud to: {output_bin}")
        
        # Save labels (.label format: uint32, lower 16 bits = semantic label)
        labels.tofile(output_labels)
        print(f"Saved labels to: {output_labels}")
        
        return len(points)
    
    def save(self, filepath):
        """Save voxel map to file."""
        save_data = {
            'voxel_size': self.voxel_size,
            'origin': self.origin,
            'voxels': dict(self.voxels),  # Convert defaultdict to dict
            'bounds_min': self.bounds_min,
            'bounds_max': self.bounds_max
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Saved voxel map to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load voxel map from file."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        voxel_map = cls(
            voxel_size=save_data['voxel_size'],
            origin=save_data['origin']
        )
        voxel_map.voxels = save_data['voxels']
        voxel_map.bounds_min = save_data['bounds_min']
        voxel_map.bounds_max = save_data['bounds_max']
        
        return voxel_map
    
    def get_statistics(self):
        """Get statistics about the voxel map."""
        if not self.voxels:
            return {}
        
        total_voxels = len(self.voxels)
        total_points = sum(v['count'] for v in self.voxels.values())
        
        # Count voxels with labels
        voxels_with_labels = sum(1 for v in self.voxels.values() if 'majority_label' in v)
        
        stats = {
            'total_voxels': total_voxels,
            'total_points': total_points,
            'voxels_with_labels': voxels_with_labels,
            'avg_points_per_voxel': total_points / total_voxels if total_voxels > 0 else 0,
            'voxel_size': self.voxel_size,
            'bounds_min': self.bounds_min.tolist() if self.bounds_min is not None else None,
            'bounds_max': self.bounds_max.tolist() if self.bounds_max is not None else None,
        }
        
        # Compute volume
        if self.bounds_min is not None and self.bounds_max is not None:
            size = self.bounds_max - self.bounds_min
            stats['map_size_m'] = size.tolist()
            stats['map_volume_m3'] = float(np.prod(size))
        
        return stats


def load_point_cloud(bin_file):
    """Load point cloud from binary file (x, y, z, intensity)."""
    scan = np.fromfile(bin_file, dtype=np.float32).reshape((-1, 4))
    points = scan[:, :3]
    intensity = scan[:, 3]
    return points, intensity


def load_labels(label_file):
    """Load semantic labels from binary file."""
    labels_raw = np.fromfile(label_file, dtype=np.uint32)
    labels = labels_raw & 0xFFFF  # Lower 16 bits = semantic label
    return labels


def load_poses(pose_file):
    """
    Load poses from CSV file.
    
    Returns:
        Dictionary mapping frame numbers to poses [x,y,z,qx,qy,qz,qw]
    """
    print(f"Loading poses from {pose_file}...")
    df = pd.read_csv(pose_file)
    
    poses = {}
    for _, row in df.iterrows():
        frame_num = int(row['num'])
        x, y, z = float(row['x']), float(row['y']), float(row['z'])
        qx, qy, qz, qw = float(row['qx']), float(row['qy']), float(row['qz']), float(row['qw'])
        poses[frame_num] = np.array([x, y, z, qx, qy, qz, qw])
    
    print(f"Loaded {len(poses)} poses")
    return poses


def transform_points_to_world(points, pose):
    """
    Transform points from sensor frame to world frame for MCD dataset.
    
    MCD requires two transformations:
    1. LiDAR frame -> Body frame (using calibration matrix)
    2. Body frame -> World frame (using pose from pose_inW.csv)
    
    Args:
        points: (N, 3) array of points in LiDAR frame
        pose: (7,) array [x, y, z, qx, qy, qz, qw] - body pose in world frame
    
    Returns:
        (N, 3) array of points in world frame
    """
    position = pose[:3]
    quat = pose[3:7]
    
    # Create body-to-world transformation
    body_rotation_matrix = R.from_quat(quat).as_matrix()
    body_to_world = np.eye(4, dtype=np.float64)
    body_to_world[:3, :3] = body_rotation_matrix
    body_to_world[:3, 3] = np.asarray(position, dtype=np.float64)
    
    # LiDAR-to-body transformation (inverse of body-to-lidar calibration)
    lidar_to_body = np.linalg.inv(BODY_TO_LIDAR_TF)
    
    # Complete transformation: LiDAR -> Body -> World
    transform_matrix = body_to_world @ lidar_to_body
    
    # Transform points
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    world_points = (transform_matrix @ points_homogeneous.T).T
    return world_points[:, :3].astype(np.float32)


def get_frame_number_from_filename(filename):
    """Extract frame number from filename."""
    stem = Path(filename).stem
    try:
        return int(stem)
    except ValueError:
        raise ValueError(f"Could not extract frame number from: {filename}")


def build_voxel_map(scan_dir, pose_file, label_dir=None, voxel_size=0.2, 
                   max_scans=None, skip=1, output_file=None, export_bin=None, 
                   export_labels=None, min_points=1):
    """
    Build a voxel map from multiple scans with MCD dataset transformations.
    
    Args:
        scan_dir: Directory containing .bin scan files
        pose_file: Path to pose CSV file (num,t,x,y,z,qx,qy,qz,qw)
        label_dir: Optional directory containing label files (supports both .label and .bin extensions)
        voxel_size: Voxel size in meters
        max_scans: Maximum number of scans to process (None = all)
        skip: Process every Nth scan (1 = all scans)
        output_file: Output file path for voxel map (.pkl)
        export_bin: Optional path to export voxel centers as .bin point cloud
        export_labels: Optional path to export majority labels as .label file
        min_points: Minimum points per voxel for export (default: 1)
    
    Note:
        Label files can have either .label or .bin extensions. The script will
        automatically check for both and use whichever exists.
    """
    scan_dir = Path(scan_dir)
    label_dir = Path(label_dir) if label_dir else None
    
    # Find all scan files
    scan_files = sorted(scan_dir.glob("*.bin"))
    print(f"Found {len(scan_files)} scan files in {scan_dir}")
    
    if not scan_files:
        print("ERROR: No scan files found!")
        return None
    
    # Apply skip and max_scans
    scan_files = scan_files[::skip]
    if max_scans:
        scan_files = scan_files[:max_scans]
    print(f"Processing {len(scan_files)} scans (skip={skip}, max={max_scans})")
    
    # Load poses
    poses = load_poses(pose_file)
    
    # Print label directory info
    if label_dir:
        print(f"Label directory: {label_dir}")
        print(f"  Supported extensions: .label, .bin")
    else:
        print("No label directory specified - building voxel map without labels")
    
    # Initialize voxel map
    voxel_map = VoxelMap(voxel_size=voxel_size)
    
    # Process each scan
    processed_count = 0
    skipped_count = 0
    
    for scan_file in tqdm(scan_files, desc="Processing scans"):
        try:
            # Get frame number
            frame_num = get_frame_number_from_filename(scan_file.name)
            
            # Check if pose exists
            if frame_num not in poses:
                print(f"WARNING: No pose for frame {frame_num}, skipping")
                skipped_count += 1
                continue
            
            # Load scan
            points, intensities = load_point_cloud(str(scan_file))
            
            # Load labels if available (supports both .label and .bin extensions)
            labels = None
            if label_dir:
                # Try various extensions
                label_file = None
                label_extensions = ['.label', '.bin', '_prediction.label', '_prediction.bin']
                
                for ext in label_extensions:
                    candidate = label_dir / f"{scan_file.stem}{ext}"
                    if candidate.exists():
                        label_file = candidate
                        break
                
                if label_file:
                    try:
                        labels = load_labels(str(label_file))
                        # Ensure same length
                        if len(labels) != len(points):
                            print(f"WARNING: Label count mismatch for frame {frame_num}, truncating")
                            min_len = min(len(labels), len(points))
                            labels = labels[:min_len]
                            points = points[:min_len]
                            intensities = intensities[:min_len]
                    except Exception as e:
                        print(f"WARNING: Failed to load labels for frame {frame_num}: {e}")
                        labels = None
                else:
                    # Only warn on first scan to avoid spam
                    if processed_count == 0:
                        print(f"INFO: No label file found for frame {frame_num}")
                        print(f"      Checked extensions: {', '.join(label_extensions)}")
            
            # Transform to world frame using MCD calibration
            pose = poses[frame_num]
            world_points = transform_points_to_world(points, pose)
            
            # Add to voxel map
            voxel_map.add_points(world_points, intensities, labels)
            
            processed_count += 1
            
        except Exception as e:
            print(f"ERROR processing {scan_file.name}: {e}")
            import traceback
            traceback.print_exc()
            skipped_count += 1
            continue
    
    print(f"\nProcessing complete:")
    print(f"  Processed: {processed_count} scans")
    print(f"  Skipped: {skipped_count} scans")
    
    # Finalize voxel map
    voxel_map.finalize()
    
    # Print statistics
    stats = voxel_map.get_statistics()
    print("\nVoxel Map Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Debug: Print label distribution in the voxel map
    if stats.get('voxels_with_labels', 0) > 0:
        print("\nLabel Distribution in Voxel Map:")
        label_counts = {}
        for voxel_data in voxel_map.voxels.values():
            if 'majority_label' in voxel_data:
                label = voxel_data['majority_label']
                label_counts[label] = label_counts.get(label, 0) + 1
        
        # Show top 10 labels
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        for label, count in sorted_labels[:10]:
            percentage = (count / stats['total_voxels']) * 100
            print(f"  Label {label:3d}: {count:6d} voxels ({percentage:5.2f}%)")
        if len(sorted_labels) > 10:
            print(f"  ... and {len(sorted_labels) - 10} more labels")
    
    # Save pickle format if output file specified
    if output_file:
        voxel_map.save(output_file)
    
    # Export to bin/label format if specified
    if export_bin and export_labels:
        voxel_map.export_to_bin(export_bin, export_labels, min_points=min_points)
    elif export_bin or export_labels:
        print("WARNING: Both --export_bin and --export_labels must be specified for export")
    
    return voxel_map


def main():
    parser = argparse.ArgumentParser(
        description="Build a voxel map from multiple LiDAR scans with poses and labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to .bin and .label format (voxel centers and majority labels) - MCD dataset
  python build_voxel_map.py --scan_dir data/scans --pose pose_inW.csv --label_dir data/labels \
      --export_bin voxels.bin --export_labels voxels.label
  
  # Also save pickle format for analysis
  python build_voxel_map.py --scan_dir data/scans --pose pose_inW.csv --label_dir data/labels \
      --export_bin voxels.bin --export_labels voxels.label --output voxel_map.pkl
  
  # With custom voxel size and filtering
  python build_voxel_map.py --scan_dir data/scans --pose pose_inW.csv --voxel_size 0.5 --min_points 5 \
      --export_bin voxels.bin --export_labels voxels.label
  
  # Process subset with skip
  python build_voxel_map.py --scan_dir data/scans --pose pose_inW.csv --skip 2 --max_scans 100 \
      --export_bin voxels.bin --export_labels voxels.label

Notes:
  - For MCD dataset, applies proper calibration (LiDAR->Body->World) using pose_inW.csv
  - Label files can have .label or .bin extensions (script checks for both automatically)
        """
    )
    
    parser.add_argument('--scan_dir', type=str, required=True,
                        help='Directory containing .bin point cloud files')
    parser.add_argument('--pose', type=str, required=True,
                        help='Path to pose CSV file (num,t,x,y,z,qx,qy,qz,qw)')
    parser.add_argument('--label_dir', type=str, default=None,
                        help='Optional directory containing label files (supports .label or .bin extensions)')
    parser.add_argument('--voxel_size', type=float, default=0.2,
                        help='Voxel size in meters (default: 0.2)')
    parser.add_argument('--max_scans', type=int, default=None,
                        help='Maximum number of scans to process (default: all)')
    parser.add_argument('--skip', type=int, default=1,
                        help='Process every Nth scan (default: 1 = all scans)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path for voxel map (.pkl, optional)')
    parser.add_argument('--export_bin', type=str, default=None,
                        help='Export voxel centers as point cloud .bin file')
    parser.add_argument('--export_labels', type=str, default=None,
                        help='Export majority labels as .label file')
    parser.add_argument('--min_points', type=int, default=1,
                        help='Minimum points per voxel for export (default: 1)')
    
    args = parser.parse_args()
    
    # At least one output must be specified
    if not args.output and not (args.export_bin and args.export_labels):
        parser.error("At least one output format must be specified: --output or (--export_bin and --export_labels)")
    
    # Validate inputs
    if not Path(args.scan_dir).exists():
        print(f"ERROR: Scan directory not found: {args.scan_dir}")
        return 1
    
    if not Path(args.pose).exists():
        print(f"ERROR: Pose file not found: {args.pose}")
        return 1
    
    if args.label_dir and not Path(args.label_dir).exists():
        print(f"ERROR: Label directory not found: {args.label_dir}")
        return 1
    
    # Build voxel map
    build_voxel_map(
        scan_dir=args.scan_dir,
        pose_file=args.pose,
        label_dir=args.label_dir,
        voxel_size=args.voxel_size,
        max_scans=args.max_scans,
        skip=args.skip,
        output_file=args.output,
        export_bin=args.export_bin,
        export_labels=args.export_labels,
        min_points=args.min_points
    )
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
