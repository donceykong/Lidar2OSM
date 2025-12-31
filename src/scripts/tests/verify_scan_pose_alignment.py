#!/usr/bin/env python3
"""
Minimal script to verify alignment between LiDAR scans and poses.
Checks that every lidar scan has a corresponding timestamp and pose.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_poses(poses_file):
    """Load UTM poses from CSV file."""
    import pandas as pd
    
    print(f"\nLoading poses from: {poses_file}")
    
    expected_data_lines = 0  # Initialize for scope
    
    # First, check how many lines are in the file
    try:
        with open(poses_file, 'r') as f:
            lines = f.readlines()
            total_lines = len(lines)
            
            # Count different line types
            comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
            empty_lines = sum(1 for line in lines if line.strip() == '')
            expected_data_lines = total_lines - comment_lines - empty_lines
            
            print(f"  Total lines in file: {total_lines}")
            print(f"  Comment lines (start with #): {comment_lines}")
            print(f"  Empty/blank lines: {empty_lines}")
            print(f"  Expected data rows: {expected_data_lines}")
            
            # Show first few lines for debugging
            print(f"\n  First 5 lines:")
            for i, line in enumerate(lines[:5]):
                line_preview = line.strip()[:80]  # Show first 80 chars
                print(f"    Line {i+1}: {line_preview}")
            
            # Show last few lines to check for empty lines at end
            print(f"\n  Last 3 lines:")
            for i in range(max(0, total_lines-3), total_lines):
                line = lines[i]
                if line.strip() == '':
                    display = "<EMPTY LINE>"
                else:
                    display = line.strip()[:80]
                has_newline = line.endswith('\n')
                newline_info = " [has \\n]" if has_newline else " [NO \\n - might cause pandas issue!]"
                print(f"    Line {i+1}: {display}{newline_info}")
    except Exception as e:
        print(f"  Error reading file: {e}")
        return {}, 0
    
    try:
        # Try reading with comment='#' and header=None to read ALL data rows
        df = pd.read_csv(poses_file, comment='#', header=None)
        num_rows = len(df)
        print(f"\n  CSV data rows after pandas read: {num_rows}")
        
        if num_rows == expected_data_lines:
            print(f"  ✓ Perfect match! All {num_rows} data rows read successfully")
        else:
            print(f"  ⚠ WARNING: pandas read {num_rows} rows but expected {expected_data_lines}")
            print(f"  Difference: {num_rows - expected_data_lines}")
    except Exception as e:
        print(f"  Error reading CSV: {e}")
        return {}, 0
    
    poses = {}
    num_valid = 0
    num_invalid = 0
    invalid_rows = []  # Track which rows are invalid
    
    for idx, row in df.iterrows():
        try:
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
                    num_invalid += 1
                    invalid_rows.append((idx, "Insufficient columns", row.tolist()[:3]))
                    continue
            
            pose = [float(x), float(y), float(z), float(qx), float(qy), float(qz), float(qw)]
            poses[float(timestamp)] = pose
            num_valid += 1
        except Exception as e:
            num_invalid += 1
            invalid_rows.append((idx, str(e), row.tolist()[:3] if hasattr(row, 'tolist') else str(row)[:50]))
            continue
    
    print(f"\n  Valid poses: {num_valid}")
    if num_invalid > 0:
        print(f"  Invalid/skipped rows: {num_invalid}")
        print(f"\n  Details of invalid rows:")
        for row_idx, error, preview in invalid_rows[:10]:  # Show first 10 invalid rows
            print(f"    Row {row_idx}: {error}")
            print(f"      Preview: {preview}")
        if len(invalid_rows) > 10:
            print(f"    ... and {len(invalid_rows) - 10} more invalid rows")
    
    # Return expected_data_lines (what should match LiDAR files) instead of num_rows
    return poses, expected_data_lines


def verify_robot_alignment(dataset_path, environment, robot):
    """
    Verify alignment between lidar scans and poses for a robot.
    
    Args:
        dataset_path: Path to dataset root
        environment: Environment name
        robot: Robot name
    
    Returns:
        Dictionary with alignment statistics
    """
    print(f"\n{'='*80}")
    print(f"Verifying {robot} in {environment}")
    print(f"{'='*80}")
    
    # Setup paths
    lidar_path = dataset_path / environment / robot / "lidar_bin/data"
    poses_file = dataset_path / environment / robot / f"{robot}_{environment}_gt_utm_poses.csv"
    
    # Check if paths exist
    if not lidar_path.exists():
        print(f"ERROR: LiDAR path not found: {lidar_path}")
        return None
    
    if not poses_file.exists():
        print(f"ERROR: Poses file not found: {poses_file}")
        return None
    
    # Load lidar files
    lidar_files = sorted(list(lidar_path.glob("*.bin")))
    print(f"\nLiDAR files: {len(lidar_files)}")
    if lidar_files:
        print(f"  First: {lidar_files[0].name}")
        print(f"  Last:  {lidar_files[-1].name}")
    
    # Load poses
    poses, num_csv_rows = load_poses(poses_file)
    # timestamps = sorted(list(poses.keys()))
    
    # # Analyze alignment
    # print(f"\n{'='*60}")
    # print("ALIGNMENT ANALYSIS")
    # print(f"{'='*60}")
    
    # num_lidar = len(lidar_files)
    # num_poses = len(timestamps)
    
    # stats = {
    #     'robot': robot,
    #     'num_lidar_files': num_lidar,
    #     'num_csv_rows': num_csv_rows,
    #     'num_pose_timestamps': num_poses,
    #     'difference': num_lidar - num_poses,
    #     'aligned': num_lidar == num_poses,
    #     'csv_aligned': num_lidar == num_csv_rows
    # }
    
    # # Check CSV rows vs LiDAR files
    # print(f"\n1. CSV Timestamps vs LiDAR Files:")
    # if num_csv_rows == num_lidar:
    #     print(f"   ✓ CSV rows ({num_csv_rows}) == LiDAR files ({num_lidar})")
    # else:
    #     print(f"   ⚠ CSV rows ({num_csv_rows}) != LiDAR files ({num_lidar})")
    #     print(f"     Difference: {num_csv_rows - num_lidar:+d}")
    
    # # Check valid poses vs LiDAR files
    # print(f"\n2. Valid Poses vs LiDAR Files:")
    # if num_lidar == num_poses:
    #     print(f"   ✓ ALIGNED: {num_lidar} lidar files == {num_poses} valid poses")
    #     stats['status'] = 'ALIGNED'
    # elif num_lidar > num_poses:
    #     print(f"   ⚠ MISMATCH: {num_lidar} lidar files > {num_poses} valid poses")
    #     print(f"     → {num_lidar - num_poses} lidar scan(s) WITHOUT corresponding valid pose")
    #     stats['status'] = 'EXTRA_LIDAR'
    # else:
    #     print(f"   ⚠ MISMATCH: {num_lidar} lidar files < {num_poses} valid poses")
    #     print(f"     → {num_poses - num_lidar} valid pose(s) WITHOUT corresponding lidar scan")
    #     stats['status'] = 'EXTRA_POSES'
    
    # # Check if any poses were invalid
    # if num_csv_rows > num_poses:
    #     print(f"\n3. Data Quality:")
    #     print(f"   ⚠ {num_csv_rows - num_poses} timestamp(s) in CSV had INVALID pose data")
    #     print(f"     (These rows exist but couldn't be parsed)")
    #     stats['num_invalid_poses'] = num_csv_rows - num_poses
    
    # # Check label files if they exist
    # labels_path = dataset_path / environment / robot / "lidar_labels"
    # if labels_path.exists():
    #     label_files = sorted(list(labels_path.glob("*.bin")))
    #     print(f"\nOriginal label files: {len(label_files)}")
    #     stats['num_label_files'] = len(label_files)
        
    #     if len(label_files) == num_lidar:
    #         print(f"✓ Labels aligned with lidar files")
    #     else:
    #         print(f"⚠ Labels ({len(label_files)}) != Lidar files ({num_lidar})")
    
    # # Check refined label files if they exist
    # refined_labels_path = dataset_path / environment / robot / f"{robot}_{environment}_refined_lidar_labels"
    # if refined_labels_path.exists():
    #     refined_label_files = sorted(list(refined_labels_path.glob("*.bin")))
    #     print(f"\nRefined label files: {len(refined_label_files)}")
    #     stats['num_refined_label_files'] = len(refined_label_files)
        
    #     expected_count = min(num_lidar, num_poses)
    #     if len(refined_label_files) == expected_count:
    #         print(f"✓ Refined labels match expected count: {len(refined_label_files)}")
    #     else:
    #         print(f"⚠ Refined labels ({len(refined_label_files)}) != Expected ({expected_count})")
    
    # return stats


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify LiDAR scan and pose alignment")
    parser.add_argument("--dataset_path", type=str,
                       default="/media/donceykong/donceys_data_ssd/datasets/CU_MULTI/data",
                       help="Path to dataset root")
    parser.add_argument("--environment", type=str, default="main_campus",
                       help="Environment name")
    parser.add_argument("--robot", type=str, default=None,
                       help="Specific robot to check (default: check all robots)")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    
    # Determine which robots to check
    if args.robot:
        robots = [args.robot]
    else:
        # Check all robots in the environment
        env_path = dataset_path / args.environment
        if not env_path.exists():
            print(f"Error: Environment path not found: {env_path}")
            return
        robots = [d.name for d in env_path.iterdir() if d.is_dir() and d.name.startswith('robot')]
        robots = sorted(robots)
    
    print(f"\n{'='*80}")
    print(f"CHECKING ALIGNMENT FOR: {', '.join(robots)}")
    print(f"{'='*80}")
    
    # Verify each robot
    all_stats = []
    for robot in robots:
        stats = verify_robot_alignment(dataset_path, args.environment, robot)
        if stats:
            all_stats.append(stats)
    
    # # Print summary
    # print(f"\n\n{'='*80}")
    # print("SUMMARY")
    # print(f"{'='*80}")
    # print(f"Format: Robot | LiDAR Files | CSV Rows | Valid Poses | Diff | Status")
    # print(f"{'-'*80}")
    
    # for stats in all_stats:
    #     status_symbol = "✓" if stats['aligned'] else "⚠"
    #     csv_symbol = "✓" if stats['csv_aligned'] else "⚠"
    #     print(f"{status_symbol} {stats['robot']:10s} | "
    #           f"Lidar: {stats['num_lidar_files']:5d} | "
    #           f"CSV: {stats['num_csv_rows']:5d} {csv_symbol} | "
    #           f"Valid: {stats['num_pose_timestamps']:5d} | "
    #           f"Diff: {stats['difference']:+3d} | "
    #           f"{stats['status']}")
    
    # # Overall verdict
    # all_aligned = all(s['aligned'] for s in all_stats)
    # print(f"\n{'='*80}")
    # if all_aligned:
    #     print("✓ ALL ROBOTS ALIGNED - Ready for relabeling!")
    # else:
    #     print("⚠ ALIGNMENT ISSUES DETECTED")
    #     print("\nRecommendation:")
    #     print("  - The relabeling script will process min(lidar_files, poses) scans")
    #     print("  - Extra lidar scans without poses will be skipped")
    #     print("  - This is usually fine if the mismatch is just 1-2 files")
    # print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

