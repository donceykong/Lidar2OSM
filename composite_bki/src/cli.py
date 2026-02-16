#!/usr/bin/env python3
"""
Command-line interface for Composite BKI (Unified).

Provides a user-friendly CLI for running semantic segmentation refinement
on LiDAR point clouds using OpenStreetMap priors.
This version uses the Continuous BKI backend.
"""

import argparse
import sys
import os
import numpy as np
import time
from pathlib import Path

# Try to import the extension
try:
    import composite_bki_cpp
except ImportError:
    # If running from source/dev, try to add current dir to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        import composite_bki_cpp
    except ImportError:
        pass # Will be handled in main

def load_scan(path):
    """Load LiDAR scan from .bin file (x, y, z, intensity)."""
    scan = np.fromfile(path, dtype=np.float32)
    return scan.reshape((-1, 4))

def load_label(path):
    """Load labels from .label or .bin file (uint32)."""
    return np.fromfile(path, dtype=np.uint32).reshape(-1)

def save_label(path, labels):
    """Save labels to file."""
    labels = labels.astype(np.uint32)
    labels.tofile(path)

def load_poses(path):
    """Load poses from KITTI format file (N x 12) or CSV format (index, timestamp, x, y, z, qx, qy, qz, qw)."""
    poses = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # Try parsing as standard KITTI format (space separated)
            try:
                values = [float(v) for v in line.split()]
                if len(values) == 12:
                    pose = np.array(values).reshape(3, 4)
                    poses.append(pose)
                    continue
            except ValueError:
                pass
            
            # Try parsing as CSV format
            try:
                parts = line.split(',')
                # Expect at least x, y, z at indices 2, 3, 4
                if len(parts) >= 5:
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                    
                    # Create identity rotation with extracted translation
                    pose = np.eye(3, 4)
                    pose[0, 3] = x
                    pose[1, 3] = y
                    pose[2, 3] = z
                    poses.append(pose)
                    continue
            except ValueError:
                pass
                
            # If both failed
            if i == 0:
                print(f"Skipping header/invalid line 0 in poses file: {line}")
            else:
                print(f"Warning: Skipping invalid line {i+1} in poses file: {line}")
                
    return poses

def compute_metrics(pred, gt, num_classes=None):
    """Compute basic accuracy and IoU metrics."""
    if len(pred) != len(gt):
        print(f"Warning: Prediction and GT length mismatch ({len(pred)} vs {len(gt)})")
        return {}
        
    mask = (gt != 0) # Ignore 0 (unlabeled)
    if not np.any(mask):
        return {"accuracy": 0.0, "miou": 0.0}
        
    pred_masked = pred[mask]
    gt_masked = gt[mask]
    
    accuracy = np.mean(pred_masked == gt_masked)
    
    # Per-class IoU
    classes = np.unique(np.concatenate((pred_masked, gt_masked)))
    ious = {}
    for c in classes:
        intersection = np.sum((pred_masked == c) & (gt_masked == c))
        union = np.sum((pred_masked == c) | (gt_masked == c))
        if union > 0:
            ious[int(c)] = intersection / union
            
    miou = np.mean(list(ious.values())) if ious else 0.0
    
    return {
        "accuracy": accuracy,
        "miou": miou,
        "class_ious": ious
    }

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Composite BKI - Semantic-Spatial Bayesian Kernel Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--scan", type=str, required=True, help="Path to .bin lidar point cloud file")
    parser.add_argument("--label", type=str, required=True, help="Path to semantic labels")
    parser.add_argument("--osm", type=str, required=True, help="Path to OSM geometries file (.bin or .osm XML)")
    parser.add_argument("--output", type=str, required=True, help="Path to save refined labels")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    
    # Optional arguments
    parser.add_argument("--ground-truth", type=str, default=None, help="Path to ground truth labels for evaluation")
    parser.add_argument("--metrics-output", type=str, default=None, help="Path to save per-class metrics CSV")
    
    # BKI parameters
    parser.add_argument("--resolution", type=float, default=0.1, help="Voxel resolution (default: 0.1)")
    parser.add_argument("--l-scale", type=float, default=0.5, help="Spatial kernel scale parameter (default: 0.5)")
    parser.add_argument("--sigma-0", type=float, default=1.0, help="Spatial kernel amplitude (default: 1.0)")
    parser.add_argument("--prior-delta", type=float, default=5.0, help="OSM prior distance scaling (default: 5.0)")
    parser.add_argument("--height-sigma", type=float, default=0.3, help="Sigma for height-based gating (default: 0.3)")
    
    parser.add_argument("--disable-spatial", action="store_true", help="Disable spatial kernel")
    parser.add_argument("--disable-semantic", action="store_true", help="Disable semantic kernel")
    
    parser.add_argument("--num-threads", type=int, default=-1, help="Number of OpenMP threads (-1 = auto)")
    
    # Mode selection
    parser.add_argument("--continuous", action="store_true", help="Enable continuous mapping mode (stateful)")
    parser.add_argument("--save-map", type=str, help="Save BKI map state to file (for continuous mode)")
    parser.add_argument("--load-map", type=str, help="Load BKI map state from file (for continuous mode)")

    args = parser.parse_args()
    
    # Check imports and ensure module is available locally
    try:
        import composite_bki_cpp
    except ImportError:
        print("Error: composite_bki_cpp extension not found. Please install the package.", file=sys.stderr)
        return 1

    # Validate inputs
    if not os.path.exists(args.scan):
        print(f"Error: Scan file not found: {args.scan}", file=sys.stderr)
        return 1
    if not os.path.exists(args.label):
        print(f"Error: Label file not found: {args.label}", file=sys.stderr)
        return 1
    if not os.path.exists(args.osm):
        print(f"Error: OSM file not found: {args.osm}", file=sys.stderr)
        return 1
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1

    # Initialize BKI
    print("Initializing BKI (Continuous / GridHash)...")
    try:
        bki = composite_bki_cpp.PyContinuousBKI(
            osm_path=args.osm,
            config_path=args.config,
            resolution=args.resolution,
            l_scale=args.l_scale,
            sigma_0=args.sigma_0,
            prior_delta=args.prior_delta,
            height_sigma=args.height_sigma,
            use_semantic_kernel=not args.disable_semantic,
            use_spatial_kernel=not args.disable_spatial,
            num_threads=args.num_threads
        )
    except Exception as e:
        print(f"Error initializing BKI: {e}", file=sys.stderr)
        return 1

    # Load/Save map state if requested
    if args.load_map and os.path.exists(args.load_map):
        print(f"Loading map state from {args.load_map}...")
        bki.load(args.load_map)

    # Load data
    print(f"Loading data from {args.scan}...")
    points = load_scan(args.scan)
    labels = load_label(args.label)

    # Ensure points match labels
    if len(points) != len(labels):
        print(f"Error: Point cloud size ({len(points)}) does not match label size ({len(labels)})", file=sys.stderr)
        return 1
        
    # Run Update
    print(f"Running BKI update on {len(points)} points...")
    start_time = time.time()
    bki.update(labels, points[:, :3])
    update_time = time.time() - start_time
    print(f"Update completed in {update_time:.3f}s")
    
    # Run Inference
    print("Running inference...")
    start_time = time.time()
    refined_labels = bki.infer(points[:, :3])
    infer_time = time.time() - start_time
    print(f"Inference completed in {infer_time:.3f}s")
    
    # Save Output
    print(f"Saving output to {args.output}...")
    save_label(args.output, refined_labels)
    
    # Optional: Save map
    if args.save_map:
        print(f"Saving map state to {args.save_map}...")
        bki.save(args.save_map)

    # Compute Metrics
    if args.ground_truth and os.path.exists(args.ground_truth):
        print("Computing metrics...")
        gt_labels = load_label(args.ground_truth)
        metrics = compute_metrics(refined_labels, gt_labels)
        
        print("=" * 40)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"mIoU:     {metrics['miou']:.4f}")
        print("=" * 40)
        
        if args.metrics_output:
            with open(args.metrics_output, "w") as f:
                f.write("class,iou\n")
                for c, iou in metrics.get('class_ious', {}).items():
                    f.write(f"{c},{iou:.4f}\n")
            print(f"Metrics saved to {args.metrics_output}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
