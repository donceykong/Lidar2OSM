#!/usr/bin/env python3
"""
Command-line interface for Composite BKI.

Provides a user-friendly CLI for running semantic segmentation refinement
on LiDAR point clouds using OpenStreetMap priors.
"""

import argparse
import sys
import os
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Composite BKI - Semantic-Spatial Bayesian Kernel Inference for LiDAR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with custom config
  composite-bki --scan scan.bin --label labels.label --osm map.bin \\
                --config configs/mcd_config.yaml --output refined.label

  # With ground truth evaluation
  composite-bki --scan scan.bin --label labels.label --osm map.bin \\
                --config configs/kitti_config.yaml --ground-truth gt.label --output refined.label

  # Custom parameters and threading
  composite-bki --scan scan.bin --label labels.label --osm map.bin \\
                --config configs/mcd_config.yaml --l-scale 5.0 --num-threads 8 --output refined.label

For more information: https://github.com/yourrepo/composite-bki
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--scan", 
        type=str, 
        required=True,
        help="Path to .bin lidar point cloud file (Nx4 float32: x,y,z,intensity)"
    )
    parser.add_argument(
        "--label", 
        type=str, 
        required=True,
        help="Path to semantic labels (.label or .bin, uint32 format)"
    )
    parser.add_argument(
        "--osm", 
        type=str, 
        required=True,
        help="Path to .bin OSM geometries file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Path to save refined labels (.label or .bin)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Path to ground truth labels for evaluation (optional)"
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default=None,
        help="Path to save per-class metrics CSV (optional)"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    
    # BKI parameters
    parser.add_argument(
        "--l-scale",
        type=float,
        default=3.0,
        help="Spatial kernel scale parameter (default: 3.0)"
    )
    parser.add_argument(
        "--sigma-0",
        type=float,
        default=1.0,
        help="Spatial kernel amplitude (default: 1.0)"
    )
    parser.add_argument(
        "--prior-delta",
        type=float,
        default=5.0,
        help="OSM prior distance scaling (default: 5.0)"
    )
    parser.add_argument(
        "--alpha-0",
        type=float,
        default=0.01,
        help="Dirichlet prior weight (default: 0.01)"
    )
    
    parser.add_argument(
        "--height-sigma",
        type=float,
        default=0.3,
        help="Sigma for height-based gating of ground classes (default: 0.3)"
    )
    
    parser.add_argument(
        "--disable-spatial",
        action="store_true",
        help="Disable spatial kernel (distance-based)"
    )
    
    parser.add_argument(
        "--disable-semantic",
        action="store_true",
        help="Disable semantic kernel (OSM priors)"
    )
    
    # Performance
    parser.add_argument(
        "--num-threads",
        type=int,
        default=-1,
        help="Number of OpenMP threads (-1 = auto-detect all cores, default: -1)"
    )
    
    # Utility
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information and exit"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show package information and exit"
    )
    
    args = parser.parse_args()
    
    # Handle version/info flags
    if args.version:
        from . import __version__
        print(f"Composite BKI C++ v{__version__}")
        return 0
    
    if args.info:
        from . import print_info
        print_info()
        return 0
    
    # Validate input files exist
    for filepath, name in [
        (args.scan, "Scan file"),
        (args.label, "Label file"),
        (args.osm, "OSM file")
    ]:
        if not os.path.exists(filepath):
            print(f"Error: {name} not found: {filepath}", file=sys.stderr)
            return 1
    
    if args.ground_truth and not os.path.exists(args.ground_truth):
        print(f"Warning: Ground truth file not found: {args.ground_truth}", file=sys.stderr)
        args.ground_truth = None
    
    # Import the module
    try:
        import composite_bki_cpp
        run_pipeline = composite_bki_cpp.run_pipeline
    except ImportError as e:
        print(f"Error: Could not import composite_bki_cpp module: {e}", file=sys.stderr)
        print("\nPlease ensure the package is properly installed:", file=sys.stderr)
        print("  python setup.py build_ext --inplace", file=sys.stderr)
        print("  or: pip install .", file=sys.stderr)
        return 1
    
    # Print configuration
    print("=" * 60)
    print("Composite BKI - Semantic Refinement")
    print("=" * 60)
    print(f"Scan:         {args.scan}")
    print(f"Labels:       {args.label}")
    print(f"OSM:          {args.osm}")
    print(f"Output:       {args.output}")
    if args.ground_truth:
        print(f"Ground Truth: {args.ground_truth}")
    print(f"Config:       {args.config}")
    print(f"Threads:      {args.num_threads if args.num_threads > 0 else 'auto'}")
    print(f"Parameters:   l_scale={args.l_scale}, sigma_0={args.sigma_0}, "
          f"prior_delta={args.prior_delta}, alpha_0={args.alpha_0}")
    print("=" * 60)
    print()
    
    # Run the pipeline
    try:
        refined_labels = run_pipeline(
            lidar_path=args.scan,
            label_path=args.label,
            osm_path=args.osm,
            config_path=args.config,
            output_path=args.output,
            ground_truth_path=args.ground_truth,
            l_scale=args.l_scale,
            sigma_0=args.sigma_0,
            prior_delta=args.prior_delta,
            alpha_0=args.alpha_0,
            height_sigma=args.height_sigma,
            use_spatial_kernel=not args.disable_spatial,
            use_semantic_kernel=not args.disable_semantic,
            num_threads=args.num_threads
        )
        
        print()
        print("=" * 60)
        print(f"✓ Success! Processed {len(refined_labels)} points")
        print(f"✓ Output saved to: {args.output}")
        if args.metrics_output:
            print(f"✓ Metrics saved to: {args.metrics_output}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\nError during processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
