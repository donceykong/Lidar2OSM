#!/usr/bin/env python3
"""
Example script for continuous mapping with multiple scans.

This demonstrates how to use composite-bki for continuous mapping,
where multiple scans are processed sequentially and the map state
accumulates over time.
"""

import composite_bki_cpp
import numpy as np
import os
import glob
from pathlib import Path


def continuous_mapping_example(scan_dir, label_dir, osm_path, config_path,
                                output_dir, map_state_file=None,
                                osm_prior_strength=0.0,
                                disable_osm_fallback=False,
                                lambda_min=0.8,
                                lambda_max=0.99):
    """
    Process multiple scans sequentially for continuous mapping.

    Args:
        scan_dir: Directory containing .bin scan files
        label_dir: Directory containing .label prediction files
        osm_path: Path to OSM geometries binary file
        config_path: Path to YAML config file
        output_dir: Directory to save refined labels
        map_state_file: Optional path to save/load map state
    """

    # Validate OSM file exists and is .bin or .osm format
    if not os.path.exists(osm_path):
        raise FileNotFoundError(f"OSM file not found: {osm_path}")
    if not (osm_path.endswith('.bin') or osm_path.endswith('.osm')):
        raise ValueError(f"OSM file must be a .bin (binary) or .osm (XML) file, got: {osm_path}")

    # Initialize BKI once (stateful)
    print("Initializing BKI (Continuous / GridHash) for continuous mapping...")
    try:
        bki = composite_bki_cpp.PyContinuousBKI(
            osm_path=osm_path,
            config_path=config_path,
            resolution=0.1,
            l_scale=0.5,  # Can be tuned for continuous mapping
            use_semantic_kernel=True,
            use_spatial_kernel=True,
            osm_prior_strength=osm_prior_strength,
            osm_fallback_in_infer=not disable_osm_fallback,
            lambda_min=lambda_min,
            lambda_max=lambda_max
        )
    except Exception as e:
        print(f"Error initializing BKI: {e}")
        print("Note: OSM file can be either .bin (binary) or .osm (XML) format")
        raise
    
    # Load existing map state if provided
    if map_state_file and os.path.exists(map_state_file):
        print(f"Loading map state from {map_state_file}...")
        bki.load(map_state_file)
        print(f"Loaded {bki.get_size()} voxels")
    
    # Find all scan files
    scan_files = sorted(glob.glob(os.path.join(scan_dir, "*.bin")))
    print(f"Found {len(scan_files)} scans to process")
    
    # Process each scan sequentially
    for i, scan_file in enumerate(scan_files):
        scan_name = Path(scan_file).stem
        
        # Try to parse scan index from filename (e.g. 000000.bin -> 0)
        try:
            scan_idx = int(scan_name)
        except ValueError:
            scan_idx = i # Fallback to sequential index
        
        # Find corresponding label file (try both .label and .bin extensions)
        label_file = os.path.join(label_dir, f"{scan_name}.label")
        if not os.path.exists(label_file):
            label_file = os.path.join(label_dir, f"{scan_name}.bin")
        
        if not os.path.exists(label_file):
            print(f"Warning: No label file for {scan_name}, skipping...")
            continue
        
        print(f"\n[{i+1}/{len(scan_files)}] Processing {scan_name} (Index: {scan_idx})...")
        
        # Load scan and labels
        points = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)
        labels = np.fromfile(label_file, dtype=np.uint32)
        
        if len(points) != len(labels):
            print(f"  Warning: Size mismatch ({len(points)} vs {len(labels)}), skipping...")
            continue
        
        # Update map with this scan (accumulates state)
        print(f"  Updating map with {len(points)} points...")
        bki.update(labels, points[:, :3])
        
        # Print map statistics
        print(f"  Map size: {bki.get_size()} voxels")
        
        # Optional: Infer and save refined labels (can be disabled for pure mapping)
        if output_dir:
            print(f"  Running inference...")
            refined_labels = bki.infer(points[:, :3])
            
            # Save refined labels
            output_file = os.path.join(output_dir, f"{scan_name}_refined.label")
            refined_labels.astype(np.uint32).tofile(output_file)
            print(f"  Saved refined labels to {output_file}")
    
    # Save final map state
    if map_state_file:
        print(f"\nSaving map state to {map_state_file}...")
        bki.save(map_state_file)
        print(f"Final map size: {bki.get_size()} voxels")
    
    print("\nContinuous mapping complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous mapping example")
    parser.add_argument("--scan-dir", required=True, help="Directory with .bin scan files")
    parser.add_argument("--label-dir", required=True, help="Directory with .label prediction files")
    parser.add_argument("--osm", required=True, help="Path to OSM geometries binary")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output-dir", default=None, help="Output directory for refined labels (optional, omit to skip inference)")
    parser.add_argument("--map-state", help="Path to save/load map state file")

    parser.add_argument("--osm-prior-strength", type=float, default=0.0, help="OSM prior strength")
    parser.add_argument("--disable-osm-fallback", type=bool, default=False, help="Disable OSM fallback during inference")
    parser.add_argument("--lambda-min", type=float, default=0.8, help="Min forgetting factor")
    parser.add_argument("--lambda-max", type=float, default=0.99, help="Max forgetting factor")
    
    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    continuous_mapping_example(
        scan_dir=args.scan_dir,
        label_dir=args.label_dir,
        osm_path=args.osm,
        config_path=args.config,
        output_dir=args.output_dir,
        map_state_file=args.map_state,
        osm_prior_strength=args.osm_prior_strength,
        disable_osm_fallback=args.disable_osm_fallback,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max
    )
