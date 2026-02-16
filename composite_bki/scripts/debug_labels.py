#!/usr/bin/env python3
"""
Debug script to inspect label files.
"""
import numpy as np
import argparse
from collections import Counter

def inspect_labels(label_file):
    """Inspect a label file to see what labels it contains."""
    print(f"\n=== Inspecting: {label_file} ===")
    
    # Load labels
    labels_raw = np.fromfile(label_file, dtype=np.uint32)
    labels = labels_raw & 0xFFFF  # Lower 16 bits = semantic label
    
    print(f"Total points: {len(labels)}")
    print(f"Raw label range: [{labels_raw.min()}, {labels_raw.max()}]")
    print(f"Semantic label range: [{labels.min()}, {labels.max()}]")
    
    # Count unique labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\nUnique labels found: {len(unique_labels)}")
    print(f"Label distribution:")
    
    # Sort by count (descending)
    sorted_indices = np.argsort(counts)[::-1]
    for idx in sorted_indices[:20]:  # Show top 20
        label = unique_labels[idx]
        count = counts[idx]
        percentage = (count / len(labels)) * 100
        print(f"  Label {label:3d}: {count:8d} points ({percentage:5.2f}%)")
    
    if len(unique_labels) > 20:
        print(f"  ... and {len(unique_labels) - 20} more labels")
    
    # Determine format
    max_label = labels.max()
    has_kitti_labels = any(l in [40, 44, 48, 50, 70, 80, 81] for l in unique_labels)
    
    print(f"\n--- Format Detection ---")
    print(f"Max label: {max_label}")
    print(f"Has KITTI-specific labels: {has_kitti_labels}")
    
    if max_label > 30 or has_kitti_labels:
        print("→ Detected as: SemanticKITTI format")
    else:
        print("→ Detected as: MCD format")
    
    # Check for potential issues
    print(f"\n--- Potential Issues ---")
    if len(unique_labels) == 1:
        print("⚠️  WARNING: Only ONE unique label found!")
        print("    This means all points have the same label - likely an error.")
    elif labels[labels == 0].sum() / len(labels) > 0.9:
        print("⚠️  WARNING: >90% of points have label 0!")
        print("    This might indicate labels weren't properly loaded.")
    else:
        print("✓ Label distribution looks reasonable")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect label files")
    parser.add_argument('label_file', type=str, help='Path to .label file')
    
    args = parser.parse_args()
    inspect_labels(args.label_file)
