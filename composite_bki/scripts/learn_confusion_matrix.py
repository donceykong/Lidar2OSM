#!/usr/bin/env python3
"""
Learn Confusion Matrix from Predicted vs Ground Truth Labels.

Computes the row-normalized confusion matrix:
    M[i, j] = P(True Class = j | Predicted Class = i)

This matrix answers: "When the sensor predicts class i, what is the probability
that the object is actually class j?"

Usage:
    python3 scripts/learn_confusion_matrix.py \
        --pred-dir /path/to/predictions \
        --gt-dir /path/to/ground_truth \
        --config configs/mcd_config.yaml
"""

import sys
import argparse
import numpy as np
from pathlib import Path
import yaml

def load_labels(path):
    """Load labels from .label or .bin file (uint32)."""
    raw = np.fromfile(path, dtype=np.uint32)
    return (raw & 0xFFFF).astype(np.uint32)

def load_config_mapping(config_path):
    """
    Load label mapping from YAML config.
    Returns:
        raw_to_matrix (dict): Maps raw ID -> Matrix Index
        num_classes (int): Total number of matrix classes
        class_names (list): Names of matrix classes (if inferable)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # We need the mapping from raw label ID to matrix index (super-class)
    if 'label_to_matrix_idx' not in config:
        print("Error: label_to_matrix_idx not found in config")
        sys.exit(1)
        
    raw_to_matrix = config['label_to_matrix_idx']
    
    # Determine number of classes (0-indexed, so max_idx + 1)
    if not raw_to_matrix:
        num_classes = 0
    else:
        num_classes = max(raw_to_matrix.values()) + 1
    
    # Try to infer class names from the config comments or structure
    # For MCD, we know the standard 8 classes, but let's try to be generic
    # or default to "Class N"
    # Hardcoded fallback for standard MCD config structure
    default_names = [
        "Roads", "Parking", "Sidewalks", "Vegetation", 
        "Buildings", "Fences", "Vehicles", "Other"
    ]
    
    if num_classes == 8:
        class_names = default_names
    else:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    return raw_to_matrix, num_classes, class_names

def compute_confusion_matrix(pred_dir, gt_dir, raw_to_matrix, num_classes):
    """
    Compute confusion matrix counts C[pred_class, true_class].
    """
    # C[i, j] = count of (pred=i, true=j)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # Find all prediction files
    pred_path_obj = Path(pred_dir)
    pred_files = sorted(list(pred_path_obj.glob("*.label"))) + sorted(list(pred_path_obj.glob("*.bin")))
    
    print(f"Found {len(pred_files)} prediction files.")
    
    # Pre-compute lookup table for fast mapping
    # Find max raw ID to size the lookup array
    max_raw_id = max(raw_to_matrix.keys()) if raw_to_matrix else 0
    lookup = np.full(max_raw_id + 1000, -1, dtype=np.int32) # Buffer for safety
    
    for raw, idx in raw_to_matrix.items():
        if raw < len(lookup):
            lookup[raw] = idx
            
    processed_count = 0
    
    for pred_path in pred_files:
        stem = pred_path.stem
        # Handle _prediction suffix if present (common in some pipelines)
        if stem.endswith("_prediction"):
            stem = stem.replace("_prediction", "")
            
        # Find corresponding GT file
        gt_path = None
        for ext in ['.label', '.bin']:
            candidate = Path(gt_dir) / f"{stem}{ext}"
            if candidate.exists():
                gt_path = candidate
                break
        
        if not gt_path:
            continue
            
        # Load data
        try:
            pred_raw = load_labels(str(pred_path))
            gt_raw = load_labels(str(gt_path))
        except Exception as e:
            print(f"Error loading {stem}: {e}")
            continue
        
        # Ensure lengths match
        if len(pred_raw) != len(gt_raw):
            n = min(len(pred_raw), len(gt_raw))
            pred_raw = pred_raw[:n]
            gt_raw = gt_raw[:n]
            
        # Map raw labels to matrix indices
        # Filter out labels that exceed our lookup table or aren't in config
        valid_mask = (pred_raw < len(lookup)) & (gt_raw < len(lookup))
        pred_raw = pred_raw[valid_mask]
        gt_raw = gt_raw[valid_mask]
        
        pred_indices = lookup[pred_raw]
        gt_indices = lookup[gt_raw]
        
        # Filter out unmapped labels (-1)
        mapped_mask = (pred_indices >= 0) & (gt_indices >= 0)
        pred_valid = pred_indices[mapped_mask]
        gt_valid = gt_indices[mapped_mask]
        
        if len(pred_valid) == 0:
            continue

        # Accumulate counts efficiently
        np.add.at(confusion, (pred_valid, gt_valid), 1)
        processed_count += 1
        
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} scans...")
            
    return confusion

def print_matrix_yaml(matrix, class_names):
    """Print matrix in YAML format ready for copy-pasting."""
    print("\n" + "="*60)
    print("COPY THIS INTO YOUR CONFIG (noise_matrix):")
    print("="*60)
    print("# Learned Noise Matrix (Row-Normalized: P(True|Pred))")
    print("# Rows: Predicted Class, Cols: True Class")
    print("noise_matrix:")
    
    rows, cols = matrix.shape
    for i in range(rows):
        # Format probabilities nicely
        row_str = ", ".join([f"{x:.4f}" for x in matrix[i]])
        name = class_names[i] if i < len(class_names) else f"Class {i}"
        print(f"  - [{row_str}] # {name}")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Learn Confusion Matrix from Pred/GT labels")
    parser.add_argument("--pred-dir", required=True, help="Directory of predicted labels")
    parser.add_argument("--gt-dir", required=True, help="Directory of ground truth labels")
    parser.add_argument("--config", required=True, help="Path to YAML config (for label mapping)")
    parser.add_argument("--output", help="Optional output path to save raw counts (.npy)")
    
    args = parser.parse_args()
    
    print(f"Loading config from {args.config}...")
    raw_to_matrix, num_classes, class_names = load_config_mapping(args.config)
    print(f"Number of super-classes: {num_classes}")
    print(f"Classes: {class_names}")
    
    print(f"\nComputing confusion counts from {args.pred_dir}...")
    counts = compute_confusion_matrix(args.pred_dir, args.gt_dir, raw_to_matrix, num_classes)
    
    # Row Normalize: P(True | Pred)
    # This answers: "Given the sensor said X, what is the probability it is Y?"
    row_sums = counts.sum(axis=1, keepdims=True)
    
    # Handle empty rows (classes never predicted) to avoid division by zero
    # If a class is never predicted, its row is all zeros. We can leave it as zeros
    # or set diagonal to 1 (trusting it perfectly if it ever happens).
    # Setting diagonal to 1 is safer for stability.
    zero_rows = (row_sums.flatten() == 0)
    if np.any(zero_rows):
        print(f"Warning: Classes {np.where(zero_rows)[0]} were never predicted.")
        row_sums[zero_rows] = 1.0 # Prevent div by zero
    
    prob_matrix = counts.astype(np.float64) / row_sums
    
    # Fix zero rows to be identity (if I ever predict this rare class, trust it)
    for i in range(num_classes):
        if zero_rows[i]:
            prob_matrix[i, i] = 1.0

    print("\n--- Raw Counts (Diagonal should be high) ---")
    print(counts)
    
    print_matrix_yaml(prob_matrix, class_names)
    
    if args.output:
        np.save(args.output, counts)
        print(f"Saved raw counts to {args.output}")

if __name__ == "__main__":
    main()