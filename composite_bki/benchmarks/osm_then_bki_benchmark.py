#!/usr/bin/env python3
"""
Benchmark: OSM-only prediction followed by BKI refinement.

This script:
1. Predicts labels from OSM priors only
2. Runs Composite BKI using those OSM-only labels as input
3. Compares both stages to ground truth
"""

import numpy as np
import sys
import csv
from pathlib import Path
from datetime import datetime

# Add repository root for composite_bki imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from composite_bki import load_osm_bin, SemanticBKI, get_config
import composite_bki_cpp


OSM_TO_MCD_LABEL = {
    0: 16,  # road
    1: 13,  # parking
    2: 18,  # sidewalk
    3: 25,  # vegetation
    4: 2,   # building
    5: 7,   # fence
}

OSM_TO_KITTI_LABEL = {
    0: 40,  # road
    1: 44,  # parking
    2: 48,  # sidewalk
    3: 70,  # vegetation
    4: 50,  # building
    5: 51,  # fence
}


def check_files_exist(file_dict):
    """Check if required files exist."""
    missing = [name for name, path in file_dict.items() if not path.exists()]

    if missing:
        print("❌ Missing required files:")
        for name in missing:
            print(f"  - {name}: {file_dict[name]}")
        return False
    return True


def calculate_metrics(pred_labels, gt_labels):
    """
    Calculate accuracy and mIoU (matches composite_bki.py::compute_metrics logic).
    """
    intersection = {}
    union = {}
    correct = {}
    total = {}

    total_correct = 0
    total_valid = 0

    unique_gt = np.unique(gt_labels)

    for cls in unique_gt:
        if cls == 0:
            continue

        gt_mask = (gt_labels == cls)
        pred_mask = (pred_labels == cls)

        inter = np.sum(gt_mask & pred_mask)
        uni = np.sum(gt_mask | pred_mask)
        count = np.sum(gt_mask)

        intersection[cls] = inter
        union[cls] = uni
        correct[cls] = inter
        total[cls] = count

        total_correct += inter
        total_valid += count

    iou_list = []

    for cls in intersection:
        if union[cls] > 0:
            val = intersection[cls] / union[cls]
            iou_list.append(val)

    miou = np.mean(iou_list) if iou_list else 0.0
    accuracy = total_correct / total_valid if total_valid > 0 else 0.0

    return {
        "accuracy": accuracy,
        "miou": miou,
    }


def predict_labels_from_osm(points, osm_path, use_kitti=False):
    """Predict labels using only OSM priors for each point."""
    config = get_config(use_kitti)
    osm_data = load_osm_bin(osm_path)
    bki = SemanticBKI(config, osm_data)

    label_map = OSM_TO_KITTI_LABEL if use_kitti else OSM_TO_MCD_LABEL

    preds = np.zeros(len(points), dtype=np.uint32)
    for i in range(len(points)):
        if i % 10000 == 0:
            print(f"  Predicted {i}/{len(points)} points...")

        m_i = bki.get_osm_prior(points[i, 0], points[i, 1])
        osm_idx = int(np.argmax(m_i))
        preds[i] = label_map.get(osm_idx, 0)

    return preds


def run_benchmark(
    lidar_path,
    gt_labels_path,
    osm_path,
    config_path,
    output_csv,
    output_labels=None,
    use_kitti=False,
):
    """
    Run the OSM-only -> BKI benchmark.
    """
    print("=" * 80)
    print("OSM-Only -> BKI Benchmark")
    print("=" * 80)
    print()

    if not check_files_exist({
        "LiDAR data": Path(lidar_path),
        "Ground truth labels": Path(gt_labels_path),
        "OSM geometries": Path(osm_path),
        "Config": Path(config_path),
    }):
        raise FileNotFoundError("Required files missing")

    print("Loading data...")
    scan = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 4))
    points = scan[:, :3]
    gt_raw = np.fromfile(gt_labels_path, dtype=np.uint32)
    gt_labels = (gt_raw & 0xFFFF).astype(np.uint32)

    print(f"  LiDAR points: {len(points)}")
    print(f"  Ground truth labels: {len(gt_labels)}")
    print(f"  Label format: {'KITTI' if use_kitti else 'MCD'}")
    print()

    print("Stage 1: Predicting labels from OSM only...")
    osm_only_labels = predict_labels_from_osm(points, osm_path, use_kitti=use_kitti)
    metrics_osm = calculate_metrics(osm_only_labels, gt_labels)

    print()
    print(f"OSM-only -> Accuracy: {metrics_osm['accuracy']*100:.2f}%, "
          f"mIoU: {metrics_osm['miou']*100:.2f}%")
    print()

    # Save OSM-only labels to disk for BKI input
    output_dir = Path(output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    osm_only_path = output_dir / "osm_only.labels"
    osm_only_labels.astype(np.uint32).tofile(osm_only_path)

    print("Stage 2: Running BKI using OSM-only labels as input...")
    refined_labels = composite_bki_cpp.run_pipeline(
        lidar_path=str(lidar_path),
        label_path=str(osm_only_path),
        osm_path=str(osm_path),
        config_path=str(config_path),
        ground_truth_path=None,
        output_path=None,
        l_scale=3.0,
        sigma_0=1.0,
        prior_delta=5.0,
        alpha_0=0.01,
    )

    metrics_bki = calculate_metrics(refined_labels, gt_labels)

    print()
    print(f"OSM+ BKI  -> Accuracy: {metrics_bki['accuracy']*100:.2f}%, "
          f"mIoU: {metrics_bki['miou']*100:.2f}%")
    print(f"Improvement -> Acc: {(metrics_bki['accuracy']-metrics_osm['accuracy'])*100:+.2f}%, "
          f"mIoU: {(metrics_bki['miou']-metrics_osm['miou'])*100:+.2f}%")
    print()

    print(f"Writing results to {output_csv}...")
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Stage", "Accuracy", "mIoU"])
        writer.writerow([
            "OSM_only",
            f"{metrics_osm['accuracy']*100:.4f}",
            f"{metrics_osm['miou']*100:.4f}",
        ])
        writer.writerow([
            "OSM_then_BKI",
            f"{metrics_bki['accuracy']*100:.4f}",
            f"{metrics_bki['miou']*100:.4f}",
        ])

    if output_labels:
        output_labels = Path(output_labels)
        output_labels.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving refined labels to {output_labels}...")
        refined_labels.astype(np.uint32).tofile(output_labels)

    print()
    print("Benchmark Complete!")
    print(f"📊 Results saved to: {output_csv}")
    if output_labels:
        print(f"🏷️  Labels saved to: {output_labels}")
    print(f"🗺️  OSM-only labels saved to: {osm_only_path}")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark OSM-only prediction then BKI refinement"
    )

    parser.add_argument(
        "--lidar",
        type=str,
        default="../example_data/mcd_scan/0000000011_transformed.bin",
        help="Path to LiDAR point cloud (.bin)",
    )

    parser.add_argument(
        "--gt-labels",
        type=str,
        default="../example_data/mcd_scan/0000000011_transformed.labels",
        help="Path to ground truth labels",
    )

    parser.add_argument(
        "--osm",
        type=str,
        default="../example_data/mcd_scan/kth_day_06_osm_geometries.bin",
        help="Path to OSM geometries",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="../configs/mcd_config.yaml",
        help="Path to configuration YAML",
    )

    parser.add_argument(
        "--kitti-labels",
        action="store_true",
        help="Treat ground truth labels as SemanticKITTI format",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: auto-generated with timestamp)",
    )

    parser.add_argument(
        "--output-labels",
        type=str,
        default=None,
        help="Optional path to save refined labels (.label/.bin)",
    )

    args = parser.parse_args()

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = Path(__file__).parent / f"osm_then_bki_{timestamp}.csv"
    else:
        output_csv = Path(args.output)

    script_dir = Path(__file__).parent
    lidar_path = (script_dir / args.lidar).resolve()
    gt_labels_path = (script_dir / args.gt_labels).resolve()
    osm_path = (script_dir / args.osm).resolve()
    config_path = (script_dir / args.config).resolve()

    run_benchmark(
        lidar_path=lidar_path,
        gt_labels_path=gt_labels_path,
        osm_path=osm_path,
        config_path=config_path,
        output_csv=output_csv,
        output_labels=args.output_labels,
        use_kitti=args.kitti_labels,
    )


if __name__ == "__main__":
    main()
