#!/usr/bin/env python3
"""
Train a continuous BKI map on the first half of scans and evaluate on the second half.

Uses the same LiDAR->Body->World transform as build_voxel_map.py when a pose file
is provided, so all scans are in a common world frame. No pre-voxelization;
points are transformed and passed directly to the BKI.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Optional: use transform from build_voxel_map if available
try:
    import pandas as pd
    from scipy.spatial.transform import Rotation as R
    _HAS_TRANSFORM = True
except ImportError:
    _HAS_TRANSFORM = False

# Body to LiDAR transformation matrix for MCD dataset (same as build_voxel_map.py)
BODY_TO_LIDAR_TF = np.array(
    [
        [0.9999135040741837, -0.011166365511073898, -0.006949579221822984, -0.04894521120494695],
        [-0.011356389542502144, -0.9995453006865824, -0.02793249526856565, -0.03126929060348084],
        [-0.006634514801117132, 0.02800900135032654, -0.999585653686922, -0.01755515794222565],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def load_poses(pose_file):
    """Load poses from CSV (num,t,x,y,z,qx,qy,qz,qw). Returns dict frame_num -> (7,) pose."""
    if not _HAS_TRANSFORM:
        raise RuntimeError("pandas and scipy are required for pose loading. pip install pandas scipy")
    df = pd.read_csv(pose_file)
    poses = {}
    for _, row in df.iterrows():
        frame_num = int(row["num"])
        x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
        qx, qy, qz, qw = float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])
        poses[frame_num] = np.array([x, y, z, qx, qy, qz, qw])
    return poses


def transform_points_to_world(points, pose):
    """LiDAR -> Body -> World using MCD calibration. points (N,3), pose (7,) [x,y,z,qx,qy,qz,qw]."""
    position = pose[:3]
    quat = pose[3:7]
    body_rotation_matrix = R.from_quat(quat).as_matrix()
    body_to_world = np.eye(4, dtype=np.float64)
    body_to_world[:3, :3] = body_rotation_matrix
    body_to_world[:3, 3] = np.asarray(position, dtype=np.float64)
    lidar_to_body = np.linalg.inv(BODY_TO_LIDAR_TF)
    transform_matrix = body_to_world @ lidar_to_body
    points_homogeneous = np.hstack([np.asarray(points, dtype=np.float64), np.ones((points.shape[0], 1))])
    world_points = (transform_matrix @ points_homogeneous.T).T
    return world_points[:, :3].astype(np.float32)


def load_scan(bin_path):
    """Load point cloud (N,4) and return (N,3) xyz, (N,) intensity."""
    scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return scan[:, :3], scan[:, 3]


def load_labels(label_path):
    """Load semantic labels; lower 16 bits (same as build_voxel_map)."""
    raw = np.fromfile(label_path, dtype=np.uint32)
    return (raw & 0xFFFF).astype(np.uint32)


def find_label_file(label_dir, scan_stem):
    """Find label file for a scan; try .label, .bin, _prediction.label, _prediction.bin."""
    exts = [".label", ".bin", "_prediction.label", "_prediction.bin"]
    for ext in exts:
        p = Path(label_dir) / f"{scan_stem}{ext}"
        if p.exists():
            return str(p)
    return None


def get_frame_number(stem):
    try:
        return int(stem)
    except ValueError:
        return None


def compute_metrics(pred, gt, ignore_label=0):
    """Accuracy and mIoU; optionally ignore a label (e.g. 0) in GT."""
    pred = np.asarray(pred, dtype=np.uint32)
    gt = np.asarray(gt, dtype=np.uint32)
    if pred.shape != gt.shape:
        return {"accuracy": 0.0, "miou": 0.0, "class_ious": {}}
    mask = gt != ignore_label
    if not np.any(mask):
        return {"accuracy": 0.0, "miou": 0.0, "class_ious": {}}
    pred_m = pred[mask]
    gt_m = gt[mask]
    accuracy = np.mean(pred_m == gt_m)
    classes = np.unique(np.concatenate([pred_m, gt_m]))
    ious = {}
    for c in classes:
        inter = np.sum((pred_m == c) & (gt_m == c))
        union = np.sum((pred_m == c) | (gt_m == c))
        if union > 0:
            ious[int(c)] = inter / union
    miou = np.mean(list(ious.values())) if ious else 0.0
    return {"accuracy": accuracy, "miou": miou, "class_ious": ious}


def main():
    parser = argparse.ArgumentParser(
        description="Train continuous BKI on first half of scans (world frame), evaluate on the same scans."
    )
    parser.add_argument("--scan-dir", required=True, help="Directory of .bin scans")
    parser.add_argument("--label-dir", required=True, help="Directory of prediction labels (.label or .bin)")
    parser.add_argument("--osm", required=True, help="Path to OSM geometries (.bin or .osm XML)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--pose", default=None, help="CSV poses (num,t,x,y,z,qx,qy,qz,qw) for world-frame transform (optional)")
    parser.add_argument("--gt-dir", default=None, help="Optional: directory of ground-truth labels for test half metrics")
    parser.add_argument("--output-dir", default=None, help="Optional: save refined labels for test scans here")
    parser.add_argument("--map-state", default=None, help="Optional: path to save trained map state")
    parser.add_argument("--resolution", type=float, default=1.0, help="BKI resolution")
    parser.add_argument("--l-scale", type=float, default=3.0, help="BKI l_scale")
    parser.add_argument("--sigma-0", type=float, default=1.0, help="BKI sigma_0")
    parser.add_argument("--prior-delta", type=float, default=0.5, help="BKI prior_delta")
    parser.add_argument("--height-sigma", type=float, default=5.0, help="BKI height_sigma (controls vertical spread of OSM prior)")
    parser.add_argument("--alpha0", type=float, default=1.0, help="BKI alpha0")
    parser.add_argument("--seed-osm-prior", type=bool, default=False, help="Enable OSM prior seeding")
    parser.add_argument("--osm-prior-strength", type=float, default=0.0, help="OSM prior strength")
    parser.add_argument("--disable-osm-fallback", type=bool, default=False, help="Disable OSM fallback during inference")
    parser.add_argument("--lambda-min", type=float, default=0.8, help="Min forgetting factor (for high confidence OSM areas)")
    parser.add_argument("--lambda-max", type=float, default=0.99, help="Max forgetting factor (for low confidence OSM areas)")
    args = parser.parse_args()

    if args.pose and not _HAS_TRANSFORM:
        print("ERROR: --pose requires pandas and scipy. pip install pandas scipy", file=sys.stderr)
        return 1

    try:
        import composite_bki_cpp
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
        try:
            import composite_bki_cpp
        except ImportError:
            print("ERROR: composite_bki_cpp not found. Install the package or run from repo root with PYTHONPATH=src", file=sys.stderr)
            return 1

    scan_dir = Path(args.scan_dir)
    label_dir = Path(args.label_dir)
    scan_files = sorted(scan_dir.glob("*.bin"))
    if not scan_files:
        print(f"No .bin scans in {scan_dir}", file=sys.stderr)
        return 1

    n_total = len(scan_files)
    # Use ALL scans for both training and testing
    n_train = n_total
    n_test = n_total
    train_files = scan_files
    test_files = scan_files

    print(f"Scans: {n_total} total -> Using ALL {n_total} scans for training and testing")

    poses = None
    if args.pose:
        if not os.path.exists(args.pose):
            print(f"Pose file not found: {args.pose}", file=sys.stderr)
            return 1
        poses = load_poses(args.pose)
        print(f"Loaded {len(poses)} poses; transforming scans to world frame.")
    else:
        print("No --pose provided; using scan coordinates as-is (sensor frame).")

    if not os.path.exists(args.osm):
        print("OSM file must exist", file=sys.stderr)
        return 1
    if not (args.osm.endswith(".bin") or args.osm.endswith(".osm")):
        print("OSM file must be .bin (binary) or .osm (XML format)", file=sys.stderr)
        return 1
    if not os.path.exists(args.config):
        print("Config file not found", file=sys.stderr)
        return 1

    def train_bki(use_semantic_kernel):
        bki = composite_bki_cpp.PyContinuousBKI(
            osm_path=args.osm,
            config_path=args.config,
            resolution=args.resolution,
            l_scale=args.l_scale,
            sigma_0=args.sigma_0,
            prior_delta=args.prior_delta,
            height_sigma=args.height_sigma,
            use_semantic_kernel=use_semantic_kernel,
            use_spatial_kernel=True,
            alpha0=args.alpha0,
            seed_osm_prior=args.seed_osm_prior,
            osm_prior_strength=args.osm_prior_strength,
            osm_fallback_in_infer=not args.disable_osm_fallback,
            lambda_min=args.lambda_min,
            lambda_max=args.lambda_max
        )
        for scan_path in train_files:
            stem = scan_path.stem
            frame = get_frame_number(stem)
            label_path = find_label_file(label_dir, stem)
            if not label_path:
                continue
            points_xyz, _ = load_scan(str(scan_path))
            labels = load_labels(label_path)
            if len(labels) != len(points_xyz):
                min_len = min(len(labels), len(points_xyz))
                points_xyz = points_xyz[:min_len]
                labels = labels[:min_len]
            if poses is not None and frame is not None and frame in poses:
                points_xyz = transform_points_to_world(points_xyz, poses[frame])
            bki.update(labels, points_xyz)
        return bki

    # Build maps: with semantic kernel, without semantic kernel, and without spatial kernel
    print("\n--- Training (first half) ---")
    print("  Training with semantic kernel...")
    bki_sem = train_bki(use_semantic_kernel=True)
    print(f"  Map size (with sem):  {bki_sem.get_size()} voxels")
    
    print("  Training without semantic kernel...")
    bki_nosem = train_bki(use_semantic_kernel=False)
    print(f"  Map size (without sem): {bki_nosem.get_size()} voxels")

    print("  Training without BOTH kernels (no spatial, no semantic)...")
    # To disable spatial kernel, we need to modify train_bki or create a new instance
    # Let's modify train_bki to accept use_spatial_kernel
    def train_bki_custom(use_semantic_kernel, use_spatial_kernel):
        bki = composite_bki_cpp.PyContinuousBKI(
            osm_path=args.osm,
            config_path=args.config,
            resolution=args.resolution,
            l_scale=args.l_scale,
            sigma_0=args.sigma_0,
            prior_delta=args.prior_delta,
            height_sigma=args.height_sigma,
            use_semantic_kernel=use_semantic_kernel,
            use_spatial_kernel=use_spatial_kernel,
            alpha0=args.alpha0,
            seed_osm_prior=args.seed_osm_prior,
            osm_prior_strength=args.osm_prior_strength,
            osm_fallback_in_infer=not args.disable_osm_fallback,
            lambda_min=args.lambda_min,
            lambda_max=args.lambda_max
        )
        for scan_path in train_files:
            stem = scan_path.stem
            frame = get_frame_number(stem)
            label_path = find_label_file(label_dir, stem)
            if not label_path:
                continue
            points_xyz, _ = load_scan(str(scan_path))
            labels = load_labels(label_path)
            if len(labels) != len(points_xyz):
                min_len = min(len(labels), len(points_xyz))
                points_xyz = points_xyz[:min_len]
                labels = labels[:min_len]
            if poses is not None and frame is not None and frame in poses:
                points_xyz = transform_points_to_world(points_xyz, poses[frame])
            bki.update(labels, points_xyz)
        return bki

    bki_nokernels = train_bki_custom(use_semantic_kernel=False, use_spatial_kernel=False)
    print(f"  Map size (no kernels): {bki_nokernels.get_size()} voxels")

    if args.map_state:
        base, ext = os.path.splitext(args.map_state)
        os.makedirs(os.path.dirname(args.map_state) or ".", exist_ok=True)
        bki_sem.save(base + "_sem" + (ext or ""))
        bki_nosem.save(base + "_nosem" + (ext or ""))
        bki_nokernels.save(base + "_nokernels" + (ext or ""))
        print(f"  Saved maps to {base}_sem / {base}_nosem / {base}_nokernels")

    # Evaluate on the same files used for training
    print("\n--- Testing (same files as training) ---")
    all_metrics_sem = []
    all_metrics_nosem = []
    all_metrics_nokernels = []
    all_metrics_baseline = []
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for scan_path in train_files:
        stem = scan_path.stem
        frame = get_frame_number(stem)
        points_xyz, _ = load_scan(str(scan_path))
        if poses is not None and frame is not None and frame in poses:
            points_xyz = transform_points_to_world(points_xyz, poses[frame])

        pred_sem = bki_sem.infer(points_xyz)
        pred_nosem = bki_nosem.infer(points_xyz)
        pred_nokernels = bki_nokernels.infer(points_xyz)

        if args.output_dir:
            pred_sem.astype(np.uint32).tofile(str(Path(args.output_dir) / f"{stem}_refined_sem.label"))
            pred_nosem.astype(np.uint32).tofile(str(Path(args.output_dir) / f"{stem}_refined_nosem.label"))
            pred_nokernels.astype(np.uint32).tofile(str(Path(args.output_dir) / f"{stem}_refined_nokernels.label"))

        if args.gt_dir:
            gt_path = find_label_file(args.gt_dir, stem)
            if gt_path:
                gt = load_labels(gt_path)
                
                # Load baseline (input) labels
                input_path = find_label_file(label_dir, stem)
                input_labels = load_labels(input_path) if input_path else None

                n = min(len(gt), len(pred_sem), len(pred_nosem), len(pred_nokernels))
                if input_labels is not None:
                    n = min(n, len(input_labels))
                
                if n > 0:
                    all_metrics_sem.append(compute_metrics(pred_sem[:n], gt[:n]))
                    all_metrics_nosem.append(compute_metrics(pred_nosem[:n], gt[:n]))
                    all_metrics_nokernels.append(compute_metrics(pred_nokernels[:n], gt[:n]))
                    if input_labels is not None:
                        all_metrics_baseline.append(compute_metrics(input_labels[:n], gt[:n]))

    if all_metrics_sem:
        acc_sem = np.mean([m["accuracy"] for m in all_metrics_sem])
        miou_sem = np.mean([m["miou"] for m in all_metrics_sem])
        acc_nosem = np.mean([m["accuracy"] for m in all_metrics_nosem])
        miou_nosem = np.mean([m["miou"] for m in all_metrics_nosem])
        acc_nokernels = np.mean([m["accuracy"] for m in all_metrics_nokernels])
        miou_nokernels = np.mean([m["miou"] for m in all_metrics_nokernels])
        
        print(f"  Test scans with GT: {len(all_metrics_sem)}")
        if all_metrics_baseline:
            acc_base = np.mean([m["accuracy"] for m in all_metrics_baseline])
            miou_base = np.mean([m["miou"] for m in all_metrics_baseline])
            print("  Baseline (Input):        Accuracy {:.4f}  mIoU {:.4f}".format(acc_base, miou_base))
        
        print("  With semantic kernel:    Accuracy {:.4f}  mIoU {:.4f}".format(acc_sem, miou_sem))
        print("  Without semantic kernel: Accuracy {:.4f}  mIoU {:.4f}".format(acc_nosem, miou_nosem))
        print("  Without ANY kernels:     Accuracy {:.4f}  mIoU {:.4f}".format(acc_nokernels, miou_nokernels))
    else:
        if args.gt_dir:
            print("  No test scans had matching GT labels.")
        else:
            print("  No --gt-dir provided; skipping metrics.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
