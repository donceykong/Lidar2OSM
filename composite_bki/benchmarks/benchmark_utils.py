import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import os

# Body to LiDAR transformation matrix for MCD dataset
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
    """Load semantic labels; lower 16 bits."""
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

def check_files_exist(file_dict):
    """Check if required files exist."""
    missing = [name for name, path in file_dict.items() if not path.exists()]
    if missing:
        print("❌ Missing required files:")
        for name in missing:
            print(f"  - {name}: {file_dict[name]}")
        return False
    return True
