#!/usr/bin/env python3
"""
Convert a .bki map file to a point cloud (.bin) and labels (.label) file.

The .bki format (version 2) stores a sparse block grid. This script reads it,
enumerates all voxel centers and their predicted class (argmax of alpha),
and writes them in standard .bin (x,y,z,0) and .label (uint32) format.

Requires --config to get num_total_classes and dense_to_raw (same YAML used when building the map).
"""

import argparse
import struct
import numpy as np
from pathlib import Path

BLOCK_SIZE = 8


def read_bki(bki_path, config_path, min_alpha_sum=1e-6):
    """
    Read .bki file and yield (points_xyz, labels_raw) as arrays.
    points_xyz: (N, 3) float32, labels_raw: (N,) uint32.
    Only includes voxels with sum(alpha) >= min_alpha_sum.
    """
    # Load config to get K and dense_to_raw mapping
    # We need to parse the YAML manually or use PyYAML if available
    try:
        import yaml
        with open(config_path) as f:
            data = yaml.safe_load(f)
        labels = data.get("labels") or {}
        raw_ids = sorted(int(k) for k in labels.keys())
        K = len(raw_ids)
        dense_to_raw = raw_ids
    except ImportError:
        # Fallback: minimal parse without PyYAML
        labels = {}
        in_labels = False
        with open(config_path) as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("#") or not stripped:
                    continue
                if stripped == "labels:":
                    in_labels = True
                    continue
                if in_labels:
                    if ":" in stripped and not stripped.startswith("-"):
                        # Top-level key (e.g. "confusion_matrix:") - stop
                        indent = len(line) - len(line.lstrip())
                        if indent < 2:
                            break
                    parts = stripped.split(":", 1)
                    if len(parts) == 2:
                        try:
                            raw_id = int(parts[0].strip())
                            labels[raw_id] = parts[1].strip()
                        except ValueError:
                            pass
        if not labels:
            raise ValueError("Could not parse labels from config; install PyYAML or use a config with 'labels:' section")
        raw_ids = sorted(labels.keys())
        K = len(raw_ids)
        dense_to_raw = raw_ids

    block_alpha_size = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE * K

    points_list = []
    labels_list = []

    with open(bki_path, "rb") as f:
        version = struct.unpack("B", f.read(1))[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported .bki version {version} (expected 2 or 3)")
        
        resolution = struct.unpack("f", f.read(4))[0]
        l_scale = struct.unpack("f", f.read(4))[0]
        sigma_0 = struct.unpack("f", f.read(4))[0]
        
        current_time = 0
        if version >= 3:
            current_time = struct.unpack("i", f.read(4))[0]
            
        num_blocks = struct.unpack("Q", f.read(8))[0]

        for _ in range(num_blocks):
            bx, by, bz = struct.unpack("iii", f.read(12))
            
            last_updated = 0
            if version >= 3:
                last_updated = struct.unpack("i", f.read(4))[0]
                
            alpha_bytes = f.read(block_alpha_size * 4)
            if len(alpha_bytes) != block_alpha_size * 4:
                break
                
            alpha = np.frombuffer(alpha_bytes, dtype=np.float32)
            alpha = alpha.reshape((BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, K))

            for lz in range(BLOCK_SIZE):
                for ly in range(BLOCK_SIZE):
                    for lx in range(BLOCK_SIZE):
                        a = alpha[lx, ly, lz, :]
                        s = float(np.sum(a))
                        if s < min_alpha_sum:
                            continue
                        
                        # Filter out unobserved voxels (uniform distribution)
                        if np.max(a) - np.min(a) < 1e-6:
                            continue

                        best = int(np.argmax(a))
                        raw = dense_to_raw[best] if best < len(dense_to_raw) else 0
                        
                        # Voxel center in world coordinates
                        vx = bx * BLOCK_SIZE + lx
                        vy = by * BLOCK_SIZE + ly
                        vz = bz * BLOCK_SIZE + lz
                        x = (vx + 0.5) * resolution
                        y = (vy + 0.5) * resolution
                        z = (vz + 0.5) * resolution
                        
                        points_list.append([x, y, z])
                        labels_list.append(raw)

    if not points_list:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.uint32)

    points_xyz = np.array(points_list, dtype=np.float32)
    labels_raw = np.array(labels_list, dtype=np.uint32)
    return points_xyz, labels_raw


def main():
    parser = argparse.ArgumentParser(
        description="Convert .bki map to .bin point cloud and .label file."
    )
    parser.add_argument("bki", help="Path to .bki map file")
    parser.add_argument("--config", required=True, help="Path to YAML config (same as used when building the map)")
    parser.add_argument("--output-bin", default=None, help="Output .bin path (default: <bki_stem>_map.bin)")
    parser.add_argument("--output-label", default=None, help="Output .label path (default: <bki_stem>_map.label)")
    parser.add_argument("--min-alpha", type=float, default=1e-6, help="Min alpha sum per voxel to export (default: 1e-6)")
    args = parser.parse_args()

    bki_path = Path(args.bki)
    if not bki_path.exists():
        raise SystemExit(f"File not found: {bki_path}")
    if not Path(args.config).exists():
        raise SystemExit(f"Config not found: {args.config}")

    stem = bki_path.stem
    out_bin = args.output_bin or str(bki_path.parent / f"{stem}_map.bin")
    out_label = args.output_label or str(bki_path.parent / f"{stem}_map.label")

    print(f"Reading {bki_path} with config {args.config}...")
    points_xyz, labels_raw = read_bki(str(bki_path), args.config, min_alpha_sum=args.min_alpha)
    n = len(points_xyz)
    print(f"Exporting {n} voxels.")

    # .bin: N x 4 (x, y, z, 0)
    cloud = np.hstack([points_xyz, np.zeros((n, 1), dtype=np.float32)])
    cloud.tofile(out_bin)
    labels_raw.tofile(out_label)
    print(f"Wrote {out_bin}")
    print(f"Wrote {out_label}")
    return 0


if __name__ == "__main__":
    main()
