from __future__ import annotations

import numpy as np


def scale_to_255(a: np.ndarray, min_val: float, max_val: float, dtype=np.float32) -> np.ndarray:
    """Scale values from [min_val, max_val] to [0, 255]."""
    return (((a - min_val) / float(max_val - min_val)) * 255).astype(dtype)


def point_cloud_2_birdseye(
    points: np.ndarray,
    *,
    res: float = 4.0,
    side_range: tuple[float, float] = (-40.0, 40.0),
    fwd_range: tuple[float, float] = (-40.0, 40.0),
    height_range: tuple[float, float] = (-4.0, 4.0),
) -> np.ndarray:
    """
    Create a 2D bird's-eye-view representation of point cloud data.

    Input points are expected as Nx3, Nx4 (x,y,z,intensity), or Nx5 (x,y,z,intensity,label).
    Output is an image of shape (H, W, C) where C is 1..3 (height, intensity, label).
    """
    num_points, num_channels = points.shape
    if num_channels < 3:
        raise ValueError(f"Expected at least 3 channels (x,y,z), got {num_channels}")

    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    intensities = points[:, 3] if num_channels > 3 else None
    labels = points[:, 4] if num_channels > 4 else None

    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    keep = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(keep).flatten()

    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]
    if intensities is not None:
        intensities = intensities[indices]
    if labels is not None:
        labels = labels[indices]

    x_img = (-y_points / res).astype(np.int32)
    y_img = (-x_points / res).astype(np.int32)

    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    z_points = np.clip(a=z_points, a_min=height_range[0], a_max=height_range[1])
    pixel_values = scale_to_255(z_points, min_val=height_range[0], max_val=height_range[1])

    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)

    out_channels = 1 + (1 if intensities is not None else 0) + (1 if labels is not None else 0)
    birdseye_view = np.zeros((y_max, x_max, out_channels), dtype=np.float32)

    birdseye_view[y_img, x_img, 0] = pixel_values
    channel = 1
    if intensities is not None:
        birdseye_view[y_img, x_img, channel] = intensities
        channel += 1
    if labels is not None:
        birdseye_view[y_img, x_img, channel] = labels

    return birdseye_view


def birdseye_to_point_cloud(
    bev_image: np.ndarray,
    *,
    res: float = 4.0,
    side_range: tuple[float, float] = (-40.0, 40.0),
    fwd_range: tuple[float, float] = (-40.0, 40.0),
    height_range: tuple[float, float] = (-4.0, 4.0),
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a bird's-eye-view image back to point data.

    Returns:
    - C==1: (N,3) xyz
    - C==2: (xyz, intensity)
    - C==3: (xyz, intensity, label)
    """
    if bev_image.ndim != 3:
        raise ValueError(f"Expected bev_image with shape (H,W,C), got {bev_image.shape}")

    y_max, x_max, c = bev_image.shape
    if c < 1:
        raise ValueError("Expected at least one channel (height).")

    bev_h = bev_image[:, :, 0]
    bev_intensity = bev_image[:, :, 1].reshape(y_max * x_max, 1) if c > 1 else None
    bev_label = bev_image[:, :, 2].reshape(y_max * x_max, 1) if c > 2 else None

    x_img, y_img = np.meshgrid(np.arange(x_max), np.arange(y_max))
    x_points = -(y_img * res + side_range[0])
    y_points = -(x_img * res - fwd_range[1])
    z_points = bev_h / 255.0 * (height_range[1] - height_range[0]) + height_range[0]

    x_points = x_points.flatten()
    y_points = y_points.flatten()
    z_points = z_points.flatten()

    valid = np.where(bev_h.flatten() > 0)
    x_points = x_points[valid]
    y_points = y_points[valid]
    z_points = z_points[valid]

    xyz = np.vstack((x_points, y_points, z_points)).T

    if c == 1:
        return xyz
    if c == 2:
        assert bev_intensity is not None
        return xyz, bev_intensity[valid]
    # c >= 3
    assert bev_intensity is not None and bev_label is not None
    return xyz, bev_intensity[valid], bev_label[valid]


