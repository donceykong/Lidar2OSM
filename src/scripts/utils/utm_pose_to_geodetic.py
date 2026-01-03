#!/usr/bin/env python3

"""
utm_pose_to_geodetic.py

Convert a UTM pose (x=easting, y=northing, z=alt) and quaternion (qx, qy, qz, qw)
to latitude, longitude, altitude (WGS84, EPSG:4326). Also computes yaw (ENU) and a
navigation-friendly heading referenced to true north (clockwise, 0..360).

Requirements:
  pip install pyproj scipy

Usage examples:
  # Example with your provided numbers (assuming UTM Zone 13N, WGS84)
  python utm_pose_to_geodetic.py \
      --zone 13 --north \
      --x 477144.87558798917 --y 4428490.761253119 --z 1630.5456520790192 \
      --qx -0.0035978446494060886 --qy -0.0011208116879074216 \
      --qz 0.9992241676214164 --qw 0.03920283326912376

  # If your pose is in the southern hemisphere, omit --north and pass --south
  python utm_pose_to_geodetic.py --zone 56 --south --x ... --y ... --z ... --qx ... --qy ... --qz ... --qw ...

Notes:
- The quaternion is assumed to be expressed in an ENU frame aligned with the local
  UTM grid axes: +x (east), +y (north), +z (up).
- "Yaw_ENU_deg" is the standard yaw about +z (up) in ENU, positive CCW from +x (east).
- "Heading_deg" is the navigation-style heading: clockwise from true north (0..360),
  derived from ENU yaw by Heading = (90 - Yaw_ENU_deg) mod 360.
- If your quaternion is in a different frame (e.g., NED, body frame), adapt the code.
"""

from dataclasses import dataclass
from typing import Tuple
from math import fmod
import argparse

from pyproj import CRS, Transformer
from scipy.spatial.transform import Rotation as R


@dataclass
class Pose:
    t: float
    x: float  # UTM easting (meters)
    y: float  # UTM northing (meters)
    z: float  # altitude (meters, ellipsoidal by default)
    qx: float
    qy: float
    qz: float
    qw: float


def utm_to_latlon(easting: float, northing: float, zone: int, northern_hemisphere: bool = True) -> Tuple[float, float]:
    """
    Convert UTM (WGS84) to latitude/longitude in degrees.
    Returns (lat, lon).
    """
    if not (1 <= zone <= 60):
        raise ValueError("UTM zone must be in [1, 60]")

    # Build CRS objects. We prefer explicit proj strings for clarity.
    hemi = "+north" if northern_hemisphere else "+south"
    utm_crs = CRS.from_proj4(f"+proj=utm +zone={zone} {hemi} +datum=WGS84 +units=m +no_defs")
    wgs84 = CRS.from_epsg(4326)  # lat/lon

    transformer = Transformer.from_crs(utm_crs, wgs84, always_xy=True)
    lon, lat = transformer.transform(easting, northing)  # always_xy=True => input (x=east, y=north), output (lon, lat)
    return lat, lon


def quat_to_yaw_heading(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    """
    Given a quaternion (x,y,z,w) in ENU, compute roll, pitch, yaw (degrees), and a navigation heading (degrees).
    - Yaw_ENU: rotation about +z (up), positive CCW from +x (east).
    - Heading: clockwise from true north (0..360). Heading = (90 - Yaw_ENU) mod 360.
    """
    # SciPy's Rotation uses (x, y, z, w)
    r = R.from_quat([qx, qy, qz, qw])

    # For ENU, a common convention is intrinsic ZYX (yaw, pitch, roll) about local axes.
    # SciPy's as_euler uses extrinsic vs intrinsic via sequence; 'zyx' returns intrinsic ZYX when default 'intrinsic' is used.
    yaw, pitch, roll = r.as_euler('zyx', degrees=True)

    # Normalize yaw to (-180, 180]; Heading to [0, 360)
    def wrap180(a):
        # Wrap to (-180, 180]
        a_wrapped = (a + 180.0) % 360.0 - 180.0
        # Ensure +180 instead of -180 when exactly at boundary
        if a_wrapped == -180.0:
            return 180.0
        return a_wrapped

    yaw_enu = wrap180(yaw)
    heading = (90.0 - yaw_enu) % 360.0
    return roll, pitch, yaw_enu, heading


def main():
    ap = argparse.ArgumentParser(description="Convert UTM pose to lat/lon/alt with yaw/heading.")
    ap.add_argument("--zone", type=int, required=True, help="UTM zone number [1..60]")
    hemi = ap.add_mutually_exclusive_group(required=True)
    hemi.add_argument("--north", action="store_true", help="Pose is in the northern hemisphere (Zone ##N)")
    hemi.add_argument("--south", action="store_true", help="Pose is in the southern hemisphere (Zone ##S)")

    ap.add_argument("--t", type=float, default=0.0, help="Timestamp (seconds)")
    ap.add_argument("--x", type=float, required=True, help="UTM easting (meters)")
    ap.add_argument("--y", type=float, required=True, help="UTM northing (meters)")
    ap.add_argument("--z", type=float, default=0.0, help="Altitude (meters)")
    ap.add_argument("--qx", type=float, required=True, help="Quaternion x (ENU)")
    ap.add_argument("--qy", type=float, required=True, help="Quaternion y (ENU)")
    ap.add_argument("--qz", type=float, required=True, help="Quaternion z (ENU)")
    ap.add_argument("--qw", type=float, required=True, help="Quaternion w (ENU)")
    args = ap.parse_args()

    pose = Pose(
        t=args.t, x=args.x, y=args.y, z=args.z,
        qx=args.qx, qy=args.qy, qz=args.qz, qw=args.qw
    )

    lat, lon = utm_to_latlon(pose.x, pose.y, args.zone, northern_hemisphere=args.north)
    roll, pitch, yaw_enu, heading = quat_to_yaw_heading(pose.qx, pose.qy, pose.qz, pose.qw)

    # Pretty print
    print("=== Geodetic Pose (WGS84) ===")
    print(f"Time (s):     {pose.t:.6f}")
    print(f"Latitude:     {lat:.12f} deg")
    print(f"Longitude:    {lon:.12f} deg")
    print(f"Altitude:     {pose.z:.6f} m\n")

    print("=== Orientation (ENU) ===")
    print(f"Quaternion:   x={pose.qx:.12f}, y={pose.qy:.12f}, z={pose.qz:.12f}, w={pose.qw:.12f}")
    print(f"Roll  (deg):  {roll:.6f}")
    print(f"Pitch (deg):  {pitch:.6f}")
    print(f"Yaw   (deg):  {yaw_enu:.6f}   # ENU, +CCW from East")
    print(f"Heading (deg):{heading:.6f}   # CW from North (0..360)")

if __name__ == "__main__":
    main()
