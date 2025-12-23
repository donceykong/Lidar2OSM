import numpy as np

def normalize_quaternion(q):
    """
    Normalize a quaternion to ensure it's valid for rotation calculations.
    Args:
        q (tuple): Quaternion (w, x, y, z).
    Returns:
        tuple: Normalized quaternion.
    """
    norm = np.sqrt(sum(v**2 for v in q))
    return tuple(v / norm for v in q)

def quaternion_to_rotation_index(q, rotation_bin_size=0.01):
    """
    Converts a quaternion to a discretized index based on rotation bin size.
    Args:
        q (tuple): Quaternion (w, x, y, z).
        rotation_bin_size (float): Size of each bin in radians.
    Returns:
        int: Discretized rotation index.
    """
    q = normalize_quaternion(q)
    yaw = 2 * np.arctan2(q[3], q[0])  # Simplified yaw calculation
    yaw_normalized = (yaw + np.pi)  # Shift to range [0, 2π]
    return int(yaw_normalized / rotation_bin_size)

def to_morton_code(ix, iy, iz, ir):
    """
    Compute Morton code by interleaving bits of x, y, z, and rotation indices.
    """
    morton = 0
    for i in range(max(ix.bit_length(), iy.bit_length(), iz.bit_length(), ir.bit_length())):
        morton |= ((ix >> i & 1) << (4 * i)) | \
                  ((iy >> i & 1) << (4 * i + 1)) | \
                  ((iz >> i & 1) << (4 * i + 2)) | \
                  ((ir >> i & 1) << (4 * i + 3))
    return morton

def morton_order(transforms, translation_bin_size=0.1, rotation_bin_size=0.01):
    """
    Orders transforms using Morton codes with dynamic grid sizing.
    Args:
        transforms (list): List of transforms [(x, y, z, (w, x, y, z))].
        translation_bin_size (float): Bin size for spatial coordinates in meters.
        rotation_bin_size (float): Bin size for rotation in radians.
    Returns:
        list: Sorted transforms by Morton code.
    """
    morton_codes = []
    for transform in transforms:
        x, y, z = transform[:3]
        q = transform[3]

        # Discretize translations (shift to non-negative range)
        ix = int((x + 10) / translation_bin_size)
        iy = int((y + 10) / translation_bin_size)
        iz = int((z + 10) / translation_bin_size)

        # Discretize rotation
        ir = quaternion_to_rotation_index(q, rotation_bin_size)

        # Compute Morton code
        morton_code = to_morton_code(ix, iy, iz, ir)
        morton_codes.append((morton_code, transform))

    # Sort by Morton code
    morton_codes.sort(key=lambda x: x[0])

    # Return sorted transforms
    return [item[1] for item in morton_codes]

# Example Usage
transforms = [
    (0.1, 0.1, 0.1, (0, 0, 0, 1)),
    (0.4, 0.2, 0.1, (0, 0, 0, 1)),
    (-0.4, -0.2, -0.1, (0, 0, 0, 1)),
    (1.3, 1.3, 1.3, (0.707, 0, 0.707, 0)),
    (3, 3, 3, (0, 0, 0, 1)),
    (0.0, 0.0, 0.0, (0, 0, 0, 1)),
    (-0.4, -0.2, -0.1, (0, 0, 0, 1)),
    (-0.1, -0.1, -0.1, (0, 0, 0, 1)),
    (-0.2, -0.2, -0.2, (0, 0, 0, 1)),
    (-10, -10, -10, (0, 0, 0, 1)),
    (10, 10, 10, (0, 0, 0, 1)),
    (10.001, 10.0001, 10.0001, (0, 0, 0, 1)),
    (20, 20, 20, (0, 0, 0, 1)),
]

sorted_transforms = morton_order(transforms)
print("Sorted Transforms:")
for t in sorted_transforms:
    print(t)
