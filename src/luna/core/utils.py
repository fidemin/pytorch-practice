from typing import Tuple

import numpy as np


def irc2xyz(
    irc_point: Tuple[int, int, int],
    xyz_origin: Tuple[float, float, float],
    xyz_spacing: Tuple[float, float, float, float, float, float, float, float, float],
    direction,
) -> Tuple[float, float, float]:
    """
    Convert (i, r, c) voxel indices to physical (x, y, z) coordinates.

    Args:
        irc_point: Tuple of (i, r, c) voxel indices.
        xyz_origin: Tuple representing the origin of the image (x0, y0, z0).
        xyz_spacing: Tuple representing the voxel size (sx, sy, sz).
        direction: 9 float tuple representing the image orientation.

    Returns:
        Tuple of (x, y, z) physical coordinates.
    """
    # Convert inputs to NumPy arrays
    irc_point = np.array(irc_point)
    xyz_origin = np.array(xyz_origin)
    xyz_spacing = np.array(xyz_spacing)
    direction = np.array(direction).reshape(3, 3)

    # Convert voxel coordinates to physical coordinates
    xyz_coords = direction @ (xyz_spacing * irc_point) + xyz_origin
    return tuple(xyz_coords.astype(np.float32))


def xyz2irc(
    xyz_point: Tuple[float, float, float],
    xyz_origin: Tuple[float, float, float],
    xyz_spacing: Tuple[float, float, float],
    direction: Tuple[float, float, float, float, float, float, float, float, float],
) -> Tuple[int, int, int]:
    """
    Convert physical (x, y, z) coordinates to (i, r, c) voxel indices.

    Args:
        xyz_point: Tuple of (x, y, z) physical coordinates.
        xyz_origin: Tuple representing the origin of the image (x0, y0, z0).
        xyz_spacing: Tuple representing the voxel size (sx, sy, sz).
        direction: 9 float tuple representing the image orientation.

    Returns:
        Tuple of (i, r, c) voxel indices.
    """
    # Convert inputs to NumPy arrays
    xyz_point = np.array(xyz_point)
    xyz_origin = np.array(xyz_origin)
    xyz_spacing = np.array(xyz_spacing)
    direction = np.array(direction).reshape(3, 3)

    # Convert physical coordinates to voxel coordinates
    voxel_coords = np.linalg.inv(direction) @ (xyz_point - xyz_origin) / xyz_spacing

    # Round to nearest integer since voxel indices must be integers
    irc = tuple(np.round(voxel_coords).astype(int))
    return irc
