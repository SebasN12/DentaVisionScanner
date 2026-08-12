"""
Internal data structures used by pairwise Bundle Adjustment.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class BAProblem:
    """
    Internal optimization representation for pairwise Bundle Adjustment.

    The first camera is fixed as the world reference camera.
    The second camera pose and the 3D points are optimized.
    """

    rotation: np.ndarray

    translation: np.ndarray

    points_3d: np.ndarray

    image_points1: np.ndarray

    image_points2: np.ndarray