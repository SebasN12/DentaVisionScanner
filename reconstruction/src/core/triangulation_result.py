"""
Stores the result of pairwise 3D triangulation.
"""

from dataclasses import dataclass

import numpy as np

from src.core.point_cloud import PointCloud


@dataclass(slots=True)
class TriangulationResult:
    """
    Stores the result of triangulating a single image pair.

    The arrays are aligned by index:

        point_cloud.points[i]
            ↕
        image_points1[i]
            ↕
        image_points2[i]

    Each index therefore represents one reconstructed 3D
    point and its corresponding observations in both images.

    This correspondence is required by pairwise
    Bundle Adjustment.
    """

    point_cloud: PointCloud

    image_points1: np.ndarray

    image_points2: np.ndarray