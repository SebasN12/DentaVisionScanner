"""
Stores a reconstructed 3D point cloud.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PointCloud:
    """
    Represents a reconstructed 3D point cloud.
    """

    points: np.ndarray

    colors: np.ndarray | None = None