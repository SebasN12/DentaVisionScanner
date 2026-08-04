"""
Stores one observation of a 3D landmark.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Observation:
    """
    One image observation of a landmark.

    Attributes
    ----------
    frame_name
        Name of the observing frame.

    keypoint_index
        Index of the detected keypoint.

    image_point
        Pixel coordinates (x, y).
    """

    frame_name: str

    keypoint_index: int

    image_point: np.ndarray