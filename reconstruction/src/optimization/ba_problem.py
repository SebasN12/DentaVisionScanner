"""
Internal data structures used by Bundle Adjustment.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class BAObservation:
    """
    Observation used internally by Bundle Adjustment.

    Uses compact indices instead of persistent IDs.
    """

    camera_index: int

    landmark_index: int

    image_point: np.ndarray


@dataclass(slots=True)
class BAProblem:
    """
    Internal optimization representation.

    Converts Reconstruction data into compact arrays
    suitable for numerical optimization.
    """

    frame_names: list[str]

    landmark_ids: list[int]

    camera_index: dict[int, int]

    landmark_index: dict[int, int]

    observations: list[BAObservation]