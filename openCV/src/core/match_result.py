from dataclasses import dataclass

import cv2
import numpy as np

from .frame import Frame


@dataclass
class MatchResult:
    """
    Stores the matching result between two frames.
    """

    frame1: Frame
    frame2: Frame

    good_matches: list[cv2.DMatch]

    homography: np.ndarray | None = None

    fundamental_matrix: np.ndarray | None = None

    essential_matrix: np.ndarray | None = None

    rotation: np.ndarray | None = None

    translation: np.ndarray | None = None