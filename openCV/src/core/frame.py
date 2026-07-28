from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Frame:
    filename: str
    path: Path
    image: np.ndarray

    keypoints: list[cv2.KeyPoint] | None = None
    descriptors: np.ndarray | None = None

    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None