from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Frame:
    filename: str
    path: Path
    image: np.ndarray

    keypoints: list | None = None
    descriptors: np.ndarray | None = None

    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None