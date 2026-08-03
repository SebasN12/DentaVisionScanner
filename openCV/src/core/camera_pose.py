"""
Stores the pose of a camera in the global coordinate system.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraPose:
    """
    Represents the pose of a camera.

    Attributes
    ----------
    rotation
        3x3 rotation matrix.

    translation
        3x1 translation vector.
    """

    rotation: np.ndarray
    translation: np.ndarray

    @staticmethod
    def identity() -> "CameraPose":
        """
        Returns the world reference camera pose.
        """

        return CameraPose(
            rotation=np.eye(3),
            translation=np.zeros((3, 1)),
        )

    def transformation_matrix(
        self,
    ) -> np.ndarray:
        """
        Returns the homogeneous 4x4 transformation matrix.
        """

        matrix = np.eye(4)

        matrix[:3, :3] = self.rotation

        matrix[:3, 3] = self.translation.reshape(3)

        return matrix