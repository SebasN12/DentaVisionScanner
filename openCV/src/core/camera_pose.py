"""
Camera pose represents world-to-camera transformation.

Transforms:

X_camera = R * X_world + t

Compatible with OpenCV projectPoints.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraPose:
    """
    Represents a world-to-camera transformation.

    The transformation is:

        X_camera = R * X_world + t

    This convention is compatible with:

        cv2.projectPoints()

    Attributes
    ----------
    rotation
        Rotation matrix R (world to camera).

    translation
        Translation vector t (world to camera).
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