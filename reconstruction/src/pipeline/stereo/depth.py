"""
Depth reconstruction from stereo disparity.
"""

import numpy as np


class DepthReconstructor:
    """
    Converts a stereo disparity map into a depth map.

    Depth is computed using the standard stereo relationship:

        Z = (f * B) / d

    where:

    - Z is the depth from the camera.
    - f is the focal length in pixels.
    - B is the stereo baseline.
    - d is the disparity in pixels.
    """

    def __init__(
        self,
        focal_length: float,
        baseline: float,
    ):
        """
        Initializes the depth reconstructor.

        Parameters
        ----------
        focal_length:
            Camera focal length in pixels.

        baseline:
            Distance between the two cameras.
            The unit determines the unit of the resulting
            depth map.
        """

        if focal_length <= 0:
            raise ValueError(
                "focal_length must be positive."
            )

        if baseline <= 0:
            raise ValueError(
                "baseline must be positive."
            )

        self.focal_length = float(focal_length)
        self.baseline = float(baseline)

    def compute(
        self,
        disparity: np.ndarray,
    ) -> np.ndarray:
        """
        Computes a depth map from a disparity map.

        Parameters
        ----------
        disparity:
            Disparity map in pixels.

            Invalid or non-positive disparity values are
            treated as invalid and produce zero depth.

        Returns
        -------
        np.ndarray
            Depth map with the same shape as the disparity map.

            Depth is expressed in the same unit as the
            baseline.
        """

        if disparity is None:
            raise ValueError(
                "Disparity must not be None."
            )

        if disparity.ndim != 2:
            raise ValueError(
                "Disparity must be a 2D array."
            )

        disparity = disparity.astype(
            np.float32,
            copy=False,
        )

        depth = np.zeros_like(
            disparity,
            dtype=np.float32,
        )

        valid_mask = (
            np.isfinite(disparity)
            & (disparity > 0)
        )

        depth[valid_mask] = (
            self.focal_length
            * self.baseline
            / disparity[valid_mask]
        )

        return depth