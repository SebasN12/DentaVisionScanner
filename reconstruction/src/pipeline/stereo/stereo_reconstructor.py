"""
3D reconstruction from stereo disparity.
"""

import cv2
import numpy as np

from src.core.point_cloud import PointCloud


class StereoReconstructor:
    """
    Reconstructs a 3D point cloud from a stereo disparity map.

    The reconstruction uses OpenCV's reprojectImageTo3D()
    function and therefore requires a reprojection matrix Q.

    The Q matrix is provided externally so that this component
    remains independent of any specific camera or dataset.
    """

    def __init__(
        self,
        reprojection_matrix: np.ndarray,
    ):
        """
        Initializes the stereo reconstructor.

        Parameters
        ----------
        reprojection_matrix:
            4x4 stereo reprojection matrix Q.
        """

        if reprojection_matrix.shape != (4, 4):
            raise ValueError(
                "reprojection_matrix must have shape (4, 4)."
            )

        self.reprojection_matrix = (
            reprojection_matrix.astype(
                np.float64,
                copy=False,
            )
        )

    def reconstruct(
        self,
        disparity: np.ndarray,
        image: np.ndarray,
        valid_mask: np.ndarray,
    ) -> PointCloud:
        """
        Reconstructs a 3D point cloud from a disparity map.

        Parameters
        ----------
        disparity:
            Rectified stereo disparity map in pixels.

        image:
            Left stereo image used to assign colors
            to the reconstructed points.

        valid_mask:
            Boolean mask indicating which disparity pixels
            should be included in the point cloud.

        Returns
        -------
        PointCloud
            Reconstructed 3D points and corresponding colors.
        """

        if disparity is None:
            raise ValueError(
                "Disparity must not be None."
            )

        if image is None:
            raise ValueError(
                "Image must not be None."
            )

        if valid_mask is None:
            raise ValueError(
                "valid_mask must not be None."
            )

        if disparity.ndim != 2:
            raise ValueError(
                "Disparity must be a 2D array."
            )

        if image.shape[:2] != disparity.shape:
            raise ValueError(
                "Image and disparity must have the same dimensions."
            )

        if valid_mask.shape != disparity.shape:
            raise ValueError(
                "valid_mask must have the same shape "
                "as disparity."
            )

        if valid_mask.dtype != np.bool_:
            raise ValueError(
                "valid_mask must be a boolean array."
            )

        disparity = disparity.astype(
            np.float32,
            copy=False,
        )

        # --------------------------------------------------------------
        # Reconstruct 3D coordinates.
        # --------------------------------------------------------------

        points_3d = cv2.reprojectImageTo3D(
            disparity,
            self.reprojection_matrix,
        )

        # --------------------------------------------------------------
        # Select valid 3D points.
        # --------------------------------------------------------------

        valid_points = points_3d[valid_mask]

        # --------------------------------------------------------------
        # Remove invalid 3D coordinates.
        # --------------------------------------------------------------

        finite_points = np.all(
            np.isfinite(valid_points),
            axis=1,
        )

        valid_points = valid_points[
            finite_points
        ]

        # --------------------------------------------------------------
        # Extract corresponding colors.
        # --------------------------------------------------------------

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        colors = rgb_image[valid_mask]

        colors = colors[
            finite_points
        ]

        colors = (
            colors.astype(np.float64)
            / 255.0
        )

        return PointCloud(
            points=valid_points,
            colors=colors,
        )