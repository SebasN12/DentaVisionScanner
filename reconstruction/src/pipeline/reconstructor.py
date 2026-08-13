"""
Builds a sparse 3D reconstruction from a single image pair.
"""

import numpy as np

from src.core.match_result import MatchResult
from src.core.point_cloud import PointCloud
from src.pipeline.triangulation import Triangulator


class Reconstructor:
    """
    Reconstructs a sparse 3D scene from a single image pair.

    The reconstruction is entirely pairwise. No information is
    accumulated across different image pairs.

    The reconstruction pipeline is:

        MatchResult
            ↓
        Triangulation
            ↓
        Sparse point cloud
    """

    def __init__(
        self,
        triangulator: Triangulator,
    ):
        self.triangulator = triangulator

    def reconstruct(
        self,
        result: MatchResult,
        camera_matrix: np.ndarray,
    ) -> tuple[
        PointCloud,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Reconstructs a single image pair.

        Parameters
        ----------
        result:
            Matching and relative pose estimation result for
            the image pair.

        camera_matrix:
            Camera intrinsic matrix.

        Returns
        -------
        tuple
            A tuple containing:

            - sparse 3D point cloud
            - image points from image 1
            - image points from image 2
        """

        triangulation = (
            self.triangulator.triangulate(
                result,
                camera_matrix,
            )
        )

        return (
            triangulation.point_cloud,
            triangulation.image_points1,
            triangulation.image_points2,
        )