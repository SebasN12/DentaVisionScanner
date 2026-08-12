"""
Builds a sparse 3D reconstruction from a single image pair.
"""

import numpy as np

from src.core.match_result import MatchResult
from src.core.point_cloud import PointCloud

from src.pipeline.triangulation import Triangulator

from src.optimization.ba_problem import BAProblem
from src.optimization.bundle_adjustment import BundleAdjustment


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
        Pairwise Bundle Adjustment
            ↓
        Optimized pairwise reconstruction
    """

    def __init__(
        self,
        triangulator: Triangulator,
        bundle_adjustment: BundleAdjustment,
    ):
        self.triangulator = triangulator
        self.bundle_adjustment = bundle_adjustment

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

            - optimized 3D point cloud
            - image points from image 1
            - image points from image 2
        """

        triangulation = (
            self.triangulator.triangulate(
                result,
                camera_matrix,
            )
        )

        problem = BAProblem(
            rotation=result.rotation,
            translation=result.translation,
            points_3d=(
                triangulation.point_cloud.points
            ),
            image_points1=(
                triangulation.image_points1
            ),
            image_points2=(
                triangulation.image_points2
            ),
        )

        optimized_problem = (
            self.bundle_adjustment.optimize(
                problem
            )
        )

        point_cloud = PointCloud(
            points=optimized_problem.points_3d,
            colors=triangulation.point_cloud.colors,
        )

        return (
            point_cloud,
            optimized_problem.image_points1,
            optimized_problem.image_points2,
        )