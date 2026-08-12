"""
Bundle Adjustment optimization for pairwise reconstruction.
"""

import cv2
import numpy as np

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from src.optimization.ba_problem import BAProblem


class BundleAdjustment:
    """
    Optimizes the relative pose of the second camera and
    the reconstructed 3D points by minimizing reprojection error.

    The first camera is fixed as the world reference camera:

        R1 = I
        t1 = 0

    The optimization variables are:

        - rotation of camera 2
        - translation of camera 2
        - 3D point positions
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
    ):
        self.camera_matrix = camera_matrix

    def optimize(
        self,
        problem: BAProblem,
    ) -> BAProblem:
        """
        Runs pairwise Bundle Adjustment.

        The first camera remains fixed. The second camera pose
        and all 3D points are optimized.

        Returns
        -------
        BAProblem
            The optimized pairwise reconstruction data.
        """

        self._validate_problem(problem)

        parameters = self._pack_parameters(
            problem
        )

        jacobian_sparsity = (
            self._create_jacobian_sparsity(
                len(problem.points_3d)
            )
        )

        result = least_squares(
            self._residuals,
            parameters,
            args=(problem,),
            jac_sparsity=jacobian_sparsity,
            method="trf",
            verbose=1,
            max_nfev=200,
        )

        if not np.all(np.isfinite(result.x)):
            raise RuntimeError(
                "Bundle Adjustment produced "
                "non-finite parameters."
            )

        return self._build_optimized_problem(
            problem,
            result.x,
        )

    def _validate_problem(
        self,
        problem: BAProblem,
    ) -> None:
        """
        Validates the data required for optimization.
        """

        if len(problem.points_3d) == 0:
            raise RuntimeError(
                "No 3D points available for Bundle Adjustment."
            )

        if len(problem.image_points1) != len(
            problem.points_3d
        ):
            raise ValueError(
                "Number of observations in image 1 "
                "does not match number of 3D points."
            )

        if len(problem.image_points2) != len(
            problem.points_3d
        ):
            raise ValueError(
                "Number of observations in image 2 "
                "does not match number of 3D points."
            )

        if problem.rotation.shape != (3, 3):
            raise ValueError(
                "Camera rotation must have shape (3, 3)."
            )

        if problem.translation.size != 3:
            raise ValueError(
                "Camera translation must contain 3 values."
            )

    def _pack_parameters(
        self,
        problem: BAProblem,
    ) -> np.ndarray:
        """
        Converts the pairwise reconstruction into
        an optimization vector.

        Layout:

            [rvec2]
            [tvec2]
            [X0]
            [X1]
            ...
            [XN]
        """

        rotation_vector, _ = cv2.Rodrigues(
            problem.rotation
        )

        parameters = [
            *rotation_vector.reshape(3),
            *problem.translation.reshape(3),
        ]

        for point in problem.points_3d:
            parameters.extend(
                point.reshape(3)
            )

        return np.asarray(
            parameters,
            dtype=np.float64,
        )

    def _residuals(
        self,
        parameters: np.ndarray,
        problem: BAProblem,
    ) -> np.ndarray:
        """
        Computes reprojection errors for both cameras.

        Camera 1 is fixed as:

            R1 = I
            t1 = 0

        Camera 2 is taken from the optimization parameters.
        """

        rotation_vector = parameters[:3]

        translation = parameters[
            3:6
        ].reshape(3, 1)

        points = parameters[
            6:
        ].reshape(-1, 3)

        projected1, _ = cv2.projectPoints(
            points,
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            self.camera_matrix,
            None,
        )

        projected2, _ = cv2.projectPoints(
            points,
            rotation_vector,
            translation,
            self.camera_matrix,
            None,
        )

        projected1 = projected1.reshape(-1, 2)
        projected2 = projected2.reshape(-1, 2)

        error1 = (
            projected1
            -
            problem.image_points1
        )

        error2 = (
            projected2
            -
            problem.image_points2
        )

        return np.concatenate(
            (
                error1.reshape(-1),
                error2.reshape(-1),
            )
        )

    def _create_jacobian_sparsity(
        self,
        n_points: int,
    ):
        """
        Creates the sparse Jacobian structure.

        Each observation depends on:

            - camera 2 parameters
            - its corresponding 3D point

        Camera 1 is fixed and therefore has no
        optimization parameters.
        """

        camera_parameter_count = 6
        point_parameter_count = 3

        n_parameters = (
            camera_parameter_count
            +
            n_points * point_parameter_count
        )

        n_residuals = (
            n_points * 4
        )

        sparsity = lil_matrix(
            (
                n_residuals,
                n_parameters,
            ),
            dtype=int,
        )

        for i in range(n_points):

            point_start = (
                camera_parameter_count
                +
                i * point_parameter_count
            )

            #
            # Camera 1 residuals
            #
            residual1_start = i * 2

            sparsity[
                residual1_start:
                residual1_start + 2,
                point_start:
                point_start + 3,
            ] = 1

            #
            # Camera 2 residuals
            #
            residual2_start = (
                n_points * 2
                +
                i * 2
            )

            sparsity[
                residual2_start:
                residual2_start + 2,
                :camera_parameter_count,
            ] = 1

            sparsity[
                residual2_start:
                residual2_start + 2,
                point_start:
                point_start + 3,
            ] = 1

        return sparsity.tocsr()

    def _build_optimized_problem(
        self,
        problem: BAProblem,
        parameters: np.ndarray,
    ) -> BAProblem:
        """
        Converts optimized parameters back into a BAProblem.
        """

        rotation_vector = parameters[
            :3
        ]

        translation = parameters[
            3:6
        ].reshape(3, 1)

        rotation, _ = cv2.Rodrigues(
            rotation_vector
        )

        points_3d = parameters[
            6:
        ].reshape(-1, 3)

        return BAProblem(
            rotation=rotation,
            translation=translation,
            points_3d=points_3d,
            image_points1=problem.image_points1,
            image_points2=problem.image_points2,
        )