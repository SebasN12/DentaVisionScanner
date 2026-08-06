"""
Bundle Adjustment optimization.
"""

import cv2
import numpy as np
from scipy.sparse import lil_matrix

from scipy.optimize import least_squares

from src.core.reconstruction import Reconstruction
from src.core.camera_pose import CameraPose

from src.optimization.ba_problem import (
    BAProblem,
    BAObservation,
)


class BundleAdjustment:
    """
    Optimizes camera poses and landmark positions
    by minimizing reprojection error.
    """


    def __init__(
        self,
        camera_matrix: np.ndarray,
    ):
        self.camera_matrix = camera_matrix


    def optimize(
        self,
        reconstruction: Reconstruction,
    ) -> None:
        """
        Runs Bundle Adjustment and updates the reconstruction.
        """

        problem = self._build_problem(
            reconstruction
        )


        if len(problem.observations) == 0:
            raise RuntimeError(
                "No observations available for Bundle Adjustment."
            )


        parameters = self._pack_parameters(
            reconstruction,
            problem,
        )

        jacobian_sparsity = (
            self._create_jacobian_sparsity(
                problem
            )
        )


        result = least_squares(
            self._residuals,
            parameters,
            args=(
                problem,
                reconstruction,
            ),
            jac_sparsity=jacobian_sparsity,
            method="trf",
            verbose=1,
            max_nfev=200
        )


        self._write_back(
            reconstruction,
            problem,
            result.x,
        )


    def _build_problem(
        self,
        reconstruction: Reconstruction,
    ) -> BAProblem:
        """
        Creates compact optimization indices.
        """

        frame_names = list(
            reconstruction.camera_poses.keys()
        )


        camera_index = {}

        for index, frame_name in enumerate(
            frame_names
        ):
            camera_id = reconstruction.camera_ids[
                frame_name
            ]

            camera_index[camera_id] = index



        landmark_ids = list(
            reconstruction.landmarks.keys()
        )


        landmark_index = {}

        for index, landmark_id in enumerate(
            landmark_ids
        ):
            landmark_index[
                landmark_id
            ] = index



        observations = []


        for landmark_id, landmark in (
            reconstruction.landmarks.items()
        ):

            if landmark_id not in landmark_index:
                continue


            l_index = landmark_index[
                landmark_id
            ]


            for obs in landmark.observations:

                if obs.camera_id not in camera_index:
                    continue


                observations.append(
                    BAObservation(
                        camera_index=camera_index[
                            obs.camera_id
                        ],

                        landmark_index=l_index,

                        image_point=obs.image_point,
                    )
                )


        return BAProblem(
            frame_names=frame_names,

            landmark_ids=landmark_ids,

            camera_index=camera_index,

            landmark_index=landmark_index,

            observations=observations,
        )


    def _pack_parameters(
        self,
        reconstruction: Reconstruction,
        problem: BAProblem,
    ) -> np.ndarray:
        """
        Converts poses and landmarks into optimization vector.
        """

        parameters = []


        # Cameras
        for frame_name in problem.frame_names:

            pose = reconstruction.camera_poses[
                frame_name
            ]


            rotation_vector, _ = cv2.Rodrigues(
                pose.rotation
            )


            parameters.extend(
                rotation_vector.reshape(3)
            )

            parameters.extend(
                pose.translation.reshape(3)
            )



        # Landmarks
        for landmark_id in problem.landmark_ids:

            landmark = reconstruction.landmarks[
                landmark_id
            ]

            parameters.extend(
                landmark.position.reshape(3)
            )


        return np.asarray(
            parameters,
            dtype=np.float64,
        )


    def _residuals(
        self,
        parameters: np.ndarray,
        problem: BAProblem,
        reconstruction: Reconstruction,
    ) -> np.ndarray:
        """
        Computes reprojection errors.
        """

        residuals = []


        camera_block_size = 6

        landmark_start = (
            len(problem.frame_names)
            *
            camera_block_size
        )


        for observation in problem.observations:


            camera_offset = (
                observation.camera_index
                *
                camera_block_size
            )


            rvec = parameters[
                camera_offset:
                camera_offset + 3
            ]


            tvec = parameters[
                camera_offset + 3:
                camera_offset + 6
            ]



            landmark_offset = (
                landmark_start
                +
                observation.landmark_index
                *
                3
            )


            point = parameters[
                landmark_offset:
                landmark_offset + 3
            ]


            projected, _ = cv2.projectPoints(
                point.reshape(1, 3),
                rvec,
                tvec.reshape(3, 1),
                self.camera_matrix,
                None,
            )


            projected = projected.reshape(2)


            error = (
                projected
                -
                observation.image_point
            )


            residuals.extend(
                error
            )


        return np.asarray(
            residuals
        )


    def _write_back(
        self,
        reconstruction: Reconstruction,
        problem: BAProblem,
        parameters: np.ndarray,
    ) -> None:
        """
        Writes optimized values back into reconstruction.
        """

        camera_block_size = 6


        for index, frame_name in enumerate(
            problem.frame_names
        ):

            offset = (
                index
                *
                camera_block_size
            )


            rvec = parameters[
                offset:
                offset + 3
            ]


            tvec = parameters[
                offset + 3:
                offset + 6
            ]


            rotation, _ = cv2.Rodrigues(
                rvec
            )


            reconstruction.camera_poses[
                frame_name
            ] = CameraPose(
                rotation=rotation,

                translation=tvec.reshape(3, 1),
            )



        landmark_start = (
            len(problem.frame_names)
            *
            camera_block_size
        )


        for index, landmark_id in enumerate(
            problem.landmark_ids
        ):

            offset = (
                landmark_start
                +
                index
                *
                3
            )


            reconstruction.landmarks[
                landmark_id
            ].position = parameters[
                offset:
                offset + 3
            ]

    def _create_jacobian_sparsity(
        self,
        problem: BAProblem,
    ):
        """
        Creates sparse Jacobian structure.

        Each observation depends only on:
        - one camera
        - one landmark
        """

        


        n_cameras = len(
            problem.frame_names
        )

        n_landmarks = len(
            problem.landmark_ids
        )


        n_parameters = (
            n_cameras * 6
            +
            n_landmarks * 3
        )


        n_residuals = (
            len(problem.observations)
            *
            2
        )


        sparsity = lil_matrix(
            (
                n_residuals,
                n_parameters,
            ),
            dtype=int,
        )


        for i, observation in enumerate(
            problem.observations
        ):

            residual_index = i * 2


            #
            # Camera parameters
            #
            camera_start = (
                observation.camera_index
                *
                6
            )


            sparsity[
                residual_index:
                residual_index + 2,
                camera_start:
                camera_start + 6,
            ] = 1



            #
            # Landmark parameters
            #
            landmark_start = (
                n_cameras * 6
                +
                observation.landmark_index
                *
                3
            )


            sparsity[
                residual_index:
                residual_index + 2,
                landmark_start:
                landmark_start + 3,
            ] = 1


        return sparsity.tocsr()