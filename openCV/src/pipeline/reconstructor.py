"""
Builds a global sparse reconstruction incrementally.
"""

import numpy as np

from src.core.camera_pose import CameraPose
from src.core.reconstruction import Reconstruction
from src.core.point_cloud import PointCloud
from src.core.match_result import MatchResult

from src.pipeline.triangulation import Triangulator


class Reconstructor:
    """
    Incrementally builds a global sparse 3D reconstruction.

    Each new frame pair updates the global model.
    """


    def __init__(
        self,
        triangulator: Triangulator,
    ):

        self.triangulator = triangulator

        self.reconstruction = Reconstruction()

        self.current_pose = CameraPose.identity()

        self.initialized = False

        #
        # Internal accumulated reconstruction data
        #
        self.global_points = []

        self.global_colors = []



    def update_pair(
        self,
        result: MatchResult,
        camera_matrix: np.ndarray,
    ) -> None:
        """
        Adds one new frame pair to the reconstruction.

        Parameters
        ----------
        result
            Matching and pose estimation result.

        camera_matrix
            Camera intrinsic matrix.
        """


        if (
            result.rotation is None
            or result.translation is None
        ):
            raise RuntimeError(
                "Camera pose has not been estimated."
            )


        #
        # Initialize first camera
        #
        if not self.initialized:

            self.reconstruction.add_camera_pose(
                result.frame1.filename,
                self.current_pose,
            )

            self.initialized = True



        #
        # Relative movement between cameras
        #
        relative_pose = CameraPose(
            rotation=result.rotation,
            translation=result.translation,
        )



        #
        # Compute new global camera pose
        #
        self.current_pose = self.compose_pose(
            self.current_pose,
            relative_pose,
        )



        self.reconstruction.add_camera_pose(
            result.frame2.filename,
            self.current_pose,
        )



        #
        # Triangulate local points
        #
        cloud = self.triangulator.triangulate(
            result,
            camera_matrix,
        )



        #
        # Transform points into global coordinates
        #
        points = self.transform_points(
            cloud.points,
            self.current_pose,
        )



        #
        # Accumulate global point cloud
        #
        self.global_points.append(
            points
        )


        if cloud.colors is not None:

            self.global_colors.append(
                cloud.colors
            )



        #
        # Update stored reconstruction
        #
        colors = None

        if len(self.global_colors) > 0:

            colors = np.vstack(
                self.global_colors
            )


        self.reconstruction.set_point_cloud(
            PointCloud(
                points=np.vstack(
                    self.global_points
                ),
                colors=colors,
            )
        )



    def reconstruct(
        self,
        results: list[MatchResult],
        camera_matrix: np.ndarray,
    ) -> Reconstruction:
        """
        Reconstructs a complete sequence.

        This is a batch wrapper around update_pair().
        """


        if len(results) == 0:

            raise ValueError(
                "No match results provided."
            )


        for result in results:

            self.update_pair(
                result,
                camera_matrix,
            )


        return self.reconstruction



    def get_reconstruction(
        self,
    ) -> Reconstruction:
        """
        Returns the current global reconstruction.
        """

        return self.reconstruction



    @staticmethod
    def compose_pose(
        global_pose: CameraPose,
        relative_pose: CameraPose,
    ) -> CameraPose:
        """
        Composes two camera poses.

        T_global_new =
            T_global_current * T_relative
        """


        rotation = (
            global_pose.rotation
            @ relative_pose.rotation
        )


        translation = (
            global_pose.rotation
            @ relative_pose.translation
            +
            global_pose.translation
        )


        return CameraPose(
            rotation=rotation,
            translation=translation,
        )



    @staticmethod
    def transform_points(
        points: np.ndarray,
        pose: CameraPose,
    ) -> np.ndarray:
        """
        Transforms points from camera coordinates
        into global coordinates.
        """


        transformed = (
            pose.rotation
            @ points.T
        ).T


        transformed += (
            pose.translation.reshape(1, 3)
        )


        return transformed