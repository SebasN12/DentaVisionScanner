"""
Builds a global sparse reconstruction from multiple frames.
"""

import numpy as np

from src.core.camera_pose import CameraPose
from src.core.reconstruction import Reconstruction
from src.core.point_cloud import PointCloud
from src.core.match_result import MatchResult

from src.pipeline.triangulation import Triangulator


class Reconstructor:
    """
    Builds a global sparse 3D reconstruction.

    This class combines relative camera poses and local
    triangulated point clouds into a single global model.
    """

    def __init__(
        self,
        triangulator: Triangulator,
    ):
        self.triangulator = triangulator


    def reconstruct(
        self,
        results: list[MatchResult],
        camera_matrix: np.ndarray,
    ) -> Reconstruction:
        """
        Creates a global reconstruction.

        Parameters
        ----------
        results
            List of MatchResult objects between consecutive frames.

        camera_matrix
            Camera intrinsic matrix.

        Returns
        -------
        Reconstruction
            Global sparse reconstruction.
        """

        reconstruction = Reconstruction()


        if len(results) == 0:
            raise ValueError(
                "No match results provided."
            )


        #
        # First camera defines the world coordinate system
        #
        first_frame = results[0].frame1

        current_pose = CameraPose.identity()

        reconstruction.add_camera_pose(
            first_frame.filename,
            current_pose,
        )


        global_points = []
        global_colors = []


        for result in results:


            #
            # Estimate pose of next camera
            #
            relative_pose = CameraPose(
                rotation=result.rotation,
                translation=result.translation,
            )


            current_pose = self.compose_pose(
                current_pose,
                relative_pose,
            )


            reconstruction.add_camera_pose(
                result.frame2.filename,
                current_pose,
            )


            #
            # Triangulate local points
            #
            cloud = self.triangulator.triangulate(
                result,
                camera_matrix,
            )


            #
            # Transform points from camera coordinates
            # into world coordinates
            #
            points = self.transform_points(
                cloud.points,
                current_pose,
            )


            global_points.append(points)


            if cloud.colors is not None:

                global_colors.append(
                    cloud.colors
                )


        #
        # Merge all point clouds
        #
        if len(global_points) > 0:

            points = np.vstack(
                global_points
            )

            colors = None

            if len(global_colors) > 0:

                colors = np.vstack(
                    global_colors
                )

            reconstruction.set_point_cloud(
                PointCloud(
                    points=points,
                    colors=colors,
                )
            )


        return reconstruction


    @staticmethod
    def compose_pose(
        global_pose: CameraPose,
        relative_pose: CameraPose,
    ) -> CameraPose:
        """
        Composes two camera poses.

        Computes:

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
        Transforms points from local camera coordinates
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