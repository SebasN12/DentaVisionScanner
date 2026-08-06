"""
Builds a global sparse reconstruction incrementally.
"""

import numpy as np

from src.core.camera_pose import CameraPose
from src.core.reconstruction import Reconstruction
from src.core.point_cloud import PointCloud
from src.core.match_result import MatchResult

from src.pipeline.triangulation import Triangulator
from src.pipeline.track_builder import TrackBuilder
from src.pipeline.landmark_manager import LandmarkManager



class Reconstructor:
    """
    Incrementally builds a global sparse 3D reconstruction.

    Images are processed sequentially.
    """



    def __init__(
        self,
        triangulator: Triangulator,
        track_builder: TrackBuilder,
        landmark_manager: LandmarkManager,
    ):

        self.triangulator = triangulator

        self.track_builder = track_builder

        self.landmark_manager = landmark_manager


        self.reconstruction = Reconstruction()


        self.current_pose = CameraPose.identity()


        self.initialized = False


        #
        # Frames known by reconstruction
        #
        self.frames = {}



    def update_pair(
        self,
        result: MatchResult,
        camera_matrix: np.ndarray,
    ) -> None:
        """
        Integrates one new consecutive image pair.
        """


        if (
            result.rotation is None
            or result.translation is None
        ):
            raise RuntimeError(
                "Camera pose has not been estimated."
            )



        #
        # Store frames
        #
        self.frames[
            result.frame1.filename
        ] = result.frame1


        self.frames[
            result.frame2.filename
        ] = result.frame2



        #
        # Add feature correspondences
        #
        self.track_builder.add_matches(
            result
        )


        #
        # Update tracks after new unions
        #
        self.track_builder.update_tracks()



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
        # Compute new global camera pose
        #
        relative_pose = CameraPose(
            rotation=result.rotation,
            translation=result.translation,
        )


        self.current_pose = self.compose_pose(
            self.current_pose,
            relative_pose,
        )


        self.reconstruction.add_camera_pose(
            result.frame2.filename,
            self.current_pose,
        )



        #
        # Triangulate current pair
        #
        cloud = self.triangulator.triangulate(
            result,
            camera_matrix,
        )



        # points = self.transform_points(
        #     cloud.points,
        #     self.current_pose,
        # )
        # -> possible bug

        frame1_pose = self.reconstruction.get_camera_pose(
            result.frame1.filename
        )

        if frame1_pose is None:
            raise RuntimeError(
                "Frame 1 pose not found."
            )

        points = self.transform_points(
            cloud.points,
            frame1_pose,
        )

        # debug
        print("PAIR")
        print(
            result.frame1.filename,
            "->",
            result.frame2.filename
        )

        print("Frame1 pose:")
        print(frame1_pose.rotation)
        print(frame1_pose.translation.reshape(3))

        print("World point:")
        print(points[0])
        # debug



        #
        # Associate triangulated points
        # with their corresponding tracks
        #
        if result.inlier_matches is not None:

            for index, match in enumerate(
                result.inlier_matches
            ):

                if index >= len(points):
                    break


                observation = (
                    result.frame1.filename,
                    match.queryIdx,
                )


                track_id = (
                    self.track_builder.get_track_id(
                        observation[0],
                        observation[1],
                    )
                )


                if track_id is None:
                    continue



                track = (
                    self.track_builder.get_track(
                        track_id
                    )
                )


                if track is None:
                    continue



                #
                # Create landmark only once
                #

                # DEBUG
                if index == 0:
                    print()
                    print("Pair:")
                    print(result.frame1.filename, "->", result.frame2.filename)

                    print("First triangulated point:")
                    print(points[0])

                    print("Frame1 global translation:")
                    print(frame1_pose.translation.reshape(3))
                # /DEBUG

                landmark = self.landmark_manager.get_or_create_from_track(
                    track,
                    points[index],
                    self.frames,
                    self.reconstruction.camera_ids,
                    (
                        cloud.colors[index]
                        if cloud.colors is not None
                        else None
                    ),
                )


                if (
                    landmark.id
                    not in self.reconstruction.landmarks
                ):

                    self.reconstruction.add_landmark(
                        landmark
                    )



        #
        # Update point cloud
        #
        self.update_point_cloud()



    def update_point_cloud(
        self,
    ) -> None:
        """
        Updates point cloud from landmarks.
        """


        landmarks = (
            self.reconstruction.landmarks.values()
        )


        points = []

        colors = []


        for landmark in landmarks:

            points.append(
                landmark.position
            )


            if landmark.color is not None:

                colors.append(
                    landmark.color
                )



        if len(points) == 0:

            return



        color_array = None


        if len(colors) == len(points):

            color_array = np.asarray(
                colors
            )



        self.reconstruction.set_point_cloud(
            PointCloud(
                points=np.asarray(points),
                colors=color_array,
            )
        )



    def reconstruct(
        self,
        results: list[MatchResult],
        camera_matrix: np.ndarray,
    ) -> Reconstruction:
        """
        Offline wrapper using sequential reconstruction.
        """


        for result in results:

            self.update_pair(
                result,
                camera_matrix,
            )


        return self.reconstruction



    def get_reconstruction(
        self,
    ) -> Reconstruction:

        return self.reconstruction



    @staticmethod
    def compose_pose(
        global_pose: CameraPose,
        relative_pose: CameraPose,
    ) -> CameraPose:
        """
        Composes:

            world -> current camera

        with:

            current camera -> next camera

        resulting in:

            world -> next camera
        """

        rotation = (
            relative_pose.rotation
            @ global_pose.rotation
        )


        translation = (
            relative_pose.rotation
            @ global_pose.translation
            +
            relative_pose.translation
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
        Converts camera coordinates into world coordinates.

        Camera pose stores:

            X_camera = R * X_world + t

        Therefore inverse is:

            X_world = R.T * (X_camera - t)
        """

        points_centered = (
            points
            -
            pose.translation.reshape(1, 3)
        )


        transformed = (
            pose.rotation.T
            @ points_centered.T
        ).T


        return transformed