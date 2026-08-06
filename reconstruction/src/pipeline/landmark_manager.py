"""
Manages the creation and storage of 3D landmarks.
"""

import numpy as np

from src.core.landmark import Landmark
from src.core.observation import Observation
from src.core.track import Track
from src.core.frame import Frame



class LandmarkManager:
    """
    Creates and manages reconstructed landmarks.

    A landmark represents a 3D point together with
    all image observations that correspond to it.
    """



    def __init__(
        self,
    ):

        self.landmarks: dict[int, Landmark] = {}

        self.next_id = 0



    def create_landmark(
        self,
        position: np.ndarray,
        color: np.ndarray | None = None,
        track_id: int | None = None,
    ) -> Landmark:
        """
        Creates and stores a new landmark.
        """


        landmark = Landmark(
            id=self.next_id,
            position=position,
            color=color,
            track_id=track_id,
        )


        self.landmarks[
            self.next_id
        ] = landmark


        self.next_id += 1


        return landmark



    def create_from_track(
        self,
        track: Track,
        position: np.ndarray,
        frames: dict[str, Frame],
        camera_ids: dict[str, int],
        color: np.ndarray | None = None,
    ) -> Landmark:
        """
        Creates a landmark from a track.
        """


        landmark = self.create_landmark(
            position,
            color,
            track.id,
        )


        track.landmark_id = landmark.id


        self.add_track_observations(
            landmark,
            track,
            frames,
            camera_ids,
        )


        return landmark


    def add_track_observations(
        self,
        landmark: Landmark,
        track: Track,
        frames: dict[str, Frame],
        camera_ids: dict[str, int],
    ) -> None:
        """
        Adds only new observations.
        """


        existing = {
            (
                obs.frame_name,
                obs.keypoint_index,
            )
            for obs in landmark.observations
        }


        for frame_name, keypoint_index in track.observations:


            if (
                frame_name,
                keypoint_index,
            ) in existing:

                continue


            frame = frames.get(
                frame_name
            )


            if frame is None:
                continue


            keypoint = frame.keypoints[
                keypoint_index
            ]


            observation = Observation(
                frame_name=frame_name,
                camera_id=camera_ids[frame_name],
                keypoint_index=keypoint_index,
                image_point=np.asarray(
                    keypoint.pt
                ),
            )


            landmark.add_observation(
                observation
            )


    def add_observation(
        self,
        landmark: Landmark,
        observation: Observation,
    ) -> None:
        """
        Adds an observation to an existing landmark.
        """


        landmark.add_observation(
            observation
        )



    def get_landmark(
        self,
        landmark_id: int,
    ) -> Landmark | None:
        """
        Returns a landmark by id.
        """


        return self.landmarks.get(
            landmark_id
        )



    def get_landmarks(
        self,
    ) -> list[Landmark]:
        """
        Returns all stored landmarks.
        """


        return list(
            self.landmarks.values()
        )
    
    def get_or_create_from_track(
        self,
        track: Track,
        position: np.ndarray,
        frames: dict[str, Frame],
        camera_ids: dict[str, int],
        color=None,
    ):
        """
        Returns existing landmark or creates a new one.

        If the landmark already exists,
        missing observations are added.
        """


        if track.landmark_id is not None:

            landmark = self.landmarks[
                track.landmark_id
            ]


            self.add_track_observations(
                landmark,
                track,
                frames,
                camera_ids,
            )


            return landmark



        return self.create_from_track(
            track,
            position,
            frames,
            camera_ids,
            color,
        )


    def clear(
        self,
    ) -> None:
        """
        Removes all landmarks.
        """

        self.landmarks.clear()

        self.next_id = 0