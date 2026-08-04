"""
Builds feature tracks from pairwise image matches.
"""

from src.core.match_result import MatchResult
from src.core.track import Track



class TrackBuilder:
    """
    Builds feature tracks using a Union-Find structure.

    Each observation is represented as:

        (frame_name, keypoint_index)

    Observations that correspond to the same 3D point
    are grouped into a single Track.
    """



    def __init__(
        self,
    ):

        self.parent: dict[
            tuple[str, int],
            tuple[str, int],
        ] = {}


        #
        # Stores:
        #
        # (frame, keypoint)
        #          |
        #          ↓
        #       track id
        #
        self.observation_to_track: dict[
            tuple[str, int],
            int,
        ] = {}


        self.tracks: dict[
            int,
            Track,
        ] = {}

        self.next_track_id = 0



    def find(
        self,
        node: tuple[str, int],
    ) -> tuple[str, int]:
        """
        Finds the representative of a set.
        """


        if node not in self.parent:

            self.parent[node] = node

            return node


        if self.parent[node] != node:

            self.parent[node] = self.find(
                self.parent[node]
            )


        return self.parent[node]



    def union(
        self,
        first: tuple[str, int],
        second: tuple[str, int],
    ) -> None:
        """
        Merges two observations.
        """


        root1 = self.find(first)

        root2 = self.find(second)


        if root1 != root2:

            self.parent[root2] = root1



    def add_matches(
        self,
        result: MatchResult,
    ) -> None:
        """
        Inserts inlier matches into the Union-Find structure.
        """


        if result.inlier_matches is None:

            return


        frame1 = result.frame1.filename

        frame2 = result.frame2.filename


        for match in result.inlier_matches:


            observation1 = (
                frame1,
                match.queryIdx,
            )


            observation2 = (
                frame2,
                match.trainIdx,
            )


            self.union(
                observation1,
                observation2,
            )



    def build_tracks_from_batch(
        self,
    ) -> list[Track]:
        """
        Converts Union-Find groups into Track objects.

        Also builds an index from observations to tracks.
        """


        groups: dict[
            tuple[str, int],
            list[tuple[str, int]],
        ] = {}


        for node in self.parent:


            root = self.find(node)


            groups.setdefault(
                root,
                [],
            ).append(node)



        self.tracks = {}

        self.observation_to_track = {}



        for index, observations in enumerate(
            groups.values()
        ):


            track = Track(
                id=index
            )


            for frame_name, keypoint_index in observations:


                track.add_observation(
                    frame_name,
                    keypoint_index,
                )


                self.observation_to_track[
                    (
                        frame_name,
                        keypoint_index,
                    )
                ] = index



            self.tracks[index] = track



        return list(
            self.tracks.values()
        )

    def update_tracks(
        self,
    ) -> None:
        """
        Updates existing tracks after new unions.

        Preserves track identity during sequential
        reconstruction.
        """


        groups = {}

        #
        # Build current connected components
        #

        for node in self.parent:

            root = self.find(node)

            groups.setdefault(
                root,
                [],
            ).append(node)



        new_observation_to_track = {}


        #
        # Process every component
        #

        for observations in groups.values():


            track_ids = set()


            for observation in observations:

                if observation in self.observation_to_track:

                    track_ids.add(
                        self.observation_to_track[
                            observation
                        ]
                    )



            #
            # Existing track
            #
            if len(track_ids) > 0:

                track_id = min(track_ids)

                track = self.tracks[
                    track_id
                ]


            #
            # New track
            #
            else:

                track_id = self.next_track_id

                track = Track(
                    id=track_id
                )

                self.tracks[
                    track_id
                ] = track


                self.next_track_id += 1



            #
            # Add observations
            #
            for observation in observations:


                if observation not in track.observations:

                    track.add_observation(
                        observation[0],
                        observation[1],
                    )


                new_observation_to_track[
                    observation
                ] = track_id



        self.observation_to_track = (
            new_observation_to_track
        )

    def get_track_id(
        self,
        frame_name: str,
        keypoint_index: int,
    ) -> int | None:
        """
        Returns the track id associated with
        one image observation.
        """


        return self.observation_to_track.get(
            (
                frame_name,
                keypoint_index,
            )
        )
    
    def get_match_track_ids(
        self,
        result: MatchResult,
    ) -> list[int | None]:
        """
        Returns the track id for every inlier match.
        """

        track_ids = []


        if result.inlier_matches is None:
            return track_ids


        for match in result.inlier_matches:

            track_id = self.get_track_id(
                result.frame1.filename,
                match.queryIdx,
            )

            track_ids.append(
                track_id
            )


        return track_ids



    def get_track(
        self,
        track_id: int,
    ) -> Track | None:
        """
        Returns a track by id.
        """


        return self.tracks.get(
            track_id
        )



    def clear(
        self,
    ) -> None:
        """
        Removes all tracks.
        """


        self.parent.clear()

        self.observation_to_track.clear()

        self.tracks.clear()

        self.next_track_id = 0