"""
Stores a feature track across multiple images.
"""

from dataclasses import dataclass, field


@dataclass
class Track:
    """
    Represents the same feature observed in multiple images.

    A Track represents a potential 3D point
    before and after triangulation.
    """

    id: int


    observations: list[tuple[str, int]] = field(
        default_factory=list
    )

    landmark_id: int | None = None



    def add_observation(
        self,
        frame_name: str,
        keypoint_index: int,
    ) -> None:
        """
        Adds one image observation.
        """


        self.observations.append(
            (
                frame_name,
                keypoint_index,
            )
        )



    @property
    def length(
        self,
    ) -> int:
        """
        Number of observations.
        """


        return len(
            self.observations
        )