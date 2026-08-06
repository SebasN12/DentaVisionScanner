"""
Stores one reconstructed 3D landmark.
"""

from dataclasses import dataclass, field

import numpy as np

from src.core.observation import Observation


@dataclass
class Landmark:
    """
    Represents one reconstructed 3D point.
    """


    id: int

    position: np.ndarray

    color: np.ndarray | None = None


    observations: list[Observation] = field(
        default_factory=list
    )


    track_id: int | None = None



    def add_observation(
        self,
        observation: Observation,
    ) -> None:
        """
        Adds one image observation.
        """


        self.observations.append(
            observation
        )