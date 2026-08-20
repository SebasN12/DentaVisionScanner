"""
Validation of stereo disparity maps.
"""

import numpy as np


class StereoValidator:
    """
    Validates a stereo disparity map before depth reconstruction.

    The validator identifies disparity values that cannot be used
    for depth reconstruction.

    The initial validation rules are intentionally general:

    - disparity must be finite;
    - disparity must be positive.

    More advanced validation, such as left-right consistency,
    can be added later without changing the responsibility of
    the depth reconstruction component.
    """

    def compute_valid_mask(
        self,
        disparity: np.ndarray,
    ) -> np.ndarray:
        """
        Computes a validity mask for a disparity map.

        Parameters
        ----------
        disparity:
            Disparity map in pixels.

        Returns
        -------
        np.ndarray
            Boolean mask with the same shape as the disparity map.

            True indicates a valid disparity.
            False indicates an invalid disparity.
        """

        if disparity is None:
            raise ValueError(
                "Disparity must not be None."
            )

        if disparity.ndim != 2:
            raise ValueError(
                "Disparity must be a 2D array."
            )

        valid_mask = (
            np.isfinite(disparity)
            & (disparity > 0)
        )

        return valid_mask