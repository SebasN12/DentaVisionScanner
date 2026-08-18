"""
Stereo matching using OpenCV StereoSGBM.
"""

import cv2
import numpy as np


class StereoMatcher:
    """
    Computes a dense disparity map from a rectified stereo image pair
    using StereoSGBM.
    """

    def __init__(
        self,
        min_disparity: int = 0,
        num_disparities: int = 128,
        block_size: int = 5,
        uniqueness_ratio: int = 10,
        speckle_window_size: int = 100,
        speckle_range: int = 2,
        disp12_max_diff: int = 1,
    ):
        """
        Initializes the StereoSGBM matcher.

        Parameters
        ----------
        min_disparity:
            Minimum possible disparity.

        num_disparities:
            Number of disparity levels to search.
            Must be divisible by 16.

        block_size:
            Matching block size. Must be an odd number.

        uniqueness_ratio:
            Margin in percentage by which the best matching
            block must be better than the second-best match.

        speckle_window_size:
            Maximum size of smooth disparity regions to consider
            as speckles.

        speckle_range:
            Maximum disparity variation within a connected
            component considered as a speckle.

        disp12_max_diff:
            Maximum allowed difference between left-to-right
            and right-to-left disparity checks.
        """

        if num_disparities <= 0 or num_disparities % 16 != 0:
            raise ValueError(
                "num_disparities must be a positive multiple of 16."
            )

        if block_size <= 0 or block_size % 2 == 0:
            raise ValueError(
                "block_size must be a positive odd number."
            )

        self.matcher = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * block_size**2,
            P2=32 * block_size**2,
            disp12MaxDiff=disp12_max_diff,
            uniquenessRatio=uniqueness_ratio,
            speckleWindowSize=speckle_window_size,
            speckleRange=speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def compute(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the disparity map for a rectified stereo pair.

        Parameters
        ----------
        left_image:
            Left stereo image.

        right_image:
            Right stereo image.

        Returns
        -------
        np.ndarray
            Disparity map as float32.
        """

        if left_image is None or right_image is None:
            raise ValueError(
                "Left and right images must not be None."
            )

        if left_image.shape[:2] != right_image.shape[:2]:
            raise ValueError(
                "Left and right images must have the same dimensions."
            )

        left_gray = self._to_grayscale(left_image)
        right_gray = self._to_grayscale(right_image)

        disparity = self.matcher.compute(
            left_gray,
            right_gray,
        )

        # OpenCV stores StereoSGBM disparity multiplied by 16.
        disparity = disparity.astype(np.float32) / 16.0

        return disparity

    @staticmethod
    def _to_grayscale(
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Converts an image to grayscale if necessary.
        """

        if image.ndim == 2:
            return image

        if image.ndim == 3:
            return cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

        raise ValueError(
            "Image must be either grayscale or BGR."
        )