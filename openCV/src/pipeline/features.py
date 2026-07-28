"""
Feature detection using OpenCV.
"""

import cv2

from ..core.frame import Frame


class FeatureDetector:
    """
    Detects image features using SIFT.
    """

    def __init__(self, max_features: int = 0):
        """
        Parameters
        ----------
        max_features
            Maximum number of features.
            0 means unlimited.
        """

        self.detector = cv2.SIFT_create(
            nfeatures=max_features
        )

    def detect(self, frame: Frame) -> Frame:
        """
        Detects SIFT keypoints and descriptors.

        Parameters
        ----------
        frame
            Input frame.

        Returns
        -------
        Frame
            Same frame with keypoints and descriptors.
        """

        keypoints, descriptors = self.detector.detectAndCompute(
            frame.image,
            None
        )

        frame.keypoints = keypoints
        frame.descriptors = descriptors

        return frame