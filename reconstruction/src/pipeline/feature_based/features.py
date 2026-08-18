"""
Feature detection using OpenCV.
"""

import cv2
from tqdm import tqdm

from src.config.settings import (
    ENABLE_PROGRESS_BAR,
    SIFT_MAX_FEATURES,
)
from src.core.frame import Frame


class FeatureDetector:
    """
    Detects image features using SIFT.
    """

    def __init__(self):

        self.detector = cv2.SIFT_create(
            nfeatures=SIFT_MAX_FEATURES
        )

    def detect(
        self,
        frame: Frame,
    ) -> Frame:
        """
        Detects SIFT keypoints and descriptors.
        """

        keypoints, descriptors = self.detector.detectAndCompute(
            frame.image,
            None,
        )

        frame.keypoints = keypoints
        frame.descriptors = descriptors

        return frame

    def detect_sequence(
        self,
        frames: list[Frame],
    ) -> list[Frame]:
        """
        Detects features for an entire sequence.
        """

        iterator = (
            tqdm(
                frames,
                desc="Detecting features",
            )
            if ENABLE_PROGRESS_BAR
            else frames
        )

        for frame in iterator:
            self.detect(frame)

        return frames