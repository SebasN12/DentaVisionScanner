"""
Camera pose estimation.
"""

import cv2
import numpy as np

from src.core.match_result import MatchResult


class PoseEstimator:
    """
    Estimates relative camera motion between two frames.

    Output convention:

        X_frame2 =
        R * X_frame1 + t

    This is NOT a global camera pose.
    """

    def estimate(
        self,
        result: MatchResult,
        camera_matrix: np.ndarray,
    ) -> MatchResult:

        points1 = np.float32(
            [
                result.frame1.keypoints[m.queryIdx].pt
                for m in result.good_matches
            ]
        )

        points2 = np.float32(
            [
                result.frame2.keypoints[m.trainIdx].pt
                for m in result.good_matches
            ]
        )

        essential_matrix, mask = cv2.findEssentialMat(
            points1,
            points2,
            camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        inlier_points1 = points1[mask.ravel() == 1]

        inlier_points2 = points2[mask.ravel() == 1]

        _, rotation, translation, pose_mask = cv2.recoverPose(
            essential_matrix,
            inlier_points1,
            inlier_points2,
            camera_matrix,
        )

        ransac_matches = [
            match
            for match, valid in zip(
                result.good_matches,
                mask.ravel(),
            )
            if valid
        ]

        inlier_matches = [
            match
            for match, valid in zip(
                ransac_matches,
                pose_mask.ravel(),
            )
            if valid
        ]

        result.essential_matrix = essential_matrix
        result.ransac_mask = mask
        result.pose_mask = pose_mask
        result.inlier_matches = inlier_matches
        result.rotation = rotation
        result.translation = translation

        return result