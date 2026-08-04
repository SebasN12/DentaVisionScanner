"""
3D point triangulation.
"""

from unittest import result

import cv2
import numpy as np

from src.core.match_result import MatchResult
from src.core.frame import Frame
from src.core.point_cloud import PointCloud


class Triangulator:
    """
    Reconstructs 3D points from two camera views.
    """

    def triangulate(
        self,
        result: MatchResult,
        camera_matrix: np.ndarray,
        track_ids: list[int] | None = None,
    ) -> PointCloud:

        if (
            result.rotation is None
            or result.translation is None
        ):
            raise RuntimeError(
                "Camera pose has not been estimated."
            )

        if (
            result.inlier_matches is None
            or len(result.inlier_matches) == 0
        ):
            raise RuntimeError(
                "No inlier matches available."
            )

        points1 = np.float32(
            [
                result.frame1.keypoints[m.queryIdx].pt
                for m in result.inlier_matches
            ]
        ).T

        points2 = np.float32(
            [
                result.frame2.keypoints[m.trainIdx].pt
                for m in result.inlier_matches
            ]
        ).T

        projection1 = camera_matrix @ np.hstack(
            (
                np.eye(3),
                np.zeros((3, 1)),
            )
        )

        projection2 = camera_matrix @ np.hstack(
            (
                result.rotation,
                result.translation,
            )
        )

        homogeneous_points = cv2.triangulatePoints(
            projection1,
            projection2,
            points1,
            points2,
        )

        points = cv2.convertPointsFromHomogeneous(
            homogeneous_points.T
        )

        points = points.reshape(-1, 3)

        colors = self._extract_colors(
            result.frame1,
            result.inlier_matches,
        )

        return PointCloud(
            points=points,
            colors=colors,
            track_ids=track_ids,
        )

    def _extract_colors(
        self,
        frame: Frame,
        matches: list[cv2.DMatch],
    ) -> np.ndarray | None:
        """
        Extracts RGB colors for every reconstructed point.

        Colors are sampled from the first image.
        """

        colors = []

        for match in matches:

            x, y = frame.keypoints[
                match.queryIdx
            ].pt

            x = int(round(x))
            y = int(round(y))

            if (
                0 <= x < frame.image.shape[1]
                and 0 <= y < frame.image.shape[0]
            ):
                bgr = frame.image[y, x]

                rgb = bgr[::-1] / 255.0

                colors.append(rgb)

        if not colors:
            return None

        return np.asarray(colors)