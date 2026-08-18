"""
3D point triangulation.
"""

import cv2
import numpy as np

from src.core.frame import Frame
from src.core.match_result import MatchResult
from src.core.point_cloud import PointCloud
from src.core.triangulation_result import TriangulationResult


class Triangulator:
    """
    Reconstructs 3D points from two camera views.
    """

    def triangulate(
        self,
        result: MatchResult,
        camera_matrix: np.ndarray,
        max_reprojection_error: float = 3.0,
    ) -> TriangulationResult:
        """
        Triangulates 3D points from a single image pair.

        The first camera is defined as the world reference
        camera:

            R1 = I
            t1 = 0

        The second camera uses the relative pose estimated
        from the pair:

            X_camera2 = R * X_world + t
        """

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

        #
        # Extract corresponding image points
        #
        points1 = np.float32(
            [
                result.frame1.keypoints[
                    match.queryIdx
                ].pt
                for match in result.inlier_matches
            ]
        ).T

        points2 = np.float32(
            [
                result.frame2.keypoints[
                    match.trainIdx
                ].pt
                for match in result.inlier_matches
            ]
        ).T

        #
        # Projection matrices
        #
        projection1 = camera_matrix @ np.hstack(
            (
                np.eye(3),
                np.zeros((3, 1)),
            )
        )

        projection2 = camera_matrix @ np.hstack(
            (
                result.rotation,
                result.translation.reshape(3, 1),
            )
        )

        #
        # Triangulate points
        #
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

        #
        # Extract point colors
        #
        colors = self._extract_colors(
            result.frame1,
            result.inlier_matches,
        )

        #
        # Remove invalid points while keeping all
        # corresponding data aligned.
        #
        (
            points,
            points1,
            points2,
            colors,
        ) = self._filter_points(
            points=points,
            points1=points1,
            points2=points2,
            projection1=projection1,
            projection2=projection2,
            rotation=result.rotation,
            translation=result.translation,
            colors=colors,
            max_reprojection_error=max_reprojection_error,
        )

        point_cloud = PointCloud(
            points=points,
            colors=colors,
        )

        return TriangulationResult(
            point_cloud=point_cloud,
            image_points1=points1,
            image_points2=points2,
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

    def _filter_points(
        self,
        points: np.ndarray,
        points1: np.ndarray,
        points2: np.ndarray,
        projection1: np.ndarray,
        projection2: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
        colors: np.ndarray | None,
        max_reprojection_error: float,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
    ]:
        """
        Removes geometrically invalid triangulated points.

        The 3D points, 2D observations, and colors are
        filtered together so that their indices remain
        aligned.
        """

        filtered_points = []
        filtered_points1 = []
        filtered_points2 = []

        filtered_colors = (
            []
            if colors is not None
            else None
        )

        for i, point in enumerate(points):

            #
            # Cheirality check
            #
            if not self._is_in_front_of_cameras(
                point,
                rotation,
                translation,
            ):
                continue

            #
            # Reprojection error check
            #
            error = self._compute_reprojection_error(
                point,
                points1[:, i],
                points2[:, i],
                projection1,
                projection2,
            )

            if error > max_reprojection_error:
                continue

            filtered_points.append(point)

            filtered_points1.append(
                points1[:, i]
            )

            filtered_points2.append(
                points2[:, i]
            )

            if colors is not None:
                filtered_colors.append(
                    colors[i]
                )

        return (
            np.asarray(filtered_points),
            np.asarray(filtered_points1),
            np.asarray(filtered_points2),
            (
                np.asarray(filtered_colors)
                if filtered_colors is not None
                else None
            ),
        )

    def _compute_reprojection_error(
        self,
        point: np.ndarray,
        observed1: np.ndarray,
        observed2: np.ndarray,
        projection1: np.ndarray,
        projection2: np.ndarray,
    ) -> float:
        """
        Computes average reprojection error in pixels.
        """

        point_h = np.hstack(
            (
                point,
                1.0,
            )
        )

        projected1 = projection1 @ point_h
        projected2 = projection2 @ point_h

        projected1 = (
            projected1[:2]
            /
            projected1[2]
        )

        projected2 = (
            projected2[:2]
            /
            projected2[2]
        )

        error1 = np.linalg.norm(
            projected1 - observed1
        )

        error2 = np.linalg.norm(
            projected2 - observed2
        )

        return (
            error1 + error2
        ) / 2.0

    def _is_in_front_of_cameras(
        self,
        point: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> bool:
        """
        Checks the cheirality condition.

        The 3D point must be in front of both cameras.
        """

        #
        # First camera
        #
        if point[2] <= 0:
            return False

        #
        # Second camera
        #
        point_camera2 = (
            rotation @ point
            +
            translation.reshape(3)
        )

        if point_camera2[2] <= 0:
            return False

        return True