"""
3D point triangulation.
"""
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
        max_reprojection_error: float = 3.0,
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
        
        points, colors, track_ids = self._filter_points(
            points,
            points1,
            points2,
            projection1,
            projection2,
            result.rotation,
            result.translation,
            colors,
            track_ids,
            max_reprojection_error,
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
        track_ids: list[int] | None,
        max_reprojection_error: float,
    ):
        """
        Removes geometrically invalid triangulated points.
        """

        filtered_points = []

        filtered_colors = [] if colors is not None else None

        filtered_track_ids = (
            []
            if track_ids is not None
            else None
        )

        for i, point in enumerate(points):

            if not self._is_in_front_of_cameras(
                point,
                rotation,
                translation,
            ):
                continue


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


            if colors is not None:
                filtered_colors.append(
                    colors[i]
                )


            if track_ids is not None:
                filtered_track_ids.append(
                    track_ids[i]
                )


        return (
            np.asarray(filtered_points),
            (
                np.asarray(filtered_colors)
                if filtered_colors is not None
                else None
            ),
            filtered_track_ids,
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
        Checks cheirality condition.

        The 3D point must be in front of both cameras.
        """

        # First camera
        if point[2] <= 0:
            return False


        # Second camera
        point_camera2 = (
            rotation @ point
            +
            translation.reshape(3)
        )

        if point_camera2[2] <= 0:
            return False


        return True