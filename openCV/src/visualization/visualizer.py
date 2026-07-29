from pathlib import Path

import cv2

from src.config.settings import SHOW_WINDOWS
from src.core.frame import Frame
from src.core.match_result import MatchResult


class Visualizer:

    @staticmethod
    def draw_keypoints(
        frame: Frame,
        output_directory: Path | str,
        rich_keypoints: bool = True,
        show: bool = SHOW_WINDOWS,
    ) -> Path:

        output_directory = Path(output_directory)
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        flags = (
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            if rich_keypoints
            else 0
        )

        image = cv2.drawKeypoints(
            frame.image,
            frame.keypoints,
            None,
            flags=flags,
        )

        output_path = (
            output_directory
            / f"{frame.path.stem}_features.jpg"
        )

        cv2.imwrite(
            str(output_path),
            image,
        )

        if show:
            cv2.imshow(frame.filename, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return output_path

    @staticmethod
    def draw_matches(
        result: MatchResult,
        output_directory: str,
        use_inliers: bool = False,
        show: bool = SHOW_WINDOWS,
    ) -> Path:

        output_directory = Path(output_directory)
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        matches = (
            result.inlier_matches
            if use_inliers
            and result.inlier_matches is not None
            else result.good_matches
        )

        image = cv2.drawMatches(
            result.frame1.image,
            result.frame1.keypoints,
            result.frame2.image,
            result.frame2.keypoints,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        output_path = (
            output_directory
            / f"{result.frame1.path.stem}"
            f"_{result.frame2.path.stem}_matches.jpg"
        )

        cv2.imwrite(
            str(output_path),
            image,
        )

        if show:
            cv2.imshow("Matches", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return output_path