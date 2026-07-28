from pathlib import Path

import cv2

from src.core.frame import Frame


class Visualizer:

    @staticmethod
    def draw_keypoints(
        frame: Frame,
        output_directory: str,
        rich_keypoints: bool = True,
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

        return output_path