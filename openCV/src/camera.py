"""
camera.py

Image acquisition module.

Currently reads images from a directory.
Later this class can be extended to acquire frames directly
from the Sirona Omnicam using mvIMPACT Acquire.
"""

from pathlib import Path

import cv2

from src.frame import Frame


class Camera:
    """
    Handles image acquisition.
    """

    SUPPORTED_EXTENSIONS = (
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.bmp",
        "*.tif",
        "*.tiff",
    )

    def __init__(self, image_directory: str):
        self.image_directory = Path(image_directory)

        if not self.image_directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {self.image_directory}"
            )

    def get_image_paths(self) -> list[Path]:
        """
        Returns a sorted list of image paths.
        """

        image_paths = []

        for extension in self.SUPPORTED_EXTENSIONS:
            image_paths.extend(self.image_directory.glob(extension))

        return sorted(image_paths)

    def load_frame(self, image_path: Path) -> Frame:
        """
        Loads a single image and returns it as a Frame.
        """

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(
                f"Unable to load image: {image_path}"
            )

        return Frame(
            filename=image_path.name,
            path=image_path,
            image=image,
        )

    def load_frames(self) -> list[Frame]:
        """
        Loads all images in the directory.

        Returns
        -------
        list[Frame]
        """

        frames = []

        for image_path in self.get_image_paths():
            frames.append(self.load_frame(image_path))

        return frames

    def count(self) -> int:
        """
        Returns the number of images found.
        """

        return len(self.get_image_paths())