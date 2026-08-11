from pathlib import Path

from src_v2.config.settings import OPENMVG_BIN_DIRECTORY
from src_v2.utils.process import run_command


class OpenMVS:
    """
    Wrapper around the OpenMVS-related command-line tools.
    """

    def __init__(
        self,
        openmvg_executable_directory: Path = OPENMVG_BIN_DIRECTORY,
    ):
        self.openmvg_executable_directory = Path(
            openmvg_executable_directory
        )

    def _openmvg_executable(self, name: str) -> Path:
        """
        Returns the path to an OpenMVG executable used for OpenMVS conversion.
        """

        path = (
            self.openmvg_executable_directory
            / f"{name}.exe"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"OpenMVG executable not found: {path}"
            )

        return path

    def convert_from_openmvg(
        self,
        sfm_data: Path,
        output_file: Path,
        undistorted_images_directory: Path,
    ) -> None:
        """
        Converts an OpenMVG SfM reconstruction into an OpenMVS scene.
        """

        executable = self._openmvg_executable(
            "openMVG_main_openMVG2openMVS"
        )

        run_command(
            executable,
            [
                "-i",
                str(sfm_data),
                "-o",
                str(output_file),
                "-d",
                str(undistorted_images_directory),
            ],
        )