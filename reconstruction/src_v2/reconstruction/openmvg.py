from pathlib import Path

from src_v2.config.settings import OPENMVG_BIN_DIRECTORY
from src_v2.utils.process import run_command


class OpenMVG:
    """
    Wrapper around the OpenMVG command-line tools.
    """

    def __init__(
        self,
        executable_directory: Path = OPENMVG_BIN_DIRECTORY,
    ):
        self.executable_directory = Path(executable_directory)

    def _executable(self, name: str) -> Path:
        """
        Returns the path to an OpenMVG executable.
        """

        path = self.executable_directory / f"{name}.exe"

        if not path.exists():
            raise FileNotFoundError(
                f"OpenMVG executable not found: {path}"
            )

        return path

    def init_image_listing(
        self,
        image_directory: Path,
        output_directory: Path,
        sensor_width_database: Path,
        camera_model: int,
    ) -> None:
        """
        Creates the initial SfM_Data file from the input images.
        """

        executable = self._executable(
            "openMVG_main_SfMInit_ImageListing"
        )

        run_command(
            executable,
            [
                "-i",
                str(image_directory),
                "-d",
                str(sensor_width_database),
                "-o",
                str(output_directory),
                "-c",
                str(camera_model),
            ],
        )

    def compute_features(
        self,
        sfm_data: Path,
        output_directory: Path,
        describer_method: str,
        describer_preset: str,
        num_threads: int,
    ) -> None:
        """
        Computes local features and descriptors.
        """

        executable = self._executable(
            "openMVG_main_ComputeFeatures"
        )

        run_command(
            executable,
            [
                "-i",
                str(sfm_data),
                "-o",
                str(output_directory),
                "-m",
                str(describer_method),
                "-p",
                str(describer_preset),
                "-n",
                str(num_threads),
            ],
        )

    def compute_matches(
        self,
        sfm_data: Path,
        output_file: Path,
        ratio: float,
    ) -> None:
        """
        Computes putative feature matches.
        """

        executable = self._executable(
            "openMVG_main_ComputeMatches"
        )

        run_command(
            executable,
            [
                "-i",
                str(sfm_data),
                "-o",
                str(output_file),
                "-r",
                str(ratio),
            ],
        )

    def geometric_filter(
        self,
        sfm_data: Path,
        input_matches: Path,
        output_matches: Path,
        geometric_model: str = "e",
    ) -> None:
        """
        Removes geometrically inconsistent matches.
        """

        executable = self._executable(
            "openMVG_main_GeometricFilter"
        )

        run_command(
            executable,
            [
                "-i",
                str(sfm_data),
                "-m",
                str(input_matches),
                "-o",
                str(output_matches),
                "-g",
                str(geometric_model),
            ],
        )

    def reconstruct(
        self,
        sfm_data: Path,
        match_directory: Path,
        output_directory: Path,
        engine: str,
        match_file: str,
        camera_model: int,
    ) -> None:
        """
        Runs the OpenMVG Structure-from-Motion reconstruction.
        """

        executable = self._executable(
            "openMVG_main_SfM"
        )

        run_command(
            executable,
            [
                "-i",
                str(sfm_data),
                "-m",
                str(match_directory),
                "-o",
                str(output_directory),
                "-s",
                str(engine),
                "-M",
                str(match_file),
                "-c",
                str(camera_model),
            ],
        )

    def compute_color(
        self,
        sfm_data: Path,
        output_file: Path,
    ) -> None:
        """
        Computes RGB colors for the reconstructed 3D points.
        """

        executable = self._executable(
            "openMVG_main_ComputeSfM_DataColor"
        )

        run_command(
            executable,
            [
                "-i",
                str(sfm_data),
                "-o",
                str(output_file),
            ],
        )