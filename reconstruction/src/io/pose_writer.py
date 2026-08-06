"""
Camera pose export utilities.
"""

from pathlib import Path
import json

from src.core.camera_pose import CameraPose



class PoseWriter:
    """
    Exports camera poses.
    """


    @staticmethod
    def write_json(
        poses: dict[str, CameraPose],
        output_path: str,
    ) -> Path:
        """
        Writes camera poses into JSON format.
        """


        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        data = {}


        for name, pose in poses.items():

            data[name] = {

                "rotation": (
                    pose.rotation.tolist()
                ),

                "translation": (
                    pose.translation.reshape(3).tolist()
                ),

            }


        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
            )


        return output_path