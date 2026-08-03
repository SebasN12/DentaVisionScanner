"""
Point cloud export utilities.
"""

from pathlib import Path

import open3d as o3d

from src.core.point_cloud import PointCloud



class PointCloudWriter:
    """
    Exports point clouds to external formats.
    """


    @staticmethod
    def write_ply(
        cloud: PointCloud,
        output_path: str,
    ) -> Path:
        """
        Writes a point cloud into a PLY file.

        Parameters
        ----------
        cloud
            Point cloud to export.

        output_path
            Destination path.

        Returns
        -------
        Path
            Path of the generated file.
        """


        output_path = Path(
            output_path
        )


        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        geometry = o3d.geometry.PointCloud()


        geometry.points = (
            o3d.utility.Vector3dVector(
                cloud.points
            )
        )


        if cloud.colors is not None:

            geometry.colors = (
                o3d.utility.Vector3dVector(
                    cloud.colors
                )
            )


        o3d.io.write_point_cloud(
            str(output_path),
            geometry,
        )


        return output_path