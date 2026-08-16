import cv2
import numpy as np

from src.pipeline.reconstructor import Reconstructor
from src.pipeline.camera import Camera
from src.pipeline.features import FeatureDetector
from src.pipeline.matching import FeatureMatcher
from src.pipeline.pose import PoseEstimator
from src.pipeline.triangulation import Triangulator
from src.visualization.visualizer import Visualizer
from src.io.point_cloud_writer import PointCloudWriter
from src.optimization.bundle_adjustment import BundleAdjustment
from src.optimization.ba_problem import BAProblem
from src.core.point_cloud import PointCloud
from src.pipeline.dense_reconstruction import DenseReconstructor


from src_v2.reconstruction.openmvg import OpenMVG
from src_v2.reconstruction.openmvs import OpenMVS
from src_v2.reconstruction.pipeline import ReconstructionPipeline

from src.config.camera import CAMERA_MATRIX

from src.config.paths import (
    DATASET_PATH,
    FEATURES_OUTPUT,
    MATCHES_OUTPUT,
    INLIERS_OUTPUT,
    RECONSTRUCTION_OUTPUT,
)

# Pipeline B: OpenMVG reconstruction

def test_openmvg_sparse_pipeline():
    openmvg = OpenMVG()

    pipeline = ReconstructionPipeline(
        openmvg=openmvg,
        clean_output=True,
    )

    point_cloud = pipeline.run_sparse()

    print("\nSparse point cloud generated at:")
    print(point_cloud)

    visualizer = Visualizer()
    visualizer.show_ply(point_cloud)

def test_openmvs_prepare_dense():
    openmvg = OpenMVG()
    openmvs = OpenMVS()

    pipeline = ReconstructionPipeline(
        openmvg=openmvg,
        openmvs=openmvs,
        clean_output=True,
    )

    scene = pipeline.prepare_dense()

    print("\nOpenMVS scene generated at:")
    print(scene)

def test_openmvs_dense_pipeline():
    openmvs = OpenMVS()

    pipeline = ReconstructionPipeline(
        openmvs=openmvs,
        clean_output=True,
    )

    point_cloud = pipeline.run_dense()

    print("\nDense point cloud generated at:")
    print(point_cloud)

    visualizer = Visualizer()
    visualizer.show_ply(point_cloud)