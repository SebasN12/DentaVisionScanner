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
from src.core.reconstruction import Reconstruction
from src.optimization.ba_problem import BAProblem
from src.core.point_cloud import PointCloud


from src_v2.reconstruction.openmvg import OpenMVG
from src_v2.reconstruction.openmvs import OpenMVS
from src_v2.reconstruction.pipeline import ReconstructionPipeline

from src.config.camera import CAMERA_MATRIX

from src.config.paths import (
    DATASET_PATH,
    FEATURES_OUTPUT,
    MATCHES_OUTPUT,
    INLIERS_OUTPUT,
    RECONSTRUCTION_OUTPUT_FILE,
)


# Pipeline A: pairwise reconstruction with OpenCV

def load_frames():

    camera = Camera(DATASET_PATH)

    frames = camera.load_frames()

    print(
        f"Loaded {len(frames)} frames.\n"
    )

    return frames


def test_camera():

    frames = load_frames()

    for frame in frames:

        print(frame.filename)



def test_features():

    frames = load_frames()

    detector = FeatureDetector()
    detector.detect_sequence(frames)

    for frame in frames:

        print(
            f"{frame.filename}: "
            f"{len(frame.keypoints)} keypoints"
        )

        output = Visualizer.draw_keypoints(
            frame,
            FEATURES_OUTPUT,
        )

        print(f"Saved: {output}")



def test_matching():

    frames = load_frames()

    detector = FeatureDetector()
    detector.detect_sequence(frames)

    matcher = FeatureMatcher()
    results = matcher.match_sequence(frames)

    for result in results:

        print(
            f"{result.frame1.filename} "
            f"<-> "
            f"{result.frame2.filename}: "
            f"{len(result.good_matches)} matches"
        )

        output = Visualizer.draw_matches(
            result,
            MATCHES_OUTPUT,
        )

        print(f"Saved: {output}")

def test_pose():
    """
    Tests relative camera pose estimation for a single image pair.
    """

    frame1, frame2, result = prepare_test_pair()

    print(
        f"Pair: "
        f"{frame1.filename} -> "
        f"{frame2.filename}"
    )

    print(
        f"Good matches: "
        f"{len(result.good_matches)}"
    )

    print(
        f"RANSAC inliers: "
        f"{result.ransac_mask.sum()}"
    )

    print(
        f"Pose inliers: "
        f"{len(result.inlier_matches)}"
    )

    Visualizer.draw_matches(
        result,
        INLIERS_OUTPUT,
        use_inliers=True,
    )

def test_triangulation():
    """
    Tests triangulation for a single image pair.
    """

    frame1, frame2, result = prepare_test_pair()

    triangulator = Triangulator()

    triangulation = triangulator.triangulate(
        result,
        CAMERA_MATRIX,
    )

    point_cloud = triangulation.point_cloud

    if len(point_cloud.points) == 0:
        raise RuntimeError(
            "Triangulation produced no 3D points."
        )

    if len(triangulation.image_points1) != len(
        point_cloud.points
    ):
        raise RuntimeError(
            "Image 1 observations do not match "
            "the number of 3D points."
        )

    if len(triangulation.image_points2) != len(
        point_cloud.points
    ):
        raise RuntimeError(
            "Image 2 observations do not match "
            "the number of 3D points."
        )

    if not np.all(
        np.isfinite(point_cloud.points)
    ):
        raise RuntimeError(
            "Triangulated points contain "
            "non-finite values."
        )

    print(
        f"Pair: "
        f"{frame1.filename} -> "
        f"{frame2.filename}"
    )

    print(
        f"Inlier matches: "
        f"{len(result.inlier_matches)}"
    )

    print(
        f"Triangulated points: "
        f"{len(point_cloud.points)}"
    )

    Visualizer.show_point_cloud(
        point_cloud
    )

def test_pairwise_reconstruction():
    """
    Tests the complete pairwise reconstruction pipeline.

    The pipeline consists of:

        Feature Detection
            ↓
        Feature Matching
            ↓
        Relative Pose Estimation
            ↓
        Triangulation
            ↓
        Pairwise Bundle Adjustment
            ↓
        Optimized Point Cloud
    """

    frame1, frame2, result = prepare_test_pair()

    triangulator = Triangulator()

    bundle_adjustment = BundleAdjustment(
        CAMERA_MATRIX
    )

    reconstructor = Reconstructor(
        triangulator,
        bundle_adjustment,
    )

    print(
        f"Pair: "
        f"{frame1.filename} -> "
        f"{frame2.filename}"
    )

    print(
        f"Inlier matches: "
        f"{len(result.inlier_matches)}"
    )

    (
        point_cloud,
        image_points1,
        image_points2,
    ) = reconstructor.reconstruct(
        result,
        CAMERA_MATRIX,
    )

    if len(point_cloud.points) == 0:
        raise RuntimeError(
            "Pairwise reconstruction produced "
            "no 3D points."
        )

    if len(image_points1) != len(
        point_cloud.points
    ):
        raise RuntimeError(
            "Image 1 observations do not match "
            "the number of reconstructed points."
        )

    if len(image_points2) != len(
        point_cloud.points
    ):
        raise RuntimeError(
            "Image 2 observations do not match "
            "the number of reconstructed points."
        )

    if not np.all(
        np.isfinite(point_cloud.points)
    ):
        raise RuntimeError(
            "Reconstructed points contain "
            "non-finite values."
        )

    print(
        f"Reconstructed points: "
        f"{len(point_cloud.points)}"
    )

    Visualizer.show_point_cloud(
        point_cloud
    )

def test_pairwise_bundle_adjustment():
    """
    Tests pairwise triangulation and Bundle Adjustment.
    """

    frame1, frame2, result = prepare_test_pair()

    triangulator = Triangulator()

    bundle_adjustment = BundleAdjustment(
        CAMERA_MATRIX
    )

    triangulation = triangulator.triangulate(
        result,
        CAMERA_MATRIX,
    )

    if len(
        triangulation.point_cloud.points
    ) == 0:
        raise RuntimeError(
            "Triangulation produced no points."
        )

    problem = BAProblem(
        rotation=result.rotation,
        translation=result.translation,
        points_3d=(
            triangulation.point_cloud.points
        ),
        image_points1=(
            triangulation.image_points1
        ),
        image_points2=(
            triangulation.image_points2
        ),
    )

    error_before = (
        compute_pairwise_reprojection_error(
            problem,
            CAMERA_MATRIX,
        )
    )

    optimized_problem = (
        bundle_adjustment.optimize(
            problem
        )
    )

    error_after = (
        compute_pairwise_reprojection_error(
            optimized_problem,
            CAMERA_MATRIX,
        )
    )

    print(
        f"Pair: "
        f"{frame1.filename} -> "
        f"{frame2.filename}"
    )

    print(
        f"Points: "
        f"{len(optimized_problem.points_3d)}"
    )

    print(
        f"Reprojection error before BA: "
        f"{error_before:.4f} px"
    )

    print(
        f"Reprojection error after BA:  "
        f"{error_after:.4f} px"
    )

    if not np.isfinite(error_before):
        raise RuntimeError(
            "Initial reprojection error is not finite."
        )

    if not np.isfinite(error_after):
        raise RuntimeError(
            "Optimized reprojection error is not finite."
        )

    if error_after > error_before:
        raise RuntimeError(
            "Bundle Adjustment increased the "
            "reprojection error."
        )

    point_cloud = PointCloud(
        points=optimized_problem.points_3d,
        colors=triangulation.point_cloud.colors,
    )

    Visualizer.show_point_cloud(
        point_cloud
    )

# Helpers

def prepare_test_pair():
    """
    Prepares the first image pair for reconstruction tests.

    Returns
    -------
    tuple
        The two frames and the estimated MatchResult.
    """

    frames = load_frames()

    if len(frames) < 2:
        raise RuntimeError(
            "At least two frames are required."
        )

    detector = FeatureDetector()
    matcher = FeatureMatcher()
    estimator = PoseEstimator()

    frame1 = frames[0]
    frame2 = frames[1]

    detector.detect(frame1)
    detector.detect(frame2)

    result = matcher.match(
        frame1,
        frame2,
    )

    estimator.estimate(
        result,
        CAMERA_MATRIX,
    )

    if result.rotation is None:
        raise RuntimeError(
            "Pose estimation did not produce a rotation."
        )

    if result.translation is None:
        raise RuntimeError(
            "Pose estimation did not produce a translation."
        )

    if result.ransac_mask is None:
        raise RuntimeError(
            "RANSAC did not produce a valid mask."
        )

    if result.inlier_matches is None:
        raise RuntimeError(
            "Pose estimation did not produce inlier matches."
        )

    if len(result.inlier_matches) == 0:
        raise RuntimeError(
            "Pose estimation produced no inlier matches."
        )

    return frame1, frame2, result

def compute_pairwise_reprojection_error(
    problem: BAProblem,
    camera_matrix: np.ndarray,
) -> float:
    """
    Computes the mean reprojection error for
    a pairwise reconstruction.
    """

    points = problem.points_3d

    projected1, _ = cv2.projectPoints(
        points,
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        camera_matrix,
        None,
    )

    projected2, _ = cv2.projectPoints(
        points,
        problem.rotation,
        problem.translation.reshape(3, 1),
        camera_matrix,
        None,
    )

    projected1 = projected1.reshape(-1, 2)
    projected2 = projected2.reshape(-1, 2)

    errors1 = np.linalg.norm(
        projected1 - problem.image_points1,
        axis=1,
    )

    errors2 = np.linalg.norm(
        projected2 - problem.image_points2,
        axis=1,
    )

    return float(
        np.mean(
            np.concatenate(
                (errors1, errors2)
            )
        )
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