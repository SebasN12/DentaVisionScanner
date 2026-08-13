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
        Sparse Point Cloud
    """

    frame1, frame2, result = prepare_test_pair()

    triangulator = Triangulator()

    reconstructor = Reconstructor(
        triangulator
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

def test_pairwise_reconstruction_sequence():
    """
    Tests pairwise reconstruction over the first 20 frames.

    The test simulates the sequential processing of consecutive
    image pairs:

        Frame 0 + Frame 1
        Frame 1 + Frame 2
        ...
        Frame 18 + Frame 19

    Each pair is reconstructed independently.
    """

    frames = load_frames()

    if len(frames) < 20:
        raise RuntimeError(
            "At least 20 frames are required."
        )

    frames = frames[:20]

    detector = FeatureDetector()
    matcher = FeatureMatcher()
    estimator = PoseEstimator()

    triangulator = Triangulator()

    reconstructor = Reconstructor(
        triangulator,
    )

    print(
        f"Loaded {len(frames)} frames for "
        f"pairwise reconstruction."
    )

    print(
        f"Processing {len(frames) - 1} consecutive pairs.\n"
    )

    reconstructed_clouds = []

    for i in range(len(frames) - 1):

        frame1 = frames[i]
        frame2 = frames[i + 1]

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

        if (
            result.rotation is None
            or result.translation is None
        ):
            raise RuntimeError(
                f"Pose estimation failed for pair "
                f"{i}."
            )

        point_cloud, _, _ = (
            reconstructor.reconstruct(
                result,
                CAMERA_MATRIX,
            )
        )

        if len(point_cloud.points) == 0:
            raise RuntimeError(
                f"Pair {i} produced no reconstructed points."
            )

        if not np.all(
            np.isfinite(point_cloud.points)
        ):
            raise RuntimeError(
                f"Pair {i} produced non-finite points."
            )
        
        output_path = (
            RECONSTRUCTION_OUTPUT
            / f"pair_{i:03d}.ply"
        )

        saved_path = PointCloudWriter.write_ply(
            point_cloud,
            output_path,
        )

        print(
            f"  Saved: {saved_path}"
        )

        reconstructed_clouds.append(
            point_cloud
        )

        print(
            f"Pair {i + 1}/{len(frames) - 1}: "
            f"{len(point_cloud.points)} points"
        )

    print(
        f"\nPairwise reconstruction completed: "
        f"{len(reconstructed_clouds)} point clouds."
    )

def test_pairwise_bundle_adjustment():
    """
    Evaluates pairwise Bundle Adjustment in terms of
    reprojection error and execution time.
    """

    import time

    frame1, frame2, result = prepare_test_pair()

    triangulator = Triangulator()

    bundle_adjustment = BundleAdjustment(
        CAMERA_MATRIX
    )

    #
    # Triangulation
    #
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

    #
    # Reprojection error before BA
    #
    error_before = (
        compute_pairwise_reprojection_error(
            problem,
            CAMERA_MATRIX,
        )
    )

    if not np.isfinite(error_before):
        raise RuntimeError(
            "Initial reprojection error is not finite."
        )

    #
    # Bundle Adjustment
    #
    start_time = time.perf_counter()

    optimized_problem = (
        bundle_adjustment.optimize(
            problem
        )
    )

    ba_time = (
        time.perf_counter()
        - start_time
    )

    #
    # Reprojection error after BA
    #
    error_after = (
        compute_pairwise_reprojection_error(
            optimized_problem,
            CAMERA_MATRIX,
        )
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

    #
    # Improvement
    #
    if error_before > 0:
        improvement = (
            (error_before - error_after)
            / error_before
            * 100.0
        )
    else:
        improvement = 0.0

    #
    # Results
    #
    print()

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

    print(
        f"Reprojection error improvement: "
        f"{improvement:.2f}%"
    )

    print(
        f"Bundle Adjustment time: "
        f"{ba_time:.2f} s"
    )

    #
    # Visualize optimized result
    #
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