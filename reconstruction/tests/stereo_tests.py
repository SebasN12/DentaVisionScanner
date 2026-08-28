"""
Tests for stereo reconstruction.
"""

import cv2
import numpy as np

from src.config.paths import (
    DATASET_PATH,
    OPENSTEREO_PATH,
    IGEV_CHECKPOINT,
)

from src.config.camera_middlebury import (
    BASELINE_MM,
    FOCAL_LENGTH,
    REPROJECTION_MATRIX,
)

from src.pipeline.camera import Camera
from src.pipeline.stereo.sgbm_matcher import StereoMatcher
from src.pipeline.stereo.igev_matcher import IGEVMatcher
from src.pipeline.stereo.depth import DepthReconstructor
from src.pipeline.stereo.validator import StereoValidator
from src.pipeline.stereo.stereo_reconstructor import StereoReconstructor
from src.visualization.visualizer import Visualizer

def test_stereo_sgbm():
    """
    Tests StereoSGBM on the Middlebury Plastic stereo dataset.

    The dataset provides view1/view5 and the corresponding
    ground-truth disparity map.
    """

    left_frame, right_frame = _load_stereo_frames()

    print(f"Left image:  {left_frame.filename}")
    print(f"Right image: {right_frame.filename}")

    # ------------------------------------------------------------------
    # Load ground truth.
    # ------------------------------------------------------------------

    disparity_path = DATASET_PATH / "disp1.png"

    ground_truth_raw = cv2.imread(
        str(disparity_path),
        cv2.IMREAD_UNCHANGED,
    )

    if ground_truth_raw is None:
        raise RuntimeError(
            f"Unable to load ground-truth disparity: "
            f"{disparity_path}"
        )

    # Middlebury third-size disparity maps use a scale factor of 3.
    ground_truth = (
        ground_truth_raw.astype(np.float32) / 3.0
    )

    valid_ground_truth = ground_truth_raw > 0

    # ------------------------------------------------------------------
    # Test different block sizes.
    # ------------------------------------------------------------------

    for block_size in (3, 5, 7):

        _evaluate_stereo(
            left_image=left_frame.image,
            right_image=right_frame.image,
            ground_truth=ground_truth,
            valid_ground_truth=valid_ground_truth,
            block_size=block_size,
        )

def test_stereo_validator():
    """
    Tests StereoValidator using disparity produced by StereoSGBM.
    """

    left_frame, right_frame = _load_stereo_frames()

    print(f"Left image:  {left_frame.filename}")
    print(f"Right image: {right_frame.filename}")

    disparity = _compute_disparity(
        left_frame.image,
        right_frame.image,
    )

    validator = StereoValidator()

    valid_mask = validator.compute_valid_mask(
        disparity
    )

    if valid_mask.shape != disparity.shape:
        raise RuntimeError(
            "StereoValidator returned a mask with an invalid shape."
        )

    valid_pixels = np.count_nonzero(valid_mask)
    invalid_pixels = disparity.size - valid_pixels

    if valid_pixels == 0:
        raise RuntimeError(
            "StereoValidator found no valid disparity values."
        )

    print()
    print("=" * 90)
    print("STEREO DISPARITY VALIDATION")
    print("=" * 90)

    print()
    print("Validation result:")

    print(
        f"  Valid pixels:        "
        f"{valid_pixels}"
    )

    print(
        f"  Invalid pixels:      "
        f"{invalid_pixels}"
    )

    print(
        f"  Valid percentage:    "
        f"{valid_pixels / disparity.size * 100:.2f}%"
    )

    print(
        f"  Removed percentage:  "
        f"{invalid_pixels / disparity.size * 100:.2f}%"
    )

def test_depth_reconstruction():
    """
    Tests depth reconstruction using the disparity produced
    by StereoSGBM on the Middlebury Plastic stereo dataset.

    The Middlebury 2006 dataset provides:

    - focal length: 3740 pixels
    - baseline: 160 mm
    - dmin: 280 pixels

    Therefore, the resulting depth is expressed in millimeters.
    """

    left_frame, right_frame = _load_stereo_frames()

    print(f"Left image:  {left_frame.filename}")
    print(f"Right image: {right_frame.filename}")

    # ------------------------------------------------------------------
    # Compute and validate disparity.
    # ------------------------------------------------------------------

    disparity = _compute_disparity(
        left_frame.image,
        right_frame.image,
    )

    validator = StereoValidator()

    valid_disparity = validator.compute_valid_mask(
        disparity
    )

    if not np.any(valid_disparity):
        raise RuntimeError(
            "StereoValidator found no valid disparity values."
        )

    # ------------------------------------------------------------------
    # Apply Middlebury disparity offset.
    #
    # The third-size Middlebury images are cropped.
    # dmin must therefore be added before depth reconstruction.
    # ------------------------------------------------------------------

    dmin = 280.0

    reconstruction_disparity = (
        disparity + dmin
    )

    # ------------------------------------------------------------------
    # Compute depth.
    # ------------------------------------------------------------------

    depth_reconstructor = DepthReconstructor(
        focal_length=3740.0,
        baseline=160.0,
    )

    depth = depth_reconstructor.compute(
        reconstruction_disparity
    )

    valid_depth = (
        np.isfinite(depth)
        & (depth > 0)
        & valid_disparity
    )

    if not np.any(valid_depth):
        raise RuntimeError(
            "Depth reconstruction produced no valid "
            "depth values."
        )

    # ------------------------------------------------------------------
    # Print disparity statistics.
    # ------------------------------------------------------------------

    valid_disparities = disparity[
        valid_disparity
    ]

    valid_reconstruction_disparities = (
        reconstruction_disparity[valid_disparity]
    )

    print()
    print("=" * 90)
    print("STEREO DISPARITY")
    print("=" * 90)

    print(
        f"  Shape:               "
        f"{disparity.shape}"
    )

    print(
        f"  Valid pixels:        "
        f"{len(valid_disparities)}"
    )

    print(
        f"  Minimum:             "
        f"{valid_disparities.min():.2f} px"
    )

    print(
        f"  Maximum:             "
        f"{valid_disparities.max():.2f} px"
    )

    print(
        f"  Mean:                "
        f"{valid_disparities.mean():.2f} px"
    )

    print(
        f"  Median:              "
        f"{np.median(valid_disparities):.2f} px"
    )

    print()
    print("Middlebury reconstruction disparity:")

    print(
        f"  dmin:                "
        f"{dmin:.2f} px"
    )

    print(
        f"  Minimum:             "
        f"{valid_reconstruction_disparities.min():.2f} px"
    )

    print(
        f"  Maximum:             "
        f"{valid_reconstruction_disparities.max():.2f} px"
    )

    print(
        f"  Mean:                "
        f"{valid_reconstruction_disparities.mean():.2f} px"
    )

    print(
        f"  Median:              "
        f"{np.median(valid_reconstruction_disparities):.2f} px"
    )

    # ------------------------------------------------------------------
    # Print depth statistics.
    # ------------------------------------------------------------------

    valid_depth_values = depth[
        valid_depth
    ]

    print()
    print("=" * 90)
    print("DEPTH")
    print("=" * 90)

    print(
        f"  Shape:               "
        f"{depth.shape}"
    )

    print(
        f"  Valid pixels:        "
        f"{len(valid_depth_values)}"
    )

    print(
        f"  Minimum:             "
        f"{valid_depth_values.min():.2f} mm"
    )

    print(
        f"  Maximum:             "
        f"{valid_depth_values.max():.2f} mm"
    )

    print(
        f"  Mean:                "
        f"{valid_depth_values.mean():.2f} mm"
    )

    print(
        f"  Median:              "
        f"{np.median(valid_depth_values):.2f} mm"
    )

    # ------------------------------------------------------------------
    # Visualize disparity.
    # ------------------------------------------------------------------

    disparity_display = np.zeros_like(
        disparity,
        dtype=np.float32,
    )

    disparity_display[valid_disparity] = (
        disparity[valid_disparity]
    )

    disparity_visualization = cv2.normalize(
        disparity_display,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    # ------------------------------------------------------------------
    # Visualize depth.
    # ------------------------------------------------------------------

    depth_values = depth[
        valid_depth
    ]

    depth_min = depth_values.min()
    depth_max = depth_values.max()

    depth_visualization = np.zeros_like(
        depth,
        dtype=np.uint8,
    )

    if depth_max > depth_min:

        depth_visualization[valid_depth] = (
            (
                (
                    depth[valid_depth]
                    - depth_min
                )
                / (depth_max - depth_min)
            )
            * 255.0
        ).astype(np.uint8)

    # ------------------------------------------------------------------
    # Print visualization range.
    # ------------------------------------------------------------------

    print()
    print("=" * 90)
    print("DEPTH VISUALIZATION")
    print("=" * 90)

    print(
        f"  Minimum disparity:  "
        f"{reconstruction_disparity[valid_disparity].min():.2f} px"
    )

    print(
        f"  Maximum disparity:  "
        f"{reconstruction_disparity[valid_disparity].max():.2f} px"
    )

    print(
        f"  Minimum depth:       "
        f"{depth_min:.2f} mm"
    )

    print(
        f"  Maximum depth:       "
        f"{depth_max:.2f} mm"
    )

    print(
        f"  Valid pixels:        "
        f"{len(depth_values)}"
    )

    # ------------------------------------------------------------------
    # Show visualizations.
    # ------------------------------------------------------------------

    cv2.imshow(
        "Disparity",
        disparity_visualization,
    )

    cv2.imshow(
        "Depth",
        depth_visualization,
    )

    print()
    print("Press any key to close the visualizations.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
def test_stereo_reconstruction():
    """
    Tests the complete stereo reconstruction pipeline
    using the Middlebury Plastic stereo dataset.

    The pipeline consists of:

        StereoSGBM
            -> disparity
            -> disparity validation
            -> 3D reconstruction
            -> point cloud visualization
    """

    left_frame, right_frame = _load_stereo_frames()

    print(f"Left image:  {left_frame.filename}")
    print(f"Right image: {right_frame.filename}")

    # ------------------------------------------------------------------
    # Compute disparity.
    # ------------------------------------------------------------------

    disparity = _compute_disparity(
        left_frame.image,
        right_frame.image,
    )

    # ------------------------------------------------------------------
    # Validate disparity.
    # ------------------------------------------------------------------

    validator = StereoValidator()

    valid_mask = validator.compute_valid_mask(
        disparity
    )

    if not np.any(valid_mask):
        raise RuntimeError(
            "StereoValidator found no valid disparity values."
        )

    # ------------------------------------------------------------------
    # Add Middlebury disparity offset.
    #
    # The third-size Middlebury images are cropped.
    # dmin must therefore be added before reconstruction.
    # ------------------------------------------------------------------

    dmin = 280.0

    reconstruction_disparity = (
        disparity + dmin
    )

    # ------------------------------------------------------------------
    # Reconstruct point cloud.
    # ------------------------------------------------------------------

    reconstructor = StereoReconstructor(
        reprojection_matrix=REPROJECTION_MATRIX,
    )

    point_cloud = reconstructor.reconstruct(
        disparity=reconstruction_disparity,
        image=left_frame.image,
        valid_mask=valid_mask,
    )

    # ------------------------------------------------------------------
    # Print reconstruction statistics.
    # ------------------------------------------------------------------

    print()
    print("=" * 90)
    print("STEREO RECONSTRUCTION")
    print("=" * 90)

    print(
        f"  Input disparity shape: "
        f"{disparity.shape}"
    )

    print(
        f"  Valid disparity pixels: "
        f"{np.count_nonzero(valid_mask)}"
    )

    print(
        f"  Reconstructed points:   "
        f"{len(point_cloud.points)}"
    )

    print(
        f"  Point dimensions:       "
        f"{point_cloud.points.shape}"
    )

    if point_cloud.colors is not None:

        print(
            f"  Point colors:           "
            f"{point_cloud.colors.shape}"
        )

    # ------------------------------------------------------------------
    # Show point cloud.
    # ------------------------------------------------------------------

    Visualizer.show_point_cloud(
        point_cloud
    )

def test_igev_disparity():
    """
    Tests IGEV disparity on the Middlebury Plastic
    stereo dataset against the ground-truth disparity.
    """

    left_frame, right_frame = _load_stereo_frames()

    print(f"Left image:  {left_frame.filename}")
    print(f"Right image: {right_frame.filename}")

    # ------------------------------------------------------------------
    # Load ground truth.
    # ------------------------------------------------------------------

    disparity_path = DATASET_PATH / "disp1.png"

    ground_truth_raw = cv2.imread(
        str(disparity_path),
        cv2.IMREAD_UNCHANGED,
    )

    if ground_truth_raw is None:
        raise RuntimeError(
            f"Unable to load ground-truth disparity: "
            f"{disparity_path}"
        )

    # Middlebury third-size disparity maps use a scale factor of 3.
    ground_truth = (
        ground_truth_raw.astype(np.float32) / 3.0
    )

    valid_ground_truth = ground_truth_raw > 0

    # ------------------------------------------------------------------
    # Compute IGEV disparity.
    # ------------------------------------------------------------------

    matcher = IGEVMatcher(
        config_path=OPENSTEREO_PATH
        / "cfgs"
        / "igev"
        / "igev_sceneflow_amp.yaml",
        checkpoint_path=IGEV_CHECKPOINT,
    )

    disparity = matcher.compute(
        left_frame.image,
        right_frame.image,
    )

    # ------------------------------------------------------------------
    # Evaluate.
    # ------------------------------------------------------------------

    _evaluate_igev(
        disparity=disparity,
        ground_truth=ground_truth,
        valid_ground_truth=valid_ground_truth,
    )

    # ------------------------------------------------------------------
    # Visualize disparity.
    # ------------------------------------------------------------------

    disparity_vis = cv2.normalize(
        disparity,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )

    cv2.imshow("IGEV Disparity", disparity_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_igev_reconstruction():
    """
    Tests dense stereo reconstruction using IGEV through OpenStereo.

    The pipeline consists of:

        IGEV
            -> disparity
            -> disparity validation
            -> depth reconstruction
            -> 3D reconstruction
            -> point cloud visualization
    """

    left_frame, right_frame = _load_stereo_frames()

    print(f"Left image:  {left_frame.filename}")
    print(f"Right image: {right_frame.filename}")

    # ------------------------------------------------------------------
    # Compute disparity.
    # ------------------------------------------------------------------

    matcher = IGEVMatcher(
        config_path=OPENSTEREO_PATH
        / "cfgs"
        / "igev"
        / "igev_sceneflow_amp.yaml",
        checkpoint_path=IGEV_CHECKPOINT,
    )

    disparity = matcher.compute(
        left_frame.image,
        right_frame.image,
    )

    print()
    print("=" * 90)
    print("IGEV DISPARITY")
    print("=" * 90)

    print(
        f"  Shape:   {disparity.shape}"
    )

    print(
        f"  Min:     {disparity.min():.4f} px"
    )

    print(
        f"  Max:     {disparity.max():.4f} px"
    )

    print(
        f"  Mean:    {disparity.mean():.4f} px"
    )

    print(
        f"  Median:  {np.median(disparity):.4f} px"
    )

    # ------------------------------------------------------------------
    # Validate disparity.
    # ------------------------------------------------------------------

    validator = StereoValidator()

    valid_mask = validator.compute_valid_mask(
        disparity
    )

    valid_pixels = np.count_nonzero(
        valid_mask
    )

    total_pixels = valid_mask.size

    print()
    print("=" * 90)
    print("DISPARITY VALIDATION")
    print("=" * 90)

    print(
        f"  Valid pixels:     {valid_pixels}"
    )

    print(
        f"  Invalid pixels:   "
        f"{total_pixels - valid_pixels}"
    )

    print(
        f"  Valid percentage: "
        f"{valid_pixels / total_pixels * 100:.2f}%"
    )

    # ------------------------------------------------------------------
    # Reconstruct depth.
    #
    # Middlebury Plastic ThirdSize:
    #
    # Focal length = 3740 / 3 px
    # Baseline     = 160 mm
    # ------------------------------------------------------------------

    depth_reconstructor = DepthReconstructor(
        focal_length=FOCAL_LENGTH,
        baseline=BASELINE_MM,
    )

    depth = depth_reconstructor.compute(
        disparity
    )

    valid_depth = depth[valid_mask]

    print()
    print("=" * 90)
    print("DEPTH RECONSTRUCTION")
    print("=" * 90)

    print(
        f"  Shape:   {depth.shape}"
    )

    print(
        f"  Min:     {valid_depth.min():.2f} mm"
    )

    print(
        f"  Max:     {valid_depth.max():.2f} mm"
    )

    print(
        f"  Mean:    {valid_depth.mean():.2f} mm"
    )

    print(
        f"  Median:  {np.median(valid_depth):.2f} mm"
    )

    print("  Unit:    mm")

    # ------------------------------------------------------------------
    # Reconstruct point cloud.
    # ------------------------------------------------------------------

    reconstructor = StereoReconstructor(
        reprojection_matrix=REPROJECTION_MATRIX,
    )

    point_cloud = reconstructor.reconstruct(
        disparity=disparity,
        image=left_frame.image,
        valid_mask=valid_mask,
    )

    # ------------------------------------------------------------------
    # Print reconstruction statistics.
    # ------------------------------------------------------------------

    print()
    print("=" * 90)
    print("IGEV STEREO RECONSTRUCTION")
    print("=" * 90)

    print(
        f"  Input disparity shape: "
        f"{disparity.shape}"
    )

    print(
        f"  Valid disparity pixels: "
        f"{valid_pixels}"
    )

    print(
        f"  Reconstructed points:   "
        f"{len(point_cloud.points)}"
    )

    print(
        f"  Point dimensions:       "
        f"{point_cloud.points.shape}"
    )

    if point_cloud.colors is not None:
        print(
            f"  Point colors:           "
            f"{point_cloud.colors.shape}"
        )

    # ------------------------------------------------------------------
    # Show point cloud.
    # ------------------------------------------------------------------

    Visualizer.show_point_cloud(
        point_cloud
    )

# Helpers

def _compute_disparity(
    left_image: np.ndarray,
    right_image: np.ndarray,
    block_size: int = 7,
) -> np.ndarray:
    """
    Computes stereo disparity using StereoSGBM.
    """

    matcher = StereoMatcher(
        min_disparity=0,
        num_disparities=128,
        block_size=block_size,
    )

    return matcher.compute(
        left_image,
        right_image,
    )

def _load_stereo_frames():
    """
    Loads the left and right frames from the Middlebury dataset.
    """

    camera = Camera(DATASET_PATH)

    frames = camera.load_frames()

    if len(frames) < 4:
        raise RuntimeError(
            "The Middlebury Plastic dataset must contain "
            "disp1.png, disp5.png, view1.png and view5.png."
        )

    return frames[2], frames[3]


def _evaluate_stereo(
    left_image: np.ndarray,
    right_image: np.ndarray,
    ground_truth: np.ndarray,
    valid_ground_truth: np.ndarray,
    block_size: int,
):
    """
    Runs StereoSGBM and evaluates the resulting disparity
    against the Middlebury ground truth.
    """

    disparity = _compute_disparity(
        left_image,
        right_image,
        block_size=block_size,
    )

    valid_disparity = disparity > 0

    valid_comparison = (
        valid_ground_truth
        & valid_disparity
    )

    predicted = disparity[valid_comparison]
    expected = ground_truth[valid_comparison]

    if len(predicted) == 0:
        raise RuntimeError(
            "There are no pixels where both StereoSGBM "
            "and the ground truth provide valid disparities."
        )

    absolute_error = np.abs(
        predicted - expected
    )

    signed_error = (
        predicted - expected
    )

    error_map = np.zeros_like(
        disparity,
        dtype=np.float32,
    )

    error_map[valid_comparison] = absolute_error

    print()
    print("=" * 90)
    print(f"BLOCK SIZE: {block_size}")
    print("=" * 90)

    print()
    print("StereoSGBM:")
    print(f"  Disparity shape:    {disparity.shape}")
    print(
        f"  Minimum disparity:  "
        f"{predicted.min():.2f}"
    )
    print(
        f"  Maximum disparity:  "
        f"{predicted.max():.2f}"
    )
    print(
        f"  Mean disparity:     "
        f"{predicted.mean():.2f}"
    )
    print(
        f"  Median disparity:   "
        f"{np.median(predicted):.2f}"
    )

    print()
    print("Ground truth:")
    print(
        f"  Minimum disparity:  "
        f"{expected.min():.2f}"
    )
    print(
        f"  Maximum disparity:  "
        f"{expected.max():.2f}"
    )
    print(
        f"  Mean disparity:     "
        f"{expected.mean():.2f}"
    )
    print(
        f"  Median disparity:   "
        f"{np.median(expected):.2f}"
    )

    print()
    print("Pixel-wise comparison:")
    print(
        f"  Valid comparison pixels: "
        f"{len(predicted)}"
    )
    print(
        f"  Mean absolute error:      "
        f"{absolute_error.mean():.2f} px"
    )
    print(
        f"  Median absolute error:    "
        f"{np.median(absolute_error):.2f} px"
    )
    print(
        f"  Maximum absolute error:   "
        f"{absolute_error.max():.2f} px"
    )
    print(
        f"  Mean signed error:        "
        f"{signed_error.mean():.2f} px"
    )
    print(
        f"  Median signed error:      "
        f"{np.median(signed_error):.2f} px"
    )

    print(
        f"  Absolute error > 1 px:    "
        f"{np.mean(absolute_error > 1.0) * 100:.2f}%"
    )
    print(
        f"  Absolute error > 2 px:    "
        f"{np.mean(absolute_error > 2.0) * 100:.2f}%"
    )
    print(
        f"  Absolute error > 5 px:    "
        f"{np.mean(absolute_error > 5.0) * 100:.2f}%"
    )

def _evaluate_igev(
    disparity: np.ndarray,
    ground_truth: np.ndarray,
    valid_ground_truth: np.ndarray,
):
    """
    Evaluates IGEV disparity against the Middlebury
    ground-truth disparity.
    """

    valid_disparity = (
        np.isfinite(disparity)
        & (disparity > 0)
    )

    valid_comparison = (
        valid_ground_truth
        & valid_disparity
    )

    predicted = disparity[valid_comparison]
    expected = ground_truth[valid_comparison]

    if len(predicted) == 0:
        raise RuntimeError(
            "There are no pixels where both IGEV "
            "and the ground truth provide valid disparities."
        )

    absolute_error = np.abs(
        predicted - expected
    )

    signed_error = (
        predicted - expected
    )

    print()
    print("=" * 90)
    print("IGEV DISPARITY EVALUATION")
    print("=" * 90)

    print()
    print("IGEV:")
    print(f"  Disparity shape:    {disparity.shape}")
    print(
        f"  Minimum disparity:  "
        f"{predicted.min():.2f}"
    )
    print(
        f"  Maximum disparity:  "
        f"{predicted.max():.2f}"
    )
    print(
        f"  Mean disparity:     "
        f"{predicted.mean():.2f}"
    )
    print(
        f"  Median disparity:   "
        f"{np.median(predicted):.2f}"
    )

    print()
    print("Ground truth:")
    print(
        f"  Minimum disparity:  "
        f"{expected.min():.2f}"
    )
    print(
        f"  Maximum disparity:  "
        f"{expected.max():.2f}"
    )
    print(
        f"  Mean disparity:     "
        f"{expected.mean():.2f}"
    )
    print(
        f"  Median disparity:   "
        f"{np.median(expected):.2f}"
    )

    print()
    print("Pixel-wise comparison:")
    print(
        f"  Valid comparison pixels: "
        f"{len(predicted)}"
    )
    print(
        f"  Mean absolute error:      "
        f"{absolute_error.mean():.2f} px"
    )
    print(
        f"  Median absolute error:    "
        f"{np.median(absolute_error):.2f} px"
    )
    print(
        f"  Maximum absolute error:   "
        f"{absolute_error.max():.2f} px"
    )
    print(
        f"  Mean signed error:        "
        f"{signed_error.mean():.2f} px"
    )
    print(
        f"  Median signed error:      "
        f"{np.median(signed_error):.2f} px"
    )

    print(
        f"  Absolute error > 1 px:    "
        f"{np.mean(absolute_error > 1.0) * 100:.2f}%"
    )
    print(
        f"  Absolute error > 2 px:    "
        f"{np.mean(absolute_error > 2.0) * 100:.2f}%"
    )
    print(
        f"  Absolute error > 5 px:    "
        f"{np.mean(absolute_error > 5.0) * 100:.2f}%"
    )